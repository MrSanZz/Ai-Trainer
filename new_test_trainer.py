import json
import random
import os
import time
import signal
import threading
import logging
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

locker = False

nltk.download('punkt')
nltk.download('punkt_tab')
texts_for_sentiment = []
labels_for_sentiment = []

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # hapus simbol asing
    text = text.lower().strip()
    return text

class NeuralChat:
    def __init__(self, model_file='ai_model.json', softmax_temperature=0.7, weight_factor=1.2,
                 top_k=20, top_p=0.9, personality_bias=None, context_window=50,
                 response_length_factor=1.0, sentiment_model_file='sentiment_model.pkl'):
        """
        personality_bias: Optional dictionary untuk mengatur kecenderungan kata tertentu.
        context_window: Jumlah token terakhir dari riwayat percakapan yang dipertimbangkan untuk prompt.
        response_length_factor: Faktor pengali untuk panjang respons (misal: 1.0 = default, >1.0 = lebih panjang).
        """
        self.model_file = model_file
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.ngram4 = defaultdict(Counter)
        self.total_trigrams = 0
        self.record_counter = 0
        self.softmax_temperature = softmax_temperature
        self.weight_factor = weight_factor
        self.top_k = top_k
        self.top_p = top_p
        self.personality_bias = personality_bias if personality_bias is not None else {}
        self.context_window = context_window
        self.response_length_factor = response_length_factor
        self.conversation_history = []
        self.lock = threading.Lock()
        ###############################
        self.word2idx = {'<unk>': 0}
        self.idx2word = {0: '<unk>'}
        self.vocab_size = 10000
        self.embedding_dim = 128
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        ###############################
        self.sentiment_model_file = sentiment_model_file
        self.sentiment_model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)  # make TF-IDF buat representasi kata

        self.load_sentiment_model()
        ################################
        self.load_model()

    def load_sentiment_model(self):
        if os.path.exists(self.sentiment_model_file):
            import pickle
            with open(self.sentiment_model_file, 'rb') as f:
                self.sentiment_model = pickle.load(f)
            logging.info("Sentiment model loaded successfully!")
        else:
            logging.warning("Sentiment model file not found, model will not be available.")

    def train_sentiment_model(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        sentiment_model = LogisticRegression(max_iter=1000)
        sentiment_model.fit(X_train, y_train)

        y_pred = sentiment_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logging.info(f"Sentiment model training completed with accuracy: {accuracy:.4f}")

        with open(self.sentiment_model_file, 'wb') as f:
            import pickle
            pickle.dump(sentiment_model, f)
            logging.info(f"Sentiment model saved to {self.sentiment_model_file}.")

        self.sentiment_model = sentiment_model

    def predict_sentiment(self, text):
        if self.sentiment_model:
            X = self.vectorizer.transform([text])
            prediction = self.sentiment_model.predict(X)
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            return sentiment
        return "Sentiment model not available."

    def train_with_embeddings(self, sentence, freeze_vocab=True):
        tokens = word_tokenize(sentence.lower())
        if len(tokens) < 2:
            return None, None

        filter_tokens = {'user', 'ai', 'user:', 'ai:', 'system'}
        tokens = [t for t in tokens if t not in filter_tokens]

        if not freeze_vocab:
            for token in tokens:
                if token not in self.word2idx and len(self.word2idx) < self.vocab_size:
                    idx = len(self.word2idx)
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token
            new_vocab_size = len(self.word2idx)
            if new_vocab_size != self.vocab_size:
                self.vocab_size = new_vocab_size
                self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
                self.linear_layer = nn.Linear(self.embedding_dim, self.embedding_dim)

        indices = [self.word2idx.get(t, 0) for t in tokens]
        idx_tensor = torch.tensor(indices, dtype=torch.long)

        max_len = 128
        if len(idx_tensor) > max_len:
            idx_tensor = idx_tensor[:max_len]

        embeddings = self.embedding_layer(idx_tensor)
        linear_output = self.linear_layer(embeddings)

        self.update_ngram_stats(" ".join(tokens))

        return embeddings.detach().cpu().numpy(), linear_output.detach().cpu().numpy()

    def _serialize_counter(self, counter_obj):
        return dict(counter_obj)

    def _serialize_ngram(self, ngram_dict):
        serialized = {}
        for key, counter_obj in ngram_dict.items():
            key_str = "|||".join(key)
            serialized[key_str] = dict(counter_obj)
        return serialized

    def _deserialize_ngram(self, data):
        ngram = {}
        for key_str, value in data.items():
            key = tuple(key_str.split("|||"))
            ngram[key] = Counter(value)
        return ngram

    def softmax(self, logits):
        logits = np.array(logits) / self.softmax_temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def top_k_top_p_sampling(self, counter):
        biased_counter = {}
        for word, count in counter.items():
            bias = self.personality_bias.get(word, 1.0)
            biased_counter[word] = count * bias

        words = list(biased_counter.keys())
        counts = np.array(list(biased_counter.values()), dtype=float)
        logits = np.log(counts * self.weight_factor + 1e-10)
        sorted_idx = np.argsort(logits)[::-1]
        sorted_words = [words[i] for i in sorted_idx]
        sorted_logits = logits[sorted_idx]

        if self.top_k > 0:
            sorted_words = sorted_words[:self.top_k]
            sorted_logits = sorted_logits[:self.top_k]

        probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
        cumulative_probs = np.cumsum(probs)
        cutoff = cumulative_probs <= self.top_p
        if not np.any(cutoff):
            cutoff[-1] = True
        final_words = [w for w, flag in zip(sorted_words, cutoff) if flag]
        final_probs = np.exp(sorted_logits[:len(final_words)])
        final_probs = final_probs / np.sum(final_probs)
        chosen = random.choices(final_words, weights=final_probs)[0]
        return chosen

    def determine_response_length(self, prompt_tokens):
        min_length, max_length = 35, 45

        if any(token in prompt_tokens for token in ['long', 'detailed', 'extended']):
            min_length, max_length = 45, 60
        elif any(token in prompt_tokens for token in ['short', 'brief']):
            min_length, max_length = 25, 35

        base_length = random.randint(min_length, max_length)
        desired_length = int(base_length * self.response_length_factor)
        return desired_length

    def train(self, sentence):
        tokens = word_tokenize(sentence.lower())
        if not tokens:
            return
        with self.lock:
            self.unigram.update(tokens)
            for i in range(len(tokens) - 1):
                self.bigram[tokens[i]].update([tokens[i + 1]])
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i + 1])
                self.trigram[key].update([tokens[i + 2]])
                self.total_trigrams += 1
            for i in range(len(tokens) - 3):
                key = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.ngram4[key].update([tokens[i + 3]])
            self.record_counter += 1
        if self.record_counter % 2500 == 0:
            threading.Thread(target=self.safe_save).start()

    def generate_response(self, prompt_tokens):
        response = []
        seed_found = False
        for n in [3, 2, 1]:
            if len(prompt_tokens) >= n:
                seed = tuple(prompt_tokens[-n:])
                if n == 3 and seed in self.ngram4 and sum(self.ngram4[seed].values()) > 0:
                    response.extend(list(seed))
                    next_token = self.top_k_top_p_sampling(self.ngram4[seed])
                    response.append(next_token)
                    seed_found = True
                    break
                elif n == 2 and seed in self.trigram and sum(self.trigram[seed].values()) > 0:
                    response.extend(list(seed))
                    next_token = self.top_k_top_p_sampling(self.trigram[seed])
                    response.append(next_token)
                    seed_found = True
                    break
                elif n == 1 and seed[0] in self.bigram and sum(self.bigram[seed[0]].values()) > 0:
                    response.append(seed[0])
                    next_token = self.top_k_top_p_sampling(self.bigram[seed[0]])
                    response.append(next_token)
                    seed_found = True
                    break

        if not seed_found:
            seed = random.choice(list(self.unigram.keys()))
            response.append(seed)
        desired_length = self.determine_response_length(prompt_tokens)

        while len(response) < desired_length:
            next_token = None
            if len(response) >= 3:
                key4 = tuple(response[-3:])
                if key4 in self.ngram4 and sum(self.ngram4[key4].values()) > 0:
                    next_token = self.top_k_top_p_sampling(self.ngram4[key4])
            if not next_token and len(response) >= 2:
                key3 = (response[-2], response[-1])
                if key3 in self.trigram and sum(self.trigram[key3].values()) > 0:
                    next_token = self.top_k_top_p_sampling(self.trigram[key3])
            if not next_token and response[-1] in self.bigram and sum(self.bigram[response[-1]].values()) > 0:
                next_token = self.top_k_top_p_sampling(self.bigram[response[-1]])
            if not next_token:
                break
            response.append(next_token)
        return " ".join(response)
    
    def update_ngram_stats(self, sentence):
        tokens = word_tokenize(sentence.lower())
        if not tokens:
            return
        with self.lock:
            self.unigram.update(tokens)
            for i in range(len(tokens) - 1):
                self.bigram[tokens[i]].update([tokens[i + 1]])
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i + 1])
                self.trigram[key].update([tokens[i + 2]])
                self.total_trigrams += 1
            for i in range(len(tokens) - 3):
                key = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.ngram4[key].update([tokens[i + 3]])
            self.record_counter += 1
        if self.record_counter % 2500 == 0:
            threading.Thread(target=self.safe_save).start()

    def update_history(self, user_message, ai_response):
        self.conversation_history.append(f"User: {user_message}")
        self.conversation_history.append(f"AI: {ai_response}")
        combined = " ".join(self.conversation_history)
        tokens = word_tokenize(combined)
        if len(tokens) > self.context_window:
            tokens = tokens[-self.context_window:]
            self.conversation_history = [" ".join(tokens)]

    def get_context_prompt(self):
        return " ".join(self.conversation_history)

    def chat(self, message):
        #sentiment = self.predict_sentiment(message)
        context_prompt = self.get_context_prompt()
        full_prompt = f"{context_prompt} {message}" if context_prompt else message
        self.update_ngram_stats(full_prompt)
        tokens = word_tokenize(full_prompt.lower())
        response = self.generate_response(tokens)
        self.update_history(message, response)
        return response

    def test(self, user_input):
        response = self.chat(user_input)
        print(f"AI: {response}")

    def safe_save(self):
        if locker != True:
            with self.lock:
                data = {
                    "unigram": self._serialize_counter(self.unigram),
                    "bigram": {k: self._serialize_counter(v) for k, v in self.bigram.items()},
                    "trigram": self._serialize_ngram(self.trigram),
                    "ngram4": self._serialize_ngram(self.ngram4),
                    "total_trigrams": self.total_trigrams,
                    "record_counter": self.record_counter,
                    "softmax_temperature": self.softmax_temperature,
                    "weight_factor": self.weight_factor,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "personality_bias": self.personality_bias
                }
                with open(self.model_file, 'w') as f:
                    json.dump(data, f, indent=4)
                logging.info("Model saved successfully!.")
        else:
            return

    def load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, 'r') as f:
                try:
                    model_data = json.load(f)
                    self.unigram = Counter(model_data.get("unigram", {}))
                    bigram_data = {k: Counter(v) for k, v in model_data.get("bigram", {}).items()}
                    self.bigram = defaultdict(Counter, bigram_data)
                    trigram_data = self._deserialize_ngram(model_data.get("trigram", {}))
                    self.trigram = defaultdict(Counter, trigram_data)
                    ngram4_data = self._deserialize_ngram(model_data.get("ngram4", {}))
                    self.ngram4 = defaultdict(Counter, ngram4_data)
                    self.total_trigrams = model_data.get("total_trigrams", 0)
                    self.record_counter = model_data.get("record_counter", 0)
                    self.softmax_temperature = model_data.get("softmax_temperature", 0.7)
                    self.weight_factor = model_data.get("weight_factor", 1.2)
                    self.top_k = model_data.get("top_k", 20)
                    self.top_p = model_data.get("top_p", 0.9)
                    self.personality_bias = model_data.get("personality_bias", {})
                except (json.JSONDecodeError, SyntaxError, TypeError) as e:
                    logging.warning("Model file corrupted. Error: " + str(e))
                    self.unigram = Counter()
                    self.bigram = defaultdict(Counter)
                    self.trigram = defaultdict(Counter)
                    self.ngram4 = defaultdict(Counter)
                    self.total_trigrams = 0
                    self.record_counter = 0

def handle_interrupt(signal_received, frame, chatbot):
    global locker
    logging.info("Saving model before exiting...")
    chatbot.safe_save()
    locker = True
    logging.info("Model saved successfully!.")
    exit(0)

def build_vocab_from_dataset(dataset, model):
    filter_tokens = {'user', 'ai', 'user:', 'ai:', 'system'}
    for row in dataset:
        text = row.get("text")
        if not text:
            continue
        tokens = [t for t in word_tokenize(text.lower()) if t not in filter_tokens]
        for token in tokens:
            if token not in model.word2idx:
                idx = len(model.word2idx)
                model.word2idx[token] = idx
                model.idx2word[idx] = token
    model.vocab_size = len(model.word2idx)
    model.embedding_layer = nn.Embedding(model.vocab_size, model.embedding_dim)
    model.linear_layer = nn.Linear(model.embedding_dim, model.embedding_dim)

def load_external_data(max_samples=100000, batch_size=8):
    from datasets import load_dataset
    from concurrent.futures import ThreadPoolExecutor, as_completed

    chatbot = NeuralChat()
    signal.signal(signal.SIGINT, lambda s, f: handle_interrupt(s, f, chatbot))
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        dataset = [row for row in dataset if row.get("text")]
        if max_samples:
            dataset = dataset[:max_samples]

        build_vocab_from_dataset(dataset, chatbot)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(chatbot.train_with_embeddings, row["text"]): idx
                for idx, row in enumerate(dataset)
            }
            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                    text = dataset[futures[future]]["text"]
                    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha()]
                    token_ids = [chatbot.word2idx.get(t, 0) for t in tokens]

                    if len(token_ids) > 2:
                        max_len = 30
                        padded = token_ids[:max_len] + [0] * max(0, max_len - len(token_ids))
                        texts_for_sentiment.append(padded)
                        labels_for_sentiment.append(1 if "good" in text.lower() else 0)

                except Exception as inner_e:
                    logging.warning(f"Error in sample {futures[future]}: {inner_e}")
                    continue

                if i % 1000 == 0 and i > 0:
                    size_kb = os.path.getsize(chatbot.model_file) / 1024 if os.path.exists(chatbot.model_file) else 0
                    logging.info(f"Processed {i} records, model size: {size_kb:.2f} KB")
                time.sleep(0.001)

    except Exception as e:
        logging.error(f"Failed to load dataset OpenAssistant: {str(e)}")
    chatbot.train_sentiment_model(texts_for_sentiment, labels_for_sentiment)
    return chatbot

def load_test_mode():
    chatbot = NeuralChat()
    print("Entering chat mode. Type 'exit' to interrupt.")
    while True:
        user_input = clean_text(input("You: "))
        if user_input.lower() == "exit":
            break
        response = clean_text(chatbot.chat(user_input))
        print(f"AI: {response}")

def validate_training():
    chatbot = NeuralChat()
    sample_inputs = ["hello", "hi", "how are you", "what's up", "good morning"]
    logging.info("Starting model validation..")
    while True:
        for inp in sample_inputs:
            response = chatbot.chat(inp)
            print(f"Input: {inp} → AI: {response}")
        time.sleep(1)
        logging.info("Validation finished.")

def main():
    mode = input("Choose mode (Train/Test/Validate): ").strip().lower()
    if mode == "train":
        chatbot = load_external_data()
        logging.info("Training finished. Model saved successfully.")
    elif mode == "test":
        load_test_mode()
    elif mode == "validate":
        validate_training()
    else:
        logging.error("Not a valid mode, please choose 'Train', 'Test', or 'Validate' mode!.")

if __name__ == "__main__":
    main()
