import json
import random
import os
import time
import signal
import threading
import logging
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter, defaultdict
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from torch.optim.lr_scheduler import LambdaLR
import math, re
locker = False

nltk.download('punkt')
nltk.download('punkt_tab')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_linear_schedule_with_warmup(optimizer, warmup_steps=100, total_steps=1000):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

def is_valid(text):
    if not text or len(text) < 10:
        return False
    if any(c in text for c in ['http', '@', '#']):
        return False
    if sum(1 for c in text if c.isupper()) > 30:
        return False
    if len(text.split()) > 100:
        return False
    if re.search(r'[^a-zA-Z0-9\s.,!?\'\"-]', text):
        return False
    return True

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def check(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class TextDataset(Dataset):
    """ambil field terus bersihin data g guna."""
    def __init__(self, dataset, max_samples=None):
        def is_valid(text):
            if not text or len(text) < 10:
                return False
            if any(c in text for c in ['http', '@', '#']):
                return False
            if sum(1 for c in text if c.isupper()) > 30:
                return False
            if len(text.split()) > 100:
                return False
            return True

        cleaned = [row["text"] for row in dataset if "text" in row and is_valid(row["text"])]
        self.samples = cleaned[:max_samples] if max_samples else cleaned

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def log_gradient_info(loss, grad_norm, count):
    if count == 0:
        return {"loss": 0, "grad_norm": 0}
    return {
        "loss": loss / count,
        "grad_norm": grad_norm / count
    }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class NeuralChat(nn.Module):
    def __init__(self, 
                 model_file='ai_model.json',
                 softmax_temperature=0.025,
                 weight_factor=2.4,
                 top_k=20,
                 top_p=1.0,
                 personality_bias=None,
                 context_window=50,
                 response_length_factor=1.0,
                 embedding_dim=256,  # Diperbesar
                 hidden_dim=512,     # Ditambahkan
                 n_layers=4,         # Ditambahkan
                 init_vocab_size=10000,  # Diperbesar
                 sentiment_model_file='sentiment_model.pkl',
                 dropout=0.2):       # Ditambahkan
        super(NeuralChat, self).__init__()
        self.model_file = model_file
        self.softmax_temperature = softmax_temperature
        self.weight_factor = weight_factor
        self.top_k = top_k
        self.top_p = top_p
        self.personality_bias = personality_bias or {}
        self.context_window = context_window
        self.response_length_factor = response_length_factor
        self.sentiment_model_file = sentiment_model_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.ngram4 = defaultdict(Counter)
        self.total_trigrams = 0
        self.record_counter = 0

        self.embedding_dim = embedding_dim
        self.vocab_size = init_vocab_size
        self.word2idx = {'<unk>': 0}
        self.idx2word = {0: '<unk>'}
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)  # Ditambahkan
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_layers
        )
        self.rnn = nn.LSTM(self.embedding_dim, self.embedding_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.vocab_size)
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
        self.optimizer = optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.lock = threading.Lock()

        self.load_model()
        self.reprocess_model_data()

        self.sentiment_model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.load_sentiment_model()

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

    def update_vocab(self, tokens):
        new_token_added = False
        for token in tokens:
            if not token.isalpha() or len(token) > 25:
                continue
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                new_token_added = True
        if new_token_added:
            logging.debug(f"[VOCAB] New vocab size: {len(self.word2idx)}")
            old_weight = self.embedding.weight.data
            new_vocab_size = len(self.word2idx)
            new_embedding = nn.Embedding(new_vocab_size, self.embedding_dim)
            new_embedding.weight.data[:old_weight.size(0)] = old_weight
            self.embedding = new_embedding

            old_fc_weight = self.fc[-1].weight.data
            new_fc = nn.Linear(self.embedding_dim, new_vocab_size)
            new_fc.weight.data[:old_fc_weight.size(0)] = old_fc_weight
            self.fc[-1] = new_fc
            self.vocab_size = new_vocab_size

    def train_batch(self, sentences):
        self.train()
        total_loss = 0
        count = 0
        total_grad_norm = 0.0
        acc_steps = 4
        
        for sentence in sentences:
            tokens = [t for t in word_tokenize(sentence.lower()) if t.isalpha()]
            if len(tokens) < 2 or len(tokens) > 150:
                continue
            self.update_vocab(tokens)
            idx = torch.tensor([self.word2idx.get(t, 0) for t in tokens], 
                            dtype=torch.long, device=self.device)
            
            if len(idx) < 2:
                continue

            self.update_ngram_stats(" ".join(tokens))
            src = self.embedding(idx[:-1].unsqueeze(0))
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src)
            logits = self.fc(output.squeeze(0))
            
            loss = self.criterion(logits, idx[1:]) / acc_steps
            loss.backward()
            total_loss += loss.item() * acc_steps
            count += 1
            
            if count % acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                total_grad_norm += grad_norm.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return total_loss, count, total_grad_norm

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
                    "personality_bias": self.personality_bias,
                    "word2idx": self.word2idx,
                    "idx2word": self.idx2word
                }
                with open(self.model_file, 'w') as f:
                    json.dump(data, f, indent=4)
                logging.info("Model saved successfully.")
        else:
            return

    def load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, 'r') as f:
                try:
                    model_data = json.load(f)
                    self.unigram = Counter(model_data.get("unigram", {}))
                    self.bigram = defaultdict(Counter, {k: Counter(v)
                                                         for k, v in model_data.get("bigram", {}).items()})
                    self.trigram = defaultdict(Counter, self._deserialize_ngram(model_data.get("trigram", {})))
                    self.ngram4 = defaultdict(Counter, self._deserialize_ngram(model_data.get("ngram4", {})))
                    self.total_trigrams = model_data.get("total_trigrams", 0)
                    self.record_counter = model_data.get("record_counter", 0)
                    self.softmax_temperature = model_data.get("softmax_temperature", 0.7)
                    self.weight_factor = model_data.get("weight_factor", 1.2)
                    self.top_k = model_data.get("top_k", 20)
                    self.top_p = model_data.get("top_p", 0.9)
                    self.personality_bias = model_data.get("personality_bias", {})
                    self.word2idx = model_data.get("word2idx", self.word2idx)
                    self.idx2word = model_data.get("idx2word", self.idx2word)
                    self.vocab_size = len(self.word2idx)
                except Exception as e:
                    logging.warning(f"Model file corrupted. Error: {e}")
                    self.unigram = Counter()
                    self.bigram = defaultdict(Counter)
                    self.trigram = defaultdict(Counter)
                    self.ngram4 = defaultdict(Counter)
                    self.total_trigrams = 0
                    self.record_counter = 0

    def reprocess_model_data(self):
        if not os.path.exists(self.model_file):
            logging.warning("404 ai_model.json not found, cancelled.")
            return

        with open(self.model_file, 'r') as f:
            model_data = json.load(f)

        self.unigram = Counter(model_data.get("unigram", {}))
        self.bigram = defaultdict(Counter, {k: Counter(v)
                                             for k, v in model_data.get("bigram", {}).items()})
        self.trigram = defaultdict(Counter, self._deserialize_ngram(model_data.get("trigram", {})))
        self.ngram4 = defaultdict(Counter, self._deserialize_ngram(model_data.get("ngram4", {})))
        self.total_trigrams = model_data.get("total_trigrams", 0)
        self.record_counter = model_data.get("record_counter", 0)
        self.word2idx = model_data.get("word2idx", self.word2idx)
        self.idx2word = model_data.get("idx2word", self.idx2word)
        self.vocab_size = len(self.word2idx)
        self.embedding = nn.Sequential(
            nn.Embedding(self.vocab_size, self.embedding_dim),
            nn.Dropout(0.1)
        )
        self.fc[-1] = nn.Linear(self.embedding_dim, self.vocab_size)
        logging.info("Reprocessing model data finished: n-gram, vocabulary, and neural components has been updated.")

    def softmax(self, logits):
        logits = np.array(logits) / self.softmax_temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def top_k_top_p_sampling2(self, counter):
        if not counter:
            return '<unk>'  # fallback

        biased = {
            word: count * self.personality_bias.get(word, 1.0)
            for word, count in counter.items()
            if word in self.word2idx
        }

        if not biased:
            return '<unk>'

        words = list(biased.keys())
        counts = np.array(list(biased.values()), dtype=float)
        logits = np.log(counts * self.weight_factor + 1e-10)

        sorted_idx = np.argsort(logits)[::-1]
        sorted_words = [words[i] for i in sorted_idx][:self.top_k]
        sorted_logits = logits[sorted_idx][:self.top_k]

        probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
        cumulative = np.cumsum(probs)
        cutoff = cumulative <= self.top_p
        if not np.any(cutoff):
            cutoff[-1] = True

        final_words = [w for w, flag in zip(sorted_words, cutoff) if flag]
        final_probs = np.exp(sorted_logits[:len(final_words)])
        final_probs /= np.sum(final_probs)

        chosen = random.choices(final_words, weights=final_probs)[0]
        return chosen
    
    def get_context_prompt(self):
        return " ".join(self.conversation_history) if hasattr(self, "conversation_history") else ""

    def update_history(self, user_message, ai_response):
        if not hasattr(self, "conversation_history"):
            self.conversation_history = []
        self.conversation_history.append(f"User: {user_message}")
        self.conversation_history.append(f"AI: {ai_response}")
        combined = " ".join(self.conversation_history)
        tokens = word_tokenize(combined)
        if len(tokens) > self.context_window:
            tokens = tokens[-self.context_window:]
            self.conversation_history = [" ".join(tokens)]

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

    def top_k_top_p_sampling(self, logits, top_k=50, top_p=0.9):
        logits = np.asarray(logits)
        if logits.size == 0:
            raise ValueError("Logits is empty, cannot perform sampling.")

        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        cumulative_probs = np.cumsum(self.softmax(sorted_logits))
        cutoff = cumulative_probs <= top_p
        cutoff[:top_k] = True
        if cutoff.size > 0:
            cutoff[-1] = True
        else:
            raise ValueError("Cutoff array is empty after thresholding.")
        biased = {word: count * self.personality_bias.get(word, 1.0)
                 for word, count in logits.items()
                 if word in self.word2idx}
        words = list(biased.keys())
        counts = np.array(list(biased.values()), dtype=float)
        logits = np.log(counts * self.weight_factor + 1e-10)
        sorted_idx = np.argsort(logits)[::-1]
        sorted_words = [words[i] for i in sorted_idx][:self.top_k]
        sorted_logits = logits[sorted_idx][:self.top_k]
        probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
        cumulative = np.cumsum(probs)
        cutoff = cumulative <= self.top_p
        if not np.any(cutoff):
            cutoff[-1] = True
        final_words = [w for w, flag in zip(sorted_words, cutoff) if flag]
        final_probs = np.exp(sorted_logits[:len(final_words)])
        final_probs /= np.sum(final_probs)
        chosen = random.choices(final_words, weights=final_probs)[0]
        return chosen

    def determine_response_length(self, prompt_tokens):
        min_length, max_length = 35, 45
        if any(token in prompt_tokens for token in ['long', 'detailed', 'extended']):
            min_length, max_length = 45, 60
        elif any(token in prompt_tokens for token in ['short', 'brief']):
            min_length, max_length = 25, 35
        base = random.randint(min_length, max_length)
        return int(base * self.response_length_factor)

    def generate_response(self, prompt, max_length=None, temperature=None):
        self.eval()
        tokens = word_tokenize(prompt.lower())
        token_ids = [self.word2idx.get(t, 0) for t in tokens if t.isalpha()]
        
        if not token_ids:
            return "I don't understand that input."
            
        max_length = max_length or self.determine_response_length(tokens)
        temp = temperature or self.softmax_temperature
        
        with torch.no_grad():
            input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            for _ in range(max_length):
                emb = self.embedding(input_ids)
                emb = self.pos_encoder(emb)
                output = self.transformer_encoder(emb)
                logits = self.fc(output[:, -1, :])
                
                probs = torch.softmax(logits / temp, dim=-1)
                top_probs, top_indices = torch.topk(probs, self.top_k)
                
                if self.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[..., indices_to_remove] = 0
                    probs = probs / probs.sum()
                
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if self.idx2word.get(next_token.item(), '') in ['.', '!', '?']:
                    break
                    
        generated_ids = input_ids[0, len(token_ids):].tolist()
        return ' '.join([self.idx2word.get(idx, '') for idx in generated_ids if idx in self.idx2word])

    def load_sentiment_model(self):
        if os.path.exists(self.sentiment_model_file):
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

    def chat(self, message):
        sentiment = self.predict_sentiment(message)
        context_prompt = self.get_context_prompt()
        full_prompt = f"{context_prompt} {message}" if context_prompt else message
        self.update_ngram_stats(full_prompt)
        
        tokens = [t for t in word_tokenize(full_prompt.lower()) if t.isalpha()]
        if len(tokens) < 2:
            return "[Input terlalu pendek untuk dihitung loss-nya.]"

        idx = torch.tensor([self.word2idx.get(t, 0) for t in tokens], dtype=torch.long, device=self.device)
        self.embedding = self.embedding.to(self.device)
        emb = self.embedding(idx[:-1])
        out = self.fc(emb)
        loss = self.criterion(out, idx[1:])

        # Logging seperti training
        logging.warning(f"[TextDebug] Token len: {len(tokens)}, Loss: {loss.item():.2f}, Text: {message[:100]}")
        
        # Return ke user juga dalam bentuk string
        return f"[TextDebug] Token len: {len(tokens)}, Loss: {loss.item():.2f}, Text: {message[:100]}"

def build_vocab_from_dataset(dataset, model):
    filter_tokens = {'user', 'ai', 'user:', 'ai:', 'system'}
    for sentence in dataset:
        tokens = [t for t in word_tokenize(sentence.lower()) if t not in filter_tokens]
        for token in tokens:
            if token not in model.word2idx:
                idx = len(model.word2idx)
                model.word2idx[token] = idx
                model.idx2word[idx] = token
    model.vocab_size = len(model.word2idx)
    model.embedding = nn.Embedding(model.vocab_size, model.embedding_dim)
    model.fc = nn.Sequential(
        nn.Linear(model.embedding_dim, model.embedding_dim),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(model.embedding_dim, model.vocab_size)
    )

def re_train_model(epochs=5, batch_size=16, max_samples=150000):
    logging.info("Loading external dataset...")
    dataset_raw = load_dataset("OpenAssistant/oasst1", split="train")
    filtered_data = [row for row in dataset_raw if row.get("role") == "assistant" and isinstance(row.get("text"), str)]
    dataset = TextDataset(filtered_data, max_samples=max_samples)

    chatbot = NeuralChat()
    build_vocab_from_dataset(dataset, chatbot)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    early_stopper = EarlyStopping(patience=3)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        grad_norm = 0.0 
        start_time = time.time()

        chatbot.train()
        for i, batch_sentences in enumerate(train_loader):
            loss, count, g = chatbot.train_batch(batch_sentences)
            epoch_loss += loss
            batch_count += count
            grad_norm += g
            if (i + 1) % 10 == 0:
                metrics = log_gradient_info(epoch_loss, grad_norm, count)
                avg_loss = metrics["loss"]
                avg_grad_norm = metrics["grad_norm"]
                elapsed = time.time() - start_time
                print(json.dumps({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "learn_rate": chatbot.optimizer.param_groups[0]['lr'],
                    "grad_norm": avg_grad_norm,
                    "elapsed": elapsed
                }))
        chatbot.scheduler.step()
        torch.save(chatbot.state_dict(), f'checkpoint_epoch{epoch+1}.pt')

        chatbot.eval()
        with torch.no_grad():
            val_loss = 0
            val_count = 0
            for batch in val_loader:
                for sentence in batch:
                    tokens = word_tokenize(sentence.lower())
                    if len(tokens) < 2:
                        continue
                    idx = torch.tensor([chatbot.word2idx.get(t, 0) for t in tokens], dtype=torch.long, device=chatbot.device)
                    if len(idx) < 2:
                        continue
                    emb = chatbot.embedding(idx[:-1])
                    out = chatbot.fc(emb)
                    loss = chatbot.criterion(out, idx[1:])
                    val_loss += loss.item()
                    val_count += 1
            val_loss /= val_count if val_count else 1
            if early_stopper.check(val_loss):
                logging.info(f"Early stopping at epoch {epoch+1} with val_loss {val_loss:.4f}")
                break

    chatbot.safe_save()
    logging.info("Re-training finished, model saved.")

def handle_interrupt(signal_received, frame, chatbot):
    global locker
    logging.info("Interrupt accepted, saving model...")
    chatbot.safe_save()
    locker = True
    logging.info("Model saved!, exiting.")
    exit(0)

def main():
    mode = input("Enter mode [chat/retrain]: ").strip().lower()

    if mode == "retrain":
        try:
            epochs = int(input("Insert Epochs [5-10]: "))
        except ValueError:
            epochs = 5

        try:
            batch_size = int(input("Insert batch size [16, 32, 64, 128]: "))
        except ValueError:
            batch_size = 16

        try:
            max_samples = int(input("Insert total sample [84400-150000]: "))
        except ValueError:
            max_samples = 150000

        temp_model = NeuralChat()
        signal.signal(signal.SIGINT, lambda s, f: handle_interrupt(s, f, temp_model))
        re_train_model(epochs=epochs, batch_size=batch_size, max_samples=max_samples)

    elif mode == "chat":
        chatbot = NeuralChat()
        print("Chat mode activated. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat mode.")
                break
            response = chatbot.generate_response(user_input)
            print(f"AI: {response}")
    else:
        print("Invalid mode. Please type 'chat' or 'retrain'.")

if __name__ == "__main__":
    main()
