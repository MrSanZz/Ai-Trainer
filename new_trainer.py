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

# Pastikan nltk telah mengunduh resource yang diperlukan
nltk.download('punkt')
nltk.download('punkt_tab')

# Set up logging for monitoring training progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NeuralChat:
    def __init__(self, model_file='ai_model.json', softmax_temperature=0.7, weight_factor=1.2,
                 top_k=20, top_p=0.9, personality_bias=None, context_window=50,
                 response_length_factor=1.0):
        """
        personality_bias: Optional dictionary untuk mengatur kecenderungan kata tertentu.
        context_window: Jumlah token terakhir dari riwayat percakapan yang dipertimbangkan untuk prompt.
        response_length_factor: Faktor pengali untuk panjang respons (misal: 1.0 = default, >1.0 = lebih panjang).
        """
        self.model_file = model_file
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.ngram4 = defaultdict(Counter)  # Opsional: n-gram 4 untuk konteks lebih panjang
        self.total_trigrams = 0
        self.record_counter = 0
        self.softmax_temperature = softmax_temperature
        self.weight_factor = weight_factor
        self.top_k = top_k
        self.top_p = top_p
        self.personality_bias = personality_bias if personality_bias is not None else {}
        self.context_window = context_window
        self.response_length_factor = response_length_factor
        self.conversation_history = []  # Menyimpan riwayat pesan dalam format string
        self.lock = threading.Lock()
        ###############################
        self.word2idx = {'<unk>': 0}
        self.idx2word = {0: '<unk>'}
        self.vocab_size = 10000  # Bisa diupdate secara dinamis
        self.embedding_dim = 128
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        ###############################
        self.load_model()

    def train_with_embeddings(self, sentence):
        tokens = word_tokenize(sentence.lower())
        if not tokens:
            return

        # Update vocabulary secara dinamis dengan batas vocab_size
        for token in tokens:
            if token not in self.word2idx:
                if len(self.word2idx) < self.vocab_size:
                    idx = len(self.word2idx)
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token
                else:
                    # Jika sudah melebihi batas, token akan dianggap <unk>
                    continue

        # Konversi token ke indeks, token yang tidak ada akan mendapat indeks <unk>
        indices = [self.word2idx.get(token, 0) for token in tokens]
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        # Dapatkan embedding dari embedding layer
        embeddings = self.embedding_layer(indices_tensor)
        embedding_matrix = embeddings.detach().numpy()

        # Proses embeddings melalui linear layer
        linear_output = self.linear_layer(embeddings)

        # Lanjutkan update n-gram model menggunakan metode train yang sudah ada
        self.train(sentence)

        return embedding_matrix, linear_output

    def _serialize_counter(self, counter_obj):
        """Serialisasi Counter ke dict."""
        return dict(counter_obj)

    def _serialize_ngram(self, ngram_dict):
        """Serialize n-gram dictionary menggunakan delimiter aman."""
        serialized = {}
        for key, counter_obj in ngram_dict.items():
            key_str = "|||".join(key)
            serialized[key_str] = dict(counter_obj)
        return serialized

    def _deserialize_ngram(self, data):
        """Deserialize n-gram dictionary."""
        ngram = {}
        for key_str, value in data.items():
            key = tuple(key_str.split("|||"))
            ngram[key] = Counter(value)
        return ngram

    def softmax(self, logits):
        logits = np.array(logits) / self.softmax_temperature
        exp_logits = np.exp(logits - np.max(logits))  # Prevent overflow
        return exp_logits / np.sum(exp_logits)

    def top_k_top_p_sampling(self, counter):
        """
        Lakukan sampling dengan teknik top-k dan nucleus (top-p).
        Terapkan juga personality bias jika ada.
        """
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
        """
        Menentukan panjang respons berdasarkan token prompt dan context.
        Jika ditemukan kata kunci seperti "long" atau "short", panjang respons akan disesuaikan.
        Jika tidak, panjang respons diambil secara acak dalam rentang default.
        """
        # Set rentang default
        min_length, max_length = 35, 45

        # Cek apakah ada petunjuk dari token prompt
        if any(token in prompt_tokens for token in ['long', 'detailed', 'extended']):
            min_length, max_length = 45, 60
        elif any(token in prompt_tokens for token in ['short', 'brief']):
            min_length, max_length = 25, 35

        # Terapkan faktor pengali untuk panjang respons
        base_length = random.randint(min_length, max_length)
        desired_length = int(base_length * self.response_length_factor)
        return desired_length

    def train(self, sentence):
        tokens = word_tokenize(sentence.lower())
        if not tokens:
            return
        with self.lock:
            self.unigram.update(tokens)
            # Bigram update
            for i in range(len(tokens) - 1):
                self.bigram[tokens[i]].update([tokens[i + 1]])
            # Trigram update
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i + 1])
                self.trigram[key].update([tokens[i + 2]])
                self.total_trigrams += 1
            # 4-gram update untuk konteks lebih panjang
            for i in range(len(tokens) - 3):
                key = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.ngram4[key].update([tokens[i + 3]])
            self.record_counter += 1
        if self.record_counter % 1000 == 0:
            threading.Thread(target=self.safe_save).start()

    def generate_response(self, prompt_tokens):
        """
        Menghasilkan respons dengan mempertimbangkan konteks penuh dari prompt.
        Alih-alih hanya menggunakan token terakhir, fungsi ini mencoba mencari
        kecocokan n-gram dengan menggunakan 3, 2, atau 1 token terakhir dari prompt.
        """
        response = []

        # Coba gunakan n-gram terbesar yang mungkin dari prompt sebagai seed.
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

        # Jika tidak ditemukan seed yang sesuai, gunakan token acak dari model.
        if not seed_found:
            seed = random.choice(list(self.unigram.keys()))
            response.append(seed)

        # Tentukan panjang respons yang diinginkan berdasarkan prompt
        desired_length = self.determine_response_length(prompt_tokens)

        # Hasilkan token berikutnya dengan menggunakan n-gram yang lebih tinggi bila memungkinkan
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

    def update_history(self, user_message, ai_response):
        """
        Update riwayat percakapan dengan format:
        User: <pesan>
        AI: <respons>
        """
        self.conversation_history.append(f"User: {user_message}")
        self.conversation_history.append(f"AI: {ai_response}")
        # Jika riwayat terlalu panjang, batasi hanya token terakhir berdasarkan context_window
        combined = " ".join(self.conversation_history)
        tokens = word_tokenize(combined)
        if len(tokens) > self.context_window:
            tokens = tokens[-self.context_window:]
            self.conversation_history = [" ".join(tokens)]

    def get_context_prompt(self):
        """
        Gabungkan riwayat percakapan menjadi satu prompt untuk konteks.
        Pastikan seluruh riwayat percakapan dimasukkan dengan baik tanpa ada pemotongan.
        """
        return " ".join(self.conversation_history)

    def chat(self, message):
        """
        Menggabungkan pesan terbaru dengan riwayat percakapan untuk digunakan sebagai prompt.
        Model akan belajar dari pesan baru sekaligus menggunakan konteks sebelumnya.
        """
        context_prompt = self.get_context_prompt()
        full_prompt = f"{context_prompt} {message}" if context_prompt else message
        self.train(full_prompt)
        tokens = word_tokenize(full_prompt.lower())
        response = self.generate_response(tokens)
        self.update_history(message, response)
        return response

    def test(self, user_input):
        """
        Dalam mode test, AI belajar dari input pengguna dan langsung memberikan respons.
        """
        response = self.chat(user_input)
        print(f"AI: {response}")

    def safe_save(self):
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
            logging.info("Model saved successfully.")

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
    """
    Menyimpan model dengan aman sebelum keluar ketika terjadi interupsi.
    """
    logging.info("Saving model before exiting...")
    chatbot.safe_save()
    logging.info("Model saved successfully!.")
    exit(0)

def load_external_data():
    from datasets import load_dataset
    from concurrent.futures import ThreadPoolExecutor
    chatbot = NeuralChat()
    signal.signal(signal.SIGINT, lambda s, f: handle_interrupt(s, f, chatbot))
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        # Batasi jumlah thread agar tidak membuat sistem overload
        with ThreadPoolExecutor(max_workers=8) as executor:
            for idx, row in enumerate(dataset):
                try:
                    text_data = row.get("text", "")
                    if text_data:
                        # Pastikan args berupa tuple, dan submit task ke executor
                        executor.submit(chatbot.train_with_embeddings, text_data)
                except Exception as inner_e:
                    logging.warning(f"Skipping row {idx} due to error: {inner_e}")
                    continue

                if idx % 1000 == 0 and idx > 0:
                    size_kb = os.path.getsize(chatbot.model_file) / 1024 if os.path.exists(chatbot.model_file) else 0
                    print(f"Processed {idx} records, model size: {size_kb:.2f} KB", end='\r')
                time.sleep(0.005)
    except Exception as e:
        logging.error(f"Failed to load dataset OpenAssistant: {str(e)}")
    return chatbot

def load_test_mode():
    chatbot = NeuralChat()  # Buat satu instance
    print("Entering chat mode. Type 'exit' to interrupt.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot.chat(user_input)  # Gunakan instance yang sama
        print(f"AI: {response}")

def validate_training():
    """
    Validasi model dengan menguji beberapa input contoh.
    """
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
    """
    Program utama untuk memilih mode.
    """
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
