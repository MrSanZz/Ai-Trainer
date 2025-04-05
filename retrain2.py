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
from concurrent.futures import ThreadPoolExecutor

# Pastikan resource NLTK sudah tersedia
nltk.download('punkt')
nltk.download('punkt_tab')

# Konfigurasi logging yang lebih detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class NeuralChat:
    def __init__(self,
                 model_file='ai_model.json',
                 softmax_temperature=0.7,
                 weight_factor=1.2,
                 top_k=20,
                 top_p=0.9,
                 personality_bias=None,
                 context_window=50,
                 response_length_factor=1.0,
                 embedding_dim=128,
                 vocab_size=10000):
        """
        Inisialisasi model NeuralChat dengan komponen n-gram, embedding, dan linear layer.
        Parameter:
          - model_file: Nama file untuk menyimpan/memuat model.
          - softmax_temperature, weight_factor, top_k, top_p: Parameter sampling.
          - personality_bias: Optional dictionary bias kata.
          - context_window: Jumlah token riwayat yang dipakai (meski untuk re-training tidak interaktif).
          - response_length_factor: Faktor pengali panjang respons.
          - embedding_dim: Dimensi embedding.
          - vocab_size: Ukuran maksimum vocabulary.
        """
        self.model_file = model_file
        self.softmax_temperature = softmax_temperature
        self.weight_factor = weight_factor
        self.top_k = top_k
        self.top_p = top_p
        self.personality_bias = personality_bias or {}
        self.context_window = context_window
        self.response_length_factor = response_length_factor

        # Inisialisasi struktur n-gram
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.ngram4 = defaultdict(Counter)
        self.total_trigrams = 0
        self.record_counter = 0

        # Komponen embedding & vocabulary
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.word2idx = {'<unk>': 0}
        self.idx2word = {0: '<unk>'}
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear_layer = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Variabel probabilitas (bigram & trigram)
        self.bigram_prob = {}
        self.trigram_prob = {}

        # Untuk thread-safe update model
        self.lock = threading.Lock()

        # Muat model jika file ada, lalu reprocess untuk update seluruh komponen
        self.load_model()
        self.reprocess_model_data()

    # --- Serialization Methods ---
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

    # --- Sampling Methods ---
    def softmax(self, logits):
        logits = np.array(logits) / self.softmax_temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def top_k_top_p_sampling(self, counter):
        # Terapkan bias ke counter
        biased = {word: count * self.personality_bias.get(word, 1.0)
                  for word, count in counter.items()}
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

    # --- Training ---
    def train(self, sentence):
        tokens = word_tokenize(sentence.lower())
        if not tokens:
            return
        with self.lock:
            self.unigram.update(tokens)
            # Update bigram
            for i in range(len(tokens) - 1):
                self.bigram[tokens[i]].update([tokens[i + 1]])
            # Update trigram
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i + 1])
                self.trigram[key].update([tokens[i + 2]])
                self.total_trigrams += 1
            # Update 4-gram
            for i in range(len(tokens) - 3):
                key = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.ngram4[key].update([tokens[i + 3]])
            self.record_counter += 1

        # Safe-save model setiap 500 record
        if self.record_counter % 500 == 0:
            threading.Thread(target=self.safe_save).start()

    # --- Save / Load & Reprocess ---
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
                "personality_bias": self.personality_bias,
                "word2idx": self.word2idx,
                "idx2word": self.idx2word
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
                except (json.JSONDecodeError, SyntaxError, TypeError) as e:
                    logging.warning(f"Model file corrupted. Error: {e}")
                    self.unigram = Counter()
                    self.bigram = defaultdict(Counter)
                    self.trigram = defaultdict(Counter)
                    self.ngram4 = defaultdict(Counter)
                    self.total_trigrams = 0
                    self.record_counter = 0

    def reprocess_model_data(self):
        """
        Proses ulang data dari file model untuk memperbarui:
          - Counters n-gram dan probabilitas bigram/trigram
          - Vocabulary serta re-inisialisasi embedding dan linear layer
        """
        if not os.path.exists(self.model_file):
            logging.warning("Model file tidak ditemukan. Reprocessing dibatalkan.")
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

        # Hitung ulang probabilitas untuk bigram
        self.bigram_prob = {
            token: {w: count / sum(counter.values())
                    for w, count in counter.items()}
            for token, counter in self.bigram.items()
        }
        # Hitung ulang probabilitas untuk trigram
        self.trigram_prob = {}
        for key, counter in self.trigram.items():
            total = sum(counter.values())
            self.trigram_prob[key] = {w: count / total for w, count in counter.items()} if total > 0 else {}

        # Perbarui vocabulary dan reinitialize embedding serta linear layer
        self.word2idx = model_data.get("word2idx", self.word2idx)
        self.idx2word = model_data.get("idx2word", self.idx2word)
        self.vocab_size = len(self.word2idx)
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear_layer = nn.Linear(self.embedding_dim, self.embedding_dim)

        logging.info("Reprocessing model data selesai: n-gram, probabilitas, vocabulary, dan embedding diperbarui.")


# --- Fungsi Re-Training Utama ---
def re_train_model(epochs=5):
    """
    Fungsi utama untuk re-training model menggunakan dataset eksternal.
    Menghitung pseudo-loss dengan probabilitas bigram untuk monitoring.
    """
    from datasets import load_dataset
    chatbot = NeuralChat()  # Muat model (atau buat baru jika tidak ada)
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    total_samples = len(dataset)
    
    logging.info(f"Mulai re-training selama {epochs} epoch dengan {total_samples} sampel...")
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs} dimulai...")
        total_loss = 0.0
        count_loss = 0
        start_time = time.time()
        for idx, row in enumerate(dataset):
            text_data = row.get("text", "")
            if not text_data:
                continue

            tokens = word_tokenize(text_data.lower())
            # Hitung pseudo-loss berbasis bigram
            for i in range(len(tokens) - 1):
                prev, curr = tokens[i], tokens[i + 1]
                if prev in chatbot.bigram:
                    total = sum(chatbot.bigram[prev].values())
                    prob = chatbot.bigram[prev][curr] / total if total > 0 else 1e-10
                    loss = -np.log(prob + 1e-10)
                    total_loss += loss
                    count_loss += 1

            chatbot.train(text_data)

            if (idx + 1) % 1000 == 0:
                avg_loss = total_loss / count_loss if count_loss > 0 else 0
                elapsed = time.time() - start_time
                logging.info(f"Epoch {epoch+1}, Step {idx+1}/{total_samples}, Avg Loss: {avg_loss:.4f}, Elapsed: {elapsed:.2f}s")
        logging.info(f"Epoch {epoch + 1} selesai.")
    chatbot.safe_save()
    logging.info("Re-training selesai. Model disimpan.")


# --- Handler untuk Interrupt ---
def handle_interrupt(signal_received, frame, chatbot):
    logging.info("Interrupt diterima. Menyimpan model sebelum keluar...")
    chatbot.safe_save()
    logging.info("Model disimpan. Program akan dihentikan.")
    exit(0)


# --- Main Program: Hanya Mode Re-Training ---
def main():
    try:
        epochs = int(input("Masukkan jumlah epoch untuk re-training: "))
    except ValueError:
        epochs = 5

    # Daftarkan interrupt handler di main thread
    temp_model = NeuralChat()  # Instance sementara untuk handler
    signal.signal(signal.SIGINT, lambda s, f: handle_interrupt(s, f, temp_model))
    re_train_model(epochs=epochs)


if __name__ == "__main__":
    main()
