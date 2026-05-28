"""RomanticGPT — Train a small LSTM text generator on romantic literature."""

import glob
import json
import os
import re
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------
MAX_VOCAB_SIZE = 10_000
SEQ_LENGTH = 40
EMBED_DIM = 64
RNN_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 3
DATA_DIR = "data"
MODEL_PATH = "romantic_gpt.keras"
TOKENIZER_PATH = "tokenizer_config.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_texts(data_dir: str) -> str:
    """Load and concatenate all .txt files in *data_dir*."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not paths:
        sys.exit(f"No .txt files found in {data_dir!r}. See data/README.md for download instructions.")
    chunks = []
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            chunks.append(f.read())
    return "\n".join(chunks)


def clean_text(text: str) -> str:
    """Lowercase and strip to letters, basic punctuation, and whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s.,;:!?'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def fit_tokenizer(text: str, max_vocab_size: int) -> Tokenizer:
    """Fit a Keras word-level tokenizer on *text*."""
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    return tokenizer


# ---------------------------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------------------------
def create_sequences(tokenizer: Tokenizer, text: str, seq_length: int):
    """Build (input, label) pairs using a sliding window over token IDs."""
    encoded = tokenizer.texts_to_sequences([text])[0]
    total = len(encoded) - seq_length
    if total <= 0:
        return np.array([]), np.array([])
    X = np.zeros((total, seq_length), dtype=np.int32)
    y = np.zeros(total, dtype=np.int32)
    for i in range(total):
        X[i] = encoded[i : i + seq_length]
        y[i] = encoded[i + seq_length]
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(vocab_size: int, seq_length: int, embed_dim: int, rnn_units: int):
    """Build and compile a small LSTM language model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length,)),
        tf.keras.layers.Embedding(vocab_size, embed_dim),
        tf.keras.layers.LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Tokenizer serialization
# ---------------------------------------------------------------------------
def save_tokenizer_config(tokenizer: Tokenizer, seq_length: int, path: str):
    """Save tokenizer metadata to JSON."""
    index_word = {str(k): v for k, v in tokenizer.index_word.items()}
    config = {
        "word_index": tokenizer.word_index,
        "index_word": index_word,
        "max_seq_length": seq_length,
        "max_vocab_size": tokenizer.num_words,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading texts …")
    raw = load_texts(DATA_DIR)
    text = clean_text(raw)
    print(f"  Corpus length: {len(text):,} characters")

    print("Tokenizing …")
    tokenizer = fit_tokenizer(text, MAX_VOCAB_SIZE)
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    print(f"  Vocabulary size: {vocab_size:,}")

    print("Creating sequences …")
    X, y = create_sequences(tokenizer, text, SEQ_LENGTH)
    print(f"  Training samples: {len(X):,}")

    print("Building model …")
    model = build_model(vocab_size, SEQ_LENGTH, EMBED_DIM, RNN_UNITS)
    model.summary()

    print("Training …")
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

    print(f"Saving model → {MODEL_PATH}")
    model.save(MODEL_PATH)

    print(f"Saving tokenizer config → {TOKENIZER_PATH}")
    save_tokenizer_config(tokenizer, SEQ_LENGTH, TOKENIZER_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
