"""Tests for tokenization and sequence creation pipeline."""

import json
import os
import tempfile

import numpy as np
import pytest


SAMPLE_TEXT = (
    "she looked into his eyes and smiled warmly at him "
    "he held her hand gently as they walked through the garden together "
    "the moonlight fell across her face and she whispered his name softly "
    "her heart beat faster when she saw him standing at the door "
    "he took her hand and led her into the room where they sat together"
)


class TestTokenizer:
    """Tests for fitting a tokenizer on text."""

    def test_fit_tokenizer_produces_word_index(self):
        """Tokenizer fitted on text should contain expected words."""
        from train import fit_tokenizer

        tokenizer = fit_tokenizer(SAMPLE_TEXT, max_vocab_size=100)
        assert "she" in tokenizer.word_index
        assert "eyes" in tokenizer.word_index
        assert "heart" in tokenizer.word_index

    def test_fit_tokenizer_respects_vocab_cap(self):
        """Tokenizer should only use IDs within max_vocab_size when encoding."""
        from train import fit_tokenizer

        tokenizer = fit_tokenizer(SAMPLE_TEXT, max_vocab_size=10)
        encoded = tokenizer.texts_to_sequences([SAMPLE_TEXT])[0]
        # All token IDs should be < num_words (the cap)
        assert all(tid < 10 for tid in encoded)

    def test_tokenizer_handles_oov_words(self):
        """Unknown words should map to the OOV token index (1)."""
        from train import fit_tokenizer

        tokenizer = fit_tokenizer(SAMPLE_TEXT, max_vocab_size=100)
        encoded = tokenizer.texts_to_sequences(["xyzzyplugh"])
        # OOV token index is 1
        assert encoded == [[1]]


class TestSequences:
    """Tests for creating training sequences from tokenized text."""

    def test_create_sequences_shape(self):
        """Sequences should have correct input/label shapes."""
        from train import create_sequences, fit_tokenizer

        tokenizer = fit_tokenizer(SAMPLE_TEXT, max_vocab_size=100)
        X, y = create_sequences(tokenizer, SAMPLE_TEXT, seq_length=5)

        assert X.ndim == 2
        assert X.shape[1] == 5  # seq_length
        assert y.ndim == 1
        assert len(X) == len(y)
        assert len(X) > 0

    def test_create_sequences_values_in_vocab_range(self):
        """All token IDs in sequences should be within vocab range."""
        from train import create_sequences, fit_tokenizer

        tokenizer = fit_tokenizer(SAMPLE_TEXT, max_vocab_size=100)
        X, y = create_sequences(tokenizer, SAMPLE_TEXT, seq_length=5)
        vocab_size = len(tokenizer.word_index) + 1

        assert np.all(X >= 0)
        assert np.all(X < vocab_size)
        assert np.all(y >= 1)  # labels should be real tokens, not padding
        assert np.all(y < vocab_size)


class TestTokenizerSerialization:
    """Tests for saving/loading tokenizer config."""

    def test_save_and_load_tokenizer_config(self):
        """Tokenizer config saved to JSON should be loadable with same word_index."""
        from train import fit_tokenizer, save_tokenizer_config
        from generate import load_tokenizer_config

        tokenizer = fit_tokenizer(SAMPLE_TEXT, max_vocab_size=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer_config.json")
            save_tokenizer_config(tokenizer, seq_length=5, path=path)

            config = load_tokenizer_config(path)

            assert config["word_index"] == tokenizer.word_index
            assert config["max_seq_length"] == 5
            assert "index_word" in config
            # index_word keys are strings in JSON
            assert config["index_word"][str(tokenizer.word_index["she"])] == "she"
