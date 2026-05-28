"""Tests for text generation pipeline."""

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


def _train_tiny_model(tmpdir):
    """Helper: train a tiny model on sample text and save artifacts."""
    from train import (
        fit_tokenizer,
        create_sequences,
        build_model,
        save_tokenizer_config,
    )

    tokenizer = fit_tokenizer(SAMPLE_TEXT, max_vocab_size=100)
    X, y = create_sequences(tokenizer, SAMPLE_TEXT, seq_length=5)
    vocab_size = len(tokenizer.word_index) + 1

    model = build_model(vocab_size, 5, embed_dim=16, rnn_units=32)
    model.fit(X, y, epochs=2, batch_size=8, verbose=0)

    model_path = os.path.join(tmpdir, "test.keras")
    config_path = os.path.join(tmpdir, "tokenizer_config.json")
    model.save(model_path)
    save_tokenizer_config(tokenizer, seq_length=5, path=config_path)

    return model_path, config_path


class TestGenerateText:
    """Tests for the text generation function."""

    def test_generate_returns_string_of_expected_length(self):
        """generate_text should return a string with roughly the requested word count."""
        from generate import generate_text, load_tokenizer_config
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path, config_path = _train_tiny_model(tmpdir)
            model = tf.keras.models.load_model(model_path)
            config = load_tokenizer_config(config_path)

            result = generate_text(model, config, "she looked", num_words=10, temperature=1.0)

            assert isinstance(result, str)
            words = result.split()
            # Should have prompt words + generated words
            assert len(words) >= 5  # at least some output

    def test_zero_temperature_is_deterministic(self):
        """Temperature near 0 should produce deterministic output."""
        from generate import generate_text, load_tokenizer_config
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path, config_path = _train_tiny_model(tmpdir)
            model = tf.keras.models.load_model(model_path)
            config = load_tokenizer_config(config_path)

            result1 = generate_text(model, config, "her heart", num_words=10, temperature=0.01)
            result2 = generate_text(model, config, "her heart", num_words=10, temperature=0.01)

            assert result1 == result2
