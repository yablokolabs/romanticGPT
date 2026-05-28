"""Tests for model building, saving, and loading."""

import os
import tempfile

import numpy as np
import pytest


class TestBuildModel:
    """Tests for model construction."""

    def test_build_model_output_shape(self):
        """Model should accept (batch, seq_length) and output (batch, vocab_size)."""
        from train import build_model

        vocab_size = 50
        seq_length = 5
        model = build_model(vocab_size, seq_length, embed_dim=16, rnn_units=32)

        # Build model by running a forward pass
        test_input = np.array([[1, 2, 3, 4, 5]])
        output = model.predict(test_input, verbose=0)
        assert output.shape == (1, vocab_size)

    def test_build_model_is_compiled(self):
        """Model should be compiled with an optimizer and loss."""
        from train import build_model

        model = build_model(50, 5, embed_dim=16, rnn_units=32)
        # A compiled model has a non-None optimizer
        assert model.optimizer is not None


class TestModelSaveLoad:
    """Tests for saving and loading the full model."""

    def test_save_and_load_keras(self):
        """Model saved as .keras should load and produce identical predictions."""
        from train import build_model

        model = build_model(50, 5, embed_dim=16, rnn_units=32)
        test_input = np.array([[1, 2, 3, 4, 5]])
        original_pred = model.predict(test_input, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.keras")
            model.save(path)

            import tensorflow as tf
            loaded = tf.keras.models.load_model(path)
            loaded_pred = loaded.predict(test_input, verbose=0)

            np.testing.assert_allclose(original_pred, loaded_pred, atol=1e-6)
