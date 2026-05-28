"""RomanticGPT — Generate romantic text continuations from a prompt."""

import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_tokenizer_config(path: str) -> dict:
    """Load tokenizer metadata from JSON."""
    with open(path) as f:
        return json.load(f)


def generate_text(
    model,
    config: dict,
    prompt: str,
    num_words: int = 50,
    temperature: float = 1.0,
) -> str:
    """Generate a text continuation from *prompt*."""
    word_index = config["word_index"]
    index_word = config["index_word"]
    seq_length = config["max_seq_length"]

    words = prompt.lower().split()

    for _ in range(num_words):
        token_ids = [word_index.get(w, 1) for w in words]  # 1 = OOV
        padded = pad_sequences([token_ids], maxlen=seq_length, padding="pre")
        probs = model.predict(padded, verbose=0)[0]

        # Temperature sampling
        probs = np.asarray(probs).astype("float64")
        if temperature < 0.05:
            next_id = int(np.argmax(probs))
        else:
            probs = np.clip(probs, 1e-10, None)
            log_probs = np.log(probs) / temperature
            log_probs -= np.max(log_probs)
            exp_probs = np.exp(log_probs)
            exp_probs /= exp_probs.sum()
            next_id = np.random.choice(len(exp_probs), p=exp_probs)

        next_word = index_word.get(str(next_id), "")
        if not next_word:
            continue
        words.append(next_word)

    return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description="Generate romantic text")
    parser.add_argument("--prompt", type=str, default="she looked into his eyes",
                        help="Starting text for generation")
    parser.add_argument("--words", type=int, default=50,
                        help="Number of words to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--model", type=str, default="romantic_gpt.keras",
                        help="Path to saved .keras model")
    parser.add_argument("--tokenizer", type=str, default="tokenizer_config.json",
                        help="Path to tokenizer config JSON")
    args = parser.parse_args()

    print(f"Loading model from {args.model} …")
    model = tf.keras.models.load_model(args.model)

    config = load_tokenizer_config(args.tokenizer)

    print(f"Prompt: {args.prompt}\n")
    result = generate_text(model, config, args.prompt,
                           num_words=args.words, temperature=args.temperature)
    print(result)


if __name__ == "__main__":
    main()
