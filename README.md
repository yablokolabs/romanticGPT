# RomanticGPT 💕

A small Keras LSTM text generator trained on public-domain romantic literature.

## What is this?

RomanticGPT is a word-level language model built with TensorFlow/Keras:

- **Architecture:** Embedding → LSTM → Dense (softmax)
- **Training data:** Public-domain romantic novels from [Project Gutenberg](https://www.gutenberg.org/)
- **Vocabulary:** ~10,000 words
- **Parameters:** ~1.4M (intentionally small, CPU-friendly)

## Files

| File | Description |
|------|-------------|
| `romantic_gpt.keras` | Trained Keras model (full model, not just weights) |
| `tokenizer_config.json` | Vocabulary and sequence length metadata |
| `train.py` | Training script |
| `generate.py` | Inference / text generation script |
| `sample_prompts.txt` | Example prompts to try |

## Quick Start

### Load the model

```python
import tensorflow as tf
model = tf.keras.models.load_model("romantic_gpt.keras")
```

### Generate text

```bash
pip install tensorflow numpy
python generate.py --prompt "she looked into his eyes" --words 50 --temperature 0.8
```

### Train from scratch

```bash
# 1. Download public-domain texts into data/
cd data
curl -o pride_and_prejudice.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt
curl -o sense_and_sensibility.txt https://www.gutenberg.org/cache/epub/161/pg161.txt
curl -o jane_eyre.txt https://www.gutenberg.org/cache/epub/1260/pg1260.txt
cd ..

# 2. Train
python train.py

# 3. Generate
python generate.py --prompt "her heart beat faster"
```

## Programmatic Inference

```python
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("romantic_gpt.keras")

with open("tokenizer_config.json") as f:
    config = json.load(f)

word_index = config["word_index"]
index_word = config["index_word"]
seq_length = config["max_seq_length"]

prompt = "she looked into his eyes"
words = prompt.lower().split()

for _ in range(50):
    token_ids = [word_index.get(w, 1) for w in words]
    padded = pad_sequences([token_ids], maxlen=seq_length, padding="pre")
    probs = model.predict(padded, verbose=0)[0]
    next_id = int(np.argmax(probs))
    next_word = index_word.get(str(next_id), "")
    if next_word:
        words.append(next_word)

print(" ".join(words))
```

## Training Data

All training texts are sourced from [Project Gutenberg](https://www.gutenberg.org/)
and are in the **public domain** in the United States. See `data/README.md` for
download instructions and recommended titles.

## Limitations

- **Very small model** — produces grammatically rough, sometimes incoherent text
- **Word-level tokenization** — cannot handle subword patterns or rare words well
- **No attention mechanism** — limited long-range coherence
- **CPU-trained** — training is feasible but slow on large corpora
- **Not suitable for production use** — this is an educational demonstration

## License

The code in this repository is provided as-is for educational purposes.
Training data is public domain (Project Gutenberg).

## Tech Stack

- Python 3.11+
- TensorFlow / Keras (CPU)
- NumPy
