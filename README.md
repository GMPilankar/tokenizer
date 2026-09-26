# tokenizer

## Overview

This repository contains a complete, from-scratch implementation of a **Byte Pair Encoding (BPE)** tokenizer written in pure Python. It is designed to be lightweight, transparent, and highly readable, making it an excellent resource for understanding the fundamental text-processing pipelines used by modern Large Language Models (LLMs).

**What is BPE?**
Byte Pair Encoding is a subword tokenization algorithm that bridges the gap between word-level and character-level tokenization. It works by initializing a base vocabulary of individual characters (or raw bytes) and iteratively merging the most frequently occurring adjacent pairs into new, unified tokens. 

**Why it matters:**
By building a custom vocabulary based on frequency, BPE effectively solves the Out-Of-Vocabulary (OOV) problem while maintaining highly efficient compression for common words. This exact mechanism (and its variants) serves as the foundational tokenization strategy for state-of-the-art models like GPT-3, GPT-4, LLaMA, and RoBERTa. 

This project strips away the heavy dependencies of standard libraries to show exactly how token training, merging, encoding, and decoding work under the hood.

## Project Structure

The repository is kept intentionally minimal and focused, consisting of four core Python scripts:

```text
.
├── bpe_v5.py           # Core BPE tokenizer class
├── train_tok_utils.py  # Data preprocessing and dataset handling
├── train_tokenizer.py  # Train and save trained tokenizer object as pickle file
└── bpe_benchmark.py    # Tokenization latency comparison against Hugging Face
```
## Installation

### 1. Clone the repository
First, clone the project to your local machine:
```bash
git clone https://github.com/GMPilankar/tokenizer.git
```
### 2. Install the required dependencies using pip
```
pip install -r requirements.txt
```

## Performance

To evaluate the tokenizer's efficiency, it was benchmarked against Hugging Face's codebert-base tokenizer (running in pure Python with use_fast=False). The evaluation corpus consisted of a highly diverse mixture of Python code, C++ source code, and natural language text to simulate real-world LLM training data. The results are as follows:
```
--------------------------------------------------
BENCHMARK RESULTS
--------------------------------------------------
Input text size: 3,887 characters

HUGGINGFACE PYTHON TOKENIZER:
  Time:       0.0013s
  Tokens:     1,405
  Throughput: 1,122,330 tokens/sec
  Raw Speed:  3,104,980 chars/sec

CUSTOM TOKENIZER:
  Time:       0.0008s
  Tokens:     936
  Throughput: 1,103,331 tokens/sec
  Raw Speed:  4,581,889 chars/sec

Speedup: Custom tokenizer is 1.48x faster based on processing time.
```

### Why the Custom BPE is Faster than HF `use_fast=False`

While Hugging Face's pure-Python tokenizer is built as an all-purpose, highly flexible engine, our custom implementation achieves a notable speedup by eliminating generic framework overhead:

* **Minimal Pipeline Overhead:** Hugging Face's `PreTrainedTokenizer` wraps execution inside a heavy abstraction layer—managing padding, truncation, attention masks, token type IDs, and converting outputs into complex dictionary structures (`BatchEncoding`). This implementation returns raw token sequences directly.
* **Simplified Pre-Tokenization:** Models like CodeBERT (based on RoBERTa/GPT-2) execute complex multi-pass Unicode regexes and character-to-byte mappings to handle edge-case whitespace and arbitrary unicode across hundreds of languages. Our implementation uses a focused regex splitting pass.

