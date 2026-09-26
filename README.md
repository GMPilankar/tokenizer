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
