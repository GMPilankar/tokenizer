# tokenizer

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Theory](#theory)
- [Performance](#performance)


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
## Theory

> **Note:** This step-by-step explanation is purely for conceptual understanding. Actual code implementation might slightly differ, particularly in how data structures are managed, how frequency ties are broken, and how end-of-word tokens are represented.

Let's train a Byte Pair Encoding (BPE) tokenizer on a single sentence: 
**"the cat sat on the mat with the rat"**

### 1. Pre-tokenization & Initialization
First, we split the sentence into words, append an End-Of-Word token (`_`) to each, and split every word into individual characters. We also count the frequency of each word in our training corpus.

**Initial Corpus State:**
| Word | Frequency | Split Form |
| :--- | :--- | :--- |
| the_ | 3 | `t` `h` `e` `_` |
| cat_ | 1 | `c` `a` `t` `_` |
| sat_ | 1 | `s` `a` `t` `_` |
| on_ | 1 | `o` `n` `_` |
| mat_ | 1 | `m` `a` `t` `_` |
| with_ | 1 | `w` `i` `t` `h` `_` |
| rat_ | 1 | `r` `a` `t` `_` |

**Initial Vocabulary:** 
`{ 't', 'h', 'e', '_', 'c', 'a',  's', 'o', 'n', 'm', 'w', 'i', 'r' }`

---

### 2. Merge Iteration 1
We scan the entire corpus to find the most frequent adjacent pair of tokens.

**Pair Frequencies (Top 3):**
*   (`a`, `t`): 4 times *(from cat, sat, mat, rat)*
*   (`t`, `_`): 4 times *(from cat, sat, mat, rat)*
*   (`t`, `h`): 4 times *(from the x3, with x1)*

*Note: In a tie, the algorithm can pick any of the top pairs based on its internal sorting logic. We will pick `(a, t)`.*

**Action:** Merge `a` and `t` into a new token `at`. Add `at` to the vocabulary.

**Updated Corpus State:**
*   `t` `h` `e` `_` (3)
*   `c` **`at`** `_` (1)
*   `s` **`at`** `_` (1)
*   `o` `n` `_` (1)
*   `m` **`at`** `_` (1)
*   `w` `i` `t` `h` `_` (1)
*   `r` **`at`** `_` (1)

**New Vocabulary:** `{ ...all initial chars..., 'at' }`

---

### 3. Merge Iteration 2
We recount the pairs based on the *updated* corpus. The old `(a, t)` pair no longer exists, but new pairs involving `at` have been created.

**Pair Frequencies (Top 3):**
*   (`at`, `_`): 4 times *(from c-at-_, s-at-_, m-at-_, r-at-_)*
*   (`t`, `h`): 4 times *(from t-h-e, w-i-t-h)*
*   (`h`, `e`): 3 times *(from t-h-e)*

**Action:** Merge `t` and `h` into `th`. Add `th` to the vocabulary.

**Updated Corpus State:**
*   **`th`** `e` `_` (3)
*   `c` `at` `_` (1)
*   `s` `at` `_` (1)
*   `o` `n` `_` (1)
*   `m` `at` `_` (1)
*   `w` `i` **`th`** `_` (1)
*   `r` `at` `_` (1)

**New Vocabulary:** `{ ..., 'at', 'th' }`

---

### 4. Merge Iteration 3
Recount the pairs again using the newly updated corpus.

**Pair Frequencies:**
*   (`at`, `_`): 4 times
*   (`th`, `e`): 3 times
*   (`e`, `_`): 3 times

**Action:** Merge `at` and `_` into `at_`. Add `at_` to the vocabulary. 
*(Notice how the end-of-word token gets absorbed, meaning `at` at the end of a word is now treated as a distinct subword).*

**Updated Corpus State:**
*   `th` `e` `_` (3)
*   `c` **`at_`** (1)
*   `s` **`at_`** (1)
*   `o` `n` `_` (1)
*   `m` **`at_`** (1)
*   `w` `i` `th` `_` (1)
*   `r` **`at_`** (1)

**New Vocabulary:** `{ ..., 'at', 'th', 'at_' }`

---

### 5. Merge Iteration 4
**Pair Frequencies:**
*   (`th`, `e`): 3 times
*   (`e`, `_`): 3 times

**Action:** Merge `th` and `e` into `the`.

**Updated Corpus State:**
*   **`the`** `_` (3)
*   `c` `at_` (1)
*   `s` `at_` (1)
*   `o` `n` `_` (1)
*   `m` `at_` (1)
*   `w` `i` `th` `_` (1)
*   `r` `at_` (1)

**Final Vocabulary:** `{ ..., 'at', 'th', 'at_', 'the' }`

### Summary of Training
In a real scenario, we repeat this process for a fixed number of iterations (a target hyperparameter, e.g., 50,000 merges). As training progresses, highly frequent subwords merge into whole words (like `the_`), while rarer words remain split into efficient, reusable subwords.
## Performance

To evaluate the tokenization latency, it was benchmarked against Hugging Face's codebert-base tokenizer (running in pure Python with use_fast=False). The evaluation corpus consisted of a highly diverse mixture of Python code, C++ source code, and natural language text to simulate real-world LLM training data. The results are as follows:
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

