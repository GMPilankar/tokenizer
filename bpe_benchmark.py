# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 00:20:24 2026

@author: gaura
"""

import time
from transformers import AutoTokenizer
from .bpe_v5 import BPE
import pickle
import gc
import argparse


def load_tokenizer(path):
    with open(path, 'rb') as file:
        tokenizer = pickle.load(file)
    
    return tokenizer


sample_texts = [
    # =========================================================================
    # 1. SHORT STRINGS (High overhead: tests function call & regex setup)
    # =========================================================================
    "def hello_world(): return 'Hello, World!'",
    "x = [i**2 for i in range(10) if i % 2 == 0]",
    "import torch\nimport torch.nn as nn",
    "SELECT users.id, users.name FROM users WHERE active = 1;",
    "const handleClick = async (e) => { e.preventDefault(); };",
    "The quick brown fox jumps over the lazy dog.",
    "BPE tokenization is a subword segmentation technique.",

    # =========================================================================
    # 2. MEDIUM STRINGS: PYTHON & JAVASCRIPT (Tests subword merges & syntax)
    # =========================================================================
    """def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1""",

    """class AttentionHead(nn.Module):
        def __init__(self, hidden_dim: int, num_heads: int):
            super().__init__()
            self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            B, S, D = x.shape
            qkv = self.qkv_proj(x).reshape(B, S, 3, -1)
            return self.out_proj(qkv[:, :, 0])""",

    """function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }""",

    # =========================================================================
    # 3. MEDIUM STRINGS: C++ / RUST (Punctuation & dense operators)
    # =========================================================================
    """template <typename T>
    class ThreadPool {
    public:
        explicit ThreadPool(size_t threads);
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args) 
            -> std::future<typename std::invoke_result<F, Args...>::type>;
        ~ThreadPool();
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable cv;
        bool stop;
    };""",

    # =========================================================================
    # 4. NATURAL LANGUAGE & DOCSTRINGS (Tests standard vocabulary and prose)
    # =========================================================================
    """Byte-pair encoding (BPE) is an algorithm originally designed for data compression 
    that was later adapted for natural language processing and tokenization in large language models. 
    It iteratively merges the most frequent pairs of bytes or characters until a fixed vocabulary size 
    is reached, balancing vocabulary size and sequence length.""",

    """Tokenizers are essential components in modern Transformer architectures. When tokenizing raw 
    text or code repositories, handling whitespaces, indentation, camelCase identifiers, snake_case 
    variables, and rare unicode characters requires robust character fallback and efficient caching.""",

    # =========================================================================
    # 5. EDGE CASES & STRESS TESTS (Crucial for testing pure Python performance)
    # =========================================================================
    # Repeated single-token words (Tests merge caching / dictionary memoization)
    "for while for while for while if else if else return return return " * 15,

    # Dense indentation and whitespace (Can break poorly tuned regex splits)
    "                x = 1\n\t\t\t\ty = 2\n            z = 3",

    # Unseen/rare characters, UTF-8 symbols, emojis (Tests byte-fallback fallback loops)
    "// 🚀 Feature: calculate ∑(x_i) for € prices with standard deviation σ ≠ 0",
    "def test_unicode(): return 'こんにちは世界' + '🔥'*5",

    # Extremely long identifiers (Worst-case iterative merge complexity)
    "get_authenticated_user_profile_information_by_external_oauth_provider_v2_endpoint()",

    # Chained symbolic operators
    ">>====>> && || != !== <<= >>= -> :: /* comment */ // line"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="benchmark custom bpe tokenizer against hf tokenizer.")
    
    # Define command line arguments
    parser.add_argument("--tok_pickle_file", type=str, default=r"bpe_sql.pkl", 
                        help="path to pickled pretrained tokenizer")
    parser.add_argument("--n_runs", type=int, default=5, 
                        help="no of runs to average processing time over.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # 1. Setup
    hf_slow_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", use_fast=False)
    custom_tokenizer = load_tokenizer(args.tok_pickle_file)
    
    # Multiplying to create a large string for testing
    large_file_text = "\n\n".join(sample_texts)# * 400)
    
    print("Warming up tokenizers...")
    # Warm-up (builds internal memoization caches)
    for text in sample_texts[:10]:
        _ = hf_slow_tokenizer.encode(text) 
        _ = custom_tokenizer.tokenize_flash(text) # Assuming this is your flag
        
    runs = args.n_runs
    print(f"Starting benchmark over {runs} runs...\n")

    # --- Benchmark HF Slow ---
    gc.disable() # Disable garbage collection for accurate timing
    hf_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        hf_tokens = hf_slow_tokenizer.encode(large_file_text) 
        hf_times.append(time.perf_counter() - t0)
    gc.enable()
    hf_avg_time = sum(hf_times) / runs

    # --- Benchmark Custom ---
    gc.disable()
    custom_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        custom_tokens = custom_tokenizer.tokenize_flash(large_file_text)
        custom_times.append(time.perf_counter() - t0)
    gc.enable()
    custom_avg_time = sum(custom_times) / runs

    #--- Sanity Check & Decoding ---
    #Optional: verify decoding works
    recov_text_hf = hf_slow_tokenizer.decode(hf_tokens)
    recov_text_custom = ''.join(custom_tokenizer.decode(custom_tokens))
    
    # --- Reporting ---
    hf_num_tokens = len(hf_tokens)
    custom_num_tokens = len(custom_tokens)
    
    # Calculate characters (or bytes) to get an objective measure
    num_chars = len(large_file_text)
    
    print("-" * 50)
    print("BENCHMARK RESULTS")
    print("-" * 50)
    print(f"Input text size: {num_chars:,} characters\n")
    
    print("HUGGINGFACE PYTHON TOKENIZER:")
    print(f"  Time:       {hf_avg_time:.4f}s")
    print(f"  Tokens:     {hf_num_tokens:,}")
    print(f"  Throughput: {hf_num_tokens/hf_avg_time:,.0f} tokens/sec")
    print(f"  Raw Speed:  {num_chars/hf_avg_time:,.0f} chars/sec\n")

    print("CUSTOM TOKENIZER:")
    print(f"  Time:       {custom_avg_time:.4f}s")
    print(f"  Tokens:     {custom_num_tokens:,}")
    print(f"  Throughput: {custom_num_tokens/custom_avg_time:,.0f} tokens/sec")
    print(f"  Raw Speed:  {num_chars/custom_avg_time:,.0f} chars/sec\n")
    
    # Speedup based on time is still the most accurate headline metric
    speedup = hf_avg_time / custom_avg_time
    print(f"Speedup: Custom tokenizer is {speedup:.2f}x faster based on processing time.")