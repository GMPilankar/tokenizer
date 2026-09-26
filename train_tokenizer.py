# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 12:24:06 2026

@author: gaura
"""

from .bpe_v5 import BPE
from .train_tok_utils import get_tokenizer_train_chunks
import pickle
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on text chunks from multiple datasets.")
    
    # Define command line arguments
    parser.add_argument("--stack_parq_dir", type=str, required=True, 
                        help="Directory containing StackOverflow parquet files")
    parser.add_argument("--synsql_data", type=str, required=True, 
                        help="Path to the SynSQL data.json file")
    parser.add_argument("--synsql_tables", type=str, required=True, 
                        help="Path to the SynSQL tables.json file")
    parser.add_argument("--py_parq_dir", type=str, required=True, 
                        help="Directory containing Python code parquet files")
    parser.add_argument("--save_file", type=str, default="bpe_sql.pkl", 
                        help="Output path for the saved tokenizer (default: bpe_sql.pkl)")
    parser.add_argument("--vocab_size", type=int, default=32768, 
                        help="Vocabulary size for the tokenizer (default: 32768)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    
    text_chunks = get_tokenizer_train_chunks(args.synSql_data, args.synsql_tables, args.stack_parq_dir, args.py_parq_dir)
    print("\n Got the chunks \nJoining them...\n")
    text = "\n".join(text_chunks)      # need to update tokenizer.train method to accept multiple files for training to save on ram
    print("Done joining")
    text_chunks = []
    
    
    tokenizer = BPE(args.vocab_size)
    tokenizer.train(text)
    
    with open(args.save_file, 'wb') as file:
        pickle.dump(tokenizer , file)