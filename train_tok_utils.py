# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 21:18:02 2026

@author: gaura
"""

import os
import pyarrow.parquet as pq
from tqdm import tqdm
import ijson
import re
import json




def process_large_json_stream(filename):
    with open(filename, 'rb') as f: # Open in binary mode for ijson
        # If it's a top-level array of objects, use 'item'.
        # This yields a Python object for each item as it is parsed.
        for item in ijson.items(f, 'item'):
            yield item


def get_strings_refinedWeb(parq_files_dir, max_token_limit = float("inf")):
    parq_files = [
        os.path.join(parq_files_dir, file)
        for file in os.listdir(parq_files_dir)
    ]

    print("Getting refined_web text...")
    
    text_chunks = []   # store pieces here
    token_count = 0
    
    for idx, file in enumerate(parq_files):
        print(f"Processing file {idx+1}/{len(parq_files)}")
        
        # Read only required column
        table = pq.read_table(file, columns=["content"])
        
        for batch in table.to_batches():
            col = batch.column(0).to_pylist()
            break_flag = 0
            for chunk in col:
                token_count += len(chunk.split())
                if token_count > max_token_limit:
                    break_flag  = 1
                    break
                text_chunks.append(chunk)
            if break_flag:
                break
        if break_flag:
            break
            #text_chunks.extend(col.to_pylist())
    
    # Join once
    #return "\n".join(text_chunks)
    print("Done with refinedWeb..")
    return text_chunks, token_count

def get_strings_stackoverflow(parq_files_dir, max_token_limit = float("inf")):
    parq_files = os.listdir(parq_files_dir)
    parq_files = [os.path.join(parq_files_dir, file) for file in parq_files if file[-7:] == "parquet"]

    print("Getting stackoverflow text...")
    
    text_chunks = []   # store pieces here
    token_count = 0
    
    for idx, file in enumerate(parq_files):
        print(f"Processing file {idx+1}/{len(parq_files)}")
        
        # Read only required column
        table = pq.read_table(file, columns=["Body"])
        
        for batch in table.to_batches():
            col = batch.column(0).to_pylist()
            break_flag = 0
            for chunk in col:
                token_count += len(chunk.split())
                if token_count > max_token_limit:
                    break_flag  = 1
                    break
                text_chunks.append(chunk)
            if break_flag:
                break
        if break_flag:
            break
            #text_chunks.extend(col.to_pylist())
    
    # Join once
    #return "\n".join(text_chunks)
    print("Done with stackoverflow..")
    return text_chunks, token_count

def get_strings_python(parq_files_dir, max_token_limit = float("inf")):
    parq_files = os.listdir(parq_files_dir)
    parq_files = [os.path.join(parq_files_dir, file) for file in parq_files if file[-7:] == "parquet"]

    print("Getting python text...")
    
    text_chunks = []   # store pieces here
    token_count = 0
    
    for idx, file in enumerate(parq_files):
        print(f"Processing file {idx+1}/{len(parq_files)}")
        
        # Read only required column
        table = pq.read_table(file, columns=["output"])
        
        for batch in tqdm(table.to_batches()):
            col = batch.column(0).to_pylist()
            break_flag = 0
            for chunk in col:
                if '```' not in chunk:
                    continue
                #result = re.search(r".*?```(?:python)?(.*?)```.*", chunk, re.DOTALL)
                result = re.search(r".*?```python(.*?)```.*", chunk, re.DOTALL)
                if result is None:
                    continue
                chunk = result[1]
                token_count += len(chunk.split())
                if token_count > max_token_limit:
                    break_flag  = 1
                    break
                text_chunks.append(chunk)
            if break_flag:
                break
        if break_flag:
            break
            #text_chunks.extend(col.to_pylist())
    
    # Join once
    #return "\n".join(text_chunks)
    print("Done with python code..")
    return text_chunks, token_count


def get_strings_schema(tables_file):
    print("processing schema ddls")
    with open(tables_file, 'r', encoding='utf-8') as file:
            # Use json.load() to parse the file content
            data = json.load(file)
    
    token_count = 0
    text_chunks = []
    
    for df in tqdm(data):
        for ddl in df["ddls"]:
            token_count += len(ddl.split())
        
        text_chunks.extend(df["ddls"])
    
    return text_chunks, token_count


def get_strings_synSql(file):
    print("Processing synSql data (2.5M samples)...")
    text_chunks = []
    token_count = 0

    for sample in tqdm(process_large_json_stream(file), total = 2.5 * 10**6):
        token_count += len(sample["question"].split())
        token_count += len(sample["sql"].split())
        text_chunks.append(sample["question"])
        text_chunks.append(sample["sql"])
    
    return text_chunks, token_count
        
        
def get_tokenizer_train_chunks(synSql_data_file, tables_file, stack_parq_files_dir, py_parq_files_dir, fraction = 0.2):
    schema_statements , token_count_1 = get_strings_schema(tables_file)
    synSql_text_chunks , token_count_2 = get_strings_synSql(synSql_data_file)
    
    token_count_synSql = token_count_1 + token_count_2
    
    #web_token_limit = fraction * token_count_synSql
    
    stack_chunks , token_count_web = get_strings_stackoverflow(stack_parq_files_dir)
    
    py_chunks , token_count_py = get_strings_python(py_parq_files_dir, max_token_limit = float("inf"))
    
    # schema_token_count = sum(list(map(len ,schema_statements)))
    # synSql_token_count = sum(list(map(len ,synSql_text_chunks)))
    # stack_token_count = sum(list(map(len ,stack_chunks)))
    # py_token_count = sum(list(map(len ,py_chunks)))
    
    schema_statements.extend(synSql_text_chunks[:int(0.28*len(synSql_text_chunks))])
    schema_statements.extend(stack_chunks[:int(0.18*len(stack_chunks))])
    schema_statements.extend(py_chunks)
    return schema_statements #, (schema_token_count, synSql_token_count, stack_token_count, py_token_count)



def add_tokens(tokenizer, tokens):
    idx = len(tokenizer.vocab)
    for tok in tokens:
        tokenizer.vocab[tok] = idx
        tokenizer.rev_vocab[idx] = tok
        idx += 1 
    return tokenizer


if __name__ == "__main__":
    stack_parq_files_dir = r"D:\work\datasets\stackoverflow"
    synSql_data_file = r"D:\work\datasets\SynSQL\data.json"
    tables_file = r"D:\work\datasets\SynSQL\tables.json"
    py_parq_files_dir = r"D:\work\datasets\python_code"
    
    
    text_chunks , (schema_token_count, synSql_token_count, stack_token_count, py_token_count) = get_tokenizer_train_chunks(synSql_data_file, tables_file, stack_parq_files_dir, py_parq_files_dir, fraction = 0.2)
    