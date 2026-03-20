import os
import re
import numpy as np
import argparse
import requests
import pyarrow.parquet as pq
import json
import time

ARTIFACTS_PATH = "data/files"
TOKENIZER_PATH = "data/bpe-tokenizer"
SHARDS_PATH = "data/bpe-shards"
EOW = "<|EOW|>"

DATASET_ENDPOINT = "datasets/HuggingFaceFW/fineweb-edu"
DATASET_PATH = "tree/main/sample/10BT"

def download_dataset(num_files):
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    print(f"Dataset: https://huggingface.co/api/{DATASET_ENDPOINT}/{DATASET_PATH}")
    files = requests.get(f"https://huggingface.co/api/{DATASET_ENDPOINT}/{DATASET_PATH}", stream=True, timeout=30).json()
    
    if len(files) > num_files:
        files = files[:num_files]
    else:
        num_files = len(files)

    print(f"Downloading {num_files} files")
    print(f"\n")

    for i, file in enumerate(files):
        file_path = file["path"]
        local_path = os.path.join(ARTIFACTS_PATH, os.path.basename(file_path))
        print(f"[{i + 1}/{num_files}] Downloading https://huggingface.co/api/{DATASET_ENDPOINT}/{DATASET_PATH}{file_path} to {local_path}")
        if os.path.exists(local_path):
            continue
        
        with requests.get(f"https://huggingface.co/{DATASET_ENDPOINT}/resolve/main/{file_path}", stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    return sorted([os.path.join(ARTIFACTS_PATH, f) for f in os.listdir(ARTIFACTS_PATH)])

def sample_text(files):
    for file in files:
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(columns=["text"], batch_size=256):
            for s in batch.columns[0].to_pylist():
                if s:
                    yield re.sub(r'\s+', ' ', s.strip())

def train(words, max_merges):
    from collections import defaultdict
    words = [list(w) + [EOW] for w in words]

    # Build initial pair counts once
    pair_counts = defaultdict(int)
    for word in words:
        for i in range(len(word) - 1):
            if word[i + 1] != EOW:
                pair_counts[(word[i], word[i + 1])] += 1

    merges = []
    start_time = time.time()
    for merge_idx in range(max_merges):
        if not pair_counts:
            break

        pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[pair] <= 0:
            break

        merged = pair[0] + pair[1]
        merges.append(pair)

        for word in words:
            i = 0
            while i < len(word) - 1:
                if word[i] == pair[0] and word[i + 1] == pair[1]:
                    # Decrement neighbor pairs that are being destroyed
                    if i > 0 and word[i - 1] != EOW:
                        pair_counts[(word[i - 1], word[i])] -= 1
                    if i + 2 < len(word) and word[i + 2] != EOW:
                        pair_counts[(word[i + 1], word[i + 2])] -= 1

                    word[i:i + 2] = [merged]

                    # Increment new neighbor pairs created by the merge
                    if i > 0 and word[i - 1] != EOW:
                        pair_counts[(word[i - 1], merged)] += 1
                    if i + 1 < len(word) and word[i + 1] != EOW:
                        pair_counts[(merged, word[i + 1])] += 1
                else:
                    i += 1

        del pair_counts[pair]

        if (merge_idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Merge {merge_idx + 1}/{max_merges} | Elapsed: {elapsed:.1f}s")

    return merges
        
def build_vocab(merges, words):
    vocab = {}
    token_id = 0
    
    base_tokens = set({c for w in words for c in w if not c.isspace()})
    for c in sorted(base_tokens):
        vocab[c] = token_id
        token_id += 1
        
    for a, b in merges:
        merged = a + b
        if merged not in vocab:
            vocab[merged] = token_id
            token_id += 1
            
    vocab[EOW] = token_id
    return vocab
        
def encode(sencence, merges, vocab):
    words = [list(word) + [EOW] for word in sencence.split(" ")]
        
    for a, b in merges:
        merged = a + b
        for word in words:
            i = 0
            while i < len(word) - 1:
                if word[i] == a and word[i + 1] == b:
                    word[i:i+2] = [merged]
                else:
                    i += 1

    tokens = []
    for word in words:
        for token in word:
            if token in vocab:
                tokens.append(vocab[token])
            else:
                raise Exception(f"Token {token} not found in the vocab")
                
    return tokens

def decode(tokens, vocab):
        inv_vocab = {v: k for k, v in vocab.items()}
        
        sentence = []
        for tok in tokens:
            key = inv_vocab[tok]
            sentence.append(" " if key == EOW else key)
            
        return "".join(sentence)

def save_tokenizer(merges, vocab):
    os.makedirs(TOKENIZER_PATH, exist_ok=True)
    with open(os.path.join(TOKENIZER_PATH, "merges.json"), "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False)
    with open(os.path.join(TOKENIZER_PATH, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

def load_tokenizer():
    with open(os.path.join(TOKENIZER_PATH, "merges.json"), "r", encoding="utf-8") as f:
        merges = json.load(f)
    with open(os.path.join(TOKENIZER_PATH, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return merges, vocab
   
def write_shard(output_dir, idx, tokens):
    header = np.zeros(256, dtype="<i4")
    header[0] = 20250320
    header[1] = 1
    header[2] = len(tokens)
    path = os.path.join(output_dir, f"train_{idx:06d}.bin")
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())
    print(f"Shard {path}: {len(tokens)} tokens")

def export_shards(files, encode_fn, shard_size, num_train_docs, shards_path=None):
    shards_path = shards_path or SHARDS_PATH
    os.makedirs(shards_path, exist_ok=True)
    buf = np.empty(shard_size, dtype=np.uint16)
    fill = 0
    shard_idx = 0
    doc_count = 0

    for i, text in enumerate(sample_text(files)):
        if i >= num_train_docs:
            break
        tokens = encode_fn(text)
        tokens = np.array(tokens, dtype=np.uint16)
        pos = 0
        while pos < len(tokens):
            take = min(shard_size - fill, len(tokens) - pos)
            buf[fill:fill + take] = tokens[pos:pos + take]
            fill += take
            pos += take
            if fill == shard_size:
                write_shard(shards_path, shard_idx, buf[:fill])
                shard_idx += 1
                fill = 0
   
        doc_count += 1

    if fill > 0:
        write_shard(shards_path, shard_idx, buf[:fill])

def load_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * 4)

if __name__ == "__main__":
    total_start = time.time()
    
    print("=== Downloading dataset ===")
    download_start = time.time()
    files = download_dataset(5)
    print(f"Download complete in {time.time() - download_start:.1f}s\n")

    print("=== Collecting words ===")
    collect_start = time.time()
    num_train_docs = 50000
    words = {}
    for i, text, in enumerate(sample_text(files)):
        if i >= num_train_docs:
            break
        for w in text.split():
            words[w] = words.get(w, 0) + 1
        if (i + 1) % (num_train_docs // 10) == 0:
            print(f"  Processed {i + 1}/{num_train_docs} docs...")
    print(f"Collected {len(words)} unique words in {time.time() - collect_start:.1f}s\n")

    print("=== Training tokenizer ===")
    train_start = time.time()
    word_list = sorted(words, key=words.get, reverse=True)
    merges = train(word_list, max_merges=5000)
    vocab = build_vocab(merges, words)
    save_tokenizer(merges, vocab)
    print(f"Vocab size: {len(vocab)} | Merges: {len(merges)} | Time: {time.time() - train_start:.1f}s\n")

    print("=== Exporting shards ===")
    export_start = time.time()
    export_shards(files, lambda text: encode(text, merges, vocab), shard_size=10000000, num_train_docs=num_train_docs)
    print(f"Export complete in {time.time() - export_start:.1f}s\n")

    print(f"=== Total time: {time.time() - total_start:.1f}s ===")
    
