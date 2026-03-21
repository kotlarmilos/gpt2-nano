import os
import time
import sentencepiece as spm
from bpe_tokenizer import download_dataset, sample_text, export_shards, ARTIFACTS_PATH

TOKENIZER_PATH = "data/sp-tokenizer"
SHARDS_PATH = "data/sp-shards"

def train_sp(files, num_train_docs, vocab_size):
    os.makedirs(TOKENIZER_PATH, exist_ok=True)
    corpus_path = os.path.join(TOKENIZER_PATH, "train_corpus.txt")

    print(f"Writing training corpus ({num_train_docs} docs)...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(sample_text(files)):
            if i >= num_train_docs:
                break
            f.write(text + "\n")

    model_prefix = os.path.join(TOKENIZER_PATH, "sp_model")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.999,
        byte_fallback=True,
        split_digits=True,
        add_dummy_prefix=False,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    )
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")


if __name__ == "__main__":
    files = download_dataset(5)
    sp = train_sp(files, num_train_docs=50000, vocab_size=1024)
    export_shards(files, lambda text: sp.encode(text, out_type=int),
        shard_size=10000000, num_train_docs=50000,
        shards_path="data/sp-shards")