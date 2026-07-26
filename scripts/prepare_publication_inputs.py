from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


REQUIRED_PATHS = {
    "checkpoints/final.pt",
    "data/bpe-tokenizer/vocab.json",
    "data/bpe-tokenizer/merges.json",
    "data/bpe-shards/train_000000.bin",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/manifest.json")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = {
        str(record["path"]): record
        for record in manifest["artifacts"]
        if record.get("source")
    }
    missing_records = REQUIRED_PATHS - records.keys()
    if missing_records:
        raise ValueError(
            "manifest lacks required source records: "
            + ", ".join(sorted(missing_records))
        )

    for relative in sorted(REQUIRED_PATHS):
        record = records[relative]
        path = Path(relative)
        expected = str(record["sha256"])
        if path.is_file() and sha256(path) == expected:
            print(f"verified {path}")
            continue
        download(str(record["source"]), path)
        actual = sha256(path)
        if actual != expected:
            path.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {path}")
        print(f"downloaded and verified {path}")


if __name__ == "__main__":
    main()
