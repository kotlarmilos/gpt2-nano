from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import time

import gradio as gr
from huggingface_hub import hf_hub_download
import torch

from data.tokenizer_runtime import decode, encode
from src.model import GPT, GPTConfig, generate


MODEL_REPO = "kotlarmilos/gpt2-nano"


@lru_cache(maxsize=1)
def load_release() -> tuple[GPT, list, dict]:
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="checkpoints/final.pt"
    )
    vocab_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="bpe-tokenizer/vocab.json"
    )
    merges_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="bpe-tokenizer/merges.json"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    merges = json.loads(Path(merges_path).read_text())
    vocab = json.loads(Path(vocab_path).read_text())
    return model, merges, vocab


def run(prompt: str, new_tokens: int) -> tuple[str, str]:
    model, merges, vocab = load_release()
    prompt_tokens = encode(prompt, merges, vocab)
    timings = {}
    outputs = {}
    for use_cache in (False, True):
        torch.manual_seed(1337)
        started = time.perf_counter()
        tokens = generate(
            model,
            prompt_tokens,
            int(new_tokens),
            use_cache=use_cache,
            do_sample=False,
        )
        timings["cached" if use_cache else "uncached"] = time.perf_counter() - started
        outputs["cached" if use_cache else "uncached"] = decode(tokens, vocab)
    return outputs["cached"], json.dumps(timings, indent=2)


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Prompt", value="The "),
        gr.Slider(1, 100, value=30, step=1, label="New tokens"),
    ],
    outputs=[
        gr.Textbox(label="Cached generation"),
        gr.Code(label="Measured seconds", language="json"),
    ],
    title="gpt2-nano KV-cache",
    description="Compare cached and uncached decoding on the signed-off checkpoint.",
)


if __name__ == "__main__":
    demo.launch()
