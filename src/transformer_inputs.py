"""XLM-R raw-review preprocessing, independent of the project's TF-IDF processor.

Pinned SentencePiece-derived FAST tokenizer; no external word segmentation,
lowercasing, punctuation/emoji removal, stopwords or custom tone rewriting.
Only token IDs, attention masks and the same optional labels go to the model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import urllib.request

import torch

from src.experiments import PROJECT_ROOT, file_fingerprint
from src.review_inputs import BatchIdentity, batch_identity, raw_strings

MODEL_ID = "FacebookAI/xlm-roberta-base"
REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "models/transformer_tokenizers/xlm_roberta_base_e73636d4"
ASSET_HASHES = {
    "config.json": "d66ed8cd4f2a93b358c245e50736fa389ed4f35c0bae7aad0b32abb20c62b579",
    "tokenizer_config.json": "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
    "tokenizer.json": "a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c",
}


def verify_assets(directory):
    directory = Path(directory)
    for name, expected in ASSET_HASHES.items():
        if file_fingerprint(directory / name)["sha256"] != expected:
            raise ValueError(f"Pinned tokenizer asset changed: {name}")


def prepare_assets(directory):
    """Download only public pinned tokenizer/config files; never model weights or remote code."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ASSET_HASHES:
        path = directory / name
        if not path.exists():
            request = urllib.request.Request(f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}",
                                             headers={"User-Agent": "sentiment-project-tokenizer-setup"})
            with urllib.request.urlopen(request, timeout=60) as response, path.open("xb") as handle:
                handle.write(response.read())
    verify_assets(directory)
    model_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    backend = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    profile = {"schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
               "input_contract": "raw_review", "external_processing": "identity",
               "external_word_segmentation": False, "lowercase": False, "remove_stopwords": False,
               "strip_punctuation_or_emoji": False, "mask_urls_or_phones": False,
               "normalization": "delegated to the pinned tokenizer backend",
               "backend_normalizer_type": backend["normalizer"]["type"],
               "backend_normalizer_sha256": hashlib.sha256(json.dumps(backend["normalizer"], sort_keys=True).encode()).hexdigest(),
               "max_length": tokenizer_config["model_max_length"], "add_special_tokens": True,
               "padding": "longest in batch", "padding_side": "right", "truncation": True,
               "truncation_side": "right", "stride": 0,
               "model_max_position_embeddings": model_config["max_position_embeddings"],
               "model_pad_token_id": model_config["pad_token_id"],
               "runtime": {name: version(name) for name in ("transformers", "tokenizers")},
               "files": {name: file_fingerprint(directory / name) for name in ASSET_HASHES}}
    profile_path = directory / "processor_config.json"
    if profile_path.exists():
        if json.loads(profile_path.read_text(encoding="utf-8")) != profile:
            raise ValueError("Existing tokenizer profile differs; use a new version/directory")
    else:
        with profile_path.open("x", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)
    return profile


@dataclass
class TransformerBatch:
    identity: BatchIdentity
    model_inputs: dict[str, torch.Tensor]
    token_lengths: tuple[int, ...]
    full_token_lengths: tuple[int, ...]
    truncated: tuple[bool, ...]
    empty_raw: tuple[bool, ...]
    no_content_tokens: tuple[bool, ...]
    unknown_token_counts: tuple[int, ...]


class XLMRReviewInputs:
    def __init__(self, directory=DEFAULT_TOKENIZER_DIR, *, max_length=512):
        directory = Path(directory)
        verify_assets(directory)
        self.profile = json.loads((directory / "processor_config.json").read_text(encoding="utf-8"))
        if (self.profile.get("schema_version") != 1 or self.profile.get("model_id") != MODEL_ID
                or self.profile.get("revision") != REVISION or self.profile.get("external_processing") != "identity"):
            raise ValueError("Unsupported Transformer preprocessing profile")
        fixed = {"external_word_segmentation": False, "lowercase": False, "remove_stopwords": False,
                 "strip_punctuation_or_emoji": False, "mask_urls_or_phones": False,
                 "padding_side": "right", "truncation_side": "right", "stride": 0,
                 "add_special_tokens": True, "truncation": True, "max_length": 512}
        if any(self.profile.get(name) != expected for name, expected in fixed.items()):
            raise ValueError("Transformer preprocessing flags differ from the locked model contract")
        if self.profile["files"] != {name: file_fingerprint(directory / name) for name in ASSET_HASHES}:
            raise ValueError("Tokenizer profile file fingerprints differ")
        for name, expected in self.profile["runtime"].items():
            if version(name) != expected:
                raise ValueError(f"Tokenizer runtime mismatch: {name}")
        model_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if model_config["model_type"] != "xlm-roberta":
            raise ValueError("This route is for XLM-R; other model families need their own preprocessing contract")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True,
                                                       trust_remote_code=False, token=False)
        if not self.tokenizer.is_fast:
            raise ValueError("Fast tokenizer required for nonduplicating overflow-length audit")
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        # XLM-R position IDs start after padding_idx; 514 entries => 512 nonpadding IDs.
        position_limit = model_config["max_position_embeddings"] - model_config["pad_token_id"] - 1
        self.model_limit = min(self.tokenizer.model_max_length, position_limit)
        if not self.special_tokens < max_length <= self.model_limit:
            raise ValueError(f"max_length must be >{self.special_tokens} and <={self.model_limit}, including special tokens")
        if len(self.tokenizer) != model_config["vocab_size"]:
            raise ValueError("Tokenizer vocabulary differs from selected model config")
        self.max_length = max_length

    def tokenize_raw(self, reviews):
        reviews = raw_strings(reviews)
        if not reviews:
            return {"input_ids": torch.empty((0, 0), dtype=torch.long),
                    "attention_mask": torch.empty((0, 0), dtype=torch.long)}, (), (), (), ()
        # Raw strings go directly to the native normalizer/subword tokenizer once.
        # Overflow encodings are inspected for audit, but never returned as extra samples.
        batch = self.tokenizer(reviews, add_special_tokens=True, truncation=True, max_length=self.max_length,
                               stride=0, padding="longest", return_tensors="pt", return_attention_mask=True)
        lengths, full_lengths = [], []
        for encoding in batch.encodings:
            length = int(sum(encoding.attention_mask))
            full = length + sum(int(sum(part.attention_mask)) - self.special_tokens for part in encoding.overflowing)
            lengths.append(length)
            full_lengths.append(full)
        inputs = {name: batch[name].to(dtype=torch.long) for name in ("input_ids", "attention_mask")}
        if inputs["input_ids"].shape[0] != len(reviews) or inputs["input_ids"].shape[1] > self.max_length:
            raise RuntimeError("Tokenizer changed sample count or exceeded model context")
        unknown = ((inputs["input_ids"] == self.tokenizer.unk_token_id) & inputs["attention_mask"].bool()).sum(dim=1)
        return inputs, tuple(lengths), tuple(full_lengths), tuple(length > self.max_length for length in full_lengths), tuple(unknown.tolist())

    def encode(self, records):
        records = tuple(records)
        identity = batch_identity(records)
        texts = [row.review for row in records]
        inputs, lengths, full_lengths, truncated, unknown = self.tokenize_raw(texts)
        if identity.labels is not None:
            inputs["labels"] = identity.labels
        return TransformerBatch(identity, inputs, lengths, full_lengths, truncated,
                                tuple(not text.strip() for text in texts),
                                tuple(length == self.special_tokens for length in lengths), unknown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Fetch/verify the three pinned public files and write preprocessing profile")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    args = parser.parse_args()
    if not args.prepare:
        parser.error("Use --prepare; encoding API is XLMRReviewInputs.encode(RawReview records)")
    print(json.dumps(prepare_assets(args.tokenizer_dir), indent=2))


if __name__ == "__main__":
    main()
