"""Load and sample prompts from a JSON prompt dataset.

Each entry is ``{"id": str, "pos": str, "neg": str, "seed": int}``.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    id: str
    pos: str
    neg: str
    seed: int


def load_prompts(path: str | Path) -> list[Prompt]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"prompt_file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"prompt_file {path} must be a JSON array")
    prompts: list[Prompt] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id", i))
        pos = item.get("pos")
        neg = item.get("neg", "")
        raw_seed = item.get("seed")
        if not pos:
            continue
        if raw_seed is None:
            raise ValueError(f"Prompt {pid!r} is missing required field 'seed'")
        prompts.append(Prompt(id=pid, pos=pos, neg=neg or "", seed=int(raw_seed)))
    if not prompts:
        raise ValueError(f"No usable prompts found in {path}")
    return prompts


def sample_prompts(prompts: list[Prompt], num: int | None, seed: int = 0) -> list[Prompt]:
    """Deterministically sample ``num`` prompts, preserving each entry's pos/neg pair."""
    if num is None or num >= len(prompts):
        return list(prompts)
    if num <= 0:
        raise ValueError("num_prompts must be positive")
    rng = random.Random(seed)
    return rng.sample(prompts, num)


def inject_trigger(pos: str, trigger: str, class_word: str) -> str:
    """Insert a DreamBooth trigger before the class word (``of man`` -> ``of sks man``)."""
    if re.search(rf"\b{re.escape(trigger)}\s+{re.escape(class_word)}\b", pos):
        return pos
    return re.sub(rf"\bof {re.escape(class_word)}\b", f"of {trigger} {class_word}", pos, count=1)


def load_training_sample_prompts(
    path: str | Path,
    *,
    trigger: str | None = None,
    class_word: str | None = None,
) -> tuple[list[str], list[str], list[int]]:
    """Load pos/neg/seed lists for lorakit training-time sampling."""
    prompts = load_prompts(path)
    pos: list[str] = []
    neg: list[str] = []
    seeds: list[int] = []
    for item in prompts:
        text = item.pos
        if trigger and class_word:
            text = inject_trigger(text, trigger, class_word)
        pos.append(text)
        neg.append(item.neg)
        seeds.append(item.seed)
    return pos, neg, seeds
