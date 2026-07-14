#!/usr/bin/env python3
"""Bake frozen CLIP text features for the CMP dataset (one-time, offline).

MLD's denoiser is conditioned on a frozen CLIP text feature. This script runs
CLIP once over every CMP caption and writes float32 rows per clip, plus
the empty-string (unconditional) feature for classifier-free guidance:

    <cmp>/text_feats/<id>.npy      # [captions, dim], same order as texts/<id>.txt
    <cmp>/text_feats/uncond.npy    # [1, dim] = CLIP feature of ""
    <cmp>/text_feats/manifest.json # model and feature contract

OaDsHumanMl3d reads these directly for OaAlm, so the OA training/sampling path
has zero runtime CLIP dependency.

text_encoded_dim is 768 in the reference configs (configs/modules/denoiser.yaml),
which is CLIP ViT-L territory. Confirm the released checkpoint's exact CLIP
variant before relying on these features for 1:1 parity; switch --model below to
match it (e.g. openai/clip-vit-large-patch14 → 768, clip-vit-base-patch32 → 512).

Usage:
    python bake_clip_text.py --cmp /path/to/CombatMotionProcessed \
        --model openai/clip-vit-large-patch14
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmp", required=True, help="CMP dataset root (contains texts/, *.txt splits)")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14",
                    help="HF CLIP text model (must match text_encoded_dim=768)")
    ap.add_argument("--device", default=None, help="torch device (default: cuda when available, else cpu)")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="captions per CLIP forward (default: 128)")
    args = ap.parse_args()

    try:
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    except ImportError as exc:
        raise SystemExit(
            "bake_clip_text.py requires torch and transformers; install them in "
            "a dedicated environment, then rerun this command") from exc
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cmp = Path(args.cmp)
    out = cmp / "text_feats"
    out.mkdir(parents=True, exist_ok=True)

    tok = CLIPTokenizer.from_pretrained(args.model)
    model = CLIPTextModelWithProjection.from_pretrained(args.model).to(device).eval()
    dim = model.config.projection_dim
    print(f"CLIP {args.model}: projected text dim={dim}")

    def feat(texts: list[str]) -> np.ndarray:
        encoded = tok(texts, padding="max_length", truncation=True,
                      max_length=tok.model_max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            return model(**encoded).text_embeds.float().cpu().numpy()

    # gather every clip id from the split files
    ids = set()
    for split in ("train", "val", "test"):
        f = cmp / f"{split}.txt"
        if f.exists():
            ids |= {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
    print(f"baking {len(ids)} clips + 1 uncond -> {out}")

    # unconditional (empty-string) feature for CFG
    np.save(out / "uncond.npy", feat([""]).astype(np.float32))

    records: list[tuple[str, list[str]]] = []
    miss = 0
    for cid in sorted(ids):
        tf = cmp / "texts" / f"{cid}.txt"
        if not tf.exists():
            miss += 1
            continue
        captions = [line.split("#", 1)[0].strip()
                    for line in tf.read_text().splitlines() if line.strip()]
        if not captions:
            miss += 1
            continue
        records.append((cid, captions))

    flat_captions = [caption for _, captions in records for caption in captions]
    all_features = np.empty((len(flat_captions), dim), dtype=np.float32)
    for start in range(0, len(flat_captions), args.batch_size):
        end = min(start + args.batch_size, len(flat_captions))
        all_features[start:end] = feat(flat_captions[start:end]).astype(np.float32)
        print(f"  encoded {end}/{len(flat_captions)} captions", end="\r", flush=True)
    if flat_captions:
        print()

    offset = 0
    for cid, captions in records:
        end = offset + len(captions)
        np.save(out / f"{cid}.npy", all_features[offset:end])
        offset = end
    (out / "manifest.json").write_text(json.dumps({
        "format": "oa_clip_text_v1",
        "model": args.model,
        "feature": "CLIPTextModelWithProjection.text_embeds",
        "dtype": "float32",
        "dim": dim,
        "caption_order": "texts/<id>.txt line order",
        "max_length": tok.model_max_length,
    }, indent=2) + "\n")
    print(f"done (missing captions: {miss})")


if __name__ == "__main__":
    main()
