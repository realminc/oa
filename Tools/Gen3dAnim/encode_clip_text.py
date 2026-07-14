#!/usr/bin/env python3
"""Encode one arbitrary prompt into the exact frozen CLIP feature contract.

The dataset manifest is authoritative: prompt inference must use the same model,
projection, dimensionality, and preprocessing as training. Output is a headerless
little-endian float32 row consumed by genalm --text-feature.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--manifest", required=True,
                        help="CMP text_feats/manifest.json used for training")
    parser.add_argument("--output", required=True,
                        help="raw little-endian float32 output row")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda when available, else cpu)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    contract = json.loads(manifest_path.read_text())
    if contract.get("format") != "oa_clip_text_v1":
        raise SystemExit(f"unsupported text feature contract: {contract.get('format')!r}")
    if contract.get("feature") != "CLIPTextModelWithProjection.text_embeds":
        raise SystemExit(f"unsupported text feature projection: {contract.get('feature')!r}")
    model_id = contract.get("model")
    expected_dim = int(contract.get("dim", 0))
    if not model_id or expected_dim <= 0:
        raise SystemExit("text feature manifest lacks model/dim")

    try:
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    except ImportError as exc:
        raise SystemExit(
            "encode_clip_text.py requires torch and transformers in the ALM venv") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPTextModelWithProjection.from_pretrained(model_id).to(device).eval()
    encoded = tokenizer(
        [args.prompt], padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        feature = model(**encoded).text_embeds.float().cpu().numpy()
    if feature.shape != (1, expected_dim):
        raise SystemExit(
            f"encoder output {feature.shape} does not match manifest dim {expected_dim}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    np.asarray(feature, dtype="<f4").tofile(temporary)
    os.replace(temporary, output)
    print(f"Encoded {model_id} prompt -> {output} ({expected_dim} float32 values)")


if __name__ == "__main__":
    main()
