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

    manifestPath = Path(args.manifest)
    contract = json.loads(manifestPath.read_text())
    if contract.get("format") != "oa_clip_text_v1":
        raise SystemExit(f"unsupported text feature contract: {contract.get('format')!r}")
    if contract.get("feature") != "CLIPTextModelWithProjection.text_embeds":
        raise SystemExit(f"unsupported text feature projection: {contract.get('feature')!r}")
    modelId = contract.get("model")
    expectedDim = int(contract.get("dim", 0))
    if not modelId or expectedDim <= 0:
        raise SystemExit("text feature manifest lacks model/dim")

    try:
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    except ImportError as exc:
        raise SystemExit(
            "encodeClipText.py requires torch and transformers in the ALM venv") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(modelId)
    model = CLIPTextModelWithProjection.from_pretrained(modelId).to(device).eval()
    encoded = tokenizer(
        [args.prompt], padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        feature = model(**encoded).text_embeds.float().cpu().numpy()
    if feature.shape != (1, expectedDim):
        raise SystemExit(
            f"encoder output {feature.shape} does not match manifest dim {expectedDim}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    np.asarray(feature, dtype="<f4").tofile(temporary)
    os.replace(temporary, output)
    print(f"Encoded {modelId} prompt -> {output} ({expectedDim} float32 values)")


if __name__ == "__main__":
    main()
