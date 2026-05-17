"""Generate text from a trained checkpoint with temperature + top-k."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import OWT_ROOT  # noqa: F401, E402

from owt_lm_eval import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = load_model(args.ckpt, device, seq_len_override=None)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    prompt_ids = enc.encode_ordinary(args.prompt) if args.prompt else [eot]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for k in range(args.n_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = enc.decode(out[0].tolist())
        print(f"\n--- sample {k + 1}/{args.n_samples} ---\n{text}")


if __name__ == "__main__":
    main()
