"""Merge a LoRA adapter into Apertus-v1.5-8B and save a standalone model (bf16 safetensors + tokenizer).

    python scripts/merge_adapter.py runs/nash-rl-v4/checkpoints/step-000517/adapter apertus-nash
    python scripts/merge_adapter.py ADAPTER OUT --weight 0.6     # base + 0.6 x the LoRA delta = 0.6 x fine-tuned + 0.4 x base

Needs the Apertus 1.5 build of transformers (the trainer image has it) and peft. The LoRA delta is linear in lora_B, so scaling
lora_B by w gives exactly the weight-space interpolation between the base and the fine-tuned model.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5Config

BASE, REV = "swiss-ai/Apertus-v1.5-8B", "a411d838600baf0e3635a3daf66fb7c55fc97bb6"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("adapter", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--weight", type=float, default=1.0, help="share of the fine-tuned model (default 1.0 = plain merge)")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        adapter = Path(tmp) / "adapter"
        shutil.copytree(args.adapter, adapter)
        if args.weight != 1.0:
            sd = load_file(str(adapter / "adapter_model.safetensors"))
            for k in sd:
                if "lora_B" in k:
                    sd[k] = (sd[k].float() * args.weight).to(sd[k].dtype)
            save_file(sd, str(adapter / "adapter_model.safetensors"), metadata={"format": "pt"})
        cfg = Apertus1p5Config.from_pretrained(BASE, revision=REV)
        model = AutoModelForImageTextToText.from_pretrained(BASE, revision=REV, config=cfg, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(BASE, revision=REV).save_pretrained(str(args.out))
    (args.out / "merge_info.json").write_text(json.dumps({"base": BASE, "revision": REV, "adapter": str(args.adapter), "weight": args.weight}, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
