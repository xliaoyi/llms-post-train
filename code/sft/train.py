"""Single-GPU full or LoRA fine-tune of the student on trajectory SFT data.

  python sft/train.py --data data/sft/train.jsonl --out checkpoints/sft-v1
  python sft/train.py --data data/sft/v4-gemma.jsonl --model <dir> --lora-r 64 --lr 1e-4 --out checkpoints/x

fp32 master weights + bf16 autocast, gradient checkpointing, batch size 1 with gradient
accumulation, AdamW with linear warmup and cosine decay. Loss only on assistant tokens
(labels from sft/prepare.py). Saves a vLLM-servable checkpoint at the end of each epoch.
With --lora-r the base stays frozen in bf16 and each saved checkpoint has the LoRA deltas merged in.
"""
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from loss import sft_loss

ROOT = Path(__file__).resolve().parents[1]
# attention + dense/shared-expert MLP projections (Qwen3.5 MoE routed experts are fused tensors, not Linear)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj"]


def merged_state_dict(model, scale):
    """Base weights with W + scale * B @ A folded in for every LoRA layer; the live model is left untouched."""
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if ".lora_" in k:
            continue
        if k.endswith(".base_layer.weight"):
            p = k[:-len(".base_layer.weight")]
            delta = sd[p + ".lora_B.default.weight"].float() @ sd[p + ".lora_A.default.weight"].float()
            v = (v.float() + scale * delta).to(v.dtype)
            k = p + ".weight"
        out[k.replace(".base_layer.", ".")] = v
    return out


def save(model, tok, base, out, lora_scale=None):
    out.mkdir(parents=True, exist_ok=True)
    if lora_scale is not None:
        model = model.get_base_model()
        state = merged_state_dict(model, lora_scale)
    else:
        state = model.state_dict()
    # weights are trained in fp32 (full) or bf16 (LoRA) but served in bf16, like the base model
    model.config.dtype = torch.bfloat16
    model.save_pretrained(out, state_dict={k: v.to(torch.bfloat16) for k, v in state.items()})
    tok.save_pretrained(out)
    # keep generation/chat/processor settings of the base model
    for f in ["generation_config.json", "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
              "video_preprocessor_config.json"]:
        if (Path(base) / f).exists():
            shutil.copy2(Path(base) / f, out / f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "models/Qwen3.5-2B"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (for testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank; 0 = full fine-tune")
    ap.add_argument("--lora-alpha", type=int, help="default 2 * rank")
    ap.add_argument("--max-len", type=int, default=0, help="skip examples longer than this (memory)")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    data = [json.loads(line) for line in open(args.data)]
    if args.max_len:
        data = [ex for ex in data if len(ex["input_ids"]) <= args.max_len]
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.lora_r else torch.float32
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN attention backward failed mid-run on Gemma 4
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lora_scale = None
    if args.lora_r:
        alpha = args.lora_alpha or 2 * args.lora_r
        lora_scale = alpha / args.lora_r
        model.enable_input_require_grads()  # gradient checkpointing needs a grad path through frozen embeddings
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=alpha, lora_dropout=0.0,
                                                 target_modules=LORA_TARGETS))
        for p in model.parameters():  # LoRA weights in fp32, frozen base in bf16
            if p.requires_grad:
                p.data = p.data.float()
        model.print_trainable_parameters()
    model.train()  # HF gradient checkpointing is only active in training mode
    params = [p for p in model.parameters() if p.requires_grad]
    lm = model.get_base_model() if args.lora_r else model

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = math.ceil(len(data) * args.epochs / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        progress = min(1.0, (s - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"{len(data)} examples, {total_steps} optimizer steps", flush=True)

    log = open(out / "train_log.jsonl", "a")
    step, t0, acc_loss, acc_n = 0, time.time(), 0.0, 0
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        random.shuffle(order)
        for i, idx in enumerate(order):
            ex = data[idx]
            ids = torch.tensor([ex["input_ids"]], device="cuda")
            labels = torch.tensor([ex["labels"]], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = sft_loss(lm, ids, labels)
            (loss / args.grad_accum).backward()
            acc_loss += loss.item()
            acc_n += 1
            if acc_n == args.grad_accum or i == len(order) - 1:
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "epoch": epoch, "loss": acc_loss / acc_n, "grad_norm": gnorm,
                       "lr": sched.get_last_lr()[0], "elapsed_min": round((time.time() - t0) / 60, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
                acc_loss, acc_n = 0.0, 0
                if step >= total_steps:
                    break
        save(model, tok, args.model, out / f"epoch{epoch + 1}", lora_scale)
        if step >= total_steps:
            break


if __name__ == "__main__":
    main()
