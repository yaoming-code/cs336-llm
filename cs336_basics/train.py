import argparse
import os
import time
import math
import numpy as np
import torch

from cs336_basics.transformer import (
    transformer_lm,
    cross_entropy,
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    model.eval()

    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(args.eval_iters):
            x, y = get_batch(
                data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
            )

            logits = model(x)
            loss = cross_entropy(logits, y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()

    # 数据路径
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)

    # checkpoint
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="ckpt.pt")
    parser.add_argument("--resume", type=str, default=None)

    # 模型超参数
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # 日志与验证
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 设备
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 可选 wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336-training")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.out_dir, args.checkpoint_name)

    device = torch.device(args.device)

    # 使用 np.memmap 加载大数据集
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    model = transformer_lm(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0

    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint {args.resume}, starting at iteration {start_iter}")

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    model.train()

    t0 = time.time()

    for iteration in range(start_iter, args.max_iters):
        # 设置当前学习率
        lr = get_lr_cosine_schedule(
            t=iteration,
            alpha_max=args.lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_iters,
            T_c=args.max_iters,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 取 batch
        x, y = get_batch(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        # forward
        logits = model(x)
        loss = cross_entropy(logits, y)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if args.grad_clip is not None and args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        # 日志
        if iteration % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()

            print(
                f"iter {iteration:6d} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.6e} | "
                f"time {dt:.2f}s"
            )

            if args.wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "lr": lr,
                        "iter": iteration,
                    },
                    step=iteration,
                )

        # 验证
        if iteration % args.eval_interval == 0 and iteration > 0:
            losses = estimate_loss(model, train_data, val_data, args, device)

            print(
                f"[eval] iter {iteration:6d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            if args.wandb:
                wandb.log(
                    {
                        "eval/train_loss": losses["train"],
                        "eval/val_loss": losses["val"],
                        "iter": iteration,
                    },
                    step=iteration,
                )

        # 保存 checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # 最后保存一次
    save_checkpoint(model, optimizer, args.max_iters, checkpoint_path)
    print(f"Training finished. Final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()