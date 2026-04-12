"""
Train the DLPM channel denoiser ε_θ_H.

The channel prior is over i.i.d. Rayleigh blocks:
    H_k ~ CN(0, I_{Nr} ⊗ I_{Nt})

Represented as 2-channel real tensors (real, imag) of shape (2, Nr, Nt).
Training generates fresh channel samples each iteration — no dataset needed.

Usage:
    python -m diffusion.train_channel --config configs/rayleigh_dlpm.yaml
    python -m diffusion.train_channel --Nr 4 --Nt 1 --alpha_H 1.8
"""
import argparse
import math
import os
import sys
import yaml
import torch
import torch.optim as optim

# Allow running as a module from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.levy_diffusion import GenerativeLevyProcess, GenerativeLIMProcess
from models.channel_net import ChannelDenoiser


def generate_rayleigh_blocks(batch_size: int, Nr: int, Nt: int, device) -> torch.Tensor:
    """
    Sample B i.i.d. complex Rayleigh blocks H_k ~ CN(0, I_{Nr} ⊗ I_{Nt}).

    Returns:
        (B, 2, Nr, Nt) float32: [real, imag] representation.
    """
    real = torch.randn(batch_size, Nr, Nt, device=device) / math.sqrt(2)
    imag = torch.randn(batch_size, Nr, Nt, device=device) / math.sqrt(2)
    return torch.stack([real, imag], dim=1)  # (B, 2, Nr, Nt)


def train(
    Nr: int = 4,
    Nt: int = 1,
    alpha_H: float = 1.8,
    model_type: str = "dlpm",
    diffusion_steps: int = 1000,
    batch_size: int = 256,
    n_iters: int = 100_000,
    lr: float = 2e-4,
    warmup: int = 1000,
    hidden_dim: int = 256,
    depth: int = 4,
    time_embed_dim: int = 128,
    clamp_a: float = 20.0,
    clamp_eps: float = 50.0,
    save_every: int = 10_000,
    checkpoint_dir: str = "checkpoints/channel",
    device_str: str = "cpu",
    log_every: int = 500,
):
    if model_type not in ("dlpm", "lim"):
        raise ValueError(f"model_type must be 'dlpm' or 'lim', got '{model_type}'")

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build generative process
    if model_type == "dlpm":
        glp = GenerativeLevyProcess(
            alpha=alpha_H,
            device=device,
            steps=diffusion_steps,
            isotropic=True,
            clamp_a=clamp_a,
            clamp_eps=clamp_eps,
        )
    else:  # lim
        glp = GenerativeLIMProcess(
            alpha=alpha_H,
            device=device,
            steps=diffusion_steps,
            isotropic=True,
            clamp_eps=clamp_eps,
        )

    # Channel denoiser — same architecture for both DLPM and LIM.
    # LIM uses continuous time t ∈ (0, T], passed as float; the ChannelDenoiser
    # embeds time via sinusoidal features and handles both int and float inputs.
    model = ChannelDenoiser(Nr=Nr, Nt=Nt, hidden_dim=hidden_dim,
                            depth=depth, time_embed_dim=time_embed_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4,
                                            end_factor=1.0, total_iters=warmup)

    print(f"Training channel {model_type.upper()}: Nr={Nr}, Nt={Nt}, α={alpha_H}, "
          f"steps={diffusion_steps}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    best_loss = float("inf")
    running_loss = 0.0
    ckpt_prefix = f"channel_{model_type}"

    for it in range(1, n_iters + 1):
        model.train()

        # Generate fresh channel blocks
        x_start = generate_rayleigh_blocks(batch_size, Nr, Nt, device)

        if model_type == "dlpm":
            loss = glp.training_loss(model, x_start, clamp_a=clamp_a, clamp_eps=clamp_eps)
        else:
            loss = glp.training_loss(model, x_start)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if it <= warmup:
            scheduler.step()

        running_loss += loss.item()

        if it % log_every == 0:
            avg = running_loss / log_every
            print(f"  iter {it:7d} | loss {avg:.5f} | lr {optimizer.param_groups[0]['lr']:.2e}")
            running_loss = 0.0

        if it % save_every == 0 or it == n_iters:
            path = os.path.join(checkpoint_dir, f"{ckpt_prefix}_iter{it}.pt")
            cfg_dict = {"Nr": Nr, "Nt": Nt, "alpha_H": alpha_H,
                        "model_type": model_type,
                        "diffusion_steps": diffusion_steps,
                        "hidden_dim": hidden_dim, "depth": depth}
            torch.save({
                "model": model.state_dict(),
                "iter": it,
                "loss": loss.item(),
                "config": cfg_dict,
            }, path)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_path = os.path.join(checkpoint_dir, f"{ckpt_prefix}_best.pt")
                torch.save({"model": model.state_dict(), "config": cfg_dict}, best_path)

    best_path = os.path.join(checkpoint_dir, f"{ckpt_prefix}_best.pt")
    print(f"Training complete. Best checkpoint: {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train channel DLPM denoiser")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--model", type=str, default="dlpm", choices=["dlpm", "lim"],
                        help="Diffusion model type: 'dlpm' (discrete) or 'lim' (continuous SDE)")
    parser.add_argument("--Nr", type=int, default=4)
    parser.add_argument("--Nt", type=int, default=1)
    parser.add_argument("--alpha_H", type=float, default=1.8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_iters", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--clamp_a", type=float, default=20.0)
    parser.add_argument("--clamp_eps", type=float, default=50.0)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/channel")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = vars(args)
    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f)
        # Merge: file values for channel training section
        ch_cfg = file_cfg.get("channel_training", {})
        cfg.update({k: v for k, v in ch_cfg.items() if k != "config"})
        # Also pull top-level channel params
        for key in ("Nr", "Nt", "alpha_H"):
            if key in file_cfg:
                cfg[key] = file_cfg[key]

    train(
        Nr=cfg["Nr"], Nt=cfg["Nt"],
        alpha_H=cfg.get("alpha_H", 1.8),
        model_type=cfg.get("model", "dlpm"),
        diffusion_steps=cfg.get("steps", 1000),
        batch_size=cfg.get("batch_size", 256),
        n_iters=cfg.get("n_iters", 100_000),
        lr=cfg.get("lr", 2e-4),
        hidden_dim=cfg.get("hidden_dim", 256),
        depth=cfg.get("depth", 4),
        clamp_a=cfg.get("clamp_a", 20.0),
        clamp_eps=cfg.get("clamp_eps", 50.0),
        save_every=cfg.get("save_every", 10_000),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/channel"),
        device_str=cfg.get("device", "cuda"),
    )


if __name__ == "__main__":
    main()
