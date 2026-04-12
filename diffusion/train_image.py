"""
Train the DLPM image denoiser ε_θ_D.

Trains a UNet noise predictor on FFHQ-256 (or any image dataset) using
the DLPM objective (heavy-tailed diffusion, Proposition 9).

Usage:
    python -m diffusion.train_image --config configs/rayleigh_dlpm.yaml
    python -m diffusion.train_image --data_root data/ffhq256 --alpha_D 1.8
"""
import argparse
import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.levy_diffusion import GenerativeLevyProcess, GenerativeLIMProcess
from models.unet import UNetModel


def build_dataset(data_root: str, image_size: int = 256, dataset_name: str = "imagefolder"):
    """Load image dataset.

    dataset_name: 'imagefolder' (default, FFHQ etc.) or 'mnist'.
    For MNIST, data_root is the download directory (e.g. 'data/').
    """
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
        ])
        return datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
        ])
        return datasets.ImageFolder(root=data_root, transform=transform)


def train(
    data_root: str = "data/ffhq256",
    dataset_name: str = "imagefolder",
    image_size: int = 256,
    in_channels: int = 3,
    alpha_D: float = 1.8,
    model_type: str = "dlpm",
    diffusion_steps: int = 1000,
    batch_size: int = 8,
    n_iters: int = 500_000,
    lr: float = 2e-4,
    warmup: int = 5000,
    model_channels: int = 128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks: int = 2,
    attention_resolutions=(16, 8),
    num_heads: int = 4,
    dropout: float = 0.0,
    clamp_a: float = 10.0,
    clamp_eps: float = 50.0,
    save_every: int = 50_000,
    checkpoint_dir: str = "checkpoints/image",
    device_str: str = "cuda",
    log_every: int = 500,
    num_workers: int = 4,
    ema_rate: float = 0.9999,
    use_smooth_l1: bool = True,
    smooth_l1_beta: float = 0.1,
):
    if model_type not in ("dlpm", "lim"):
        raise ValueError(f"model_type must be 'dlpm' or 'lim', got '{model_type}'")

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset
    if dataset_name != "mnist" and not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset not found at {data_root}. "
            "Download FFHQ-256 and place images in data/ffhq256/images/ (ImageFolder structure), "
            "or use --dataset mnist."
        )
    dataset = build_dataset(data_root, image_size, dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True, pin_memory=True)

    # Build generative process
    if model_type == "dlpm":
        glp = GenerativeLevyProcess(
            alpha=alpha_D,
            device=device,
            steps=diffusion_steps,
            isotropic=True,
            clamp_a=clamp_a,
            clamp_eps=clamp_eps,
        )
    else:  # lim
        glp = GenerativeLIMProcess(
            alpha=alpha_D,
            device=device,
            steps=diffusion_steps,
            isotropic=True,
            clamp_eps=clamp_eps,
        )

    # UNet model
    model = UNetModel(
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=in_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_heads=num_heads,
        use_checkpoint=True,
    ).to(device)

    # EMA shadow model
    ema_model = UNetModel(
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=in_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_heads=num_heads,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4,
                                                end_factor=1.0, total_iters=warmup)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, n_iters - warmup), eta_min=1e-6
    )

    ckpt_prefix = f"image_{model_type}"
    loss_name = "Smooth-L1" if use_smooth_l1 else "MSE"
    print(f"Training image {model_type.upper()}: size={image_size}, α={alpha_D}, steps={diffusion_steps}")
    print(f"  Loss: {loss_name}" + (f" (β={smooth_l1_beta})" if use_smooth_l1 else ""))
    print(f"  LR schedule: linear warmup ({warmup} iters) → cosine annealing → 1e-6")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Device: {device}")

    best_loss = float("inf")
    running_loss = 0.0
    it = 0
    data_iter = iter(loader)

    while it < n_iters:
        model.train()
        try:
            x_batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x_batch, _ = next(data_iter)

        x_batch = x_batch.to(device)
        if model_type == "dlpm":
            loss = glp.training_loss(
                model, x_batch,
                clamp_a=clamp_a, clamp_eps=clamp_eps,
                use_smooth_l1=use_smooth_l1, smooth_l1_beta=smooth_l1_beta,
            )
        else:
            loss = glp.training_loss(model, x_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if it < warmup:
            warmup_sched.step()
        else:
            cosine_sched.step()

        # EMA update
        with torch.no_grad():
            for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                p_ema.mul_(ema_rate).add_(p_model, alpha=1.0 - ema_rate)

        running_loss += loss.item()
        it += 1

        if it % log_every == 0:
            avg = running_loss / log_every
            print(f"  iter {it:7d} | loss {avg:.5f} | lr {optimizer.param_groups[0]['lr']:.2e}")
            running_loss = 0.0

        if it % save_every == 0 or it == n_iters:
            path = os.path.join(checkpoint_dir, f"{ckpt_prefix}_iter{it}.pt")
            cfg_dict = {"alpha_D": alpha_D, "model_type": model_type,
                        "diffusion_steps": diffusion_steps,
                        "model_channels": model_channels, "channel_mult": list(channel_mult)}
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "iter": it,
                "loss": loss.item(),
                "config": cfg_dict,
            }, path)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({"ema": ema_model.state_dict(), "config": cfg_dict},
                           os.path.join(checkpoint_dir, f"{ckpt_prefix}_best.pt"))

    print(f"Training complete. Best checkpoint: "
          f"{os.path.join(checkpoint_dir, f'{ckpt_prefix}_best.pt')}")


def main():
    parser = argparse.ArgumentParser(description="Train image DLPM denoiser (UNet)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="dlpm", choices=["dlpm", "lim"],
                        help="Diffusion model type: 'dlpm' (discrete) or 'lim' (continuous SDE)")
    parser.add_argument("--dataset", type=str, default="imagefolder",
                        choices=["imagefolder", "mnist"],
                        help="Dataset: 'imagefolder' (FFHQ etc.) or 'mnist'")
    parser.add_argument("--data_root", type=str, default="data/ffhq256")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Image size (default: 256 for imagefolder, 28 for mnist)")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Image channels (default: 3 for imagefolder, 1 for mnist)")
    parser.add_argument("--alpha_D", type=float, default=1.8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_iters", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model_channels", type=int, default=128)
    parser.add_argument("--clamp_a", type=float, default=10.0)
    parser.add_argument("--clamp_eps", type=float, default=50.0)
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/image")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_smooth_l1", action="store_true",
                        help="Disable Smooth-L1 and use MSE instead (default: Smooth-L1 on)")
    parser.add_argument("--smooth_l1_beta", type=float, default=0.1,
                        help="Smooth-L1 beta transition point (default: 0.1)")
    args = parser.parse_args()

    cfg = vars(args)
    if args.config is not None:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f)
        img_cfg = file_cfg.get("image_training", {})
        cfg.update({k: v for k, v in img_cfg.items() if k != "config"})
        for key in ("alpha_D",):
            if key in file_cfg:
                cfg[key] = file_cfg[key]

    # Resolve dataset-specific defaults
    dataset_name = cfg.get("dataset", "imagefolder")
    is_mnist = (dataset_name == "mnist")
    image_size = cfg.get("image_size") or (28 if is_mnist else 256)
    in_channels = cfg.get("in_channels") or (1 if is_mnist else 3)

    # UNet defaults tuned per dataset.
    # attention_resolutions uses downsampling factor ds (1→2→4→...), NOT pixel size.
    # MNIST 28×28 with channel_mult=(1,2,2): ds=4 at 7×7 level → att at (4,).
    # RGB 256×256 with channel_mult=(1,2,2,2): ds=16 at 16×16 level → att at (16,).
    if is_mnist:
        default_model_channels   = 32
        default_channel_mult     = (1, 2, 2)
        default_num_res_blocks   = 2
        default_attention_res    = (4,)
        default_steps            = 500
        default_batch            = 128
        default_n_iters          = 20_000
        default_save_every       = 5_000
    else:
        default_model_channels   = 128
        default_channel_mult     = (1, 2, 2, 2)
        default_num_res_blocks   = 2
        default_attention_res    = (16,)
        default_steps            = 1000
        default_batch            = 8
        default_n_iters          = 500_000
        default_save_every       = 50_000

    train(
        data_root=cfg.get("data_root", "data/" if is_mnist else "data/ffhq256"),
        dataset_name=dataset_name,
        image_size=image_size,
        in_channels=in_channels,
        alpha_D=cfg.get("alpha_D", 1.8),
        model_type=cfg.get("model", "dlpm"),
        diffusion_steps=cfg.get("steps", default_steps),
        batch_size=cfg.get("batch_size", default_batch),
        n_iters=cfg.get("n_iters", default_n_iters),
        lr=cfg.get("lr", 2e-4),
        model_channels=cfg.get("model_channels", default_model_channels),
        channel_mult=tuple(cfg.get("channel_mult", default_channel_mult)),
        num_res_blocks=cfg.get("num_res_blocks", default_num_res_blocks),
        attention_resolutions=tuple(cfg.get("attention_resolutions", default_attention_res)),
        clamp_a=cfg.get("clamp_a", 10.0),
        clamp_eps=cfg.get("clamp_eps", 50.0),
        save_every=cfg.get("save_every", default_save_every),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/image"),
        device_str=cfg.get("device", "cuda"),
        num_workers=cfg.get("num_workers", 4),
        use_smooth_l1=not args.no_smooth_l1,
        smooth_l1_beta=args.smooth_l1_beta,
    )


if __name__ == "__main__":
    main()
