"""
Train the MNIST DJSCC encoder/decoder pair end-to-end.

The encoder f_γ maps images to complex MIMO symbols; the decoder g_β maps
received signals back to images. They are trained jointly through a simulated
Rayleigh channel with AWGN noise, minimising MSE reconstruction loss.

In the PVD pipeline the decoder is discarded at inference — only f_γ is kept.
Training through the channel forces f_γ to produce symbols that are
informative after passing through fading + noise, which makes the PVD
likelihood gradient meaningful.

Usage:
    python -m encoder.train_mnist_encoder
    python -m encoder.train_mnist_encoder --Nr 4 --Nt 1 --K 4 --T 8 --snr_db 10
    python -m encoder.train_mnist_encoder --config configs/mnist_dlpm.yaml
"""
import argparse
import math
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mnist_encoder import MNISTEncoder, MNISTDecoder


# ---------------------------------------------------------------------------
# Channel simulation (inline — no dependency on noise.stable_noise)
# ---------------------------------------------------------------------------

def rayleigh_block_diagonal(B: int, Nr: int, Nt: int, K: int,
                             device: torch.device) -> torch.Tensor:
    """
    Sample (B, NrK, NtK) block-diagonal Rayleigh channel.
    Each block H_k ~ CN(0, I_{Nr} x I_{Nt}).
    """
    NrK, NtK = Nr * K, Nt * K
    H = torch.zeros(B, NrK, NtK, dtype=torch.complex64, device=device)
    for k in range(K):
        re = torch.randn(B, Nr, Nt, device=device) / math.sqrt(2)
        im = torch.randn(B, Nr, Nt, device=device) / math.sqrt(2)
        r0, c0 = k * Nr, k * Nt
        H[:, r0:r0+Nr, c0:c0+Nt] = torch.complex(re, im)
    return H


def awgn(H: torch.Tensor, X: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Y = H @ X + N,  sigma_n chosen to hit snr_db on average.

    H: (B, NrK, NtK)  X: (B, NtK, T)  →  Y: (B, NrK, T)
    """
    HX = torch.bmm(H, X)                                      # (B, NrK, T)
    signal_power = (HX.abs() ** 2).mean().item()
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma_n = math.sqrt(max(signal_power / snr_lin, 1e-8))

    noise_re = torch.randn_like(HX.real) * sigma_n / math.sqrt(2)
    noise_im = torch.randn_like(HX.imag) * sigma_n / math.sqrt(2)
    N = torch.complex(noise_re, noise_im)
    return HX + N


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    Nr: int = 4,
    Nt: int = 1,
    K: int = 4,
    T: int = 8,
    Nu: int = 1,
    in_channels: int = 1,
    img_size: int = 28,
    base_ch: int = 32,
    snr_db: float = 10.0,
    snr_min: float = 0.0,
    snr_max: float = 20.0,
    random_snr: bool = True,
    data_root: str = "data/",
    batch_size: int = 128,
    n_iters: int = 20_000,
    lr: float = 1e-3,
    warmup: int = 500,
    save_every: int = 5_000,
    checkpoint_dir: str = "checkpoints/encoder",
    device_str: str = "cuda",
    log_every: int = 200,
):
    """
    Args:
        snr_db:      Fixed SNR to train at (used when random_snr=False).
        snr_min/max: Range for random SNR training (used when random_snr=True).
        random_snr:  If True, sample a fresh SNR uniformly in [snr_min, snr_max]
                     each iteration — makes the encoder robust across SNRs.
    """
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    NtK = Nt * K
    NrK = Nr * K

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
    ])
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, drop_last=True, pin_memory=True)

    # Models
    encoder = MNISTEncoder(in_channels=in_channels, NtK=NtK, T=T,
                           Nu=Nu, base_ch=base_ch).to(device)
    decoder = MNISTDecoder(NrK=NrK, T=T, in_channels=in_channels,
                           base_ch=base_ch).to(device)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"Nr={Nr}  Nt={Nt}  K={K}  T={T}  NtK={NtK}  NrK={NrK}")
    print(f"Device: {device}")
    if random_snr:
        print(f"Training with random SNR in [{snr_min}, {snr_max}] dB")
    else:
        print(f"Training at fixed SNR = {snr_db} dB")

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                             end_factor=1.0, total_iters=warmup)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    running_loss = 0.0
    it = 0
    data_iter = iter(loader)

    while it < n_iters:
        encoder.train()
        decoder.train()

        try:
            imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)

        D0 = imgs.to(device)                              # (B, 1, 28, 28)
        B_cur = D0.shape[0]

        # Encode
        X = encoder(D0)                                   # (B, 1, NtK, T)
        X_u = X[:, 0]                                     # (B, NtK, T)

        # Channel
        H = rayleigh_block_diagonal(B_cur, Nr, Nt, K, device)  # (B, NrK, NtK)
        snr = (torch.FloatTensor(1).uniform_(snr_min, snr_max).item()
               if random_snr else snr_db)
        Y = awgn(H, X_u, snr)                            # (B, NrK, T)

        # Decode and compute loss
        D_hat = decoder(Y)                                # (B, 1, 28, 28)
        loss = loss_fn(D_hat, D0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if it < warmup:
            scheduler.step()

        running_loss += loss.item()
        it += 1

        if it % log_every == 0:
            avg = running_loss / log_every
            print(f"  iter {it:6d} | loss {avg:.5f} | lr {optimizer.param_groups[0]['lr']:.2e}")
            running_loss = 0.0

        if it % save_every == 0 or it == n_iters:
            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "iter": it,
                "loss": loss.item(),
                "config": {"Nr": Nr, "Nt": Nt, "K": K, "T": T, "Nu": Nu,
                           "in_channels": in_channels, "base_ch": base_ch},
            }
            path = os.path.join(checkpoint_dir, f"mnist_enc_iter{it}.pt")
            torch.save(ckpt, path)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(ckpt, os.path.join(checkpoint_dir, "mnist_enc_best.pt"))

    print(f"Done. Best checkpoint: {os.path.join(checkpoint_dir, 'mnist_enc_best.pt')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MNIST DJSCC encoder/decoder")
    parser.add_argument("--config", type=str, default=None, help="YAML config (mnist_dlpm.yaml)")
    parser.add_argument("--Nr", type=int, default=4)
    parser.add_argument("--Nt", type=int, default=1)
    parser.add_argument("--K",  type=int, default=4)
    parser.add_argument("--T",  type=int, default=8)
    parser.add_argument("--Nu", type=int, default=1)
    parser.add_argument("--base_ch", type=int, default=32,
                        help="CNN base channel width (must match eval config encoder_base_ch)")
    parser.add_argument("--snr_db",  type=float, default=10.0,
                        help="Fixed SNR in dB (ignored when --random_snr)")
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--random_snr", action="store_true", default=True,
                        help="Sample random SNR each iter (recommended)")
    parser.add_argument("--fixed_snr", dest="random_snr", action="store_false",
                        help="Train at a single fixed --snr_db")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_iters",   type=int, default=20_000)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--save_every",type=int, default=5_000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/encoder")
    parser.add_argument("--device",    type=str, default="cuda")
    args = parser.parse_args()

    # Pull channel dims from YAML if provided
    cfg_ovr = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ch = cfg.get("channel", {})
        cfg_ovr = {
            "Nr": ch.get("Nr", args.Nr),
            "Nt": ch.get("Nt", args.Nt),
            "K":  ch.get("K",  args.K),
            "T":  ch.get("T",  args.T),
            "Nu": ch.get("Nu", args.Nu),
            "base_ch":        cfg.get("image", {}).get("encoder_base_ch", args.base_ch),
            "data_root":      cfg.get("image", {}).get("data_root", args.data_root),
            "checkpoint_dir": cfg.get("encoder", {}).get("checkpoint_dir",
                                                          args.checkpoint_dir),
        }

    train(
        Nr=cfg_ovr.get("Nr", args.Nr),
        Nt=cfg_ovr.get("Nt", args.Nt),
        K=cfg_ovr.get("K",  args.K),
        T=cfg_ovr.get("T",  args.T),
        Nu=cfg_ovr.get("Nu", args.Nu),
        base_ch=cfg_ovr.get("base_ch", args.base_ch),
        snr_db=args.snr_db,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        random_snr=args.random_snr,
        data_root=cfg_ovr.get("data_root", args.data_root),
        batch_size=args.batch_size,
        n_iters=args.n_iters,
        lr=args.lr,
        save_every=args.save_every,
        checkpoint_dir=cfg_ovr.get("checkpoint_dir", args.checkpoint_dir),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
