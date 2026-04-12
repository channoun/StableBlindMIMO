"""
DLPM-PVD Evaluation Script.

Evaluates blind MIMO channel estimation using DLPM priors and alpha-stable noise.

Usage:
    python eval.py --config configs/rayleigh_dlpm.yaml --snr 10
    python eval.py --config configs/rayleigh_dlpm.yaml --all_snr
"""
import argparse
import math
import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channels.rayleigh import generate_rayleigh_channel, apply_channel_stable_noise, set_sigma_n_for_snr
from encoder.swin_jscc import DJSCCEncoder
from diffusion.levy_diffusion import GenerativeLevyProcess
from models.channel_net import ChannelDenoiser
from models.mnist_encoder import MNISTEncoder
from models.unet import UNetModel
from noise.stable_noise import SubGaussianStableNoise
from pvd.dlpm_pvd import DLPMPVDSolver
from metrics.ms_ssim import ms_ssim
from metrics.nmse import nmse_db
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_encoder(cfg: dict, device: torch.device) -> DJSCCEncoder:
    ch = cfg["channel"]
    enc_cfg = cfg["encoder"]
    enc = DJSCCEncoder(
        embed_dim=enc_cfg.get("embed_dim", 96),
        depths=enc_cfg.get("depths", [2, 2, 6, 2]),
        num_heads=enc_cfg.get("num_heads", [3, 6, 12, 24]),
        window_size=enc_cfg.get("window_size", 8),
        Nt=ch["Nt"], K=ch["K"], T=ch["T"], Nu=ch["Nu"],
        power=enc_cfg.get("power", 1.0),
    ).to(device).eval()
    ckpt = enc_cfg.get("checkpoint")
    if ckpt and os.path.isfile(ckpt):
        enc.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        print(f"  Loaded encoder from {ckpt}")
    else:
        print(f"  WARNING: encoder checkpoint not found at {ckpt}, using random weights.")
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def build_channel_net(cfg: dict, device: torch.device) -> ChannelDenoiser:
    ch = cfg["channel"]
    net_cfg = cfg["channel_net"]
    net = ChannelDenoiser(
        Nr=ch["Nr"], Nt=ch["Nt"],
        hidden_dim=net_cfg.get("hidden_dim", 256),
        depth=net_cfg.get("depth", 4),
        time_embed_dim=net_cfg.get("time_embed_dim", 128),
    ).to(device).eval()
    ckpt = cfg["eval"].get("channel_ckpt")
    if ckpt and os.path.isfile(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        print(f"  Loaded channel denoiser from {ckpt}")
    else:
        print(f"  WARNING: channel denoiser checkpoint not found at {ckpt}, using random weights.")
    return net


def build_image_net(cfg: dict, device: torch.device) -> UNetModel:
    net_cfg = cfg["image_net"]
    net = UNetModel(
        in_channels=3,
        model_channels=net_cfg.get("model_channels", 128),
        out_channels=3,
        num_res_blocks=net_cfg.get("num_res_blocks", 2),
        attention_resolutions=net_cfg.get("attention_resolutions", [16, 8]),
        dropout=net_cfg.get("dropout", 0.0),
        channel_mult=net_cfg.get("channel_mult", [1, 2, 2, 2]),
        num_heads=net_cfg.get("num_heads", 4),
        use_checkpoint=True,
    ).to(device).eval()
    ckpt = cfg["eval"].get("image_ckpt")
    if ckpt and os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location=device)
        key = "ema" if "ema" in state else "model"
        net.load_state_dict(state[key])
        print(f"  Loaded image denoiser from {ckpt}")
    else:
        print(f"  WARNING: image denoiser checkpoint not found at {ckpt}, using random weights.")
    return net


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _build_mnist_loader(data_root: str, batch_size: int) -> DataLoader:
    """Infinite MNIST loader (test split, no augmentation)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
    ])
    dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def evaluate_snr(
    cfg: dict,
    snr_db: float,
    device: torch.device,
    f_gamma, eps_theta_H, eps_theta_D, noise_model,
    glp_H, glp_D,
    data_iter=None,
) -> dict:
    """Run n_trials Monte Carlo trials at a given SNR."""
    ch = cfg["channel"]
    Nr, Nt, K, Nu, T = ch["Nr"], ch["Nt"], ch["K"], ch["Nu"], ch["T"]
    pvd_cfg = cfg["pvd"]
    eval_cfg = cfg["eval"]

    img_cfg = cfg.get("image", {})
    img_channels = img_cfg.get("channels", 1)
    img_size     = img_cfg.get("size", 28)

    n_trials = eval_cfg.get("n_trials", 300)
    batch_size = eval_cfg.get("batch_size", 4)
    n_batches = math.ceil(n_trials / batch_size)

    solver = DLPMPVDSolver(
        f_gamma=f_gamma,
        eps_theta_H=eps_theta_H,
        eps_theta_D=eps_theta_D,
        glp_H=glp_H,
        glp_D=glp_D,
        noise_model=noise_model,
        Nr=Nr, Nt=Nt, K=K, T=T, Nu=Nu,
        J=pvd_cfg.get("J", 1000),
        lambda_H=pvd_cfg.get("lambda_H", 1.0),
        lambda_D=pvd_cfg.get("lambda_D", 1.0),
        L_A=pvd_cfg.get("L_A", 20),
        device=device,
        use_checkpoint=pvd_cfg.get("use_checkpoint", True),
        use_analytical_channel_prior=pvd_cfg.get("analytical_channel_prior", False),
        img_channels=img_channels,
        img_size=img_size,
    )

    all_nmse = []
    all_msssim = []

    for batch_idx in tqdm(range(n_batches), desc=f"SNR={snr_db:+.0f}dB", leave=False):
        B = min(batch_size, n_trials - batch_idx * batch_size)

        # Generate channel
        H0 = generate_rayleigh_channel(B, Nu, Nr, Nt, K, device)  # (B, Nu, NrK, NtK)

        # Source images — use real MNIST batches if loader provided, else random
        if data_iter is not None:
            try:
                imgs, _ = next(data_iter)
            except StopIteration:
                # Loader exhausted — will be refreshed by caller; use random fallback
                imgs = torch.rand(B, img_channels, img_size, img_size) * 2 - 1
            D0 = imgs[:B].to(device)
        else:
            D0 = torch.rand(B, img_channels, img_size, img_size, device=device) * 2 - 1

        # Encode
        with torch.no_grad():
            X = f_gamma(D0)  # (B, Nu, NtK, T)

        # Set noise scale to hit target SNR
        H0_single = H0[:, 0]  # (B, NrK, NtK) for SNR computation
        X_single = X[:, 0]    # (B, NtK, T)
        sigma_n = set_sigma_n_for_snr(H0, X, snr_db)
        noise_model.sigma_n = sigma_n

        # Apply channel with stable noise
        Y, N = apply_channel_stable_noise(H0, X, snr_db, noise_model)  # (B, NrK, T)

        # DLPM-PVD solve
        H_hat, D_hat = solver.solve(Y, verbose=False)

        # Metrics
        with torch.no_grad():
            # NMSE on channel
            H0_full = H0[:, 0]  # (B, NrK, NtK) complex (Nu=1)
            nmse_val = nmse_db(H_hat, H0_full).item()
            all_nmse.append(nmse_val)

            # MS-SSIM on image (map [-1,1] → [0,1])
            D0_01 = (D0 + 1.0) / 2.0
            D_hat_01 = (D_hat + 1.0) / 2.0
            msssim_val = ms_ssim(D0_01, D_hat_01).mean().item()
            all_msssim.append(msssim_val)

    return {
        "snr_db": snr_db,
        "nmse_db_mean": float(np.mean(all_nmse)),
        "nmse_db_std": float(np.std(all_nmse)),
        "ms_ssim_mean": float(np.mean(all_msssim)),
        "ms_ssim_std": float(np.std(all_msssim)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DLPM-PVD blind MIMO receiver")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--snr", type=float, default=None, help="Single SNR in dB")
    parser.add_argument("--all_snr", action="store_true", help="Evaluate all SNRs from config")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device_str = args.device or cfg["eval"].get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    print(f"Device: {device}")

    # Image config
    img_cfg = cfg.get("image", {})
    img_channels = img_cfg.get("channels", 1)
    img_size     = img_cfg.get("size", 28)
    use_mnist    = img_cfg.get("dataset", "mnist") == "mnist"

    # Build models
    print("Loading models...")
    ch = cfg["channel"]
    if use_mnist:
        NtK = ch["Nt"] * ch["K"]
        f_gamma = MNISTEncoder(
            in_channels=img_channels,
            NtK=NtK,
            T=ch["T"],
            Nu=ch["Nu"],
            base_ch=img_cfg.get("encoder_base_ch", 32),
        ).to(device).eval()
        enc_ckpt = cfg.get("encoder", {}).get("checkpoint")
        if enc_ckpt and os.path.isfile(enc_ckpt):
            f_gamma.load_state_dict(torch.load(enc_ckpt, map_location=device)["model"])
            print(f"  Loaded MNISTEncoder from {enc_ckpt}")
        else:
            print("  MNISTEncoder: using random weights (train encoder for better results)")
        for p in f_gamma.parameters():
            p.requires_grad_(False)
    else:
        f_gamma = build_encoder(cfg, device)

    eps_theta_H = build_channel_net(cfg, device)
    eps_theta_D = build_image_net(cfg, device)

    # Noise model
    noise_cfg = cfg["noise"]
    noise_model = SubGaussianStableNoise(
        alpha=noise_cfg.get("alpha", 1.8),
        sigma_n=noise_cfg.get("sigma_n", 1.0),
    )

    # DLPM processes
    def _make_glp(key):
        c = cfg[key]
        return GenerativeLevyProcess(
            alpha=c.get("alpha", 1.8),
            device=device,
            steps=c.get("diffusion_steps", 1000),
            isotropic=c.get("isotropic", True),
            scale=c.get("scale", "scale_preserving"),
            clamp_a=c.get("clamp_a", 20.0),
            clamp_eps=c.get("clamp_eps", 50.0),
        )

    glp_H = _make_glp("dlpm_H")
    glp_D = _make_glp("dlpm_D")

    # SNR sweep
    if args.all_snr:
        snrs = cfg["eval"].get("snr_range", [0, 5, 10, 15, 20])
    elif args.snr is not None:
        snrs = [args.snr]
    else:
        snrs = cfg["eval"].get("snr_range", [10])

    # Build MNIST data iterator if needed
    data_iter = None
    if use_mnist:
        data_root = img_cfg.get("data_root", "data/")
        mnist_loader = _build_mnist_loader(data_root, cfg["eval"].get("batch_size", 4))
        data_iter = iter(mnist_loader)
        print(f"  MNIST test set loaded from {data_root}")

    print(f"\nEvaluating SNRs: {snrs}")
    results = []
    for snr_db in snrs:
        res = evaluate_snr(cfg, snr_db, device, f_gamma,
                           eps_theta_H, eps_theta_D, noise_model, glp_H, glp_D,
                           data_iter=data_iter)
        results.append(res)
        print(f"  SNR={snr_db:+.0f}dB | NMSE={res['nmse_db_mean']:.2f}±{res['nmse_db_std']:.2f} dB"
              f" | MS-SSIM={res['ms_ssim_mean']:.4f}±{res['ms_ssim_std']:.4f}")

    # Summary table
    print("\n--- Results Summary ---")
    print(f"{'SNR(dB)':>8} | {'NMSE(dB)':>12} | {'MS-SSIM':>10}")
    print("-" * 38)
    for r in results:
        print(f"{r['snr_db']:>8.0f} | {r['nmse_db_mean']:>+9.2f}±{r['nmse_db_std']:.2f}"
              f" | {r['ms_ssim_mean']:.4f}±{r['ms_ssim_std']:.4f}")


if __name__ == "__main__":
    main()
