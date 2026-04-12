"""
Debug runner for DLPM-PVD.

Runs a small synthetic batch through the full DLPM-PVD pipeline with
verbose diagnostics so you can detect instabilities before committing
to full training runs.

Usage (MNIST, analytical channel prior — fastest sanity check):
    python debug_pvd.py --dataset mnist --analytical-channel-prior --J 20

Usage (MNIST, learned channel denoiser with random weights):
    python debug_pvd.py --dataset mnist --J 20

Usage (LIM model, MNIST):
    python debug_pvd.py --dataset mnist --model lim --J 20 --analytical-channel-prior

Usage (original 256x256 RGB mode):
    python debug_pvd.py --J 20 --analytical-channel-prior

What to look for:
  - H_t and D_t should stay O(1) throughout (no blowup).
  - grad_H, grad_D should be non-zero and finite.
  - lik_norm_H / lik_norm_D should be O(1).
  - H_hat (final) should have abs mean close to 1/sqrt(2) ≈ 0.71 for CN(0,I).
  - D_hat should be in [-1, 1].
"""
import argparse
import math
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channels.rayleigh import (
    generate_rayleigh_channel,
    apply_channel_stable_noise,
    set_sigma_n_for_snr,
)
from diffusion.levy_diffusion import GenerativeLevyProcess, GenerativeLIMProcess
from models.channel_net import ChannelDenoiser
from models.unet import UNetModel
from models.mnist_encoder import MNISTEncoder
from noise.stable_noise import SubGaussianStableNoise
from pvd.dlpm_pvd import DLPMPVDSolver
from metrics.nmse import nmse_db


# ---------------------------------------------------------------------------
# Tiny stub encoder for RGB/large images (no trained weights needed)
# ---------------------------------------------------------------------------

class _StubEncoderRGB(torch.nn.Module):
    """Maps D0 (B, C, H, W) → X (B, 1, NtK, T) complex via a fixed linear projection."""
    def __init__(self, in_channels: int, img_size: int, NtK: int, T: int):
        super().__init__()
        self.NtK = NtK
        self.T = T
        self.proj = torch.nn.Linear(in_channels * img_size * img_size, 2 * NtK * T, bias=False)

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        B = D.shape[0]
        flat = D.reshape(B, -1)
        out = self.proj(flat)
        re = out[:, :self.NtK * self.T].reshape(B, self.NtK, self.T)
        im = out[:, self.NtK * self.T:].reshape(B, self.NtK, self.T)
        X_c = torch.complex(re, im)
        pwr = X_c.abs().pow(2).mean(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        X_c = X_c / pwr.sqrt()
        return X_c.unsqueeze(1)  # (B, 1, NtK, T)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MNIST_DEFAULTS = dict(
    in_channels=1, img_size=28,
    model_channels=32, channel_mult=(1, 2, 2),
    num_res_blocks=1,
    # attention_resolutions uses downsampling factor (ds), not pixel size.
    # For 28→14→7, ds=4 at the 7×7 level. Use () for no attention (faster debug).
    attention_resolutions=(4,),
    diffusion_steps=500,
    NtK_default=4, T_default=8,
)

RGB256_DEFAULTS = dict(
    in_channels=3, img_size=256,
    model_channels=64, channel_mult=(1, 2, 2, 2),
    num_res_blocks=1,
    # ds=16 at the 16×16 level for 256×256 with 4 levels of downsampling
    attention_resolutions=(16,),
    diffusion_steps=1000,
    NtK_default=4, T_default=16,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DLPM-PVD debug runner")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "rgb256"],
                        help="'mnist' (28x28 grayscale) or 'rgb256' (256x256 RGB)")
    parser.add_argument("--Nr", type=int, default=4)
    parser.add_argument("--Nt", type=int, default=1)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--T", type=int, default=None,
                        help="Time slots (default: 8 for MNIST, 16 for rgb256)")
    parser.add_argument("--Nu", type=int, default=1)
    parser.add_argument("--B", type=int, default=2, help="Batch size")
    parser.add_argument("--snr", type=float, default=10.0, help="SNR in dB")
    parser.add_argument("--alpha_H", type=float, default=1.8)
    parser.add_argument("--alpha_D", type=float, default=1.8)
    parser.add_argument("--alpha_noise", type=float, default=1.8,
                        help="Noise stability index (2.0 = Gaussian AWGN)")
    parser.add_argument("--J", type=int, default=20, help="Diffusion steps")
    parser.add_argument("--L_A", type=int, default=3,
                        help="Monte Carlo samples for A posterior")
    parser.add_argument("--lambda_H", type=float, default=1.0)
    parser.add_argument("--lambda_D", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="dlpm", choices=["dlpm", "lim"],
                        help="Diffusion model type")
    parser.add_argument("--analytical-channel-prior", action="store_true",
                        help="Use analytical Wiener-filter Tweedie for H (bypasses channel denoiser)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable gradient checkpointing on encoder")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dset = args.dataset
    defaults = MNIST_DEFAULTS if dset == "mnist" else RGB256_DEFAULTS

    in_channels   = defaults["in_channels"]
    img_size      = defaults["img_size"]
    model_channels = defaults["model_channels"]
    channel_mult  = defaults["channel_mult"]
    num_res_blocks = defaults["num_res_blocks"]
    attn_res      = defaults["attention_resolutions"]
    diff_steps    = defaults["diffusion_steps"]

    B   = args.B
    Nr  = args.Nr
    Nt  = args.Nt
    K   = args.K
    T   = args.T if args.T is not None else defaults["T_default"]
    Nu  = args.Nu
    NrK = Nr * K
    NtK = Nt * K

    print("=" * 60)
    print(f"DLPM-PVD Debug Run  [{dset}]")
    print(f"  img={in_channels}×{img_size}×{img_size}  Nr={Nr}  Nt={Nt}  K={K}  T={T}  Nu={Nu}  B={B}")
    print(f"  SNR={args.snr} dB  α_H={args.alpha_H}  α_D={args.alpha_D}"
          f"  α_noise={args.alpha_noise}")
    print(f"  model={args.model}  J={args.J}  L_A={args.L_A}")
    print(f"  analytical_channel_prior={args.analytical_channel_prior}")
    print(f"  device={device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    print("\n[1] Generating synthetic data...")
    torch.manual_seed(42)

    D0 = torch.rand(B, in_channels, img_size, img_size, device=device) * 2.0 - 1.0
    print(f"  D0: {list(D0.shape)}  range=[{D0.min():.2f}, {D0.max():.2f}]")

    H0 = generate_rayleigh_channel(B, Nu, Nr, Nt, K, device=device)  # (B, Nu, NrK, NtK)
    H0_u = H0[:, 0]  # (B, NrK, NtK)
    print(f"  H0: {list(H0.shape)}  abs_mean={H0.abs().mean():.3f}")

    # Build encoder
    if dset == "mnist":
        enc = MNISTEncoder(in_channels=in_channels, NtK=NtK, T=T, Nu=Nu).to(device)
    else:
        enc = _StubEncoderRGB(in_channels, img_size, NtK, T).to(device)

    with torch.no_grad():
        X = enc(D0)   # (B, Nu, NtK, T)
    X_u = X[:, 0]
    print(f"  X:  {list(X.shape)}  abs_mean={X_u.abs().mean():.3f}")

    sigma_n = set_sigma_n_for_snr(H0, X, args.snr)
    noise_model = SubGaussianStableNoise(alpha=args.alpha_noise, sigma_n=sigma_n)
    Y, _ = apply_channel_stable_noise(H0, X, args.snr, noise_model)  # (B, NrK, T)
    print(f"  Y:  {list(Y.shape)}  abs_mean={Y.abs().mean():.3f}  sigma_n={sigma_n:.4f}")

    # ------------------------------------------------------------------
    # 2. Diffusion processes
    # ------------------------------------------------------------------
    print(f"\n[2] Building {args.model.upper()} processes (J={args.J})...")
    glp_H = GenerativeLevyProcess(alpha=args.alpha_H, device=device, steps=args.J)
    glp_D = GenerativeLevyProcess(alpha=args.alpha_D, device=device, steps=args.J)

    # ------------------------------------------------------------------
    # 3. Denoisers (random weights)
    # ------------------------------------------------------------------
    print("\n[3] Building denoisers (random weights)...")

    eps_theta_H = ChannelDenoiser(Nr=Nr, Nt=Nt).to(device).eval()
    print(f"  ChannelDenoiser: {sum(p.numel() for p in eps_theta_H.parameters()):,} params")

    eps_theta_D = UNetModel(
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=in_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_res,
        channel_mult=channel_mult,
        num_heads=1,
        use_checkpoint=False,
    ).to(device).eval()
    print(f"  UNetModel: {sum(p.numel() for p in eps_theta_D.parameters()) / 1e6:.2f}M params")

    # ------------------------------------------------------------------
    # 4. Solver
    # ------------------------------------------------------------------
    print("\n[4] Building DLPMPVDSolver...")
    solver = DLPMPVDSolver(
        f_gamma=enc,
        eps_theta_H=eps_theta_H,
        eps_theta_D=eps_theta_D,
        glp_H=glp_H,
        glp_D=glp_D,
        noise_model=noise_model,
        Nr=Nr, Nt=Nt, K=K, T=T, Nu=Nu,
        J=args.J,
        lambda_H=args.lambda_H,
        lambda_D=args.lambda_D,
        L_A=args.L_A,
        device=device,
        use_checkpoint=not args.no_checkpoint,
        use_analytical_channel_prior=args.analytical_channel_prior,
        img_channels=in_channels,
        img_size=img_size,
    )

    # ------------------------------------------------------------------
    # 5. Run PVD
    # ------------------------------------------------------------------
    print("\n[5] Running DLPM-PVD (debug=True)...")
    try:
        H_hat, D_hat = solver.solve(Y, verbose=False, debug=True)
    except Exception as e:
        print(f"\n[ERROR] PVD failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ------------------------------------------------------------------
    # 6. Results
    # ------------------------------------------------------------------
    print("\n[6] Results:")
    nmse = nmse_db(H_hat, H0_u)
    print(f"  NMSE(H): {nmse:.2f} dB")
    print(f"  D_hat: {list(D_hat.shape)}  range=[{D_hat.min():.3f}, {D_hat.max():.3f}]")
    print(f"  H_hat abs mean={H_hat.abs().mean():.3f}  "
          f"(expected ~{1/math.sqrt(2):.3f} for CN(0,I))")

    h_ok = torch.isfinite(H_hat).all().item()
    d_ok = torch.isfinite(D_hat).all().item()
    print(f"\n  H_hat finite: {h_ok}  D_hat finite: {d_ok}")
    if not h_ok:
        print("  [FAIL] H_hat has NaN/Inf — check likelihood step.")
    if not d_ok:
        print("  [FAIL] D_hat has NaN/Inf — check image score / likelihood step.")
    if h_ok and d_ok:
        print("  [PASS] All outputs are finite.")

    print("\nDebug run complete.")


if __name__ == "__main__":
    main()
