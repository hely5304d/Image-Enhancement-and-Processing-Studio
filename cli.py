#!/usr/bin/env python3
"""
Image Enhancement & Processing Studio - CLI
Command Line Interface for single-image and batch image processing.

Examples:
    # 1. Apply a one-click studio preset:
    python cli.py enhance --input samples/low_light.png --preset night_sight --output out.png

    # 2. Chain custom enhancement algorithms:
    python cli.py enhance --input samples/hazy_scene.png --dehaze --clahe --unsharp 1.2 --output out.png

    # 3. Quantitative comparison between two images:
    python cli.py evaluate --ref samples/ground_truth_base.png --target samples/noisy_photo.png

    # 4. Batch process an entire folder:
    python cli.py batch --input-dir samples --output-dir processed --preset landscape_vivid
"""

import os
import sys
import argparse
from typing import Optional
import cv2
import numpy as np

from core import (
    ImagePipeline,
    enhancement,
    restoration,
    filters,
    geometry,
    metrics,
    presets,
)


def print_banner():
    banner = r"""
========================================================================
    * IMAGE ENHANCEMENT & PROCESSING STUDIO *
    Powered by Python, NumPy, OpenCV, and Pillow
========================================================================
    """
    print(banner)


def run_enhance(args):
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    print(f"[*] Loading input image: {args.input}")
    pipe = ImagePipeline.from_file(args.input)
    orig_h, orig_w = pipe.original.shape[:2]
    print(f"[*] Input resolution: {orig_w}x{orig_h} px")

    # If preset requested
    if args.preset:
        if args.preset not in presets.PRESETS_REGISTRY:
            print(f"[ERROR] Unknown preset '{args.preset}'. Available: {list(presets.PRESETS_REGISTRY.keys())}")
            sys.exit(1)
        print(f"[*] Applying preset recipe: '{args.preset}'...")
        preset_func = presets.PRESETS_REGISTRY[args.preset]["function"]
        pipe._apply_op(f"preset_{args.preset}", preset_func)

    # Custom operations chain
    if args.clahe:
        print(f"[*] Applying CLAHE (clip_limit={args.clahe_clip})...")
        pipe.clahe(clip_limit=args.clahe_clip)

    if args.gamma and args.gamma != 1.0:
        print(f"[*] Applying Gamma Correction (gamma={args.gamma})...")
        pipe.gamma(gamma=args.gamma)

    if args.retinex:
        print("[*] Applying Multi-Scale Retinex with Color Restoration (MSRCR)...")
        pipe.retinex()

    if args.white_balance:
        print(f"[*] Applying White Balance ({args.white_balance})...")
        pipe.white_balance(method=args.white_balance)

    if args.auto_contrast:
        print("[*] Applying Dynamic Range Auto-Contrast...")
        pipe.auto_contrast()

    if args.denoise_nlm:
        print("[*] Applying Non-Local Means Denoising...")
        pipe.denoise_nlm(h_lum=args.nlm_strength, h_col=args.nlm_strength)

    if args.denoise_bilateral:
        print("[*] Applying Edge-Preserving Bilateral Filter...")
        pipe.denoise_bilateral()

    if args.dehaze:
        print("[*] Applying Dark Channel Prior Dehazing...")
        pipe.dehaze()

    if args.document_clean:
        print("[*] Applying Document Background Shadow Removal...")
        pipe.document_clean()

    if args.unsharp and args.unsharp > 0:
        print(f"[*] Applying Unsharp Masking (strength={args.unsharp})...")
        pipe.unsharp_mask(strength=args.unsharp)

    if args.vignette and args.vignette > 0:
        print(f"[*] Applying Optical Vignette (strength={args.vignette})...")
        pipe.vignette(strength=args.vignette)

    if args.resize:
        w, h = map(int, args.resize.split("x"))
        print(f"[*] Resizing image to {w}x{h} with Lanczos-4...")
        pipe.resize(width=w, height=h, interpolation="lanczos")

    if args.rotate and args.rotate != 0:
        print(f"[*] Rotating by {args.rotate} degrees...")
        pipe.rotate(angle=args.rotate)

    output_path = args.output or f"enhanced_{os.path.basename(args.input)}"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pipe.save(output_path, quality=args.quality)
    print(f"[SUCCESS] Enhanced image saved to: {output_path}")

    # Evaluation
    eval_stats = pipe.evaluate()
    print("\n--- Image Quality Assessment (IQA) ---")
    print(f"  PSNR:               {eval_stats['psnr_db']} dB")
    print(f"  SSIM:               {eval_stats['ssim']}")
    print(f"  Laplacian Sharpness: {eval_stats['original_sharpness']} -> {eval_stats['enhanced_sharpness']} ({eval_stats['sharpness_change_percent']}%)")
    print(f"  Shannon Entropy:    {eval_stats['original_entropy']} -> {eval_stats['enhanced_entropy']}")
    print(f"  RMS Contrast:       {eval_stats['original_contrast']} -> {eval_stats['enhanced_contrast']} ({eval_stats['contrast_change_percent']}%)")
    print("--------------------------------------\n")


def run_batch(args):
    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] Directory not found: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in supported_exts]

    if not files:
        print(f"[WARNING] No supported images found in {args.input_dir}")
        return

    print(f"[*] Found {len(files)} image(s) for batch processing using preset '{args.preset}'...")
    preset_func = presets.PRESETS_REGISTRY[args.preset]["function"]

    for idx, fname in enumerate(files, 1):
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, f"enhanced_{fname}")
        print(f"  [{idx}/{len(files)}] Processing {fname}...", end=" ")
        try:
            pipe = ImagePipeline.from_file(in_path)
            pipe._apply_op(f"preset_{args.preset}", preset_func)
            pipe.save(out_path)
            print("DONE")
        except Exception as e:
            print(f"FAILED ({e})")

    print(f"[SUCCESS] All processed images written to: {args.output_dir}")


def run_evaluate(args):
    if not os.path.isfile(args.ref) or not os.path.isfile(args.target):
        print("[ERROR] Both --ref and --target images must exist.")
        sys.exit(1)

    ref = cv2.imread(args.ref)
    target = cv2.imread(args.target)
    eval_stats = metrics.evaluate_enhancement_quality(ref, target)

    print("\n================ QUANTITATIVE BENCHMARK ================")
    print(f" Reference: {args.ref}")
    print(f" Target:    {args.target}")
    print("--------------------------------------------------------")
    print(f" PSNR (Peak Signal-to-Noise Ratio):     {eval_stats['psnr_db']} dB")
    print(f" SSIM (Structural Similarity Index):    {eval_stats['ssim']}")
    print(f" Reference Sharpness (Laplacian Var):   {eval_stats['original_sharpness']}")
    print(f" Target Sharpness (Laplacian Var):      {eval_stats['enhanced_sharpness']} ({eval_stats['sharpness_change_percent']}%)")
    print(f" Reference Shannon Entropy:             {eval_stats['original_entropy']} bits")
    print(f" Target Shannon Entropy:                {eval_stats['enhanced_entropy']} bits")
    print(f" Reference Contrast (RMS):              {eval_stats['original_contrast']}")
    print(f" Target Contrast (RMS):                 {eval_stats['enhanced_contrast']} ({eval_stats['contrast_change_percent']}%)")
    print("========================================================\n")


def run_list_presets():
    print("\n--- Available Studio Presets ---")
    for key, p in presets.PRESETS_REGISTRY.items():
        print(f" • {key.ljust(18)} : {p['title']} - {p['description']}")
    print("--------------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Image Enhancement & Processing Studio CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: enhance
    p_enhance = subparsers.add_parser("enhance", help="Process and enhance a single image")
    p_enhance.add_argument("--input", "-i", required=True, help="Input image path")
    p_enhance.add_argument("--output", "-o", help="Output image path")
    p_enhance.add_argument("--preset", choices=list(presets.PRESETS_REGISTRY.keys()), help="Apply studio preset")
    p_enhance.add_argument("--clahe", action="store_true", help="Apply CLAHE")
    p_enhance.add_argument("--clahe-clip", type=float, default=2.5, help="CLAHE clip limit")
    p_enhance.add_argument("--gamma", type=float, default=1.0, help="Gamma factor")
    p_enhance.add_argument("--retinex", action="store_true", help="Apply Retinex MSRCR")
    p_enhance.add_argument("--white-balance", choices=["gray_world", "white_patch"], help="White balance")
    p_enhance.add_argument("--auto-contrast", action="store_true", help="Auto contrast stretch")
    p_enhance.add_argument("--denoise-nlm", action="store_true", help="Non-local means denoising")
    p_enhance.add_argument("--nlm-strength", type=float, default=10.0, help="NLM filter strength")
    p_enhance.add_argument("--denoise-bilateral", action="store_true", help="Bilateral filter")
    p_enhance.add_argument("--dehaze", action="store_true", help="Dark Channel Prior dehaze")
    p_enhance.add_argument("--document-clean", action="store_true", help="Document shadow removal")
    p_enhance.add_argument("--unsharp", type=float, default=0.0, help="Unsharp mask strength")
    p_enhance.add_argument("--vignette", type=float, default=0.0, help="Vignette strength")
    p_enhance.add_argument("--resize", help="Resize image (e.g. 1920x1080)")
    p_enhance.add_argument("--rotate", type=float, default=0.0, help="Rotate angle in degrees")
    p_enhance.add_argument("--quality", type=int, default=95, help="Output image quality (1-100)")

    # Command: batch
    p_batch = subparsers.add_parser("batch", help="Batch process a directory of images")
    p_batch.add_argument("--input-dir", required=True, help="Input directory")
    p_batch.add_argument("--output-dir", required=True, help="Output directory")
    p_batch.add_argument("--preset", default="landscape_vivid", choices=list(presets.PRESETS_REGISTRY.keys()), help="Preset to apply")

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Benchmark image quality metrics between two images")
    p_eval.add_argument("--ref", required=True, help="Reference / ground truth image")
    p_eval.add_argument("--target", required=True, help="Target / enhanced image")

    # Command: presets
    subparsers.add_parser("presets", help="List all available presets")

    args = parser.parse_args()
    print_banner()

    if args.command == "enhance":
        run_enhance(args)
    elif args.command == "batch":
        run_batch(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "presets":
        run_list_presets()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
