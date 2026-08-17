#!/usr/bin/env python3
"""
evaluate.py - Standalone inference script for the KLA semiconductor image
restoration challenge (Problem 1: denoise + super-resolve).

USAGE
    python evaluate.py <input_dir> <output_dir> [options]

    <input_dir>   directory of degraded test images
    <output_dir>  directory to write restored images to (created if missing)

OPTIONS
    --weights PATH    path to the trained checkpoint
                       (default: model_weights.pt next to this script)
    --batch-size N    max images per inference batch, images are grouped
                       by shape before batching (default: 8)
    --device DEV      'cuda' or 'cpu' (default: cuda if available)
    --fp16            run inference in fp16 on GPU (off by default -- only
                       turn on after numerically validating it against fp32,
                       see the "Benchmark and validate fp16" step)

WHAT THIS SCRIPT NEEDS TO RUN ON A FRESH MACHINE
    1. A trained checkpoint saved as {'model_state_dict': ...} (or a raw
       state_dict) at the --weights path. Put it at ./model_weights.pt next
       to this script, or pass --weights explicitly.
    2. Python packages: torch, numpy. Pillow is only imported (lazily) if the
       input directory contains non-.npy image files. No other packages --
       this script deliberately does NOT depend on lpips / pytorch-msssim /
       ipywidgets, since it only needs to run inference and write files, not
       score anything.

INPUT / OUTPUT FORMAT
    Supports .npy (raw float32 arrays, loaded exactly as the training
    pipeline did -- no extra rescaling) and standard image files
    (.png/.tif/.tiff/.jpg/.jpeg/.bmp, 8-bit or 16-bit, single channel).
    Each restored output is written back out in the SAME format as its
    corresponding input file, under the same base filename, in
    <output_dir>.

    NOTE: the exact test file format wasn't specified anywhere in the
    problem statement provided. This script auto-detects per-file
    extension and handles both .npy and common image formats, so it
    should work whichever format the benchmarking set turns out to use.
    If you know for certain it's one or the other, no changes are needed
    either way -- just double check on a handful of real test files before
    submitting.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pillow is only needed for non-.npy image files -- imported lazily so a
# pure-.npy test set never requires it to be installed.
_PIL_IMAGE = None


# ============================================================
# MODEL DEFINITION
# Copied verbatim from the training notebook so this script is fully
# self-contained -- zero dependency on the training code / environment.
# ============================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        dw_c = c * expand
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, dw_c, 1)
        self.dwconv = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_c // 2)
        self.conv2 = nn.Conv2d(dw_c // 2, c, 1)

        self.norm2 = nn.GroupNorm(1, c)
        self.conv3 = nn.Conv2d(c, dw_c, 1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(dw_c // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        x = x + y * self.gamma
        return x


def icnr_init(weight, scale=2, init=nn.init.kaiming_normal_):
    out_c, in_c, kh, kw = weight.shape
    base_c = out_c // (scale ** 2)
    base = torch.zeros(base_c, in_c, kh, kw)
    init(base)
    kernel = base.repeat_interleave(scale ** 2, dim=0)
    weight.data.copy_(kernel)


class NAFNetSR(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, width=32, enc_blocks=(2, 2, 4),
                 mid_blocks=4, dec_blocks=(2, 2, 2), scale=2):
        super().__init__()
        self.scale = scale
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2))
            c *= 2

        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(mid_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))

        self.sr_head = nn.Sequential(
            nn.Conv2d(c, c * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale)
        )
        icnr_init(self.sr_head[0].weight, scale=scale)

        self.outro = nn.Conv2d(c, out_ch, 3, padding=1)

    def forward(self, x):
        lr_input = x

        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.sr_head(x)
        x = self.outro(x)

        # Global residual: predict a correction on top of a bicubic-upsampled
        # LR base rather than reconstructing the whole image from scratch.
        base = F.interpolate(lr_input, scale_factor=self.scale, mode='bicubic',
                              align_corners=False)
        x = x + base
        return x


# ============================================================
# I/O HELPERS
# ============================================================

IMG_EXTENSIONS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')


def load_input(path):
    """Load one degraded test image as a float32 (H, W) numpy array.

    Returns (array, meta) where meta describes how to re-encode the
    restored output in the same format on the way out.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] in (1, 3) else arr[0]
        # Matches training: LR .npy arrays are used exactly as stored,
        # no additional rescaling (out-of-range speckle values are signal).
        return arr, {'kind': 'npy'}

    if ext in IMG_EXTENSIONS:
        global _PIL_IMAGE
        if _PIL_IMAGE is None:
            from PIL import Image
            _PIL_IMAGE = Image
        img = _PIL_IMAGE.open(path)
        raw = np.array(img)
        if raw.ndim == 3:
            raw = raw.mean(axis=2)  # collapse to single channel if needed
        maxval = 65535.0 if raw.dtype == np.uint16 else 255.0
        arr = raw.astype(np.float32) / maxval
        return arr, {'kind': 'img', 'ext': ext, 'maxval': maxval}

    raise ValueError(f"unsupported file extension: {ext}")


def save_output(path_no_ext, arr, meta):
    """Write a restored (H, W) float32 array back out in the input's format."""
    arr = np.clip(arr, 0.0, 1.0)

    if meta['kind'] == 'npy':
        np.save(path_no_ext + '.npy', arr.astype(np.float32))
        return

    global _PIL_IMAGE
    if _PIL_IMAGE is None:
        from PIL import Image
        _PIL_IMAGE = Image
    maxval = meta['maxval']
    dtype = np.uint16 if maxval == 65535.0 else np.uint8
    out = (arr * maxval).round().astype(dtype)
    _PIL_IMAGE.fromarray(out).save(path_no_ext + meta['ext'])


def list_input_files(input_dir):
    files = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext == '.npy' or ext in IMG_EXTENSIONS:
            files.append(fname)
    return files


# ============================================================
# PADDING (network downsamples by 8x internally -- 3 stages of stride 2)
# ============================================================

PAD_MULTIPLE = 8


def pad_to_multiple(x, multiple=PAD_MULTIPLE):
    """Reflect-pad a (B,1,H,W) tensor so H and W are multiples of `multiple`."""
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    try:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    except RuntimeError:
        # reflect padding needs pad < dim size; falls back for tiny/odd inputs
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x, (pad_h, pad_w)


def unpad_output(x, pad_hw, scale):
    """Crop the model's output back down to match the un-padded input size,
    scaled up by `scale`."""
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return x
    out_h, out_w = x.shape[-2:]
    return x[..., :out_h - pad_h * scale, :out_w - pad_w * scale]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with the trained restoration model.")
    parser.add_argument('input_dir', type=str,
                         help='Directory containing degraded test images')
    parser.add_argument('output_dir', type=str,
                         help='Directory to write restored images to')
    parser.add_argument(
        '--weights', type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'model_weights.pt'),
        help='Path to the trained checkpoint '
             '(default: model_weights.pt next to this script)')
    parser.add_argument('--batch-size', type=int, default=8,
                         help='Max images per inference batch, images are '
                              'grouped by shape before batching (default: 8)')
    parser.add_argument('--device', type=str, default=None,
                         help="'cuda' or 'cpu' (default: cuda if available)")
    parser.add_argument('--fp16', action='store_true',
                         help='Run inference in fp16 on GPU (validate against '
                              'fp32 output before relying on this)')
    args = parser.parse_args()

    t_start = time.time()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(args.weights):
        print(f"ERROR: model weights not found at {args.weights}\n"
              f"Place your trained checkpoint there, or pass --weights.",
              file=sys.stderr)
        sys.exit(1)

    model = NAFNetSR().to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if (
        isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    filenames = list_input_files(args.input_dir)
    if not filenames:
        print(f"WARNING: no supported image files found in {args.input_dir}",
              file=sys.stderr)

    # Load everything up front and group by shape so same-shaped images can
    # be batched together (resolutions in the test set are mixed).
    loaded = {}
    shape_groups = defaultdict(list)
    for fname in filenames:
        try:
            arr, meta = load_input(os.path.join(args.input_dir, fname))
        except Exception as e:
            print(f"WARNING: failed to load {fname}: {e}", file=sys.stderr)
            continue
        loaded[fname] = (arr, meta)
        shape_groups[arr.shape].append(fname)

    n_processed = 0
    t_infer_start = time.time()

    with torch.no_grad():
        for shape, group_fnames in shape_groups.items():
            for i in range(0, len(group_fnames), args.batch_size):
                batch_fnames = group_fnames[i:i + args.batch_size]
                try:
                    batch_arrs = [loaded[f][0] for f in batch_fnames]
                    batch = torch.from_numpy(np.stack(batch_arrs)) \
                        .unsqueeze(1).to(device)  # (B,1,H,W)

                    batch_padded, pad_hw = pad_to_multiple(batch)

                    if args.fp16 and device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            pred = model(batch_padded)
                        pred = pred.float()
                    else:
                        pred = model(batch_padded)

                    pred = unpad_output(pred, pad_hw, model.scale)
                    pred = pred.clamp(0.0, 1.0).cpu().numpy()
                except Exception as e:
                    print(f"WARNING: inference failed for batch {batch_fnames}: {e}",
                          file=sys.stderr)
                    continue

                for j, fname in enumerate(batch_fnames):
                    out_arr = pred[j, 0]
                    _, meta = loaded[fname]
                    base = os.path.splitext(fname)[0]
                    try:
                        save_output(os.path.join(args.output_dir, base),
                                    out_arr, meta)
                        n_processed += 1
                    except Exception as e:
                        print(f"WARNING: failed to save output for {fname}: {e}",
                              file=sys.stderr)

    t_end = time.time()
    print(f"Processed {n_processed}/{len(filenames)} images")
    print(f"Total time: {t_end - t_start:.2f}s "
          f"| Inference-only time: {t_end - t_infer_start:.2f}s")
    if n_processed:
        print(f"Avg per-image time: "
              f"{(t_end - t_infer_start) / n_processed * 1000:.1f} ms")


if __name__ == '__main__':
    main()
