"""
Generate demo outputs and full-resolution (2048) side-by-side comparisons.

Usage (from BETA/):
  python collect_demo_inputs.py
  python make_demo.py
"""
import argparse
import os
import sys
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from enhance import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SR_CHECKPOINT,
    DEFAULT_SR_WORK_SIZE,
    center_crop_square,
    enhance_image,
    list_images,
    load_fusion_model,
    load_generator,
)

LABELS_ENHANCE = ("Input@2048", "Enhanced@2048")
LABELS_FUSION = ("Input@2048", "TPOT@2048", "Fusion@2048")


def _load_font(size: int = 48):
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype("uint8")
    return Image.fromarray(array)


def crop_to_work_size(image: Image.Image, work_size: int) -> Image.Image:
    crop = center_crop_square(image)
    return crop.resize((work_size, work_size), Image.BICUBIC)


def add_label(image: Image.Image, text: str, font_size: int = 48) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    font = _load_font(font_size)
    margin = 16
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box = (margin, margin, margin + text_w + 24, margin + text_h + 16)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((margin + 12, margin + 6), text, fill=(255, 255, 255), font=font)
    return labeled


def stitch_panels(panels: List[Image.Image], gap: int = 12) -> Image.Image:
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), color=(24, 24, 24))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


def prepare_fusion_tensors(
    image_path: str,
    device: torch.device,
    work_size: int = DEFAULT_SR_WORK_SIZE,
    lr_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    crop = center_crop_square(image)
    x_hr = ToTensor()(crop.resize((work_size, work_size), Image.BICUBIC)).unsqueeze(0).to(device)
    x_lr = ToTensor()(crop.resize((lr_size, lr_size), Image.BICUBIC)).unsqueeze(0).to(device)
    return x_hr, x_lr


@torch.no_grad()
def run_demo(
    input_dir: str,
    output_root: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    fusion_checkpoint_path: str = DEFAULT_SR_CHECKPOINT,
    device: str = None,
    work_size: int = DEFAULT_SR_WORK_SIZE,
    jpeg_quality: int = 95,
) -> List[str]:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enhance_dir = os.path.join(output_root, "enhance")
    fusion_dir = os.path.join(output_root, "fusion")
    compare_dir = os.path.join(output_root, "compare")
    for folder in (enhance_dir, fusion_dir, compare_dir):
        os.makedirs(folder, exist_ok=True)

    generator = load_generator(checkpoint_path, resolved_device)
    fusion_model = load_fusion_model(
        checkpoint_path,
        fusion_checkpoint_path,
        resolved_device,
        work_size=work_size,
    )

    image_files = list_images(input_dir)
    compare_paths: List[str] = []

    for image_path in tqdm(image_files, desc="Demo"):
        stem, ext = os.path.splitext(os.path.basename(image_path))
        ext = ext or ".jpg"

        original = Image.open(image_path).convert("RGB")
        input_panel = crop_to_work_size(original, work_size)

        enhanced, _ = enhance_image(
            model=generator,
            image_path=image_path,
            device=resolved_device,
            restore_size=True,
        )
        save_image(enhanced, os.path.join(enhance_dir, stem + ext))

        enhanced_image = tensor_to_pil(enhanced)
        enhanced_panel = crop_to_work_size(enhanced_image, work_size)
        enhance_compare = stitch_panels([
            add_label(input_panel, LABELS_ENHANCE[0]),
            add_label(enhanced_panel, LABELS_ENHANCE[1]),
        ])
        enhance_compare_path = os.path.join(compare_dir, f"{stem}_enhance_compare.jpg")
        enhance_compare.save(enhance_compare_path, quality=jpeg_quality)
        compare_paths.append(enhance_compare_path)

        fusion_output, _ = enhance_image(
            model=generator,
            image_path=image_path,
            device=resolved_device,
            restore_size=False,
            fusion_model=fusion_model,
            work_size=work_size,
        )
        save_image(fusion_output, os.path.join(fusion_dir, stem + ext))

        x_hr, x_lr = prepare_fusion_tensors(image_path, resolved_device, work_size=work_size)
        _, _, enhanced_hr = fusion_model(x_hr, x_lr)
        fusion_compare = stitch_panels([
            add_label(tensor_to_pil(x_hr.squeeze(0).cpu()), LABELS_FUSION[0]),
            add_label(tensor_to_pil(enhanced_hr.squeeze(0).cpu()), LABELS_FUSION[1]),
            add_label(tensor_to_pil(fusion_output), LABELS_FUSION[2]),
        ])
        fusion_compare_path = os.path.join(compare_dir, f"{stem}_fusion_compare.jpg")
        fusion_compare.save(fusion_compare_path, quality=jpeg_quality)
        compare_paths.append(fusion_compare_path)

    return compare_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate BETA demo outputs and 2048 comparisons")
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "demo", "input"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "demo"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fusion", default=DEFAULT_SR_CHECKPOINT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--work_size", type=int, default=DEFAULT_SR_WORK_SIZE)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isdir(args.input):
        print(f"Error: demo input folder not found: {args.input}", file=sys.stderr)
        print("Run: python collect_demo_inputs.py", file=sys.stderr)
        return 1

    compare_paths = run_demo(
        input_dir=args.input,
        output_root=args.output,
        checkpoint_path=args.checkpoint,
        fusion_checkpoint_path=args.fusion,
        device=args.device,
        work_size=args.work_size,
        jpeg_quality=args.jpeg_quality,
    )
    print(
        f"Done. {len(compare_paths)} comparison image(s) at {args.work_size}px "
        f"saved under: {os.path.join(args.output, 'compare')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())