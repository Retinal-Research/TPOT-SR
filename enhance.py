"""
TPOT 眼底图像增强 — 医院交付用推理入口

用法:
  单张: python enhance.py --input path/to/image.jpg --output path/to/out_dir
  批量: python enhance.py --input path/to/images/ --output path/to/out_dir
  融合: python enhance.py --input path/to/image.jpg --output path/to/out_dir --sr
"""
import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from model.fusion_head import GeneratorWithSR
from model.model_LC import _NetG

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "weights", "best_SSIM.pth")
DEFAULT_SR_CHECKPOINT = os.path.join(os.path.dirname(__file__), "weights", "fusion_head_best.pth")
DEFAULT_SR_WORK_SIZE = 2048


def list_images(path: str) -> List[str]:
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {path}")
        return [path]

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input not found: {path}")

    files = [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if os.path.isfile(os.path.join(path, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No images found in: {path}")
    return files


def resolve_generator_state_dict(checkpoint) -> dict:
    """Extract _NetG weights from various checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("adapter_"):
            continue
        name = key[5:] if key.startswith("base.") else key
        cleaned[name] = value
    return cleaned


def center_crop_square(image: Image.Image) -> Image.Image:
    short_side = min(image.size)
    return image.crop((
        (image.width - short_side) // 2,
        (image.height - short_side) // 2,
        (image.width + short_side) // 2,
        (image.height + short_side) // 2,
    ))


def load_generator(checkpoint_path: str, device: torch.device) -> _NetG:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = resolve_generator_state_dict(checkpoint)

    model = _NetG().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_fusion_model(
    checkpoint_path: str,
    fusion_checkpoint_path: str,
    device: torch.device,
    work_size: int = DEFAULT_SR_WORK_SIZE,
) -> GeneratorWithSR:
    if not os.path.isfile(fusion_checkpoint_path):
        raise FileNotFoundError(f"Fusion checkpoint not found: {fusion_checkpoint_path}")

    base = load_generator(checkpoint_path, device)
    model = GeneratorWithSR(base, work_size=work_size).to(device)
    model.fusion_head.load_state_dict(
        torch.load(fusion_checkpoint_path, map_location=device),
        strict=True,
    )
    model.eval()
    return model


@torch.no_grad()
def enhance_image(
    model: _NetG,
    image_path: str,
    device: torch.device,
    image_size: int = 256,
    restore_size: bool = True,
    fusion_model: Optional[GeneratorWithSR] = None,
    work_size: int = DEFAULT_SR_WORK_SIZE,
) -> Tuple[torch.Tensor, str]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    if fusion_model is not None:
        crop = center_crop_square(image)
        x_hr = ToTensor()(crop.resize((work_size, work_size), Image.BICUBIC)).unsqueeze(0).to(device)
        x_lr = ToTensor()(crop.resize((image_size, image_size), Image.BICUBIC)).unsqueeze(0).to(device)
        output, _, _ = fusion_model(x_hr, x_lr)
        output = output.squeeze(0).cpu()
    else:
        resized = image.resize((image_size, image_size), Image.BICUBIC)
        tensor = ToTensor()(resized).unsqueeze(0).to(device)
        enhanced = torch.clamp(model(tensor), 0.0, 1.0)
        output = enhanced.squeeze(0).cpu()

    if restore_size and original_size != (output.shape[-1], output.shape[-2]):
        out_pil = Image.fromarray((output.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        out_pil = out_pil.resize(original_size, Image.BICUBIC)
        output = ToTensor()(out_pil)

    return output, os.path.basename(image_path)


def run(
    input_path: str,
    output_dir: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    image_size: int = 256,
    restore_size: bool = True,
    device: Optional[str] = None,
    sr_checkpoint_path: Optional[str] = None,
    work_size: int = DEFAULT_SR_WORK_SIZE,
) -> List[str]:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_files = list_images(input_path)

    fusion_model: Optional[GeneratorWithSR] = None
    model: Union[_NetG, torch.nn.Module]
    if sr_checkpoint_path:
        fusion_model = load_fusion_model(
            checkpoint_path,
            sr_checkpoint_path,
            resolved_device,
            work_size=work_size,
        )
        model = fusion_model.base
    else:
        model = load_generator(checkpoint_path, resolved_device)

    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []
    desc = "Enhancing+SR" if fusion_model is not None else "Enhancing"

    for image_path in tqdm(image_files, desc=desc):
        output, filename = enhance_image(
            model=model,
            image_path=image_path,
            device=resolved_device,
            image_size=image_size,
            restore_size=restore_size,
            fusion_model=fusion_model,
            work_size=work_size,
        )
        save_path = os.path.join(output_dir, filename)
        save_image(output, save_path)
        saved_paths.append(save_path)

    return saved_paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPOT fundus image enhancement (hospital inference)")
    parser.add_argument("--input", "-i", required=True, help="Input image file or folder")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument(
        "--checkpoint", "-c",
        default=DEFAULT_CHECKPOINT,
        help=f"Model weights (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument("--image_size", type=int, default=256, help="Model input size (default: 256)")
    parser.add_argument(
        "--no_restore_size",
        action="store_true",
        help="Keep output at model resolution instead of resizing to original",
    )
    parser.add_argument(
        "--restore_size",
        action="store_true",
        help="With --sr, resize 2048 output back to original image size",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if omitted)")
    parser.add_argument(
        "--sr",
        action="store_true",
        help="Fuse TPOT enhancement with native detail at 2048 (center-crop square)",
    )
    parser.add_argument(
        "--sr_checkpoint",
        default=DEFAULT_SR_CHECKPOINT,
        help=f"Fusion head weights used with --sr (default: {DEFAULT_SR_CHECKPOINT})",
    )
    parser.add_argument(
        "--sr_size",
        type=int,
        default=DEFAULT_SR_WORK_SIZE,
        help=f"Fusion working resolution with --sr (default: {DEFAULT_SR_WORK_SIZE})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.sr:
        restore_size = args.restore_size
    else:
        restore_size = not args.no_restore_size

    try:
        saved = run(
            input_path=args.input,
            output_dir=args.output,
            checkpoint_path=args.checkpoint,
            image_size=args.image_size,
            restore_size=restore_size,
            device=args.device,
            sr_checkpoint_path=args.sr_checkpoint if args.sr else None,
            work_size=args.sr_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    mode = "enhanced+SR" if args.sr else "enhanced"
    print(f"Done. {len(saved)} image(s) saved to: {args.output} ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())