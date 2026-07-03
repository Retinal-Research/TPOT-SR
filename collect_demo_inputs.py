"""
Collect high-resolution demo images from E:/Data/Retina into demo/input.

Usage (from BETA/):
  python collect_demo_inputs.py
  python collect_demo_inputs.py --num 30
"""
import argparse
import os
import shutil
from typing import List, Tuple

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

SOURCES = (
    (r"E:/Data/Retina/MobileLab/all_images", "ML", 12),
    (r"E:/Data/Retina/MobileLab_2026/processed_all", "ML26", 12),
    (r"E:/Data/Retina/UK_CFP/all_images", "UK", 8),
)


def list_source_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )


def evenly_sample(paths: List[str], count: int) -> List[str]:
    if not paths:
        return []
    if len(paths) <= count:
        return paths
    step = len(paths) / count
    return [paths[int(i * step)] for i in range(count)]


def collect(num_total: int, output_dir: str) -> List[Tuple[str, str]]:
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            os.remove(path)

    collected: List[Tuple[str, str]] = []
    for source_dir, prefix, quota in SOURCES:
        sampled = evenly_sample(list_source_images(source_dir), quota)
        for src in sampled:
            base = os.path.basename(src)
            dst_name = f"{prefix}_{base}"
            dst = os.path.join(output_dir, dst_name)
            shutil.copy2(src, dst)
            collected.append((src, dst))

    if len(collected) < num_total:
        raise RuntimeError(f"Only collected {len(collected)} images, expected at least {num_total}")

    print(f"Collected {len(collected)} images into {output_dir}")
    for src, dst in collected[:3]:
        print(f"  {os.path.basename(dst)}  <=  {src}")
    if len(collected) > 3:
        print(f"  ... ({len(collected) - 3} more)")
    return collected


def parse_args():
    parser = argparse.ArgumentParser(description="Collect high-res demo inputs from E:/Data/Retina")
    parser.add_argument("--num", type=int, default=30, help="Minimum number of images to collect")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "demo", "input"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collect(num_total=args.num, output_dir=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())