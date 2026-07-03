Pre-generated demos (see README.md for previews)

input/     32 high-res images from E:/Data/Retina
enhance/   enhancement outputs (restored to original size)
fusion/    --sr fusion outputs (2048x2048)
compare/   full-resolution comparisons (2048 px per panel, not downscaled)
  *_enhance_compare.jpg   Input@2048 | Enhanced@2048
  *_fusion_compare.jpg     Input@2048 | TPOT@2048 | Fusion@2048

Refresh:
  python collect_demo_inputs.py
  python make_demo.py