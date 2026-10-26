# Security

## Known issues

Last checked: 2026-10-26.

These dependencies still have open advisories because other packages cap their versions. Upgrade them once the cap is lifted.

| Package | Version | Fixed in | Held back by | Impact and mitigation |
|---|---|---|---|---|
| pillow | 12.3.0 | 13.0.0 | gradio (`pillow<13`) | Crafted image files (for example TIFF or PDF) can crash the process or read out-of-bounds memory when decoded. Run the Gradio app locally only (no `share=True`, no public port) and open only images you trust. |

## GPU

PyTorch is installed from the CUDA 12.6 wheel index (torch 2.14.1+cu126), which needs an NVIDIA driver that supports CUDA 12.6 or newer. Inference was verified on a local GPU with the real model weights.
