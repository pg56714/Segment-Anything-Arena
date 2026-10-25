# Security Log

English | [繁體中文](security-log.zh-TW.md)

- Date: 2026-10-25
- Branch: `fix/security-upgrades`
- Scope: `requirements.txt`

## Summary

gradio was moved off 4.19.2 to a patched release. pillow still has open advisories because gradio caps its version.

## Changes

| Item | Before | After | Reason |
|---|---|---|---|
| gradio | 4.19.2 | >=6.16 (6.30.0) | File read, SSRF and path traversal advisories (fixed in 6.16) |

## What you need to know

- The original comment noted that gradio 4.19.2 ran faster; the UI may feel slower on 6.x.

## Remaining issues

### Dependency versions held back upstream

| Package | Version | Advisories | Why it is not fixed | Impact and mitigation |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 (fixed in 13.0.0) | gradio >= 6.16 requires `pillow<13`; no gradio release allows Pillow 13 yet. | Crafted image files (for example TIFF or PDF) can crash the process or read out-of-bounds memory when decoded. Run the Gradio app locally only (no `share=True`, no public port) and open only images you trust. Upgrade once gradio allows Pillow 13. |

## How it was verified

- osv-scanner on the resolved `requirements.txt`; only pillow remains.
- The UI builds and serves on gradio 6.30.0 with the EfficientSAM / EfficientViT-SAM models stubbed out.
- `annotate_image_with_box_prompt_result()` runs with supervision 0.30.9.

## Not verified

- Inference on a GPU with the real model weights (the test environment has no GPU and no weights).
