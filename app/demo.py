#!/usr/bin/env python3
"""Gradio demo - the thing that goes in the video.

    python app/demo.py --checkpoint checkpoints/detector_robust.pt

Three tabs, in the order the story should be told:

1. **Score an image** - upload, get a calibrated score and the occlusion map.
2. **Damage it live** - the tab that makes the point. Move a slider, watch the
   image degrade, watch the score hold. This is the demo; the other two support
   it.
3. **Score a folder** - the batch path, showing `predict.py`'s output format
   without leaving the browser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from aigcdet.aug.transforms import (  # noqa: E402
    center_crop,
    color_jitter,
    downscale_upscale,
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
)
from aigcdet.explain.saliency import occlusion_saliency, overlay_heatmap  # noqa: E402
from aigcdet.models.detector import Detector  # noqa: E402
from aigcdet.utils.io import iter_image_files  # noqa: E402

DETECTOR: Detector | None = None

VERDICT_CSS = """
.verdict {font-size: 1.5rem; font-weight: 700; padding: .6rem .9rem; border-radius: .5rem;}
.verdict.ai   {background:#fdece5; color:#8f3311;}
.verdict.real {background:#e6f0fc; color:#17457d;}
"""


def _verdict_html(score: float, threshold: float) -> str:
    is_ai = score >= threshold
    cls = "ai" if is_ai else "real"
    word = "Likely AI-generated" if is_ai else "Likely authentic"
    return (
        f'<div class="verdict {cls}">{word} &mdash; {score:.3f}</div>'
        f'<p style="color:#52514e;font-size:.85rem;margin-top:.4rem">'
        f"Score is P(AI-generated), calibrated on held-out data. The decision threshold is "
        f"{threshold:.3f}, chosen so that no more than 5% of authentic images are flagged. "
        f"A score near {threshold:.2f} means the model is genuinely unsure &mdash; treat it as "
        f"'send to human review', not as an answer.</p>"
    )


def score_single(image, want_saliency: bool):
    if image is None:
        return "", None
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    score = float(DETECTOR.score_images([img])[0])
    heat = None
    if want_saliency:
        smap = occlusion_saliency(DETECTOR, img, grid=8)
        heat = overlay_heatmap(img, smap, alpha=0.55, size=384)
    return _verdict_html(score, DETECTOR.threshold), heat


def damage_and_score(image, jpeg_q, blur_sigma, scale, noise_sigma, color_strength, crop_keep):
    """Apply the chosen damage chain, then score - the robustness demo."""
    if image is None:
        return None, "", ""
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    clean_score = float(DETECTOR.score_images([img])[0])

    damaged, applied = img, []
    if crop_keep < 1.0:
        damaged = center_crop(damaged, crop_keep); applied.append(f"crop {crop_keep:.2f}")
    if scale < 1.0:
        damaged = downscale_upscale(damaged, scale); applied.append(f"rescale {scale:.2f}x")
    if blur_sigma > 0:
        damaged = gaussian_blur(damaged, blur_sigma); applied.append(f"blur sigma {blur_sigma:.1f}")
    if noise_sigma > 0:
        damaged = gaussian_noise(damaged, noise_sigma); applied.append(f"noise {noise_sigma:.3f}")
    if color_strength > 0:
        damaged = color_jitter(damaged, color_strength); applied.append(f"colour +-{color_strength:.0%}")
    if jpeg_q < 100:
        damaged = jpeg_compress(damaged, int(jpeg_q)); applied.append(f"JPEG q{int(jpeg_q)}")

    damaged_score = float(DETECTOR.score_images([damaged])[0])
    delta = damaged_score - clean_score
    chain = " -> ".join(applied) if applied else "no damage applied"

    table = (
        f"| | score | verdict |\n|---|---|---|\n"
        f"| original | {clean_score:.3f} | {'AI' if clean_score >= DETECTOR.threshold else 'authentic'} |\n"
        f"| damaged | {damaged_score:.3f} | {'AI' if damaged_score >= DETECTOR.threshold else 'authentic'} |\n"
        f"| change | {delta:+.3f} | {'verdict held' if (clean_score >= DETECTOR.threshold) == (damaged_score >= DETECTOR.threshold) else '**verdict flipped**'} |\n"
    )
    return damaged, f"**Damage chain:** {chain}", table


def score_folder(folder_path: str, limit: int = 200):
    if not folder_path:
        return "Enter a directory path.", None
    p = Path(folder_path).expanduser()
    if not p.exists():
        return f"No such directory: {p}", None
    paths = [str(q) for q in iter_image_files(p)][:limit]
    if not paths:
        return f"No images found under {p}", None
    preds = DETECTOR.predict_paths(paths, progress=False)
    records = [x.as_record() for x in preds]
    rows = [[r["image_path"], r["pred"], "AI" if r["pred"] >= DETECTOR.threshold else "authentic"]
            for r in records]
    return json.dumps(records[:20], indent=2), rows


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Robust AIGC Image Detector", css=VERDICT_CSS) as ui:
        gr.Markdown(
            "# Robust AI-generated image detector\n"
            "A frozen CLIP encoder plus a small calibrated head, trained on deliberately "
            "damaged images. The claim this demo is here to test is not *'it is accurate'* - "
            "it is *'it stays accurate after the image has been compressed, cropped, "
            "resized and reposted.'*"
        )

        with gr.Tab("1 · Score an image"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="pil", label="Image")
                    sal = gr.Checkbox(value=True, label="Show occlusion map (slower)")
                    btn = gr.Button("Score", variant="primary")
                with gr.Column():
                    out_html = gr.HTML()
                    out_heat = gr.Image(label="Occlusion map — red pushed the score toward 'AI'")
            btn.click(score_single, [inp, sal], [out_html, out_heat])

        with gr.Tab("2 · Damage it and score again"):
            gr.Markdown(
                "Apply the transformations from the problem statement and watch what happens "
                "to the score. The baseline model that saw only pristine training images "
                "flips its verdict here; this one is meant not to."
            )
            with gr.Row():
                with gr.Column():
                    d_in = gr.Image(type="pil", label="Image")
                    jpeg_q = gr.Slider(20, 100, value=100, step=5, label="JPEG quality (100 = none)")
                    blur = gr.Slider(0, 3, value=0, step=0.1, label="Gaussian blur σ")
                    scale = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Rescale factor")
                    noise = gr.Slider(0, 0.15, value=0, step=0.005, label="Gaussian noise σ")
                    colr = gr.Slider(0, 0.4, value=0, step=0.02, label="Colour jitter ±")
                    crop = gr.Slider(0.5, 1.0, value=1.0, step=0.05, label="Centre crop keep")
                    d_btn = gr.Button("Damage and score", variant="primary")
                with gr.Column():
                    d_out = gr.Image(label="Damaged image")
                    d_chain = gr.Markdown()
                    d_table = gr.Markdown()
            d_btn.click(
                damage_and_score,
                [d_in, jpeg_q, blur, scale, noise, colr, crop],
                [d_out, d_chain, d_table],
            )

        with gr.Tab("3 · Score a folder"):
            gr.Markdown(
                "The same code path as `predict.py`. Output is the exact submission format: "
                "a JSON array of `{image_path, pred}`."
            )
            f_in = gr.Textbox(label="Directory path", placeholder="data/synthetic/real")
            f_btn = gr.Button("Score folder", variant="primary")
            f_json = gr.Code(label="predictions.json (first 20)", language="json")
            f_table = gr.Dataframe(headers=["image_path", "pred", "verdict"], label="All results")
            f_btn.click(score_folder, [f_in], [f_json, f_table])

        gr.Markdown(
            "---\n**Limitations, stated up front:** this is a prototype trained on public "
            "research datasets. It has not seen every generator, it degrades on image types "
            "under-represented in its training data (illustration, screenshots, heavy "
            "post-processing), and a score near the threshold means *uncertain*, not *innocent*. "
            "It is a triage aid for human reviewers, not an arbiter."
        )
    return ui


def main() -> int:
    global DETECTOR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "detector_robust.pt")
    ap.add_argument("--backbone", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--share", action="store_true", help="public Gradio link (needed in Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"checkpoint not found: {args.checkpoint}\n"
            "Train one first - see the Quickstart in the README."
        )
    DETECTOR = Detector(args.checkpoint, backbone_name=args.backbone, device=args.device)
    build_ui().launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
