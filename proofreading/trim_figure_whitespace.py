#!/usr/bin/env python3
"""Trim surrounding whitespace from a PNG, JPG, or PDF figure.

Examples:
    python proofreading/trim_figure_whitespace.py chapters/chapter4/figs/LDA.png
    python proofreading/trim_figure_whitespace.py fig.jpg --in-place
    python proofreading/trim_figure_whitespace.py fig.pdf -o fig_trimmed.pdf --margin 6

Raster images are cropped by removing white or transparent borders.
PDFs are cropped with ``pdfcrop`` so vector content stays vector.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PDF_SUFFIXES = {".pdf"}
PDF_RENDER_DPI = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input PNG, JPG, or PDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to <input>_trimmed.<ext>.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a new file.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="Pixels with grayscale values at or above this value are treated as white background.",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=0,
        help="Pixels with alpha at or below this value are treated as transparent background.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Extra border to keep. Units are pixels for images and bp for PDFs.",
    )
    args = parser.parse_args()

    if args.output and args.in_place:
        parser.error("choose either --output or --in-place")
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if not 0 <= args.alpha_threshold <= 255:
        parser.error("--alpha-threshold must be between 0 and 255")
    if args.margin < 0:
        parser.error("--margin must be non-negative")

    return args


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_trimmed{input_path.suffix}")


def has_transparent_corners(image: Image.Image, alpha_threshold: int) -> bool:
    if "A" not in image.getbands():
        return False

    alpha = image.getchannel("A")
    width, height = alpha.size
    corners = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ]
    return all(alpha.getpixel(corner) <= alpha_threshold for corner in corners)


def content_bbox(
    image: Image.Image,
    threshold: int,
    alpha_threshold: int,
) -> tuple[int, int, int, int] | None:
    if has_transparent_corners(image, alpha_threshold):
        return image.getchannel("A").point(
            lambda value: 255 if value > alpha_threshold else 0
        ).getbbox()

    non_white_mask = image.convert("L").point(
        lambda value: 255 if value < threshold else 0
    )
    if "A" in image.getbands():
        opaque_mask = image.getchannel("A").point(
            lambda value: 255 if value > alpha_threshold else 0
        )
        non_white_mask = ImageChops.multiply(non_white_mask, opaque_mask)
    return non_white_mask.getbbox()


def expand_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = size
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )


def image_save_kwargs(
    source_image: Image.Image,
    source_format: str | None,
) -> dict[str, object]:
    save_kwargs: dict[str, object] = {}
    if "icc_profile" in source_image.info:
        save_kwargs["icc_profile"] = source_image.info["icc_profile"]

    exif = source_image.getexif()
    if exif:
        exif[274] = 1
        save_kwargs["exif"] = exif.tobytes()

    if source_format == "JPEG":
        save_kwargs["quality"] = 95
        save_kwargs["subsampling"] = 0
    return save_kwargs


def trim_image(
    input_path: Path,
    output_path: Path,
    threshold: int,
    alpha_threshold: int,
    margin: float,
) -> None:
    with Image.open(input_path) as original:
        image = ImageOps.exif_transpose(original)
        source_format = original.format
        bbox = content_bbox(image, threshold, alpha_threshold)
        if bbox is None:
            raise ValueError("no non-background content found")

        crop_margin = int(round(margin))
        cropped = image.crop(expand_bbox(bbox, image.size, crop_margin))
        save_kwargs = image_save_kwargs(original, source_format)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=output_path.suffix,
            dir=output_path.parent,
        ) as tmp:
            temp_path = Path(tmp.name)

        try:
            cropped.save(temp_path, format=source_format, **save_kwargs)
            temp_path.replace(output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def format_pdf_margin(margin: float) -> str:
    return str(margin).rstrip("0").rstrip(".") if margin else "0"


def format_pdf_bbox(bbox: tuple[float, float, float, float]) -> str:
    return " ".join(
        f"{value:.3f}".rstrip("0").rstrip(".")
        for value in bbox
    )


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{command[0]} was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        detail = stderr or stdout or f"{command[0]} exited with status {exc.returncode}"
        raise RuntimeError(detail) from exc


def pdf_page_sizes(input_path: Path) -> list[tuple[float, float]]:
    info = run_command(
        ["pdfinfo", "-f", "1", "-l", "999999", str(input_path)]
    ).stdout
    sizes = [
        (float(width), float(height))
        for width, height in re.findall(
            r"^Page\s+\d+\s+size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
            info,
            flags=re.MULTILINE,
        )
    ]
    if sizes:
        return sizes

    match = re.search(
        r"^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
        info,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("could not parse page size from pdfinfo output")
    return [(float(match.group(1)), float(match.group(2)))]


def rendered_pdf_bbox(
    input_path: Path,
    threshold: int,
) -> tuple[float, float, float, float] | None:
    if shutil.which("pdftoppm") is None:
        return None

    page_sizes = pdf_page_sizes(input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "page"
        run_command(
            [
                "pdftoppm",
                "-png",
                "-r",
                str(PDF_RENDER_DPI),
                str(input_path),
                str(prefix),
            ]
        )
        image_paths = sorted(
            Path(tmpdir).glob("page-*.png"),
            key=lambda path: int(path.stem.rsplit("-", 1)[1]),
        )
        if not image_paths:
            raise RuntimeError("pdftoppm did not produce any rendered pages")

        if len(page_sizes) == 1 and len(image_paths) > 1:
            page_sizes = page_sizes * len(image_paths)
        if len(page_sizes) != len(image_paths):
            raise RuntimeError(
                "pdfinfo page metadata did not match the rendered page count"
            )

        union_bbox: tuple[float, float, float, float] | None = None
        for image_path, (page_width, page_height) in zip(image_paths, page_sizes):
            with Image.open(image_path) as page_image:
                bbox = content_bbox(
                    page_image,
                    threshold=threshold,
                    alpha_threshold=0,
                )
                if bbox is None:
                    continue

                left, top, right, bottom = bbox
                image_width, image_height = page_image.size
                pdf_bbox = (
                    left * page_width / image_width,
                    page_height - (bottom * page_height / image_height),
                    right * page_width / image_width,
                    page_height - (top * page_height / image_height),
                )

            if union_bbox is None:
                union_bbox = pdf_bbox
            else:
                union_bbox = (
                    min(union_bbox[0], pdf_bbox[0]),
                    min(union_bbox[1], pdf_bbox[1]),
                    max(union_bbox[2], pdf_bbox[2]),
                    max(union_bbox[3], pdf_bbox[3]),
                )

    return union_bbox


def trim_pdf(
    input_path: Path,
    output_path: Path,
    margin: float,
    threshold: int,
) -> None:
    if shutil.which("pdfcrop") is None:
        raise RuntimeError(
            "pdfcrop was not found on PATH. Install TeX Live's pdfcrop command to trim PDFs."
        )

    rendered_bbox = rendered_pdf_bbox(input_path, threshold=threshold)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=output_path.suffix,
        dir=output_path.parent,
    ) as tmp:
        temp_path = Path(tmp.name)

    command = [
        "pdfcrop",
        "--quiet",
        "--hires",
        "--margins",
        format_pdf_margin(margin),
    ]
    if rendered_bbox is not None:
        command.extend(["--bbox", format_pdf_bbox(rendered_bbox)])
    command.extend([str(input_path), str(temp_path)])

    try:
        run_command(command)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    args = parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    suffix = input_path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES | PDF_SUFFIXES:
        print(
            "Error: supported formats are PNG, JPG, JPEG, and PDF.",
            file=sys.stderr,
        )
        return 1

    output_path = (
        input_path
        if args.in_place
        else (args.output or default_output_path(input_path))
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if suffix in IMAGE_SUFFIXES:
            trim_image(
                input_path=input_path,
                output_path=output_path,
                threshold=args.threshold,
                alpha_threshold=args.alpha_threshold,
                margin=args.margin,
            )
        else:
            trim_pdf(
                input_path=input_path,
                output_path=output_path,
                margin=args.margin,
                threshold=args.threshold,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
