#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare-figures.py — compare test-generated SVG figures against the references.

The tests in `src/mesh/draw.rs` (and a few others) save SVG figures to
`/tmp/gemlab`. The committed reference figures live in `data/figures`.

Two comparison modes are available:

1) Text mode (default)
   Matplotlib embeds a timestamp and its own version in every SVG (`<metadata>`),
   and different matplotlib releases render slightly different coordinates. This
   mode *normalises* the SVGs before comparing — it drops the DOCTYPE, comments
   and the `<metadata>` block, and (by default) rounds floating-point numbers —
   so it reports real **content** differences instead of noise.

2) Raster mode (`--raster`)
   Rasterises both SVGs to PNG at the same pixel width (using `rsvg-convert`,
   `inkscape` or ImageMagick — whichever is installed) and compares pixels.
   This is tolerant to text/coordinate noise and anti-aliasing; it reports the
   mean absolute error, the percentage of differing pixels and the max delta.

Typical use (from the repository root):

    # text compare of the figures produced by mesh/draw.rs tests
    ./zscripts/compare-figures.py --pattern 'test_draw_*.svg' --diff

    # pixel compare (visual, tolerant to matplotlib version differences)
    ./zscripts/compare-figures.py --pattern 'test_draw_*.svg' --raster \
        --pixel-tol 24 --max-diff-pct 1.0 --diff-image /tmp/gemlab/diffs

    # refresh the committed references from the current output
    # (review the diff WITHOUT --update first!)
    ./zscripts/compare-figures.py --pattern 'test_draw_*.svg' --update

Exit status: 0 when every compared figure is identical (or within tolerance in
raster mode) and none is missing; 1 otherwise. With --update the status reflects
the comparison *before* copying.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# defaults / paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED = Path("/tmp/gemlab")
DEFAULT_REFERENCE = REPO_ROOT / "data" / "figures"

# statuses
OK = "IDENTICAL"
DIFF = "DIFF"
MISSING = "MISSING"

# raster engines, in order of preference
ENGINE_BIN = {"rsvg": "rsvg-convert", "inkscape": "inkscape", "magick": "magick"}

# ---------------------------------------------------------------------------
# SVG normalisation (text mode)
# ---------------------------------------------------------------------------

# matches signed decimals with a fractional part and/or an exponent, but not
# fragments of identifiers such as "v3.11.1"
FLOAT_RE = re.compile(
    r"(?<![0-9A-Za-z_.])"
    r"-?(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)"
    r"(?![0-9A-Za-z_])"
)


def _round(match: "re.Match[str]", decimals: int) -> str:
    try:
        value = float(match.group(0))
    except ValueError:  # pragma: no cover - regex guarantees parseable input
        return match.group(0)
    if value == 0.0:
        value = 0.0  # collapse "-0.0" and "0.0"
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


# Matplotlib uses random identifiers for clip paths, markers and images (e.g.
# id="p55578eea23", id="md60f825453", id="image192e7172cd", together with the
# matching clip-path="url(#...)" / xlink:href="#..."). They are not stable
# between runs but do not affect rendering, so mask both the definition and its
# reference. Deterministic ids contain an underscore (e.g. "patch_1",
# "line2d_19") and are left untouched.
RANDOM_ID_DEF_RE = re.compile(r'id="[A-Za-z0-9_]*[0-9a-fA-F]{8,}"')
RANDOM_ID_REF_RE = re.compile(r'(url\(#|href="#)[A-Za-z0-9_]*[0-9a-fA-F]{8,}')


def normalize(text: str, decimals: int) -> str:
    """Remove volatile parts and (optionally) round floating-point numbers."""
    text = re.sub(r"<!DOCTYPE.*?>", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<metadata>.*?</metadata>", "", text, flags=re.S)
    text = RANDOM_ID_DEF_RE.sub('id="ID"', text)
    text = RANDOM_ID_REF_RE.sub(r"\1ID", text)
    if decimals >= 0:
        text = FLOAT_RE.sub(lambda m: _round(m, decimals), text)
    return text


def canonical(text: str, decimals: int) -> str:
    """Whitespace-insensitive form, used for equality checking."""
    return " ".join(normalize(text, decimals).split())


def diff_lines(text: str, decimals: int) -> list[str]:
    """Significant lines, used for similarity ratio and unified diffs."""
    return [line.strip() for line in normalize(text, decimals).splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# rasterisation (raster mode)
# ---------------------------------------------------------------------------


def detect_engine(preferred: str) -> str:
    candidates = ["rsvg", "inkscape", "magick"] if preferred == "auto" else [preferred]
    for engine in candidates:
        if shutil.which(ENGINE_BIN[engine]):
            return engine
    raise RuntimeError(
        "no SVG rasteriser found; install one of: "
        + ", ".join(ENGINE_BIN.values())
    )


def render_svg(svg: Path, png: Path, width: int, engine: str) -> None:
    if engine == "rsvg":
        cmd = ["rsvg-convert", "--width", str(width), "--output", str(png), str(svg)]
    elif engine == "inkscape":
        cmd = [
            "inkscape",
            str(svg),
            "--export-type=png",
            f"--export-width={width}",
            f"--export-filename={png}",
        ]
    elif engine == "magick":
        cmd = ["magick", "-background", "white", str(svg), "-resize", f"{width}x", "-flatten", str(png)]
    else:  # pragma: no cover - guarded by detect_engine
        raise RuntimeError(f"unknown raster engine: {engine}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{engine} failed for {svg.name}: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )


def raster_metrics(
    generated: Path,
    reference: Path,
    engine: str,
    width: int,
    pixel_tol: int,
    diff_dir: Path | None,
) -> dict:
    """Rasterise both SVGs and return pixel-difference metrics."""
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"raster mode requires numpy and Pillow: {exc}")

    with tempfile.TemporaryDirectory(prefix="compare-figures-") as tmp:
        tmp_dir = Path(tmp)
        gen_png = tmp_dir / "gen.png"
        ref_png = tmp_dir / "ref.png"
        render_svg(generated, gen_png, width, engine)
        render_svg(reference, ref_png, width, engine)

        gen_img = Image.open(gen_png).convert("RGB")
        ref_img = Image.open(ref_png).convert("RGB")

        canvas_w = max(gen_img.width, ref_img.width)
        canvas_h = max(gen_img.height, ref_img.height)

        def pad(img):
            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            canvas.paste(img, (0, 0))
            return canvas

        a = np.asarray(pad(gen_img), dtype=np.int16)
        b = np.asarray(pad(ref_img), dtype=np.int16)
        delta = np.abs(a - b)
        mask = delta.max(axis=2) > pixel_tol

        metrics = {
            "engine": engine,
            "width": canvas_w,
            "height": canvas_h,
            "height_generated": gen_img.height,
            "height_reference": ref_img.height,
            "mae": float(delta.mean()),
            "max_delta": int(delta.max()),
            "diff_pct": 100.0 * float(mask.mean()),
        }

        if diff_dir is not None and mask.any():
            diff_dir.mkdir(parents=True, exist_ok=True)
            stem = generated.stem
            heat = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
            heat[mask] = (255, 0, 0)
            Image.fromarray(heat).save(diff_dir / f"{stem}.diff.png")
            side = Image.new("RGB", (2 * canvas_w, canvas_h), (255, 255, 255))
            side.paste(gen_img, (0, 0))
            side.paste(ref_img, (canvas_w, 0))
            side.save(diff_dir / f"{stem}.side.png")

        return metrics


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


class Result:
    def __init__(self, name: str, generated: Path, reference: Path, status: str,
                 ratio: float = 0.0, diff: str = "", metrics: dict | None = None) -> None:
        self.name = name
        self.generated = generated
        self.reference = reference
        self.status = status
        self.ratio = ratio
        self.diff = diff
        self.metrics = metrics or {}


def compare_text(generated: Path, reference: Path, decimals: int, want_diff: bool,
                 max_lines: int) -> Result:
    gen_text = generated.read_text(encoding="utf-8", errors="replace")
    ref_text = reference.read_text(encoding="utf-8", errors="replace")

    if canonical(gen_text, decimals) == canonical(ref_text, decimals):
        return Result(generated.name, generated, reference, OK, 1.0)

    gen_lines = diff_lines(gen_text, decimals)
    ref_lines = diff_lines(ref_text, decimals)
    ratio = difflib.SequenceMatcher(None, gen_lines, ref_lines).ratio()

    diff_text = ""
    if want_diff:
        diff_text = "\n".join(
            difflib.unified_diff(
                ref_lines,
                gen_lines,
                fromfile=f"reference/{reference.name}",
                tofile=f"generated/{generated.name}",
                lineterm="",
            )
        )
        lines = diff_text.splitlines()
        if len(lines) > max_lines:
            diff_text = "\n".join(lines[:max_lines] + [f"… ({len(lines) - max_lines} more lines)"])

    return Result(generated.name, generated, reference, DIFF, ratio, diff_text)


def compare_raster(generated: Path, reference: Path, args: argparse.Namespace,
                   engine: str) -> Result:
    metrics = raster_metrics(
        generated,
        reference,
        engine,
        args.raster_width,
        args.pixel_tol,
        args.diff_image,
    )
    diff_pct = metrics["diff_pct"]
    ratio = 1.0 - diff_pct / 100.0
    status = OK if diff_pct <= args.max_diff_pct else DIFF
    details = (
        f"mae={metrics['mae']:.3f} max={metrics['max_delta']} "
        f"diff={diff_pct:.3f}% (tol {args.pixel_tol}, max {args.max_diff_pct}%) "
        f"size={metrics['width']}x{metrics['height']} "
        f"(gen {metrics['height_generated']}px, ref {metrics['height_reference']}px)"
    )
    return Result(generated.name, generated, reference, status, ratio, details, metrics)


def compare_file(generated: Path, reference: Path, args: argparse.Namespace,
                 engine: str | None) -> Result:
    if engine is not None:
        return compare_raster(generated, reference, args, engine)
    return compare_text(generated, reference, args.decimals, args.diff, args.max_lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare test-generated SVGs with the reference figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--generated", type=Path, default=DEFAULT_GENERATED,
                   help=f"directory with generated SVGs (default: {DEFAULT_GENERATED})")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                   help=f"directory with reference SVGs (default: {DEFAULT_REFERENCE})")
    p.add_argument("--pattern", default="*.svg",
                   help="glob for generated files (default: *.svg)")
    p.add_argument("--recursive", action="store_true",
                   help="search the generated directory recursively (for projects that write "
                        "figures into sub-directories, e.g. /tmp/plotpy/{doc_tests,integ_tests}); "
                        "files are still matched to references by basename")
    p.add_argument("--diff", action="store_true",
                   help="text mode: print a unified diff for the files that differ")
    p.add_argument("--max-lines", type=int, default=40,
                   help="text mode: max diff lines printed per file (default: 40)")
    p.add_argument("--decimals", type=int, default=6,
                   help="text mode: round floats to N decimals; -1 disables (default: 6)")
    p.add_argument("--raster", action="store_true",
                   help="pixel-compare rasterised SVGs instead of comparing text")
    p.add_argument("--raster-engine", choices=["auto", "rsvg", "inkscape", "magick"],
                   default="auto", help="SVG rasteriser (default: auto)")
    p.add_argument("--raster-width", type=int, default=800,
                   help="raster mode: output pixel width (default: 800)")
    p.add_argument("--pixel-tol", type=int, default=16,
                   help="raster mode: per-channel 0-255 tolerance (default: 16)")
    p.add_argument("--max-diff-pct", type=float, default=1.0,
                   help="raster mode: max %% of differing pixels to still pass (default: 1.0)")
    p.add_argument("--diff-image", type=Path, default=None,
                   help="raster mode: directory where diff/side-by-side PNGs are written")
    p.add_argument("--json", action="store_true",
                   help="print a machine-readable JSON summary to stdout")
    p.add_argument("--update", action="store_true",
                   help="copy each generated file over its reference (also creates missing ones)")
    p.add_argument("--quiet", action="store_true",
                   help="only print the final summary")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.generated.is_dir():
        print(f"error: generated directory not found: {args.generated}", file=sys.stderr)
        return 2
    if not args.reference.is_dir():
        print(f"error: reference directory not found: {args.reference}", file=sys.stderr)
        return 2

    engine: str | None = None
    if args.raster:
        try:
            engine = detect_engine(args.raster_engine)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.recursive:
        generated_files = sorted(p for p in args.generated.rglob(args.pattern) if p.is_file())
    else:
        generated_files = sorted(p for p in args.generated.glob(args.pattern) if p.is_file())
    if not generated_files:
        print(f"error: no files match {args.pattern!r} in {args.generated}", file=sys.stderr)
        return 2

    # guard against basename collisions when searching recursively
    seen: set[str] = set()
    for gen in generated_files:
        if gen.name in seen:
            print(f"error: duplicate basename {gen.name!r} under {args.generated}", file=sys.stderr)
            return 2
        seen.add(gen.name)

    results: list[Result] = []
    for gen in generated_files:
        ref = args.reference / gen.name
        if not ref.is_file():
            results.append(Result(gen.name, gen, ref, MISSING))
            continue
        try:
            results.append(compare_file(gen, ref, args, engine))
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    identical = [r for r in results if r.status == OK]
    different = [r for r in results if r.status == DIFF]
    missing = [r for r in results if r.status == MISSING]

    if not args.quiet and not args.json:
        header = "raster" if args.raster else "text"
        for r in results:
            if r.status == DIFF:
                print(f"  {r.status:<9} {r.name}  (similarity {r.ratio * 100:5.1f}%)")
                if args.raster:
                    if r.diff:
                        print(f"    {r.diff}")
                elif r.diff:
                    print()
                    print(r.diff)
                    print()
            elif r.status == MISSING:
                print(f"  {r.status:<9} {r.name}  (no reference file)")
        print(f"mode: {header}")

    if args.update:
        copied = 0
        for r in results:
            r.reference.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r.generated, r.reference)
            copied += 1
        print(f"updated {copied} reference file(s) in {args.reference}")

    if args.json:
        payload = {
            "generated_dir": str(args.generated),
            "reference_dir": str(args.reference),
            "pattern": args.pattern,
            "mode": "raster" if args.raster else "text",
            "summary": {
                "total": len(results),
                "identical": len(identical),
                "different": len(different),
                "missing": len(missing),
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "similarity": round(r.ratio, 6),
                    "generated": str(r.generated),
                    "reference": str(r.reference),
                    "metrics": r.metrics,
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print()
        print(
            f"summary: {len(results)} compared | "
            f"{len(identical)} identical | {len(different)} different | {len(missing)} missing"
        )

    return 0 if not different and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
