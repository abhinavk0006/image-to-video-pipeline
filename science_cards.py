"""Readable 16:9 science cards. CPU-only Pillow layout with optional mathtext.

Text between $...$ or $$...$$ is rendered as a separate mathematical block.
This supports matplotlib's LaTeX subset, not arbitrary TeX or mhchem commands.
Long content is paginated; scientific text is never silently truncated.
"""
from __future__ import annotations

import io
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont

THEMES = {
    "chemistry": ((85, 115, 255), (147, 91, 244)),
    "physics": ((30, 186, 214), (76, 117, 248)),
    "biology": ((40, 190, 145), (75, 160, 215)),
}
BACKGROUND = (13, 14, 21)
TEXT = (245, 246, 252)
MUTED = (177, 186, 211)
SIZE = (1280, 720)


def _font(size: int, bold: bool = False):
    names = (["DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"] if bold
             else ["DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _wrap(text: str, font, width: int) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            if line and draw.textlength(line + " " + word, font=font) > width:
                lines.append(line)
                line = ""
            # Split a very long token rather than allowing it to leave the canvas.
            if draw.textlength(word, font=font) > width:
                for char in word:
                    if draw.textlength(line + char, font=font) > width:
                        lines.append(line)
                        line = ""
                    line += char
            else:
                line = (line + " " + word).strip()
        lines.append(line)
    return lines


def _math(expression: str, width: int) -> Image.Image:
    try:
        from matplotlib.mathtext import math_to_image
        from matplotlib.font_manager import FontProperties
    except ImportError as exc:
        raise RuntimeError("Mathematical cards require matplotlib; install it before rendering") from exc
    output = io.BytesIO()
    try:
        math_to_image("$" + expression.strip() + "$", output,
                      prop=FontProperties(size=27), dpi=144, format="png", color="black")
    except (ValueError, RuntimeError) as exc:
        raise ValueError("Unsupported card formula: use mathtext syntax such as \\mathrm{AgCl}; full TeX/mhchem is not supported") from exc
    output.seek(0)
    # Handle either a transparent or opaque backend background.
    import numpy as np
    coverage = np.asarray(Image.open(output).convert("RGBA"))
    pixels = np.zeros_like(coverage)
    pixels[:, :, :3] = TEXT
    pixels[:, :, 3] = ((255 - coverage[:, :, :3].min(axis=2)).astype(float) * coverage[:, :, 3] / 255).astype("uint8")
    image = Image.fromarray(pixels)
    if image.width > width:
        image = image.resize((width, max(1, round(image.height * width / image.width))), Image.Resampling.LANCZOS)
    return image


def render_cards(directory: str | Path, name: str, heading: str, body: str = "", *,
                 subject: str = "chemistry", kind: str = "procedure", footer: str = "") -> list[Path]:
    """Render one or more cards and return every page in presentation order."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Card name must be a simple filename stem")
    accent, secondary = THEMES.get(subject.lower(), THEMES["chemistry"])
    title = kind == "title"
    font_size = 62 if title else 47
    while True:
        heading_font = _font(font_size, True)
        headings = _wrap(heading, heading_font, 1104)
        if len(headings) <= 3 or font_size <= 32:
            break
        font_size -= 2
    if len(headings) > 3:
        raise ValueError("Card heading is too long; use a shorter heading and move detail into the body")
    line_height = font_size + 17
    heading_y = (max(140, (720 - len(headings) * line_height) // 2 - 12)
                 if title and not body.strip() else 180 if title else 116)
    body_y = heading_y + len(headings) * line_height + 36
    bottom = 618
    available = bottom - body_y
    if available < 60 and body:
        heading_y = 100
        body_y = heading_y + len(headings) * line_height + 28
        available = bottom - body_y
    body_font = _font(33)
    blocks = []
    for index, part in enumerate(re.split(r"\$\$?([^$]+)\$\$?", body)):
        if index % 2:
            formula = _math(part, 1056)
            blocks.append((formula, formula.height + 28))
        elif part.strip():
            for line in _wrap(part.strip(), body_font, 1056):
                blocks.append((line, 51))
    pages, page, used = [], [], 0
    for block, height in blocks:
        if height > available:
            raise ValueError("Formula or content block exceeds card height; simplify the expression")
        if page and used + height > available:
            pages.append(page)
            page, used = [], 0
        page.append((block, height))
        used += height
    pages.append(page)
    paths = []
    for page_index, content in enumerate(pages, 1):
        image = Image.new("RGB", SIZE, BACKGROUND)
        draw = ImageDraw.Draw(image)
        for x in range(SIZE[0]):
            ratio = x / (SIZE[0] - 1)
            colour = tuple(round(a + (b - a) * ratio) for a, b in zip(accent, secondary))
            draw.line((x, 0, x, 9), fill=colour)
        label = f"{subject.upper()}  /  {kind.upper()}"
        draw.text((88, 53), label, font=_font(20, True), fill=accent)
        for index, line in enumerate(headings):
            x = (1280 - draw.textlength(line, font=heading_font)) / 2 if title else 88
            draw.text((x, heading_y + index * line_height), line, font=heading_font, fill=TEXT)
        y = body_y
        if content:
            draw.rounded_rectangle((72, body_y - 12, 1208, bottom + 12), radius=18, fill=(21, 25, 40))
        for block, height in content:
            if isinstance(block, Image.Image):
                image.paste(block, ((1280 - block.width) // 2, y + 6), block)
            else:
                draw.text((104, y), block, font=body_font, fill=MUTED)
            y += height
        draw.line((88, 657, 1192, 657), fill=(48, 53, 74), width=1)
        if footer:
            footer_font = _font(18)
            footer_lines = _wrap(footer, footer_font, 940)
            if len(footer_lines) > 1:
                raise ValueError("Footer is too long; move detail into card body")
            draw.text((88, 674), footer, font=footer_font, fill=MUTED)
        if len(pages) > 1:
            draw.text((1114, 674), f"{page_index}/{len(pages)}", font=_font(18), fill=MUTED)
        path = directory / f"{name}{'_' + str(page_index) if len(pages) > 1 else ''}.png"
        image.save(path)
        paths.append(path)
    return paths
