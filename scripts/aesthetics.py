"""Shared plotting defaults for notebooks and project code.

Drop this file into a repo and import it from plotting code that should share
the same figure sizing, rcParams, tick formatters, and colors.

Typical usage:

```python
import matplotlib.pyplot as plt
import seaborn as sns

import aesthetics as aes

fig, ax = plt.subplots(figsize=(aes.PAPER_WIDTH_IN, aes.FIG_HEIGHT_SINGLE_ROW_IN))
ax.yaxis.set_major_formatter(aes.NICE_FORMATTER)
ax.plot(x, y, color=aes.NONSEMANTIC_COLOR)
```

If you are iterating on a local copy of this module inside Jupyter, enable
autoreload in a notebook cell so edits to `aesthetics.py` are picked up when
you rerun imports:

```python
%load_ext autoreload
%autoreload 2

import aesthetics as aes
```

With `%autoreload 2`, changes saved to this file are reloaded automatically
before each cell execution.
"""

from __future__ import annotations

import colorsys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeGuard, cast, overload

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

Color = tuple[float, float, float]
type ColorLike = str | Color
Palette = dict[str, Color]
type PaletteNameFormatter = Callable[[str, str], str]

# Widths are text block widths in inches, taken from the official conference
# LaTeX styles/templates current as of March 10, 2026.
_CM_TO_IN: float = 1.0 / 2.54

PAPER_WIDTH_IN: float = 5.5
FIG_HEIGHT_SINGLE_ROW_IN: float = 1.3
FIG_HEIGHT_DOUBLE_ROW_IN: float = 2.8

# ACL-style venues use A4 paper with 7.7 cm columns and a 0.6 cm gutter.
ACL_COLUMN_WIDTH_IN: float = 7.7 * _CM_TO_IN
ACL_PAPER_WIDTH_IN: float = (7.7 * 2.0 + 0.6) * _CM_TO_IN

# COLM and ICLR use the OpenReview single-column template with 5.5 in text width.
COLM_COLUMN_WIDTH_IN: float = 5.5
COLM_PAPER_WIDTH_IN: float = COLM_COLUMN_WIDTH_IN
ICLR_COLUMN_WIDTH_IN: float = 5.5
ICLR_PAPER_WIDTH_IN: float = ICLR_COLUMN_WIDTH_IN

# ICML uses a wider single-column layout.
ICML_COLUMN_WIDTH_IN: float = 6.75
ICML_PAPER_WIDTH_IN: float = ICML_COLUMN_WIDTH_IN

# NeurIPS uses a single-column 5.5 in text width.
NEURIPS_COLUMN_WIDTH_IN: float = 5.5
NEURIPS_PAPER_WIDTH_IN: float = NEURIPS_COLUMN_WIDTH_IN

NICE_FORMATTER = mtick.EngFormatter(places=0, sep="")
PCT_FORMATTER = mtick.PercentFormatter(1.0)


def format_compact_number(value: float, _: float | None = None) -> str:
    """Format values with compact decimal suffixes like 1K, 1M, 1B, and 1T."""

    abs_value = abs(value)
    if abs_value < 1_000:
        return f"{value:g}"

    suffixes = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]
    for scale, suffix in suffixes:
        if abs_value >= scale:
            scaled = value / scale
            if float(scaled).is_integer():
                return f"{int(scaled)}{suffix}"
            return f"{scaled:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:g}"


COMPACT_NUMBER_FORMATTER = mtick.FuncFormatter(format_compact_number)

DEFAULT_RCS: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.labelsize": "medium",
    "axes.titlelocation": "left",
    "axes.titlesize": "medium",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xaxis.labellocation": "left",
    "xtick.alignment": "center",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "figure.dpi": 300,
}
# Alias used by notebooks/scripts that want an explicit rc payload.
rcs = DEFAULT_RCS

sns.set_theme(style="ticks", context="paper", rc=DEFAULT_RCS)


def _to_rgb_tuple(color: ColorLike) -> Color:
    if isinstance(color, str):
        rgb = sns.color_palette([color])[0]
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    c = tuple(float(x) for x in color)
    if len(c) != 3:
        raise ValueError(f"Expected RGB tuple with 3 values, got {color!r}")
    if any(x > 1.0 for x in c):
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
    return (c[0], c[1], c[2])


def _darken_color(color: ColorLike, by: float) -> Color:
    by = min(max(float(by), 0.0), 1.0)
    rgb = _to_rgb_tuple(color)
    h, lightness, s = colorsys.rgb_to_hls(*rgb)
    darker = colorsys.hls_to_rgb(h, lightness * (1.0 - by), s)
    return (float(darker[0]), float(darker[1]), float(darker[2]))


def _palette_family_key(key: str, family: str) -> str:
    key_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in key).strip("_")
    if not key_slug:
        raise ValueError("key must contain at least one alphanumeric character")
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in family).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    if not slug:
        raise ValueError("family must contain at least one alphanumeric character")
    return f"{key_slug}_{slug}"


def _palette_display_name(family: str, name: str) -> str:
    family_clean = family.strip()
    name_clean = name.strip()
    if name_clean.lower().startswith(family_clean.lower()):
        return name_clean
    return f"{family_clean} {name_clean}"


def _model_display_name(family: str, model: str) -> str:
    return _palette_display_name(family, model)


def _dataset_display_name(family: str, dataset: str) -> str:
    return _palette_display_name(family, dataset)


def _relative_luminance(color: Color) -> float:
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def _is_color_tuple(color: object) -> TypeGuard[Color]:
    return (
        isinstance(color, tuple)
        and len(color) == 3
        and all(isinstance(x, (int, float)) for x in color)
    )


def _darken_mapping(colors: Mapping[str, ColorLike], by: float) -> dict[str, Color]:
    return {name: _darken_color(color, by) for name, color in colors.items()}


def _darken_many(colors: Iterable[ColorLike], by: float) -> list[Color]:
    return [_darken_color(color, by) for color in colors]


def text_color_for_background(
    color: ColorLike, *, dark: str = "white", light: str = "black"
) -> str:
    return dark if _relative_luminance(_to_rgb_tuple(color)) < 0.6 else light


@overload
def darken(color: ColorLike, by: float = 0.2) -> Color: ...


@overload
def darken(color: Mapping[str, ColorLike], by: float = 0.2) -> dict[str, Color]: ...


@overload
def darken(color: Iterable[ColorLike], by: float = 0.2) -> list[Color]: ...


def darken(
    color: ColorLike | Mapping[str, ColorLike] | Iterable[ColorLike], by: float = 0.2
) -> Color | dict[str, Color] | list[Color]:
    if isinstance(color, Mapping):
        return _darken_mapping(cast(Mapping[str, ColorLike], color), by)
    if isinstance(color, str):
        return _darken_color(color, by)
    if _is_color_tuple(color):
        return _darken_color(color, by)
    return _darken_many(cast(Iterable[ColorLike], color), by)


PALETTES: dict[str, Any] = {"models": {}}


def _palette_colors(color: str, names: list[str]) -> Palette:
    if not names:
        raise ValueError("names must be a non-empty list")

    # Assign darker shades to earlier models, independent of seaborn's
    # palette ordering for a particular color ramp.
    palette_colors = sorted(
        (
            _to_rgb_tuple(rgb)
            for rgb in sns.color_palette(color, n_colors=len(names) + 2)[1:-1]
        ),
        key=_relative_luminance,
    )
    return {name: rgb for name, rgb in zip(names, palette_colors, strict=True)}


def _set_palette_entries(
    *,
    key: str,
    family: str,
    family_palette: Palette,
    replace: bool,
    display_name: PaletteNameFormatter = _palette_display_name,
) -> Palette:
    aggregate_palette = PALETTES.setdefault(key, {})
    if not isinstance(aggregate_palette, dict):
        raise TypeError(f"PALETTES[{key!r}] must be a dict")

    family_key = _palette_family_key(key, family)
    existing_family_palette = PALETTES.get(family_key, {})
    existing_family_palette_dict = (
        dict(existing_family_palette)
        if isinstance(existing_family_palette, Mapping)
        else {}
    )

    if replace:
        for name in existing_family_palette_dict:
            aggregate_palette.pop(display_name(family, name), None)
        merged_family_palette = family_palette
    else:
        merged_family_palette = {**existing_family_palette_dict, **family_palette}

    PALETTES[family_key] = merged_family_palette
    aggregate_palette.update(
        {display_name(family, name): rgb for name, rgb in family_palette.items()}
    )
    return merged_family_palette


def _clear_palette_namespace(key: str) -> None:
    key_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in key).strip("_")
    if not key_slug:
        raise ValueError("key must contain at least one alphanumeric character")
    key_prefix = f"{key_slug}_"
    PALETTES[key] = {}
    for palette_key in list(PALETTES):
        if palette_key.startswith(key_prefix):
            PALETTES.pop(palette_key, None)


def set_palette(
    key: str,
    family: str,
    color: str,
    names: list[str],
    *,
    display_name: PaletteNameFormatter = _palette_display_name,
) -> Palette:
    _clear_palette_namespace(key)
    family_palette = _palette_colors(color, names)
    return _set_palette_entries(
        key=key,
        family=family,
        family_palette=family_palette,
        replace=False,
        display_name=display_name,
    )


def update_palette(
    key: str,
    family: str,
    color: str,
    names: list[str],
    *,
    display_name: PaletteNameFormatter = _palette_display_name,
) -> Palette:
    family_palette = _palette_colors(color, names)
    return _set_palette_entries(
        key=key,
        family=family,
        family_palette=family_palette,
        replace=False,
        display_name=display_name,
    )


def set_model_palette(family: str, color: str, models: list[str]) -> Palette:
    if not models:
        raise ValueError("models must be a non-empty list of model names")
    return set_palette(
        key="models",
        family=family,
        color=color,
        names=models,
        display_name=_model_display_name,
    )


def update_model_palette(family: str, color: str, models: list[str]) -> Palette:
    if not models:
        raise ValueError("models must be a non-empty list of model names")
    return update_palette(
        key="models",
        family=family,
        color=color,
        names=models,
        display_name=_model_display_name,
    )


def set_dataset_palette(family: str, color: str, datasets: list[str]) -> Palette:
    return set_palette(
        key="dataset",
        family=family,
        color=color,
        names=datasets,
        display_name=_dataset_display_name,
    )


def update_dataset_palette(family: str, color: str, datasets: list[str]) -> Palette:
    return update_palette(
        key="dataset",
        family=family,
        color=color,
        names=datasets,
        display_name=_dataset_display_name,
    )


update_model_palette(
    "Claude", "Oranges", ["Opus 4", "Sonnet 4", "Sonnet 3.7", "Haiku 3.5"]
)
update_model_palette(
    "OpenAI", "Greens", ["GPT-5.2", "GPT-5 mini", "GPT-4.1", "gpt-oss-120b"]
)
update_model_palette(
    "Gemini", "Blues", ["2.5 Pro", "2.5 Flash", "2.5 Flash-Lite", "2.0 Flash"]
)
update_model_palette(
    "Gemma", "blend:#d6fbff,#00bcd4", ["3 27B", "3 12B", "3 4B", "3 1B"]
)
update_model_palette(
    "OLMo", "Purples", ["3 32B Instruct", "3 32B Think", "3 32B Base", "3 7B Base"]
)


NONSEMANTIC_COLOR: Color = darken("#ffcc66", by=0.3)


def set_figure_title(
    fig,
    title: str,
    subtitle: str | None = None,
    *,
    x: float = 0.0,
    y: float = 0.995,
    title_kwargs: Mapping[str, Any] | None = None,
    subtitle_kwargs: Mapping[str, Any] | None = None,
):
    """Add a left-aligned bold figure title and optional subtitle."""

    title_y = y if subtitle is not None else y - 0.07
    title_text = fig.suptitle(
        title,
        x=x,
        y=title_y,
        ha="left",
        fontweight="bold",
        **dict(title_kwargs or {}),
    )
    subtitle_text = None
    if subtitle is not None:
        subtitle_defaults = {
            "fontsize": plt.rcParams["font.size"],
            "fontweight": "normal",
            "ha": "left",
            "va": "top",
        }
        subtitle_defaults.update(dict(subtitle_kwargs or {}))
        subtitle_text = fig.text(x, y - 0.07, subtitle, **subtitle_defaults)
    return title_text, subtitle_text


def save_figure(
    path: str | Path,
    *,
    fig=None,
    tight: bool = True,
    transparent: bool = True,
    save_png: bool = False,
    png_dpi: int = 300,
    **savefig_kwargs: Any,
) -> tuple[Path, Path | None]:
    """Save the current figure as PDF and optionally as PNG.

    Pass a path stem like ``figures/plot_name`` or a filename such as
    ``figures/plot_name.pdf``; the suffix is normalized and the PDF is always
    written. Set ``save_png=True`` to also write a PNG alongside it.
    """

    figure = fig if fig is not None else plt.gcf()
    output_base = Path(path)
    if output_base.suffix.lower() in {".pdf", ".png"}:
        output_base = output_base.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = dict(savefig_kwargs)
    save_kwargs.setdefault("transparent", transparent)
    if tight:
        save_kwargs.setdefault("bbox_inches", "tight")

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")

    figure.savefig(pdf_path, **save_kwargs)
    if save_png:
        figure.savefig(png_path, dpi=png_dpi, **save_kwargs)
        return pdf_path, png_path
    return pdf_path, None


__all__ = [
    "Color",
    "ColorLike",
    "Palette",
    "PAPER_WIDTH_IN",
    "FIG_HEIGHT_SINGLE_ROW_IN",
    "FIG_HEIGHT_DOUBLE_ROW_IN",
    "ACL_COLUMN_WIDTH_IN",
    "ACL_PAPER_WIDTH_IN",
    "COLM_COLUMN_WIDTH_IN",
    "COLM_PAPER_WIDTH_IN",
    "ICLR_COLUMN_WIDTH_IN",
    "ICLR_PAPER_WIDTH_IN",
    "ICML_COLUMN_WIDTH_IN",
    "ICML_PAPER_WIDTH_IN",
    "NEURIPS_COLUMN_WIDTH_IN",
    "NEURIPS_PAPER_WIDTH_IN",
    "NICE_FORMATTER",
    "PCT_FORMATTER",
    "format_compact_number",
    "COMPACT_NUMBER_FORMATTER",
    "DEFAULT_RCS",
    "rcs",
    "PALETTES",
    "set_palette",
    "update_palette",
    "set_model_palette",
    "update_model_palette",
    "set_dataset_palette",
    "update_dataset_palette",
    "set_figure_title",
    "save_figure",
    "text_color_for_background",
    "NONSEMANTIC_COLOR",
]
