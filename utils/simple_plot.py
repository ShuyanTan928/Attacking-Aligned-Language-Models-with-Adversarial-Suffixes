from pathlib import Path
from typing import Iterable, Sequence, Tuple


def _scale(value: float, data_min: float, data_max: float, min_px: float, max_px: float) -> float:
    if data_max == data_min:
        return (min_px + max_px) / 2
    return min_px + (value - data_min) * (max_px - min_px) / (data_max - data_min)


def save_line_plot_svg(
    data: Sequence[Tuple[float, float]],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 700,
    height: int = 450,
) -> None:
    if not data:
        raise ValueError("No data provided for plotting")

    path.parent.mkdir(parents=True, exist_ok=True)

    xs, ys = zip(*data)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    margin_left = 80
    margin_right = 30
    margin_top = 60
    margin_bottom = 70

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    points = []
    for x, y in data:
        px = _scale(x, min_x, max_x, margin_left, margin_left + plot_width)
        py = _scale(y, min_y, max_y, margin_top + plot_height, margin_top)
        points.append(f"{px},{py}")

    svg_parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Arial,sans-serif;} .axis{stroke:#333;stroke-width:2;} .grid{stroke:#ccc;stroke-width:1;stroke-dasharray:4 4;}" "</style>",
        f"<text x='{width/2}' y='{margin_top/2}' text-anchor='middle' font-size='20'>{title}</text>",
    ]

    # Axes
    svg_parts.append(
        f"<line class='axis' x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{margin_top + plot_height}' />"
    )
    svg_parts.append(
        f"<line class='axis' x1='{margin_left}' y1='{margin_top + plot_height}' x2='{margin_left + plot_width}' y2='{margin_top + plot_height}' />"
    )

    # Axis labels
    svg_parts.append(
        f"<text x='{margin_left + plot_width / 2}' y='{height - margin_bottom / 3}' text-anchor='middle' font-size='16'>{xlabel}</text>"
    )
    svg_parts.append(
        f"<text transform='rotate(-90 {margin_left / 3} {margin_top + plot_height / 2})' x='{margin_left / 3}' y='{margin_top + plot_height / 2}' text-anchor='middle' font-size='16'>{ylabel}</text>"
    )

    # Grid lines
    for fraction in (0.25, 0.5, 0.75):
        y = margin_top + plot_height * (1 - fraction)
        svg_parts.append(
            f"<line class='grid' x1='{margin_left}' y1='{y}' x2='{margin_left + plot_width}' y2='{y}' />"
        )

    # Polyline for data
    svg_parts.append(
        f"<polyline fill='none' stroke='#1f77b4' stroke-width='3' points='{' '.join(points)}' />"
    )

    # Data points markers
    for x, y in data:
        px = _scale(x, min_x, max_x, margin_left, margin_left + plot_width)
        py = _scale(y, min_y, max_y, margin_top + plot_height, margin_top)
        svg_parts.append(f"<circle cx='{px}' cy='{py}' r='4' fill='#1f77b4' />")

    svg_parts.append("</svg>")

    path.write_text("\n".join(svg_parts), encoding="utf-8")
