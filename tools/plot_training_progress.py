"""Render a lightweight SVG plot from training metrics JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def render_svg(metrics_path: Path, output_path: Path) -> dict:
    rows = _read_rows(metrics_path)
    series_names = sorted({row.get("series") for row in rows if row.get("series")})
    points_by_series = {
        name: [
            (float(row.get("cycle", 0)), float(row.get("mean_delta", row.get("score", 0.0))))
            for row in rows
            if row.get("series") == name
        ]
        for name in series_names
    }
    all_points = [point for points in points_by_series.values() for point in points]
    width, height = 980, 560
    left, right, top, bottom = 70, 30, 35, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    if all_points:
        min_x = min(x for x, _y in all_points)
        max_x = max(x for x, _y in all_points)
        min_y = min(y for _x, y in all_points)
        max_y = max(y for _x, y in all_points)
    else:
        min_x = min_y = 0.0
        max_x = max_y = 1.0
    if abs(max_x - min_x) < 1e-9:
        max_x += 1.0
    if abs(max_y - min_y) < 1e-9:
        max_y += 1.0
    y_pad = max(1.0, (max_y - min_y) * 0.12)
    min_y -= y_pad
    max_y += y_pad

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Training EV Progress</text>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="Arial" font-size="13">Cycle / generation</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13">Mean chip delta / EV proxy</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_value = min_y + frac * (max_y - min_y)
        y = sy(y_value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_value:.0f}</text>')
    for index, name in enumerate(series_names):
        points = points_by_series[name]
        color = colors[index % len(colors)]
        if len(points) >= 2:
            path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
            body.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        legend_y = top + 18 + index * 20
        body.append(f'<rect x="{left + 12}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        body.append(f'<text x="{left + 30}" y="{legend_y}" font-family="Arial" font-size="12">{name}</text>')
    body.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")
    return {"input": str(metrics_path), "output": str(output_path), "series": series_names, "points": len(all_points)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training EV/progress metrics as SVG")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = render_svg(Path(args.metrics), Path(args.output))
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)


if __name__ == "__main__":
    main()
