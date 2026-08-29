from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


STAGES = ("base", "sft", "dpo", "rlvr")


def _escape(value: object) -> str:
    return html.escape(str(value))


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(text)
    temporary.replace(path)


def _fmt(value: float) -> str:
    return f"{value:.2f}".replace("-", "−")


def render_stage_trajectory(summary: dict, source_hash: str) -> str:
    psi = {stage: float(summary["estimands"]["psi"][stage]) for stage in STAGES}
    width, height = 960, 560
    left, right, top, bottom = 112, 900, 126, 442
    y_high, y_low = 0.25, -2.5
    xs = {stage: 165 + index * 225 for index, stage in enumerate(STAGES)}

    def y(value: float) -> float:
        return top + (y_high - value) * (bottom - top) / (y_high - y_low)

    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Stagewise reasoning-cue source contrast</title>',
        '<desc id="desc">The contrast is near zero at Base and SFT, becomes strongly negative at DPO, and remains negative at RLVR.</desc>',
        f'<metadata>confirmation_summary_sha256={source_hash}</metadata>',
        '<rect width="960" height="560" rx="18" fill="#ffffff"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.title{font-size:25px;font-weight:700;fill:#0f172a}.subtitle{font-size:14px;fill:#475569}.axis{font-size:12px;fill:#64748b}.stage{font-size:14px;font-weight:650;fill:#1e293b}.value{font-size:14px;font-weight:700;fill:#0f172a}.note{font-size:12px;fill:#475569}.callout{font-size:12px;font-weight:650;fill:#5b21b6}</style>',
        '<text id="title" class="title" x="48" y="46">Stagewise reasoning-cue source contrast</text>',
        '<text class="subtitle" x="48" y="72">Tulu 3 8B · GSM8K · negative ψ means stronger REQUEST control</text>',
        f'<rect x="{xs["sft"] - 44}" y="{top}" width="{xs["dpo"] - xs["sft"] + 88}" height="{bottom - top}" rx="12" fill="#f5f3ff"/>',
    ]

    for tick in (0.0, -0.5, -1.0, -1.5, -2.0, -2.5):
        yy = y(tick)
        stroke = "#94a3b8" if tick == 0 else "#e2e8f0"
        dash = ' stroke-dasharray="5 5"' if tick == 0 else ""
        rows.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="{stroke}" stroke-width="1"{dash}/>'
        )
        rows.append(
            f'<text class="axis" x="96" y="{yy + 4:.1f}" text-anchor="end">{_fmt(tick)}</text>'
        )

    rows.extend(
        [
            f'<text class="axis" x="28" y="{(top + bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 28 {(top + bottom) / 2:.1f})">ψ = S(notes cue) − S(request cue)</text>',
            f'<text class="callout" x="{(xs["sft"] + xs["dpo"]) / 2:.1f}" y="108" text-anchor="middle">largest observed stagewise shift</text>',
            f'<line x1="{xs["sft"] + 18}" y1="116" x2="{xs["dpo"] - 18}" y2="116" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#arrow)"/>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#7c3aed"/></marker></defs>',
        ]
    )

    points = " ".join(f'{xs[stage]},{y(psi[stage]):.1f}' for stage in STAGES)
    rows.append(
        f'<polyline points="{points}" fill="none" stroke="#334155" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    colors = {"base": "#64748b", "sft": "#0ea5e9", "dpo": "#7c3aed", "rlvr": "#2563eb"}
    labels = {"base": "Base", "sft": "SFT", "dpo": "DPO", "rlvr": "RLVR"}
    for stage in STAGES:
        xx, yy = xs[stage], y(psi[stage])
        rows.extend(
            [
                f'<circle cx="{xx}" cy="{yy:.1f}" r="8" fill="{colors[stage]}" stroke="#ffffff" stroke-width="3"/>',
                f'<text class="value" x="{xx}" y="{yy - 18:.1f}" text-anchor="middle">{_fmt(psi[stage])}</text>',
                f'<text class="stage" x="{xx}" y="474" text-anchor="middle">{labels[stage]}</text>',
            ]
        )

    rows.extend(
        [
            '<text class="note" x="112" y="516">Each point averages 1,319 items and both block orders. Stage localization is exploratory.</text>',
            '<text class="note" x="900" y="516" text-anchor="end">More REQUEST dominance ↓</text>',
            '</svg>\n',
        ]
    )
    return "".join(rows)


def render_experiment_design() -> str:
    rows = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="500" viewBox="0 0 1000 500" role="img" aria-labelledby="title desc">',
        '<title id="title">Paired source-swap experiment</title>',
        '<desc id="desc">The same problem is shown with the reasoning and direct-answer directives swapped between REQUEST and WORKING NOTES. The model scores reasoning versus direct mode, and the difference is compared across post-training stages.</desc>',
        '<rect width="1000" height="500" rx="18" fill="#ffffff"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.title{font-size:25px;font-weight:700;fill:#0f172a}.subtitle{font-size:14px;fill:#475569}.card-title{font-size:14px;font-weight:700;fill:#0f172a}.label{font-size:11px;font-weight:700;letter-spacing:.08em;fill:#64748b}.body{font-size:13px;fill:#1e293b}.formula{font-size:16px;font-weight:650;fill:#0f172a}.note{font-size:12px;fill:#475569}</style>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>',
        '<text class="title" x="50" y="44">Paired source-swap design</text>',
        '<text class="subtitle" x="50" y="70">The problem and directive wording stay fixed; only the source carrying each directive changes.</text>',
    ]

    def card(x: int, title: str, request_reason: bool) -> list[str]:
        request_fill = "#dbeafe" if request_reason else "#f1f5f9"
        notes_fill = "#dbeafe" if not request_reason else "#f1f5f9"
        request_text = "Reason step by step" if request_reason else "Answer directly"
        notes_text = "Answer directly" if request_reason else "Reason step by step"
        return [
            f'<rect x="{x}" y="100" width="390" height="216" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>',
            f'<text class="card-title" x="{x + 20}" y="128">{_escape(title)}</text>',
            f'<rect x="{x + 20}" y="145" width="350" height="58" rx="9" fill="{request_fill}"/>',
            f'<text class="label" x="{x + 34}" y="166">REQUEST</text>',
            f'<text class="body" x="{x + 34}" y="188">{request_text}</text>',
            f'<rect x="{x + 20}" y="218" width="350" height="58" rx="9" fill="{notes_fill}"/>',
            f'<text class="label" x="{x + 34}" y="239">WORKING NOTES</text>',
            f'<text class="body" x="{x + 34}" y="261">{notes_text}</text>',
            f'<text class="note" x="{x + 20}" y="298">Score S = reason logit − direct logit</text>',
        ]

    rows.extend(card(50, "request_cue", True))
    rows.extend(card(560, "notes_cue", False))
    rows.extend(
        [
            '<line x1="245" y1="316" x2="390" y2="362" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>',
            '<line x1="755" y1="316" x2="610" y2="362" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)"/>',
            '<rect x="300" y="350" width="400" height="108" rx="14" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>',
            '<text class="formula" x="500" y="386" text-anchor="middle">ψ(stage) = mean[S(notes_cue) − S(request_cue)]</text>',
            '<text class="formula" x="500" y="418" text-anchor="middle">Θ = ψ(RLVR) − ψ(Base)</text>',
            '<text class="note" x="500" y="444" text-anchor="middle">Negative Θ means post-training increased REQUEST dominance.</text>',
            '</svg>\n',
        ]
    )
    return "".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render public README figures")
    parser.add_argument(
        "--confirmation-summary",
        type=Path,
        default=Path(".external/repair-v2/evidence/confirmation/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    payload = args.confirmation_summary.read_bytes()
    summary = json.loads(payload)
    if summary.get("phase") != "confirmation":
        raise ValueError("Expected confirmation summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(
        args.output_dir / "psi-by-stage.svg",
        render_stage_trajectory(summary, hashlib.sha256(payload).hexdigest()),
    )
    _write(args.output_dir / "experiment-design.svg", render_experiment_design())


if __name__ == "__main__":
    main()
