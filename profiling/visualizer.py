from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .memory_data import MEMORY_CATEGORIES, MemoryProfile
from .timing import StepTiming


CATEGORY_LABELS = {
    "parameters": "Parameters",
    "optimizer_state": "Optimizer state",
    "input": "Inputs",
    "temporary": "Temporaries",
    "activation": "Activations",
    "gradient": "Gradients",
    "autograd_detail": "Autograd details",
    "unclassified": "Unclassified",
}

CATEGORY_COLORS = {
    "parameters": "#636EFA",
    "optimizer_state": "#EF553B",
    "input": "#00CC96",
    "temporary": "#AB63FA",
    "activation": "#FFA15A",
    "gradient": "#19D3F3",
    "autograd_detail": "#FF6692",
    "unclassified": "#B6E880",
}


def write_memory_visualizer(
    profile: MemoryProfile,
    output_path: Path,
    timing: StepTiming | None = None,
) -> None:
    """Write an interactive memory visualizer as a standalone HTML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = _build_figure(profile, timing)
    figure.write_html(output_path, include_plotlyjs=True)


def _build_figure(profile: MemoryProfile, timing: StepTiming | None = None) -> go.Figure:
    memory_rows = len(MEMORY_CATEGORIES) + 1
    row_count = memory_rows + (1 if timing is not None else 0)
    summary_peaks = [profile.peak_memory_bytes]
    if profile.peak_allocated_bytes is not None:
        summary_peaks.append(profile.peak_allocated_bytes)
    if profile.peak_reserved_bytes is not None:
        summary_peaks.append(profile.peak_reserved_bytes)
    unit, divisor = _memory_unit(max(summary_peaks))
    figure = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=(
            "Total memory",
            *(CATEGORY_LABELS.values()),
            *( ("Benchmark timing",) if timing is not None else () ),
        ),
    )

    times = [point.time_ms for point in profile.points]
    total_memory = [point.total_bytes / divisor for point in profile.points]
    for category in MEMORY_CATEGORIES:
        figure.add_trace(
            go.Scatter(
                x=times,
                y=[point.categories[category] / divisor for point in profile.points],
                mode="lines",
                name=CATEGORY_LABELS[category],
                stackgroup="memory",
                line={"color": CATEGORY_COLORS[category]},
                fillcolor=CATEGORY_COLORS[category],
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Scatter(
            x=times,
            y=total_memory,
            mode="lines",
            name="Total",
            line={"color": "black", "width": 2},
        ),
        row=1,
        col=1,
    )

    if profile.peak_allocated_bytes is not None:
        figure.add_hline(
            y=profile.peak_allocated_bytes / divisor,
            line_dash="dash",
            line_color="#444444",
            annotation_text="Peak allocated",
            row=1,
            col=1,
        )
    if profile.peak_reserved_bytes is not None:
        figure.add_hline(
            y=profile.peak_reserved_bytes / divisor,
            line_dash="dot",
            line_color="#444444",
            annotation_text="Peak reserved",
            row=1,
            col=1,
        )

    for category_index, category in enumerate(MEMORY_CATEGORIES, start=2):
        values = [
            point.categories[category] / divisor
            for point in profile.points
        ]
        figure.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=CATEGORY_LABELS[category],
                showlegend=False,
                fill="tozeroy",
                fillcolor=CATEGORY_COLORS[category],
                line={"color": CATEGORY_COLORS[category]},
            ),
            row=category_index,
            col=1,
        )

    for phase in profile.phases:
        for row in range(1, row_count + 1):
            figure.add_vline(
                x=phase.start_ms,
                line_dash="dash",
                line_color="gray",
                opacity=0.7,
                row=row,
                col=1,
            )
            figure.add_vline(
                x=phase.end_ms,
                line_dash="dot",
                line_color="gray",
                opacity=0.7,
                row=row,
                col=1,
            )
        for row in range(1, row_count + 1):
            yref = "y domain" if row == 1 else f"y{row} domain"
            figure.add_annotation(
                x=phase.start_ms,
                y=1.04,
                xref="x",
                yref=yref,
                text=phase.name,
                showarrow=False,
                textangle=-45,
                font={"size": 9},
            )

    figure.update_yaxes(title_text=unit, row=1, col=1)
    for row in range(2, row_count + 1):
        figure.update_yaxes(title_text=unit, row=row, col=1)
    if timing is not None:
        timing_row = row_count
        figure.add_trace(
            go.Bar(
                x=list(timing.regions_ms),
                y=list(timing.regions_ms.values()),
                name="Benchmark region time",
                marker_color="#636EFA",
            ),
            row=timing_row,
            col=1,
        )
        figure.update_yaxes(title_text="Milliseconds", row=timing_row, col=1)
        figure.update_xaxes(title_text="Benchmark region", row=timing_row, col=1)
    else:
        timing_row = None
    figure.update_xaxes(title_text="Elapsed time (ms)", row=memory_rows, col=1)
    summary = [
        f"device: {profile.device}",
        f"timeline peak: {profile.peak_memory_bytes / divisor:.2f} {unit}",
    ]
    if profile.peak_allocated_bytes is not None:
        summary.append(f"peak allocated: {profile.peak_allocated_bytes / divisor:.2f} {unit}")
    if profile.peak_reserved_bytes is not None:
        summary.append(f"peak reserved: {profile.peak_reserved_bytes / divisor:.2f} {unit}")
    if timing is not None:
        summary.append(f"benchmark step: {timing.step_time_ms:.2f} ms")

    figure.update_layout(
        title="GPT memory profile<br><sup>" + " | ".join(summary) + "</sup>",
        height=300 * row_count,
        hovermode="x unified",
        template="plotly_white",
    )
    return figure


def _memory_unit(peak_memory_bytes: int) -> tuple[str, int]:
    if peak_memory_bytes >= 1_000_000_000:
        return "GB", 1_000_000_000
    return "MB", 1_000_000
