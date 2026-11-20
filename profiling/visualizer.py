from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .memory_data import MEMORY_CATEGORIES, MemoryProfile


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


def write_memory_visualizer(profile: MemoryProfile, output_path: Path) -> None:
    """Write an interactive memory visualizer as a standalone HTML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = _build_figure(profile)
    figure.write_html(output_path, include_plotlyjs=True)


def _build_figure(profile: MemoryProfile) -> go.Figure:
    row_count = len(MEMORY_CATEGORIES) + 1
    unit, divisor = _memory_unit(profile.peak_memory_bytes)
    figure = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=("Total memory", *(CATEGORY_LABELS.values())),
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
    figure.update_xaxes(title_text="Elapsed time (ms)", row=row_count, col=1)
    figure.update_layout(
        title=(
            "GPT memory profile "
            f"(peak: {profile.peak_memory_bytes / divisor:.2f} {unit})"
        ),
        height=300 * row_count,
        hovermode="x unified",
        template="plotly_white",
    )
    return figure


def _memory_unit(peak_memory_bytes: int) -> tuple[str, int]:
    if peak_memory_bytes >= 1_000_000_000:
        return "GB", 1_000_000_000
    return "MB", 1_000_000
