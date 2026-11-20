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


def write_memory_visualizer(profile: MemoryProfile, output_path: Path) -> None:
    """Write an interactive memory visualizer as a standalone HTML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = _build_figure(profile)
    figure.write_html(output_path, include_plotlyjs=True)


def _build_figure(profile: MemoryProfile) -> go.Figure:
    row_count = len(MEMORY_CATEGORIES) + 1
    figure = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=("Total memory", *(CATEGORY_LABELS.values())),
    )

    times = [point.time_ms for point in profile.points]
    total_memory = [_to_mib(point.total_bytes) for point in profile.points]
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
            _to_mib(point.categories[category])
            for point in profile.points
        ]
        figure.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=CATEGORY_LABELS[category],
                showlegend=False,
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
        figure.add_annotation(
            x=phase.start_ms,
            y=1.02,
            xref="x",
            yref="paper",
            text=phase.name,
            showarrow=False,
            textangle=-45,
        )

    figure.update_yaxes(title_text="MiB", row=1, col=1)
    for row in range(2, row_count + 1):
        figure.update_yaxes(title_text="MiB", row=row, col=1)
    figure.update_xaxes(title_text="Elapsed time (ms)", row=row_count, col=1)
    figure.update_layout(
        title=(
            "GPT memory profile "
            f"(peak: {_to_mib(profile.peak_memory_bytes):.2f} MiB)"
        ),
        height=300 * row_count,
        hovermode="x unified",
        template="plotly_white",
    )
    return figure


def _to_mib(byte_count: int) -> float:
    return byte_count / (1024 * 1024)
