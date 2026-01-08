from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import plotly.graph_objects as go

from building_blocks import CONNECTED_PERM, ColumnField

_CELL_STATE_COLORS = {
    "inactive": "#1f2933",
    "predictive": "#0ea5e9",
    "active": "#22c55e",
    "bursting": "#f97316",
}
_SEGMENT_COLOR = "#a855f7"


class _CellPosition:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _SegmentPosition:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def render_column_field_3d(
    column_field: ColumnField,
    columns_per_row: int,
    show_only_connected_synapses: bool,
    permanence_threshold: float,
) -> go.Figure:
    columns_per_row = _normalize_columns_per_row(column_field, columns_per_row)
    cell_positions = _build_cell_positions(column_field, columns_per_row)
    segment_positions, segment_points = _build_segment_positions(column_field, cell_positions)

    fig = go.Figure()
    _add_cell_traces(fig, column_field, cell_positions)
    _add_segment_trace(fig, segment_points)
    _add_synapse_traces(
        fig,
        column_field,
        cell_positions,
        segment_positions,
        show_only_connected_synapses,
        permanence_threshold,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Column X", backgroundcolor="#0f172a", gridcolor="#1e293b", color="#e2e8f0"),
            yaxis=dict(title="Column Y", backgroundcolor="#0f172a", gridcolor="#1e293b", color="#e2e8f0"),
            zaxis=dict(title="Cell Z", backgroundcolor="#0f172a", gridcolor="#1e293b", color="#e2e8f0"),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        height=620,
        showlegend=True,
    )
    return fig


def _normalize_columns_per_row(column_field: ColumnField, columns_per_row: int) -> int:
    if columns_per_row <= 0:
        return int(math.ceil(math.sqrt(len(column_field.columns))))
    return columns_per_row


def _build_cell_positions(column_field: ColumnField, columns_per_row: int) -> Dict[object, _CellPosition]:
    positions: Dict[object, _CellPosition] = {}
    for idx, column in enumerate(column_field.columns):
        x = float(idx % columns_per_row)
        y = float(idx // columns_per_row)
        for cell_index, cell in enumerate(column.cells):
            positions[cell] = _CellPosition(x, y, float(cell_index))
    return positions


def _build_segment_positions(
    column_field: ColumnField,
    cell_positions: Dict[object, _CellPosition],
) -> tuple[Dict[object, _SegmentPosition], list[_SegmentPosition]]:
    positions: Dict[object, _SegmentPosition] = {}
    points: list[_SegmentPosition] = []
    for column in column_field.columns:
        for cell in column.cells:
            base = cell_positions.get(cell)
            if base is None:
                continue
            segment_count = len(cell.segments)
            for idx, segment in enumerate(cell.segments):
                angle = (2 * math.pi * idx / max(1, segment_count))
                radius = 0.28
                x = base.x + radius * math.cos(angle)
                y = base.y + radius * math.sin(angle)
                z = base.z + 0.2
                position = _SegmentPosition(x, y, z)
                positions[segment] = position
                points.append(position)
    return positions, points


def _add_cell_traces(
    fig: go.Figure,
    column_field: ColumnField,
    cell_positions: Dict[object, _CellPosition],
) -> None:
    grouped_positions: Dict[str, list[_CellPosition]] = {
        "inactive": [],
        "predictive": [],
        "active": [],
        "bursting": [],
    }
    for column in column_field.columns:
        for cell in column.cells:
            position = cell_positions.get(cell)
            if position is None:
                continue
            state = _cell_state(column, cell)
            grouped_positions[state].append(position)

    for state, positions in grouped_positions.items():
        if not positions:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=[pos.x for pos in positions],
                y=[pos.y for pos in positions],
                z=[pos.z for pos in positions],
                mode="markers",
                marker=dict(size=5, color=_CELL_STATE_COLORS[state]),
                name=f"Cells ({state})",
            )
        )


def _cell_state(column, cell) -> str:
    if column.bursting:
        return "bursting"
    if cell.active:
        return "active"
    if cell.predictive:
        return "predictive"
    return "inactive"


def _add_segment_trace(fig: go.Figure, positions: Iterable[_SegmentPosition]) -> None:
    positions_list = list(positions)
    if not positions_list:
        return
    fig.add_trace(
        go.Scatter3d(
            x=[pos.x for pos in positions_list],
            y=[pos.y for pos in positions_list],
            z=[pos.z for pos in positions_list],
            mode="markers",
            marker=dict(size=3, color=_SEGMENT_COLOR),
            name="Segments",
        )
    )


def _add_synapse_traces(
    fig: go.Figure,
    column_field: ColumnField,
    cell_positions: Dict[object, _CellPosition],
    segment_positions: Dict[object, _SegmentPosition],
    show_only_connected: bool,
    permanence_threshold: float,
) -> None:
    for column in column_field.columns:
        for cell in column.cells:
            for segment in cell.segments:
                segment_position = segment_positions.get(segment)
                if segment_position is None:
                    continue
                for synapse in segment.synapses:
                    if synapse.source_cell is None:
                        continue
                    permanence = synapse.permanence
                    if permanence < permanence_threshold:
                        continue
                    if show_only_connected and permanence < CONNECTED_PERM:
                        continue
                    source_position = cell_positions.get(synapse.source_cell)
                    if source_position is None:
                        continue
                    color, width = _synapse_style(permanence)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[source_position.x, segment_position.x],
                            y=[source_position.y, segment_position.y],
                            z=[source_position.z, segment_position.z],
                            mode="lines",
                            line=dict(color=color, width=width),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )


def _synapse_style(permanence: float) -> tuple[str, float]:
    clamped = max(0.0, min(1.0, permanence))
    color = _lerp_rgba((239, 68, 68), (34, 197, 94), clamped, 0.2 + 0.8 * clamped)
    width = 1.0 + 3.0 * clamped
    return color, width


def _lerp_rgba(
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    t: float,
    alpha: float,
) -> str:
    r = int(start[0] + (end[0] - start[0]) * t)
    g = int(start[1] + (end[1] - start[1]) * t)
    b = int(start[2] + (end[2] - start[2]) * t)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"
