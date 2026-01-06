from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Optional

try:  # Plotly optionally leans on pandas; provide a stub when pandas is broken
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - environment specific
    class _PandasStub:  # minimal interface used by plotly validators
        class Series(list):
            pass

        class Index(list):
            pass

    sys.modules.setdefault("pandas", _PandasStub())
else:
    if not hasattr(_pd, "Series"):
        class _SeriesStub(list):
            pass

        _pd.Series = _SeriesStub  # type: ignore[attr-defined]
    if not hasattr(_pd, "Index"):
        class _IndexStub(list):
            pass

        _pd.Index = _IndexStub  # type: ignore[attr-defined]

import plotly.graph_objects as go
import streamlit as st
sys.path.append(str(Path(__file__).parent.parent))

from visualizer.interactive import (
    InteractiveContext,
    ManualValueFeed,
    StepSnapshot,
    advance_context,
    column_state_codes,
    create_interactive_context,
    describe_cell_segments,
    describe_column_cells,
    describe_segment_synapses,
    sine_value_generator,
)
from visualizer.simulator import SimulationConfig, SimulationResult, run_simulation

st.set_page_config(page_title="HTM ColumnField Visualizer", layout="wide")


def main() -> None:
    st.title("HTM ColumnField Visualizer")
    st.caption("Interactive diagnostics for the Sine sequence scenario from tests/test_real_data.py")

    default_config = SimulationConfig()
    _ensure_session_state(default_config)
    sidebar_config = _render_sidebar(default_config)

    if st.sidebar.button("Run simulation", type="primary"):
        st.session_state["visualizer_config"] = sidebar_config
        st.session_state["visualizer_result"] = _cached_simulation(sidebar_config)

    result: SimulationResult = st.session_state["visualizer_result"]
    _render_summary_metrics(result)
    _render_training_charts(result)
    _render_structure_charts(result)
    _render_interactive_explorer(sidebar_config)

    st.subheader("ColumnField stats snapshot")
    st.code(result.stats_text)

    st.info(
        "Run locally via `.venv/bin/python -m streamlit run visualizer/app.py` to explore with live code updates.",
        icon="ℹ️",
    )


@st.cache_data(show_spinner="Simulating ColumnField dynamics...")
def _cached_simulation(config: SimulationConfig) -> SimulationResult:
    return run_simulation(config)


def _ensure_session_state(default_config: SimulationConfig) -> None:
    if "visualizer_result" not in st.session_state:
        st.session_state["visualizer_result"] = _cached_simulation(default_config)
        st.session_state["visualizer_config"] = default_config


def _render_sidebar(default_config: SimulationConfig) -> SimulationConfig:
    st.sidebar.header("Simulation parameters")
    config = SimulationConfig(
        num_columns=st.sidebar.slider("Columns", min_value=64, max_value=512, value=default_config.num_columns, step=64),
        cells_per_column=st.sidebar.slider("Cells / Column", min_value=4, max_value=32, value=default_config.cells_per_column, step=2),
        active_bits=st.sidebar.slider("RDSE Active Bits", min_value=8, max_value=32, value=default_config.active_bits, step=2),
        resolution=st.sidebar.number_input("RDSE Resolution", min_value=0.01, max_value=0.5, value=default_config.resolution, step=0.01),
        cycle_length=st.sidebar.slider("Sine Cycle Length", min_value=16, max_value=256, value=default_config.cycle_length, step=16),
        total_steps=st.sidebar.slider("Training Steps", min_value=100, max_value=1500, value=default_config.total_steps, step=50),
        python_seed=st.sidebar.number_input("Python RNG Seed", min_value=0, max_value=10_000, value=default_config.python_seed, step=1),
        rdse_seed=st.sidebar.number_input("RDSE Seed", min_value=0, max_value=10_000, value=default_config.rdse_seed, step=1),
        noise_std=st.sidebar.number_input("Noise Std Dev", min_value=0.0, max_value=0.5, value=default_config.noise_std, step=0.01),
        evaluation_cycles=st.sidebar.slider("Evaluation Cycles", min_value=1, max_value=5, value=default_config.evaluation_cycles, step=1),
    )
    st.sidebar.caption("Parameters mirror the configuration in tests/test_real_data.py with additional knobs for exploration.")
    st.sidebar.caption("Click 'Run simulation' after changing values to update the plots.")
    return config


def _render_summary_metrics(result: SimulationResult) -> None:
    st.subheader("Learning health checks")
    metrics = result.metrics
    cols = st.columns(4)
    cols[0].metric("Final burst count", f"{metrics['final_burst_count']:.0f}")
    cols[1].metric("Avg bursts (final quartile)", f"{metrics['mean_burst_last_quarter']:.2f}")
    cols[2].metric("Predictive column share", f"{metrics['predictive_column_share'] * 100:.1f}%")
    cols[3].metric("Evaluation burst-free ratio", f"{metrics['burst_free_ratio'] * 100:.1f}%")
    st.caption("Healthy runs should drive the final bursts toward zero and keep evaluation bursting within a narrow tolerance.")


def _render_training_charts(result: SimulationResult) -> None:
    st.subheader("Temporal evolution")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=result.burst_counts, name="Bursting columns"))
        fig.add_trace(go.Scatter(y=result.active_columns, name="Active columns"))
        fig.add_trace(go.Scatter(y=result.predictive_columns, name="Predictive columns"))
        fig.update_layout(
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Column count",
            xaxis_title="Training step",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_eval = go.Figure()
        fig_eval.add_trace(
            go.Scatter(
                x=list(range(len(result.evaluation_bursts))),
                y=result.evaluation_bursts,
                mode="lines+markers",
                name="Evaluation bursts",
            )
        )
        fig_eval.update_layout(
            height=360,
            xaxis_title="Evaluation step",
            yaxis_title="Burst count",
        )
        st.plotly_chart(fig_eval, use_container_width=True)


def _render_structure_charts(result: SimulationResult) -> None:
    st.subheader("Structural diagnostics")
    cols = st.columns(2)
    with cols[0]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=result.column_duty_cycles, nbinsx=40, name="Columns"))
        fig.add_trace(go.Histogram(x=result.cell_duty_cycles, nbinsx=40, name="Cells"))
        fig.update_layout(
            barmode="overlay",
            height=360,
            xaxis_title="Duty cycle",
            yaxis_title="Count",
        )
        fig.update_traces(opacity=0.65)
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        fig_perm = go.Figure()
        fig_perm.add_trace(go.Histogram(x=result.permanence_values, nbinsx=40, name="Synapses"))
        fig_perm.update_layout(
            height=360,
            xaxis_title="Permanence",
            yaxis_title="Synapse count",
        )
        st.plotly_chart(fig_perm, use_container_width=True)


_INTERACTIVE_CONTEXT_KEY = "interactive_ctx"
_SINE_FEED_LABEL = "Sine cycle (tests/test_real_data.py)"
_MANUAL_FEED_LABEL = "Manual value stream"
_COLUMN_STATE_LABELS = {
    0: "Inactive",
    1: "Predictive",
    2: "Active",
    3: "Bursting",
}


def _render_interactive_explorer(config: SimulationConfig) -> None:
    st.subheader("Interactive Column Explorer")
    st.caption("Drive the ColumnField with generator-backed inputs and inspect columns, cells, segments, and synapses in real time.")

    feed_mode = st.selectbox(
        "Input generator",
        options=[_SINE_FEED_LABEL, _MANUAL_FEED_LABEL],
        help="Sine feed mirrors tests/test_real_data.py; manual feed streams the current slider value each step.",
    )
    learn = st.toggle(
        "Enable learning while stepping",
        value=True,
        help="Disable to probe the learned model without updating permanences.",
    )
    manual_value = 0.0
    if feed_mode == _MANUAL_FEED_LABEL:
        manual_value = st.slider(
            "Manual input value",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help="Value streamed into the RDSE encoder on each manual-feed step.",
        )

    ctx = _ensure_interactive_context(config, feed_mode, learn, manual_value)

    steps_per_advance = int(
        st.number_input(
            "Steps per advance",
            min_value=1,
            max_value=256,
            value=1,
            step=1,
            key="interactive_step_input",
        )
    )
    control_cols = st.columns(3)

    def _advance(steps: int) -> None:
        try:
            advance_context(ctx, steps)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state[_INTERACTIVE_CONTEXT_KEY] = ctx

    if control_cols[0].button("Advance feed", use_container_width=True):
        _advance(steps_per_advance)
    if control_cols[1].button("Advance ×10", use_container_width=True):
        _advance(min(steps_per_advance * 10, 1024))
    if control_cols[2].button("Reset interactive state", use_container_width=True):
        st.session_state.pop(_INTERACTIVE_CONTEXT_KEY, None)
        st.experimental_rerun()

    snapshot: StepSnapshot | None = ctx.last_snapshot
    if snapshot is None:
        st.info("Use the advance controls to stream values through the ColumnField.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Step", f"{snapshot.step}")
    metric_cols[1].metric("Input value", f"{snapshot.input_value:.3f}")
    metric_cols[2].metric("Active columns", f"{snapshot.active_columns}")
    metric_cols[3].metric("Bursting columns", f"{snapshot.bursting_columns}")
    st.caption("Metrics update each time the generator feeds a new value into the ColumnField.")

    grid_cols = st.slider(
        "Columns per row",
        min_value=1,
        max_value=max(1, config.num_columns),
        value=min(32, config.num_columns),
        step=1,
        help="Controls the heatmap layout for the column state overview.",
    )
    _render_column_heatmap(ctx.column_field, grid_cols)

    column_count = len(ctx.column_field.columns)
    inspect_cols = st.columns(2)
    selected_column = int(
        inspect_cols[0].number_input(
            "Inspect column",
            min_value=0,
            max_value=max(0, column_count - 1),
            value=0,
            step=1,
        )
    )
    cell_count = len(ctx.column_field.columns[selected_column].cells)
    selected_cell = int(
        inspect_cols[1].number_input(
            "Inspect cell",
            min_value=0,
            max_value=max(0, cell_count - 1),
            value=0,
            step=1,
        )
    )

    st.markdown(f"Cells inside column {selected_column}")
    st.dataframe(
        describe_column_cells(ctx.column_field, selected_column),
        hide_index=True,
        use_container_width=True,
    )

    segments = describe_cell_segments(ctx.column_field, selected_column, selected_cell)
    if segments:
        st.markdown(f"Segments for cell {selected_cell}")
        st.dataframe(segments, hide_index=True, use_container_width=True)
        segment_options = list(range(len(segments)))
        segment_key = f"segment_select_{selected_column}_{selected_cell}"
        selected_segment = st.selectbox(
            "Segment to inspect",
            options=segment_options,
            format_func=str,
            key=segment_key,
        )
        synapses = describe_segment_synapses(
            ctx.column_field,
            selected_column,
            selected_cell,
            selected_segment,
        )
        if synapses:
            st.markdown("Synapses on selected segment")
            st.dataframe(synapses, hide_index=True, use_container_width=True)
        else:
            st.info("Selected segment has not formed synapses yet.")
    else:
        st.info("Selected cell has not grown distal segments yet.")

    st.session_state[_INTERACTIVE_CONTEXT_KEY] = ctx


def _ensure_interactive_context(
    config: SimulationConfig,
    feed_mode: str,
    learn: bool,
    manual_value: float,
) -> InteractiveContext:
    ctx: InteractiveContext | None = st.session_state.get(_INTERACTIVE_CONTEXT_KEY)
    needs_reset = (
        ctx is None
        or ctx.feed_label != feed_mode
        or ctx.learn != learn
        or ctx.config != config
    )
    if needs_reset:
        feed_iter, manual_feed = _build_feed(feed_mode, config, manual_value)
        ctx = create_interactive_context(
            config=config,
            feed_iter=feed_iter,
            learn=learn,
            feed_label=feed_mode,
            manual_feed=manual_feed,
        )
        st.session_state[_INTERACTIVE_CONTEXT_KEY] = ctx
    elif feed_mode == _MANUAL_FEED_LABEL and ctx.manual_feed is not None:
        ctx.manual_feed.update(manual_value)
    return ctx


def _build_feed(
    feed_mode: str,
    config: SimulationConfig,
    manual_value: float,
) -> tuple[Iterator[float], Optional[ManualValueFeed]]:
    if feed_mode == _MANUAL_FEED_LABEL:
        manual_feed = ManualValueFeed(value=manual_value)
        return manual_feed.generator(), manual_feed
    return sine_value_generator(config.cycle_length), None


def _render_column_heatmap(column_field, columns_per_row: int) -> None:
    values = column_state_codes(column_field)
    grid, text_labels, hover_labels = _reshape_for_heatmap(values, columns_per_row)
    colorscale = [
        [0.0, "#1f2933"],
        [0.32, "#1f2933"],
        [0.33, "#0ea5e9"],
        [0.65, "#0ea5e9"],
        [0.66, "#22c55e"],
        [0.98, "#22c55e"],
        [0.99, "#f97316"],
        [1.0, "#f97316"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            text=text_labels,
            customdata=hover_labels,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            colorbar=dict(
                title="Column state",
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=list(_COLUMN_STATE_LABELS.values()),
            ),
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _reshape_for_heatmap(values: list[int], columns_per_row: int) -> tuple[list[list[float]], list[list[str]], list[list[str]]]:
    rows: list[list[float]] = []
    text_rows: list[list[str]] = []
    hover_rows: list[list[str]] = []
    pad_value = float("nan")
    for start in range(0, len(values), columns_per_row):
        chunk = values[start : start + columns_per_row]
        indices = list(range(start, start + len(chunk)))
        while len(chunk) < columns_per_row:
            chunk.append(None)
            indices.append(-1)
        row_values: list[float] = []
        text_row: list[str] = []
        hover_row: list[str] = []
        for value, idx in zip(chunk, indices):
            if value is None or idx < 0:
                row_values.append(pad_value)
                text_row.append(" ")
                hover_row.append(" ")
                continue
            row_values.append(float(value))
            text_row.append(str(idx))
            hover_row.append(f"Column {idx} – {_COLUMN_STATE_LABELS[value]}")
        rows.append(row_values)
        text_rows.append(text_row)
        hover_rows.append(hover_row)
    return rows, text_rows, hover_rows


if __name__ == "__main__":
    main()
