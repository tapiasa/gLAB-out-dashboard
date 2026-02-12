from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from glabdash.ingest import ingest_out_file


def _load_manifest(cache_dir: Path) -> dict | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_table_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not {"year", "doy", "sod"}.issubset(df.columns):
        return df
    # gLAB tags are in GPS time; we build a naive datetime for plotting.
    y = df["year"].astype(int).astype(str)
    j = df["doy"].astype(int).astype(str).str.zfill(3)
    base = pd.to_datetime(y + j, format="%Y%j", errors="coerce")
    df = df.copy()
    df["datetime"] = base + pd.to_timedelta(df["sod"], unit="s", errors="coerce")
    return df


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _find_interface_file() -> Path | None:
    for name in ("gLAB_out_interface.txt", "gLAB_out_interface_description.txt"):
        path = Path(name)
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def _load_interface_doc() -> dict[str, dict[str, object]]:
    path = _find_interface_file()
    if path is None:
        return {}

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    section_re = re.compile(r"^\s{4}([A-Z][A-Z0-9]+)\b")
    field_re = re.compile(r"^\s{8}Field\s+(\d+):\s*(.+)$")

    docs: dict[str, dict[str, object]] = {}
    idx = 0
    while idx < len(lines):
        header_match = section_re.match(lines[idx])
        if header_match is None:
            idx += 1
            continue

        msg = header_match.group(1)
        end_idx = idx + 1
        while end_idx < len(lines) and section_re.match(lines[end_idx]) is None:
            end_idx += 1

        block = lines[idx + 1 : end_idx]
        summary_parts: list[str] = []
        fields: dict[int, str] = {}

        for block_line in block:
            stripped = block_line.strip()
            if stripped and not stripped.startswith("Field "):
                summary_parts.append(stripped)
            if stripped.startswith("Field "):
                break

        for block_line in block:
            field_match = field_re.match(block_line)
            if field_match is None:
                continue
            fields[int(field_match.group(1))] = field_match.group(2).strip()

        docs[msg] = {
            "summary": " ".join(summary_parts),
            "fields": fields,
            "source_file": str(path),
        }
        idx = end_idx

    return docs


def _column_description(message: str, col: str, columns: list[str], docs: dict[str, dict[str, object]]) -> str:
    if col == "datetime":
        return "Derived field in the dashboard: timestamp built from year + doy + sod."
    if col == "used":
        return "Derived field in the dashboard: 1 means message used, 0 means message with '*' suffix."

    msg_doc = docs.get(message)
    if msg_doc is None:
        return "No description found in interface file."

    fields = msg_doc.get("fields", {})
    if isinstance(fields, dict):
        try:
            field_number = columns.index(col) + 1
        except ValueError:
            field_number = -1
        if field_number in fields:
            return f"Field {field_number}: {fields[field_number]}"

    meas_base_cols = {
        "year",
        "doy",
        "sod",
        "time",
        "system",
        "prn",
        "sat_block",
        "svn",
        "arc",
        "arc_length",
        "elev_deg",
        "azim_deg",
        "n_meas",
        "meas_list",
        "extra_json",
    }
    if message == "MEAS" and col not in meas_base_cols:
        return f"Measurement value column from MEAS list: {col}."

    return "No specific field description found."


def _series_colors(series: list[str], key_prefix: str) -> dict[str, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_map: dict[str, str] = {}
    for idx, series_name in enumerate(series):
        key = f"{key_prefix}_{series_name}".replace(" ", "_")
        color_map[series_name] = st.color_picker(
            f"Color: {series_name}",
            value=palette[idx % len(palette)],
            key=key,
        )
    return color_map


def main() -> None:
    st.set_page_config(page_title="gLAB Output Dashboard", layout="wide")
    st.title("gLAB Output Dashboard")

    with st.sidebar:
        out_path_str = st.text_input("gLAB .out path", value="TLSA00615.out")
        cache_dir = Path(st.text_input("Cache dir", value=".glabdash_cache"))
        only_types_str = st.text_input("Only message types (optional)", value="")
        max_lines = st.number_input("Max lines (0 = all)", min_value=0, value=0, step=10000)

        only_types = {t.strip() for t in only_types_str.split() if t.strip()} or None
        max_lines_opt = int(max_lines) if max_lines and int(max_lines) > 0 else None

        if st.button("Build / refresh cache"):
            with st.spinner("Parsing .out and writing CSV tables..."):
                ingest_out_file(
                    Path(out_path_str),
                    cache_dir,
                    max_lines=max_lines_opt,
                    only_types=only_types,
                )
            st.success("Cache ready.")

    manifest = _load_manifest(cache_dir)
    if not manifest:
        st.info("No cache found. Use the sidebar to build it.")
        return

    tables = manifest.get("tables", {})
    msg_types = sorted(tables.keys())
    if not msg_types:
        st.warning("Cache contains no tables.")
        return

    msg = st.selectbox("MESSAGE type", msg_types)
    table_path = tables[msg]["path"]

    with st.spinner(f"Loading {msg}..."):
        df = _read_table_csv(table_path)
    df = _add_datetime(df)

    st.caption(f"{msg}: {len(df):,} rows")

    interface_docs = _load_interface_doc()
    msg_doc = interface_docs.get(msg)
    if msg_doc and msg_doc.get("summary"):
        st.caption(str(msg_doc["summary"]))
    else:
        st.caption("No interface description found for this MESSAGE.")

    filters_col, plot_col = st.columns([1, 2])

    with filters_col:
        st.subheader("Filters")

        if "used" in df.columns:
            used_only = st.checkbox("Only used (exclude '*')", value=True)
            if used_only:
                df = df[df["used"] == 1]

        for col_name in ("system", "prn", "meas"):
            if col_name in df.columns:
                values = sorted(df[col_name].dropna().unique().tolist())
                selected = st.multiselect(col_name.upper(), values, default=[])
                if selected:
                    df = df[df[col_name].isin(selected)]

        if "sod" in df.columns:
            sod_min = float(pd.to_numeric(df["sod"], errors="coerce").min())
            sod_max = float(pd.to_numeric(df["sod"], errors="coerce").max())
            r = st.slider("Seconds-of-day range", min_value=sod_min, max_value=sod_max, value=(sod_min, sod_max))
            df = df[(df["sod"] >= r[0]) & (df["sod"] <= r[1])]

    with plot_col:
        st.subheader("Plot")
        numeric_cols = _numeric_columns(df)
        if not numeric_cols:
            st.warning("No numeric columns to plot for this MESSAGE.")
            return

        plot_kind = st.radio("Type", ["Time series", "Scatter", "Histogram"], horizontal=True)

        x_candidates = []
        if "datetime" in df.columns:
            x_candidates.append("datetime")
        if "sod" in df.columns and "sod" not in x_candidates:
            x_candidates.append("sod")
        for col in numeric_cols:
            if col not in x_candidates:
                x_candidates.append(col)
        x_col = st.selectbox("X", x_candidates, index=0 if x_candidates else None)

        if plot_kind == "Histogram":
            y_cols = st.multiselect("Histogram columns", numeric_cols, default=numeric_cols[:1])
        else:
            y_candidates = [col for col in numeric_cols if col != x_col]
            y_cols = st.multiselect("Y columns", y_candidates, default=y_candidates[:1])

        if not y_cols:
            st.warning("Select at least one Y column.")
            return

        st.markdown("**Series colors**")
        color_map = _series_colors(y_cols, key_prefix=f"{msg}_{plot_kind}")

        plot_df = df
        max_points = st.number_input("Max points", min_value=1_000, value=80_000, step=10_000)
        if plot_kind != "Histogram" and len(plot_df) > int(max_points):
            plot_df = plot_df.sort_values(x_col)
            step = max(1, len(plot_df) // int(max_points))
            plot_df = plot_df.iloc[::step]

        fig = go.Figure()
        if plot_kind == "Histogram":
            bins = st.slider("Bins", min_value=10, max_value=200, value=80, step=5)
            for y_col in y_cols:
                fig.add_trace(
                    go.Histogram(
                        x=plot_df[y_col],
                        name=y_col,
                        marker_color=color_map[y_col],
                        opacity=0.55,
                        nbinsx=bins,
                    )
                )
            fig.update_layout(barmode="overlay")
        elif plot_kind == "Time series":
            for y_col in y_cols:
                fig.add_trace(
                    go.Scatter(
                        x=plot_df[x_col],
                        y=plot_df[y_col],
                        mode="lines",
                        name=y_col,
                        line={"color": color_map[y_col]},
                    )
                )
        else:
            for y_col in y_cols:
                fig.add_trace(
                    go.Scatter(
                        x=plot_df[x_col],
                        y=plot_df[y_col],
                        mode="markers",
                        name=y_col,
                        marker={"color": color_map[y_col]},
                    )
                )

        fig.update_layout(xaxis_title=x_col, yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Selected column descriptions**")
        described_cols = y_cols if plot_kind == "Histogram" else [x_col, *y_cols]
        description_rows = []
        for col in described_cols:
            description_rows.append(
                {
                    "column": col,
                    "description": _column_description(msg, col, list(df.columns), interface_docs),
                }
            )
        st.dataframe(pd.DataFrame(description_rows), hide_index=True, use_container_width=True)

        st.markdown("**Preview**")
        preview_cols = y_cols if plot_kind == "Histogram" else [x_col, *y_cols]
        ordered_preview_cols = []
        for col in preview_cols:
            if col in plot_df.columns and col not in ordered_preview_cols:
                ordered_preview_cols.append(col)
        st.dataframe(plot_df[ordered_preview_cols].head(50), use_container_width=True)


if __name__ == "__main__":
    main()
