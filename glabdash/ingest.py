from __future__ import annotations

import argparse
import csv
import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: tuple[str, ...]
    build_row: Callable[[list[str]], list[Any]]


def _tokenize(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    if '"' in line:
        return shlex.split(line, posix=True)
    return line.split()


def _message_and_used(raw_token: str) -> tuple[str, int]:
    token = raw_token.strip()
    if token.endswith("*"):
        return token[:-1], 0
    return token, 1


def _fixed_width_spec(name: str, columns: Iterable[str]) -> TableSpec:
    columns_tuple = tuple(columns)

    def build_row(tokens: list[str]) -> list[Any]:
        msg, used = _message_and_used(tokens[0])
        if msg != name:
            raise ValueError(f"Unexpected message {tokens[0]!r} for {name!r}")
        row = [used, *tokens[1:]]
        if len(row) != len(columns_tuple):
            raise ValueError(f"{name}: expected {len(columns_tuple)} fields, got {len(row)}")
        return row

    return TableSpec(name=name, columns=columns_tuple, build_row=build_row)


def _epochsat_spec() -> TableSpec:
    # v6 EPOCHSAT has fixed columns until field 38; satellite list starts at field 39 and is variable length.
    columns = (
        "used",
        "year",
        "doy",
        "sod",
        "time",
        "meas",
        "smooth_meas",
        "total_sats",
        "selected_sats",
        "gps_tag",
        "gps_selected",
        "gal_tag",
        "gal_selected",
        "glo_tag",
        "glo_selected",
        "geo_tag",
        "geo_selected",
        "bds_tag",
        "bds_selected",
        "qzss_tag",
        "qzss_selected",
        "irnss_tag",
        "irnss_selected",
        "unselected_sats",
        "gps_tag2",
        "gps_unselected",
        "gal_tag2",
        "gal_unselected",
        "glo_tag2",
        "glo_unselected",
        "geo_tag2",
        "geo_unselected",
        "bds_tag2",
        "bds_unselected",
        "qzss_tag2",
        "qzss_unselected",
        "irnss_tag2",
        "irnss_unselected",
        "satellites",
    )

    def build_row(tokens: list[str]) -> list[Any]:
        msg, used = _message_and_used(tokens[0])
        if msg != "EPOCHSAT":
            raise ValueError(f"Unexpected message {tokens[0]!r} for 'EPOCHSAT'")
        if len(tokens) < 39 or ":" not in tokens[4]:
            raise ValueError("EPOCHSAT: expected v6 format with time field and fixed 38 fields")
        return [used, *tokens[1:38], " ".join(tokens[38:])]

    return TableSpec(name="EPOCHSAT", columns=columns, build_row=build_row)


def _meas_spec_from_first_row(tokens: list[str]) -> TableSpec:
    # MEAS is the only table with dynamic trailing columns (measurement list at field 15).
    msg, _used = _message_and_used(tokens[0])
    if msg != "MEAS":
        raise ValueError(f"Unexpected message {tokens[0]!r} while building MEAS spec")
    if len(tokens) < 15:
        raise ValueError("MEAS: row shorter than required base fields")

    meas_ids = [value for value in tokens[14].split(":") if value]
    seen: dict[str, int] = {}
    meas_cols: list[str] = []
    for meas_id in meas_ids:
        count = seen.get(meas_id, 0) + 1
        seen[meas_id] = count
        meas_cols.append(meas_id if count == 1 else f"{meas_id}_{count}")

    base_cols = (
        "used",
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
    )
    columns = (*base_cols, *meas_cols, "extra_json")

    def build_row(row_tokens: list[str]) -> list[Any]:
        row_msg, used = _message_and_used(row_tokens[0])
        if row_msg != "MEAS":
            raise ValueError(f"Unexpected message {row_tokens[0]!r} for 'MEAS'")
        if len(row_tokens) < 15:
            raise ValueError("MEAS: row shorter than required base fields")

        row_base = [used, *row_tokens[1:15]]
        row_ids = [value for value in row_tokens[14].split(":") if value]
        row_values = row_tokens[15:]

        mapping: dict[str, str] = {}
        extras: dict[str, str] = {}

        for idx, row_id in enumerate(row_ids):
            value = row_values[idx] if idx < len(row_values) else ""
            if row_id in mapping:
                suffix = 2
                while f"{row_id}_{suffix}" in mapping:
                    suffix += 1
                mapping[f"{row_id}_{suffix}"] = value
            else:
                mapping[row_id] = value

        if len(row_values) > len(row_ids):
            for idx, value in enumerate(row_values[len(row_ids) :], start=1):
                extras[f"extra_{idx}"] = value

        row_meas_values = [mapping.get(col, "") for col in meas_cols]
        extra_json = json.dumps(extras, sort_keys=True) if extras else ""
        row = [*row_base, *row_meas_values, extra_json]

        if len(row) != len(columns):
            raise ValueError(f"MEAS: expected {len(columns)} fields, got {len(row)}")
        return row

    return TableSpec(name="MEAS", columns=tuple(columns), build_row=build_row)


def _v6_specs() -> dict[str, TableSpec]:
    # Column order follows gLAB_out_interface_description.txt for v6.0.0.
    return {
        "FILTER": _fixed_width_spec(
            "FILTER",
            (
                "used",
                "year",
                "doy",
                "sod",
                "time",
                "ref_clock",
                "n_isb",
                "n_unknowns",
                "x_m",
                "y_m",
                "z_m",
                "clock_m",
                "gps_isb_m",
                "gal_isb_m",
                "glo_isb_m",
                "geo_isb_m",
                "bds_isb_m",
                "qzss_isb_m",
                "irnss_isb_m",
            ),
        ),
        "OUTPUT": _fixed_width_spec(
            "OUTPUT",
            (
                "used",
                "year",
                "doy",
                "sod",
                "time",
                "proc_mode",
                "proc_dir",
                "n_sats",
                "n_const",
                "constellations",
                "conv",
                "x_m",
                "y_m",
                "z_m",
                "dx_m",
                "dy_m",
                "dz_m",
                "sx_m",
                "sy_m",
                "sz_m",
                "lat_deg",
                "lon_deg",
                "height_m",
                "north_err_m",
                "east_err_m",
                "up_err_m",
                "north_sigma_m",
                "east_sigma_m",
                "up_sigma_m",
                "horizontal_err_m",
                "vertical_err_abs_m",
                "err_3d_m",
                "ref_clock",
                "clock_m",
                "clock_sigma_m",
                "gdop",
                "pdop",
                "tdop",
                "hdop",
                "vdop",
                "ztd_m",
                "zwd_m",
                "ztd_sigma_m",
            ),
        ),
        "INPUT": _fixed_width_spec(
            "INPUT",
            (
                "used",
                "year",
                "doy",
                "sod",
                "time",
                "system",
                "prn",
                "arc",
                "arc_length",
                "c1_m",
                "p1_m",
                "p2_m",
                "l1_m",
                "l2_m",
            ),
        ),
        "MODEL": _fixed_width_spec(
            "MODEL",
            (
                "used",
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
                "meas",
                "brdc_msg_type",
                "brdc_iod",
                "glo_k",
                "leap_or_bds_offset_s",
                "meas_m",
                "flight_time_s",
                "sat_x_m",
                "sat_y_m",
                "sat_z_m",
                "sat_vx_mps",
                "sat_vy_mps",
                "sat_vz_mps",
                "geom_range_m",
                "sat_clock_corr_m",
                "sat_apc_proj_m",
                "sat_apcv_proj_m",
                "rec_pco_proj_m",
                "rec_pcv_proj_m",
                "rec_arp_proj_m",
                "relativity_m",
                "windup_m",
                "tropo_m",
                "iono_m",
                "grav_delay_m",
                "solid_tides_m",
                "reserved_40",
                "reserved_41",
                "reserved_42",
                "isb_m",
                "p1c1_dcb_m",
                "tgd_m",
                "isc_m",
                "bds_dcb_m",
                "dcb_sum_m",
                "snr_dbhz",
                "full_model_m",
            ),
        ),
        "PREFIT": _fixed_width_spec(
            "PREFIT",
            (
                "used",
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
                "meas",
                "smooth_meas",
                "prefit_m",
                "meas_m",
                "smooth_meas_m",
                "model_m",
                "los_x",
                "los_y",
                "los_z",
                "los_t",
                "sigma_m",
                "trop_wet_mapping",
                "wavelength_m",
                "glo_k",
                "tecu_to_m",
            ),
        ),
        "POSTFIT": _fixed_width_spec(
            "POSTFIT",
            (
                "used",
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
                "meas",
                "smooth_meas",
                "postfit_m",
                "meas_m",
                "smooth_meas_m",
                "model_m",
                "isb_m",
                "ambiguity_m",
                "reserved_22",
                "reserved_23",
                "reserved_24",
            ),
        ),
        "SATSEL": _fixed_width_spec(
            "SATSEL",
            (
                "used",
                "year",
                "doy",
                "sod",
                "time",
                "system",
                "prn",
                "error_code",
                "reason",
            ),
        ),
        "EPOCHSAT": _epochsat_spec(),
    }


class _CsvTableWriter:
    def __init__(self, path: Path, columns: Iterable[str]) -> None:
        self.path = path
        self._fp = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(list(columns))
        self.rows = 0

    def writerow(self, row: list[Any]) -> None:
        self._writer.writerow(row)
        self.rows += 1

    def close(self) -> None:
        self._fp.close()


def ingest_out_file(
    in_path: Path,
    out_dir: Path,
    *,
    max_lines: int | None = None,
    only_types: set[str] | None = None,
) -> dict[str, Any]:
    specs = _v6_specs()
    out_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, _CsvTableWriter] = {}
    skipped: list[dict[str, Any]] = []

    def get_writer(msg: str, columns: Iterable[str]) -> _CsvTableWriter:
        writer = writers.get(msg)
        if writer is not None:
            return writer
        writer = _CsvTableWriter(out_dir / f"{msg}.csv", columns)
        writers[msg] = writer
        return writer

    info_writer = get_writer("INFO", ("line_no", "text"))

    with in_path.open("r", encoding="utf-8", errors="replace") as fp:
        for line_no, line in enumerate(fp, 1):
            if max_lines is not None and line_no > max_lines:
                break

            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue

            if raw_line.startswith("INFO "):
                if only_types is None or "INFO" in only_types:
                    info_writer.writerow([line_no, raw_line[5:]])
                continue

            tokens = _tokenize(raw_line)
            if not tokens:
                continue

            msg, _used = _message_and_used(tokens[0])
            if only_types is not None and msg not in only_types:
                continue

            spec = specs.get(msg)
            if spec is None and msg == "MEAS":
                try:
                    spec = _meas_spec_from_first_row(tokens)
                except Exception as exc:  # noqa: BLE001
                    skipped.append({"line_no": line_no, "msg": msg, "error": str(exc), "raw": raw_line[:5000]})
                    continue
                specs[msg] = spec

            if spec is None:
                skipped.append(
                    {
                        "line_no": line_no,
                        "msg": msg,
                        "error": "Unsupported message for v6 parser",
                        "raw": raw_line[:5000],
                    }
                )
                continue

            try:
                row = spec.build_row(tokens)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"line_no": line_no, "msg": msg, "error": str(exc), "raw": raw_line[:5000]})
                continue

            get_writer(msg, spec.columns).writerow(row)

    tables: dict[str, Any] = {}
    for msg, writer in writers.items():
        writer.close()
        tables[msg] = {"path": str(writer.path), "rows": writer.rows}

    manifest = {
        "source": {
            "path": str(in_path),
            "size_bytes": in_path.stat().st_size if in_path.exists() else None,
            "mtime": in_path.stat().st_mtime if in_path.exists() else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": tables,
        "skipped": skipped[:200],
        "skipped_total": len(skipped),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a gLAB v6 .out file into per-message CSV tables.")
    parser.add_argument("out_file", type=Path, help="Path to the gLAB .out file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".glabdash_cache"),
        help="Directory where CSV tables and manifest.json are written",
    )
    parser.add_argument("--max-lines", type=int, default=None, help="Only ingest the first N lines")
    parser.add_argument(
        "--types",
        nargs="*",
        default=None,
        help="Only ingest these message types (example: OUTPUT FILTER MODEL)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    only_types = set(args.types) if args.types else None
    ingest_out_file(args.out_file, args.out_dir, max_lines=args.max_lines, only_types=only_types)
    print(f"Wrote tables to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
