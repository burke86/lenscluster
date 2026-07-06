import argparse
from contextlib import nullcontext
import os
import re
import resource
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import FITSFixedWarning
import numpy as np

from .jax_cosmology import kpc_per_arcsec_from_config

try:
    from rich.console import Console
    from rich.progress import Progress as _RichProgress
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal test environments
    RICH_AVAILABLE = False
    class _FallbackSpan:
        def __init__(self, style: str | None) -> None:
            self.style = style or ""

    class Text:
        def __init__(self) -> None:
            self.plain = ""
            self.spans: list[_FallbackSpan] = []

        def append(self, value: str, style: str | None = None) -> None:
            self.plain += str(value)
            if style:
                self.spans.append(_FallbackSpan(style))

    class Console:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def print(self, value: Any) -> None:
            print(getattr(value, "plain", value))

    class _RichProgress:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "_RichProgress":
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
            return False

        def add_task(self, description: str, total: int | None = None) -> int:
            return 0

        def update(self, task_id: int, **kwargs: Any) -> None:
            pass

        def advance(self, task_id: int, advance: int = 1) -> None:
            pass

    class Table:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.rows: list[tuple[str, ...]] = []

        def add_column(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_row(self, *values: Any, **kwargs: Any) -> None:
            self.rows.append(tuple(str(value) for value in values))

_DEBUG_LOG_PATH: Path | None = None
_DEBUG_LOG_HANDLE = None
_DEBUG_LOG_STDOUT_ENABLED = True
_CONSOLE = Console(highlight=False)
_TAG_STYLES = {
    "main": "bold cyan",
    "load": "bold blue",
    "input": "bold blue",
    "input-archive": "bold blue",
    "runtime": "bold cyan",
    "stage": "bold magenta",
    "model": "bold green",
    "parameters": "bold green",
    "surrogate": "bold yellow",
    "posterior": "bold cyan",
    "smc": "bold yellow",
    "svi": "bold magenta",
    "nuts": "bold red",
    "validation": "bold yellow",
    "approximations": "bold yellow",
    "output": "bold green",
    "plots-only": "bold cyan",
    "phase": "bold white",
    "compile": "bold blue",
    "done": "bold green",
    "exception": "bold red",
    "traceback": "red",
}
_CONSOLE_VISIBLE_TAGS = {
    "main",
    "runtime",
    "stage",
    "load",
    "input-archive",
    "model",
    "compile",
    "smc",
    "svi",
    "nuts",
    "validation",
    "approximations",
    "output",
    "plots-only",
    "resume",
    "done",
    "exception",
}
_STAGE_BANNER_RE = re.compile(r"^(\s*)(=+)(\s+)(.*?)(\s+)(=+)$")
_RADECSYS_WARNING_RE = r"(?s).*RADECSYS.*deprecated.*RADESYS.*"


class _NoOpProgress:
    def __enter__(self) -> "_NoOpProgress":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False

    def add_task(self, description: str, *, total: int | None = None) -> int:
        return 0

    def update(self, task_id: int, **kwargs: Any) -> None:
        pass

    def advance(self, task_id: int, advance: int = 1) -> None:
        pass


class _NotebookTqdmProgress:
    def __init__(self) -> None:
        self._bars: dict[int, Any] = {}
        self._next_task_id = 0

    def __enter__(self) -> "_NotebookTqdmProgress":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        for bar in list(self._bars.values()):
            try:
                bar.close()
            except Exception:
                pass
        self._bars.clear()
        return False

    def add_task(self, description: str, *, total: int | None = None) -> int:
        from tqdm.notebook import tqdm

        self._next_task_id += 1
        task_id = self._next_task_id
        self._bars[task_id] = tqdm(total=total, desc=str(description), leave=False)
        return task_id

    def update(self, task_id: int, **kwargs: Any) -> None:
        bar = self._bars.get(int(task_id))
        if bar is None:
            return
        if "description" in kwargs:
            bar.set_description_str(str(kwargs["description"]))
        if "total" in kwargs and kwargs["total"] is not None:
            bar.total = int(kwargs["total"])
        try:
            bar.refresh()
        except Exception:
            pass

    def advance(self, task_id: int, advance: int = 1) -> None:
        bar = self._bars.get(int(task_id))
        if bar is None:
            return
        bar.update(int(advance))


DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN = 4.30091e-6


def log10_dpie_mass_msun(v_disp: Any, cut_radius_kpc: Any) -> np.ndarray:
    """log10 of the dPIE total mass M = (3/2) * pi * sigma^2 * r_cut / G in Msun.

    Eliasdottir et al. 2007 eq. A11 (M_TOT = 2*pi*Sigma0*a*s, equivalently
    lenstronomy PseudoJaffe.mass_tot) expressed with the lenstool/E0 fiducial
    sigma_dPIE — the convention of every v_disp in this pipeline (the amplitude
    packing is E0 = 6*pi*sigma^2*(DLS/DS)/c^2, eq. A24). The 3/2 converts from the
    isothermal-sigma form pi*sigma_iso^2*r_cut/G via sigma_iso^2 = 1.5*sigma_dPIE^2.
    Neglects the exact (1 - r_core^2/r_cut^2) factor, <= 6e-4 for every component
    used here. The single mass definition shared by mock-truth recording, per-run
    recovery tables, and plotting — keep every subhalo-mass comparison on it.
    """
    v_disp_array = np.asarray(v_disp, dtype=float)
    cut_array = np.asarray(cut_radius_kpc, dtype=float)
    mass = (
        1.5
        * np.pi
        * np.square(v_disp_array)
        * np.maximum(cut_array, 0.0)
        / DPIE_MASS_GRAVITATIONAL_CONSTANT_KPC_KMS2_PER_MSUN
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass > 0.0, np.log10(mass), np.nan)


def is_notebook_environment() -> bool:
    try:
        get_ipython = __import__("IPython").get_ipython
        shell = get_ipython()
    except Exception:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def progress_context(args: argparse.Namespace | None, *columns: Any, **kwargs: Any) -> Any:
    if bool(getattr(args, "quiet", False)):
        return nullcontext(None)
    if is_notebook_environment():
        return _NotebookTqdmProgress()
    return _RichProgress(*columns, **kwargs)


def install_astropy_wcs_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=_RADECSYS_WARNING_RE,
        category=FITSFixedWarning,
    )


def parse_bool_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def jax_cpu_worker_count() -> int:
    try:
        import jax

        devices = jax.devices("cpu")
        if devices:
            return max(1, int(len(devices)))
    except Exception:
        pass
    try:
        env_count = int(os.environ.get("JAX_NUM_CPU_DEVICES", ""))
        if env_count > 0:
            return env_count
    except (TypeError, ValueError):
        pass
    return max(1, int(os.cpu_count() or 1))


def process_memory_snapshot() -> dict[str, float | None]:
    rss_mb: float | None = None
    vms_mb: float | None = None
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024.0
                elif line.startswith("VmSize:"):
                    vms_mb = float(line.split()[1]) / 1024.0
        except OSError:
            pass
    ru_maxrss_mb: float | None = None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ru_maxrss_mb = float(usage.ru_maxrss) / 1024.0
    except Exception:  # pragma: no cover
        ru_maxrss_mb = None
    return {
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "ru_maxrss_mb": ru_maxrss_mb,
    }


def format_memory_snapshot() -> str:
    snapshot = process_memory_snapshot()
    parts = []
    for key in ("rss_mb", "vms_mb", "ru_maxrss_mb"):
        value = snapshot.get(key)
        if value is None or not np.isfinite(value):
            parts.append(f"{key}=na")
        else:
            parts.append(f"{key}={value:.1f}")
    return " ".join(parts)


def close_debug_log() -> None:
    global _DEBUG_LOG_HANDLE
    if _DEBUG_LOG_HANDLE is not None:
        try:
            _DEBUG_LOG_HANDLE.flush()
            _DEBUG_LOG_HANDLE.close()
        finally:
            _DEBUG_LOG_HANDLE = None


def debug_log_line(line: str) -> None:
    global _DEBUG_LOG_HANDLE
    if _DEBUG_LOG_HANDLE is None or _DEBUG_LOG_HANDLE.closed:
        if _DEBUG_LOG_PATH is None:
            return
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DEBUG_LOG_HANDLE = _DEBUG_LOG_PATH.open("a", encoding="utf-8")
    _DEBUG_LOG_HANDLE.write(f"{line}\n")
    _DEBUG_LOG_HANDLE.flush()


def set_debug_log_path(path: Path) -> None:
    global _DEBUG_LOG_PATH
    path = path.resolve()
    if _DEBUG_LOG_PATH == path and _DEBUG_LOG_HANDLE is not None:
        return
    previous_path = _DEBUG_LOG_PATH
    previous_handle = _DEBUG_LOG_HANDLE
    _DEBUG_LOG_PATH = path
    if previous_handle is not None:
        try:
            previous_handle.flush()
        except Exception:  # pragma: no cover
            pass
    if previous_path is not None and previous_path != path and previous_path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(previous_path.read_text(encoding="utf-8"), encoding="utf-8")
    close_debug_log()
    debug_log_line(f"{datetime.now().isoformat(timespec='seconds')} [debug-log] path={path}")


def configure_debug_log(args: argparse.Namespace, run_name: str, run_dir: Path | None = None) -> Path:
    if run_dir is None:
        target = Path(args.output_dir) / f"{run_name}.debug.log"
    else:
        target = run_dir / "run_debug.log"
    set_debug_log_path(target)
    return target


def log_exception(context: str, exc: BaseException) -> None:
    debug_log_line(
        f"{datetime.now().isoformat(timespec='seconds')} [exception] context={context} "
        f"type={type(exc).__name__} {str(exc)} {format_memory_snapshot()}"
    )
    for line in traceback.format_exc().rstrip().splitlines():
        debug_log_line(f"{datetime.now().isoformat(timespec='seconds')} [traceback] {line}")


def should_log(args: argparse.Namespace | None) -> bool:
    return args is None or not getattr(args, "quiet", False)


def _style_for_tag(tag: str) -> str:
    normalized = tag.strip("[]").split(":", 1)[0]
    return _TAG_STYLES.get(normalized, "bold")


def _message_tag(message: str) -> tuple[str, str] | None:
    if not message.startswith("[") or "]" not in message:
        return None
    tag = message[1 : message.find("]")]
    return tag, tag.split(":", 1)[0]


def _should_log_to_console(message: str) -> bool:
    parsed = _message_tag(message)
    if parsed is None:
        return True
    tag, base_tag = parsed
    if base_tag == "phase":
        return " error " in message
    if ":" in tag:
        return False
    if base_tag in {"parameters", "posterior"}:
        return False
    if base_tag == "input":
        return message.startswith("[input] dropped singleton")
    if base_tag == "runtime":
        return message.startswith("[runtime] python=")
    if base_tag == "surrogate":
        return "active_by_potfile" not in message
    return base_tag in _CONSOLE_VISIBLE_TAGS


def _rich_log_text(timestamp: str, message: str) -> Text:
    text = Text()
    text.append(timestamp, style="dim")
    text.append(" ")
    if message.startswith("[") and "]" in message:
        tag_end = message.find("]") + 1
        tag = message[:tag_end]
        text.append(tag, style=_style_for_tag(tag))
        remainder = message[tag_end:]
        if tag == "[stage]":
            banner_match = _STAGE_BANNER_RE.match(remainder)
            if banner_match is not None:
                prefix, left_rule, left_space, title, right_space, right_rule = banner_match.groups()
                text.append(prefix)
                text.append(left_rule, style="bold magenta")
                text.append(left_space)
                text.append(title, style="bold white on magenta")
                text.append(right_space)
                text.append(right_rule, style="bold magenta")
            else:
                text.append(remainder)
        else:
            text.append(remainder)
    else:
        text.append(message)
    return text


def log_message(args: argparse.Namespace | None, message: str, *, renderable: Any | None = None) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    memory = format_memory_snapshot()
    line = f"{timestamp} {message} {memory}"
    if should_log(args) and _should_log_to_console(message):
        if renderable is not None and RICH_AVAILABLE:
            _CONSOLE.print(_rich_log_text(timestamp, str(message).splitlines()[0]))
            _CONSOLE.print(renderable)
        else:
            _CONSOLE.print(_rich_log_text(timestamp, message))
    debug_log_line(line)


def format_stage_banner(title: str, details: str | None = None, width: int = 78) -> list[str]:
    title_text = " ".join(str(title or "STAGE").split())
    core = f" {title_text} "
    banner_width = max(int(width), len(core) + 8)
    left = max(4, (banner_width - len(core)) // 2)
    right = max(4, banner_width - len(core) - left)
    lines = [f"[stage] {'=' * left}{core}{'=' * right}"]
    if details:
        lines.append(f"[stage] {' '.join(str(details).split())}")
    return lines


def log_stage_banner(args: argparse.Namespace | None, title: str, details: str | None = None) -> None:
    for line in format_stage_banner(title, details):
        log_message(args, line)


def fmt_seconds(value: float) -> str:
    return f"{value:.2f}s"


def run_logged_phase(args: argparse.Namespace | None, phase: str, fn, *, detail: str | None = None):
    start = time.time()
    suffix = f" {detail}" if detail else ""
    log_message(args, f"[phase] {phase} start{suffix}")
    try:
        result = fn()
    except Exception as exc:
        log_message(args, f"[phase] {phase} error elapsed={fmt_seconds(time.time() - start)} {type(exc).__name__}: {exc}")
        raise
    log_message(args, f"[phase] {phase} end elapsed={fmt_seconds(time.time() - start)}")
    return result


def make_run_name(par_path: str | Path) -> str:
    stem = Path(par_path).stem
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}"


def load_lensmodel_dat(filepath):
    """Load a Lenstool-style .dat catalog into a pandas DataFrame."""
    columns = ["id", "ra", "dec", "a", "b", "theta", "mag", "z"]
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        header=None,
        names=columns,
    )

    if df.shape[1] != len(columns):
        raise ValueError(
            f"Expected {len(columns)} columns ({columns}), found {df.shape[1]} in '{filepath}'."
        )

    df["id"] = df["id"].astype(int)
    return df


def save_lensmodel_dat(df, filepath, reference=0):
    """Save a DataFrame to Lenstool-style .dat catalog format."""
    columns = ["id", "ra", "dec", "a", "b", "theta", "mag", "z"]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    data = df.loc[:, columns].copy()

    nan_or_empty_cols = []
    for col in data.columns:
        has_nan = data[col].isna().any()
        has_empty = pd.api.types.is_string_dtype(data[col]) and data[col].fillna("").astype(str).str.strip().eq("").any()
        if has_nan or has_empty:
            nan_or_empty_cols.append(col)

    if nan_or_empty_cols:
        print(f"Columns with NaN or empty values: {nan_or_empty_cols}")
    
    data["id"] = data["id"].astype(int)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"#REFERENCE {int(reference)}\n")
        for row in data.itertuples(index=False):
            f.write(
                f"{int(row.id):5d} "
                f"{float(row.ra):12.6f} "
                f"{float(row.dec):11.6f} "
                f"{float(row.a):10.6f} "
                f"{float(row.b):10.6f} "
                f"{float(row.theta):8.2f} "
                f"{float(row.mag):8.4f} "
                f"{float(row.z):5.1f}\n"
            )

def merge_radec(df1, df2, radec1=("ra", "dec"), radec2=("ra", "dec"),
                                         tol_arcsec=0.2, suffixes=("_1", "_2"),
                                         add_sep_col=True, ensure_unique=True,
                                         verbose=True, fields=None, join="inner"):
    """
    Nearest-neighbor match df1 -> df2 on-sky (astropy), keep matches within tol_arcsec.
    Supports inner-style output (matched rows only), left-style output (all df1 rows),
    or outer-style output (all df1 rows plus unmatched df2 rows).

    Returns:
      matched_df, unmatched_df1, unmatched_df2

    Notes:
    - Matching is done from df1 to df2 using match_to_catalog_sky (nearest neighbor).
    - If ensure_unique=True, df2 rows are used at most once (greedy by smallest separation).
    """
    ra1, dec1 = radec1
    ra2, dec2 = radec2

    if ra1 not in df1.columns or dec1 not in df1.columns:
        raise ValueError(f"df1 must contain columns {radec1}")
    if ra2 not in df2.columns or dec2 not in df2.columns:
        raise ValueError(f"df2 must contain columns {radec2}")

    if join not in {"inner", "left", "outer"}:
        raise ValueError("join must be 'inner', 'left', or 'outer'")
    if not isinstance(suffixes, (tuple, list)) or len(suffixes) != 2:
        raise ValueError("suffixes must be a 2-item tuple/list")

    suffixes_norm = tuple("" if s is None else str(s) for s in suffixes)
    coalesce_mode = suffixes_norm[0] == "" and suffixes_norm[1] == ""

    if fields is None:
        fields = list(df2.columns)
    else:
        fields = list(fields)
        missing_fields = [f for f in fields if f not in df2.columns]
        if missing_fields:
            raise ValueError(f"df2 is missing requested field(s): {missing_fields}")

    c1 = SkyCoord(df1[ra1].to_numpy() * u.deg, df1[dec1].to_numpy() * u.deg, frame="icrs")
    c2 = SkyCoord(df2[ra2].to_numpy() * u.deg, df2[dec2].to_numpy() * u.deg, frame="icrs")

    idx2, sep2d, _ = c1.match_to_catalog_sky(c2)
    tol = tol_arcsec * u.arcsec
    within = sep2d <= tol

    # Candidate matches: rows in df1 that have a close neighbor in df2
    cand_i1 = pd.Index(df1.index[within])
    cand_i2 = pd.Index(df2.index[idx2[within]])
    cand_sep = sep2d[within].arcsec

    if ensure_unique and len(cand_i1) > 0:
        # Greedy unique assignment by ascending separation
        order = pd.Series(cand_sep).sort_values().index.to_numpy()
        used2 = set()
        keep_pos = []
        for p in order:
            j2 = int(cand_i2[p])
            if j2 in used2:
                continue
            used2.add(j2)
            keep_pos.append(p)
        keep_pos = sorted(keep_pos)
        i1 = cand_i1[keep_pos]
        i2 = cand_i2[keep_pos]
        sep_arcsec = pd.Series(cand_sep).iloc[keep_pos].to_numpy()
    else:
        i1 = cand_i1
        i2 = cand_i2
        sep_arcsec = cand_sep

    # Unmatched (kept stable across join modes)
    unmatched_df1 = df1.drop(index=i1).copy()
    unmatched_df2 = df2.drop(index=i2).copy()

    # Determine overlapping non-coordinate columns to suffix (pandas-merge style)
    common = set(df1.columns) & set(fields)
    coord_cols = {ra1, dec1, ra2, dec2}
    common_noncoord = common - coord_cols

    left_rename = {c: f"{c}{suffixes_norm[0]}" for c in common_noncoord}
    right_rename = {c: f"{c}{suffixes_norm[1]}" for c in common_noncoord}

    # matched subsets
    left_match = df1.loc[i1].copy().rename(columns=left_rename)
    right_match = df2.loc[i2, fields].copy().rename(columns=right_rename)

    # If coordinate names collide, suffix df2 coordinate columns
    coord_rename = {}
    if ra2 in right_match.columns and ra2 in left_match.columns:
        coord_rename[ra2] = f"{ra2}{suffixes_norm[1]}"
    if dec2 in right_match.columns and dec2 in left_match.columns:
        coord_rename[dec2] = f"{dec2}{suffixes_norm[1]}"
    if coord_rename:
        right_match = right_match.rename(columns=coord_rename)

    def _empty_mask(s):
        empty = s.isna()
        if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
            stripped_empty = s.fillna("").astype(str).str.strip() == ""
            empty = empty | stripped_empty
        return empty

    def _merge_columns(left_df, right_df):
        out = left_df.copy()
        for col in right_df.columns:
            if col in out.columns:
                left_empty = _empty_mask(out[col])
                right_empty = _empty_mask(right_df[col])
                use_right = left_empty & (~right_empty)
                if use_right.any():
                    # `Series.where` promotes dtypes when needed, which avoids
                    # pandas warnings from assigning strings/bools into float
                    # placeholder columns during outer joins.
                    out[col] = out[col].where(~use_right, right_df[col])
            else:
                out[col] = right_df[col]
        return out

    if join == "inner":
        left_out = left_match.reset_index(drop=True)
        right_out = right_match.reset_index(drop=True)
        if coalesce_mode:
            matched = _merge_columns(left_out, right_out)
        else:
            matched = pd.concat([left_out, right_out], axis=1)
        if add_sep_col:
            matched["match_sep_arcsec"] = sep_arcsec
    else:
        left_out = df1.copy().rename(columns=left_rename)
        # Build an empty typed frame rather than a float NaN block so later
        # assignment of object/bool columns does not hit incompatible-dtype warnings.
        right_out = right_match.iloc[:0].reindex(left_out.index)
        if len(i1) > 0:
            right_values = right_match.reset_index(drop=True)
            right_values.index = i1
            right_out.loc[i1, right_values.columns] = right_values

        left_out_reset = left_out.reset_index(drop=True)
        right_out_reset = right_out.reset_index(drop=True)
        if coalesce_mode:
            matched = _merge_columns(left_out_reset, right_out_reset)
        else:
            matched = pd.concat([left_out_reset, right_out_reset], axis=1)
        if add_sep_col:
            sep_series = pd.Series(np.nan, index=left_out.index, dtype=float)
            if len(i1) > 0:
                sep_series.loc[i1] = sep_arcsec
            matched["match_sep_arcsec"] = sep_series.reset_index(drop=True).to_numpy()

        if join == "outer":
            left_nan_block = left_out.iloc[:0].reindex(unmatched_df2.index)
            right_unmatched = unmatched_df2.loc[:, fields].copy().rename(columns=right_rename)
            if coord_rename:
                right_unmatched = right_unmatched.rename(columns=coord_rename)
            left_nan_block_reset = left_nan_block.reset_index(drop=True)
            right_unmatched_reset = right_unmatched.reset_index(drop=True)
            if coalesce_mode:
                outer_tail = _merge_columns(left_nan_block_reset, right_unmatched_reset)
            else:
                outer_tail = pd.concat([left_nan_block_reset, right_unmatched_reset], axis=1)
            if add_sep_col:
                outer_tail["match_sep_arcsec"] = np.nan
            matched = pd.concat([matched, outer_tail], axis=0, ignore_index=True)

    if verbose:
        print(f"Total df1: {len(df1)}, Total df2: {len(df2)}")
        print(f"Matched: {len(matched)}, Unmatched df1: {len(unmatched_df1)}, Unmatched df2: {len(unmatched_df2)}")

    return matched, unmatched_df1, unmatched_df2

def radec_to_offsets(ra, dec, ra0, dec0, z, cosmo):
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    c0 = SkyCoord(ra=ra0*u.deg, dec=dec0*u.deg, frame="icrs")

    # small-angle tangent-plane offsets, with correct sign conventions:
    # +x = East (increasing RA), +y = North (increasing Dec)
    dlon, dlat = c.spherical_offsets_to(c0)  # returns (lon_offset, lat_offset)
    x_arcsec = (dlon.to(u.arcsec)).value    
    y_arcsec = (-dlat.to(u.arcsec)).value    # flip sign so +y is North (Dec increasing)

    if isinstance(cosmo, dict):
        z_values = np.asarray(z, dtype=float)
        arcsec2kpc = np.vectorize(lambda z_item: kpc_per_arcsec_from_config(float(z_item), cosmo))(z_values)
    else:
        arcsec2kpc = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec).value
    x_kpc = x_arcsec * arcsec2kpc
    y_kpc = y_arcsec * arcsec2kpc

    return x_kpc, y_kpc, x_arcsec, y_arcsec
