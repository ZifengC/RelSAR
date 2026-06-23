from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
import matplotlib.pyplot as plt


CHANNEL_ORDER = {"R": 0, "S": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate intent-pattern analyses from intermediate CSV exports."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="intermediate/pcsar_intent_features_test.csv",
        help="Path to the exported intermediate CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Root directory for figures and tables.",
    )
    parser.add_argument(
        "--exploration-column",
        type=str,
        default="history_src_share",
        help="Column used as the exploration score proxy.",
    )
    parser.add_argument(
        "--user-key",
        type=str,
        default="user_id",
        help="Column used to build per-user sequences.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def channel_sort_key(series: pd.Series) -> pd.Series:
    return series.map(CHANNEL_ORDER).fillna(99).astype(int)


def normalize_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    min_v = np.nanmin(values[finite])
    max_v = np.nanmax(values[finite])
    if np.isclose(max_v, min_v):
        return np.zeros_like(values)
    return (values - min_v) / (max_v - min_v)


def safe_entropy(pi: np.ndarray) -> float:
    pi = np.asarray(pi, dtype=np.float64)
    pi = np.clip(pi, 1e-12, 1.0)
    pi = pi / np.sum(pi)
    k = max(pi.shape[-1], 1)
    denom = np.log(k) if k > 1 else 1.0
    return float(-(pi * np.log(pi)).sum() / denom)


def js_distance(pi_a: np.ndarray, pi_b: np.ndarray) -> float:
    a = np.asarray(pi_a, dtype=np.float64)
    b = np.asarray(pi_b, dtype=np.float64)
    a = np.clip(a, 1e-12, 1.0)
    b = np.clip(b, 1e-12, 1.0)
    a = a / a.sum()
    b = b / b.sum()
    m = 0.5 * (a + b)
    kl_am = np.sum(a * (np.log(a) - np.log(m)))
    kl_bm = np.sum(b * (np.log(b) - np.log(m)))
    return float(np.sqrt(max(0.5 * (kl_am + kl_bm), 0.0)))


def pi_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    def suffix_value(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 10**9
    return sorted(cols, key=suffix_value)


def prepare_frame(df: pd.DataFrame, exploration_column: str, user_key: str) -> pd.DataFrame:
    if exploration_column not in df.columns:
        fallback = "history_rec_share" if "history_rec_share" in df.columns else None
        if fallback is None:
            raise KeyError(
                f"Missing exploration column '{exploration_column}' and no fallback column found."
            )
        exploration_column = fallback

    numeric_cols = [
        "timestamp",
        "sample_index",
        "global_dominant_intent_prob",
        "global_intent_entropy",
        "global_posterior_uncertainty",
        "global_belief_uncertainty_mean",
        "rec_src_intent_shift_js",
        "attribution_confidence_gap",
    ]
    df = coerce_numeric(df.copy(), numeric_cols)
    df["exploration_score"] = pd.to_numeric(df[exploration_column], errors="coerce")
    df["channel_order"] = channel_sort_key(df["channel"])
    sort_cols = [user_key]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    sort_cols.extend(["channel_order"])
    if "sample_index" in df.columns:
        sort_cols.append("sample_index")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df


def contiguous_runs(values: np.ndarray) -> list[int]:
    values = np.asarray(values)
    if len(values) == 0:
        return []
    runs: list[int] = []
    run_len = 1
    for prev, cur in zip(values[:-1], values[1:]):
        if prev == cur:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    return runs


def mean_consecutive_jsd(pi_matrix: np.ndarray) -> float:
    if len(pi_matrix) < 2:
        return 0.0
    vals = [js_distance(a, b) for a, b in zip(pi_matrix[:-1], pi_matrix[1:])]
    return float(np.mean(vals)) if vals else 0.0


def mean_abs_diff(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


def build_session_summary(df: pd.DataFrame, user_key: str) -> pd.DataFrame:
    pi_cols = pi_columns(df, "global_pi_")
    if not pi_cols:
        raise KeyError("Could not find global_pi_* columns in the input CSV.")

    rows = []
    for user_id, g in df.groupby(user_key, sort=False):
        g = g.reset_index(drop=True)
        dom = pd.to_numeric(g["global_dominant_intent"], errors="coerce").fillna(-1).to_numpy()
        intent_entropy = pd.to_numeric(g["global_intent_entropy"], errors="coerce").to_numpy()
        dominant_prob = pd.to_numeric(g["global_dominant_intent_prob"], errors="coerce").to_numpy()
        uncertainty = pd.to_numeric(g["global_posterior_uncertainty"], errors="coerce").to_numpy()
        attr_gap = pd.to_numeric(g["attribution_confidence_gap"], errors="coerce").to_numpy()
        pi = g[pi_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(pi) == 0:
            continue

        runs = contiguous_runs(dom)
        row = {
            user_key: user_id,
            "sequence_length": int(len(g)),
            "run_length": int(max(runs) if runs else 0),
            "mean_run_length": float(np.mean(runs) if runs else 0.0),
            "switch_count": int(max(len(runs) - 1, 0)),
            "mean_intent_entropy": float(np.nanmean(intent_entropy)),
            "mean_dominant_intent_prob": float(np.nanmean(dominant_prob)),
            "posterior_dispersion": mean_consecutive_jsd(pi),
            "mean_uncertainty": float(np.nanmean(uncertainty)),
            "attribution_dispersion": mean_abs_diff(pd.Series(attr_gap).fillna(0.0).to_numpy()),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_transition_summary(df: pd.DataFrame, user_key: str) -> pd.DataFrame:
    pi_cols = pi_columns(df, "global_pi_")
    if not pi_cols:
        raise KeyError("Could not find global_pi_* columns in the input CSV.")

    rows = []
    for user_id, g in df.groupby(user_key, sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        dom = pd.to_numeric(g["global_dominant_intent"], errors="coerce").fillna(-1).to_numpy()
        uncertainty = pd.to_numeric(g["global_posterior_uncertainty"], errors="coerce").to_numpy()
        attr_gap = pd.to_numeric(g["attribution_confidence_gap"], errors="coerce").to_numpy()
        proxy = g["attribution_source_proxy"].astype(str).replace("nan", np.nan).to_numpy()
        pi = g[pi_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        for i in range(1, len(g)):
            prev_proxy = proxy[i - 1]
            cur_proxy = proxy[i]
            if pd.isna(prev_proxy) or pd.isna(cur_proxy):
                continue
            rows.append(
                {
                    user_key: user_id,
                    "anchor_transition": f"{prev_proxy}->{cur_proxy}",
                    "intent_shift": js_distance(pi[i - 1], pi[i]),
                    "dominant_intent_change": float(dom[i - 1] != dom[i]),
                    "uncertainty": float(uncertainty[i]),
                    "attribution_shift": float(abs(attr_gap[i] - attr_gap[i - 1])),
                }
            )

    return pd.DataFrame(rows)


def decile_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    valid = df[np.isfinite(df[score_col])].copy()
    if valid.empty:
        return pd.DataFrame()
    n_bins = min(10, valid[score_col].nunique())
    if n_bins < 1:
        return pd.DataFrame()
    valid["decile"] = pd.qcut(
        valid[score_col].rank(method="first"),
        q=n_bins,
        labels=False,
        duplicates="drop",
    ) + 1
    out = (
        valid.groupby("decile", as_index=False)
        .agg(
            exploration_score=(score_col, "mean"),
            intent_entropy=("global_intent_entropy", "mean"),
            belief_uncertainty=("global_posterior_uncertainty", "mean"),
            dominant_intent_prob=("global_dominant_intent_prob", "mean"),
            n=("decile", "size"),
        )
        .sort_values("decile")
    )
    return out


def aggregate_by_x(df: pd.DataFrame, x_col: str, metric_cols: list[str]) -> pd.DataFrame:
    out = (
        df.groupby(x_col, as_index=False)
        .agg({col: "mean" for col in metric_cols})
        .sort_values(x_col)
        .reset_index(drop=True)
    )
    return out


def plot_stacked_lines(
    summary: pd.DataFrame,
    x_col: str,
    metric_cols: list[str],
    titles: list[str],
    out_path: Path,
    x_label: str,
) -> None:
    ensure_dir(out_path.parent)
    plt.style.use("ggplot")
    fig, axes = plt.subplots(
        len(metric_cols),
        1,
        figsize=(11, 3.2 * len(metric_cols)),
        sharex=True,
        constrained_layout=True,
    )
    if len(metric_cols) == 1:
        axes = [axes]
    palette = plt.get_cmap("tab10")

    for idx, (ax, metric, title) in enumerate(zip(axes, metric_cols, titles)):
        ax.plot(
            summary[x_col],
            summary[metric],
            marker="o",
            linewidth=2,
            color=palette(idx % 10),
        )
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel(x_label)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_bar_panel(
    summary: pd.DataFrame,
    x_col: str,
    metric_cols: list[str],
    titles: list[str],
    out_path: Path,
    x_label: str,
) -> None:
    ensure_dir(out_path.parent)
    plt.style.use("ggplot")
    fig, axes = plt.subplots(
        len(metric_cols),
        1,
        figsize=(11, 3.1 * len(metric_cols)),
        sharex=True,
        constrained_layout=True,
    )
    if len(metric_cols) == 1:
        axes = [axes]
    palette = plt.get_cmap("tab10")

    x = np.arange(len(summary))
    for idx, (ax, metric, title) in enumerate(zip(axes, metric_cols, titles)):
        ax.bar(x, summary[metric], color=palette(idx % 10), alpha=0.85)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(summary[x_col].tolist())
    axes[-1].set_xlabel(x_label)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path, low_memory=False)
    df = prepare_frame(raw, args.exploration_column, args.user_key)

    # Figure 1: exploration score decile -> intent ambiguity
    fig1 = decile_summary(df, "exploration_score")
    fig1_dir = output_root / "1"
    save_table(fig1, fig1_dir / "exploration_intent_ambiguity.csv")
    plot_stacked_lines(
        fig1,
        "decile",
        ["intent_entropy", "belief_uncertainty", "dominant_intent_prob"],
        [
            "Exploration -> Intent Entropy",
            "Exploration -> Belief Uncertainty",
            "Exploration -> Dominant Intent Probability",
        ],
        fig1_dir / "exploration_intent_ambiguity.png",
        "exploration_score decile",
    )

    # Figure 2: run length -> intent consolidation
    session_summary = build_session_summary(df, args.user_key)
    fig2 = aggregate_by_x(
        session_summary,
        "run_length",
        ["mean_intent_entropy", "mean_dominant_intent_prob", "switch_count"],
    )
    fig2_dir = output_root / "2"
    save_table(fig2, fig2_dir / "run_length_intent_consolidation.csv")
    plot_stacked_lines(
        fig2,
        "run_length",
        ["mean_intent_entropy", "mean_dominant_intent_prob", "switch_count"],
        [
            "Run Length -> Mean Intent Entropy",
            "Run Length -> Mean Dominant Intent Probability",
            "Run Length -> Intent Switch Count",
        ],
        fig2_dir / "run_length_intent_consolidation.png",
        "run_length",
    )

    # Figure 3: run length -> intent dispersion
    fig3 = aggregate_by_x(
        session_summary,
        "run_length",
        ["posterior_dispersion", "mean_uncertainty", "attribution_dispersion"],
    )
    fig3_dir = output_root / "3"
    save_table(fig3, fig3_dir / "run_length_intent_dispersion.csv")
    plot_stacked_lines(
        fig3,
        "run_length",
        ["posterior_dispersion", "mean_uncertainty", "attribution_dispersion"],
        [
            "Run Length -> Posterior Dispersion",
            "Run Length -> Uncertainty",
            "Run Length -> Attribution Dispersion",
        ],
        fig3_dir / "run_length_intent_dispersion.png",
        "run_length",
    )

    # Figure 4: anchor transition -> intent shift
    transition_summary = build_transition_summary(df, args.user_key)
    order = ["R->R", "R->S", "S->R", "S->S"]
    if not transition_summary.empty:
        transition_summary["anchor_transition"] = pd.Categorical(
            transition_summary["anchor_transition"],
            categories=order,
            ordered=True,
        )
        transition_summary = (
            transition_summary.groupby("anchor_transition", as_index=False)
            .agg(
                intent_shift=("intent_shift", "mean"),
                dominant_intent_change=("dominant_intent_change", "mean"),
                uncertainty=("uncertainty", "mean"),
                attribution_shift=("attribution_shift", "mean"),
            )
            .sort_values("anchor_transition")
            .reset_index(drop=True)
        )

    fig4_dir = output_root / "4"
    save_table(transition_summary, fig4_dir / "transition_type_intent_shift.csv")
    plot_bar_panel(
        transition_summary,
        "anchor_transition",
        [
            "intent_shift",
            "dominant_intent_change",
            "uncertainty",
            "attribution_shift",
        ],
        [
            "Transition Type -> Mean Intent Shift",
            "Transition Type -> Dominant Intent Change Rate",
            "Transition Type -> Mean Uncertainty",
            "Transition Type -> Attribution Shift",
        ],
        fig4_dir / "transition_type_intent_shift.png",
        "anchor_transition",
    )

    print(f"Saved analysis tables and figures under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
