from __future__ import annotations

"""One-shot plots for sequence_statistics_contact_gate_benchmark_live.py.

Run AFTER the benchmark has finished:
    python plot_live_benchmark.py

The script reads all rows currently saved in contact_gate_benchmark_live.jsonl,
creates the summary charts once, saves a PNG, and exits after the chart window
is closed. It does not watch or continuously monitor the result file.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE / "contact_gate_benchmark_live.jsonl"
OUTPUT_IMAGE = HERE / "contact_gate_benchmark_summary.png"


def load_rows() -> list[dict]:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"找不到結果檔案：{RESULTS_FILE}\n"
            "請先執行 sequence_statistics_contact_gate_benchmark_live.py，"
            "並確認 Benchmark 已經開始產生結果。"
        )

    rows: list[dict] = []
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告：第 {line_number} 行不是完整 JSON，已略過。")
                continue
            if isinstance(row, dict):
                rows.append(row)

    if not rows:
        raise ValueError(
            f"結果檔案目前沒有可用的實驗資料：{RESULTS_FILE}\n"
            "請確認 Benchmark 已完整執行至少一次實驗。"
        )
    return rows


def values_for_user(rows: list[dict], user: str, field: str) -> list[float]:
    values = []
    for row in rows:
        if str(row.get("template_user", "?")) != user:
            continue
        value = row.get(field)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return values


def mean_percent(rows: list[dict], user: str, field: str) -> float:
    values = values_for_user(rows, user, field)
    return float(np.mean(values) * 100) if values else np.nan


def draw_summary(rows: list[dict]) -> None:
    users = list(dict.fromkeys(str(r.get("template_user", "?")) for r in rows))
    x = np.arange(len(users))
    width = 0.19

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Sequence Statistics + Contact Gate Benchmark Summary\n"
        f"{len(rows)} completed experiments",
        fontsize=15,
    )
    ax1, ax2, ax3, ax4 = axes.flat

    # 1. Standalone and end-to-end recognition metrics.
    for offset, field, label in [
        (-1.5 * width, "standalone_gar", "Standalone GAR"),
        (-0.5 * width, "gar", "E2E GAR"),
        (0.5 * width, "standalone_far", "Standalone FAR"),
        (1.5 * width, "far", "E2E FAR"),
    ]:
        vals = [mean_percent(rows, user, field) for user in users]
        ax1.bar(x + offset, vals, width, label=label)
    ax1.set_title("Standalone vs E2E recognition")
    ax1.set_ylabel("Rate (%)")
    ax1.set_xticks(x, users, rotation=30, ha="right")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(fontsize=8)

    # 2. Rejection and gate error rates.
    for offset, field, label in [
        (-1.5 * width, "frr", "E2E FRR"),
        (-0.5 * width, "contact_miss_rate", "Contact Gate miss"),
        (0.5 * width, "no_contact_far", "No-contact FAR"),
        (1.5 * width, "standalone_frr", "Standalone FRR"),
    ]:
        vals = [mean_percent(rows, user, field) for user in users]
        ax2.bar(x + offset, vals, width, label=label)
    ax2.set_title("False rejection / gate error rates")
    ax2.set_ylabel("Rate (%)")
    ax2.set_xticks(x, users, rotation=30, ha="right")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(fontsize=8)

    # 3. Contact gate performance.
    for field, label in [
        ("contact_rate", "Genuine contact rate"),
        ("no_contact_reject", "No-contact rejection"),
    ]:
        vals = [mean_percent(rows, user, field) for user in users]
        ax3.plot(users, vals, marker="o", label=label)
    ax3.set_title("Contact Gate performance")
    ax3.set_ylabel("Rate (%)")
    ax3.set_ylim(0, 105)
    ax3.tick_params(axis="x", rotation=30)
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8)

    # 4. Metrics for each completed experiment.
    sequence = np.arange(1, len(rows) + 1)
    gar = np.array([float(r.get("gar", np.nan)) * 100 for r in rows])
    far = np.array([float(r.get("far", np.nan)) * 100 for r in rows])
    ax4.plot(sequence, gar, marker=".", linewidth=1, label="E2E GAR")
    ax4.plot(sequence, far, marker=".", linewidth=1, label="E2E FAR")
    ax4.set_title("Metrics by experiment")
    ax4.set_xlabel("Completed experiment index")
    ax4.set_ylabel("Rate (%)")
    ax4.set_ylim(0, 105)
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_IMAGE, dpi=180, bbox_inches="tight")
    print(f"已讀取 {len(rows)} 筆實驗結果。")
    print(f"圖表已儲存至：{OUTPUT_IMAGE}")
    print("圖表視窗關閉後，程式就會結束；不會持續監控結果檔案。")
    plt.show()


def main() -> None:
    try:
        rows = load_rows()
        draw_summary(rows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"錯誤：{exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
