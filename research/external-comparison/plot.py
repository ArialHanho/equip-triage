"""Turn the frozen benchmark CSV into a three-panel publication plot."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator, PercentFormatter

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parents[1] / "docs" / "assets" / "external-benchmark.svg"
MODELS = [
    ("counts", "Event counts"),
    ("trigram", "Trigram"),
    ("mlm", "Custom MLM"),
    ("order", "Custom order head"),
    ("tiny_pretrained", "BERT-Tiny"),
    ("mini_pretrained", "BERT-Mini"),
    ("smol", "SmolLM2 · 4-shot"),
    ("milestone_order", "Custom head + milestones"),
    ("mini_milestones", "BERT-Mini + milestones"),
    ("precedence", "Learned precedence"),
]


def make_figure(data):
    ink, subtle, grid = "#182632", "#47596c", "#dde5ed"
    teal, blue, rust, purple = "#097c70", "#295eb0", "#a94c32", "#6147a0"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "equip-triage-comp-v2", "svg.fonttype": "none"})
    fig, (axr, axf, axt) = plt.subplots(
        1, 3, figsize=(12.2, 6.6), sharey=True,
        gridspec_kw={"width_ratios": [1.36, 0.93, 1.23], "wspace": 0.19},
    )
    fig.subplots_adjust(left=.223, right=.968, top=.744, bottom=.182)
    fig.patch.set_facecolor("white")
    pos = list(range(len(MODELS)))
    axr.set_yticks(pos, [name for _, name in MODELS])
    for ax in (axr, axf, axt):
        ax.set_axisbelow(True)
        ax.set_ylim(len(pos)-.4, -.57)
        ax.axhspan(6.5, len(pos)-.4, color="#edf7f4", zorder=0)
        ax.axhline(6.5, color="#a6c3bc", lw=.9, zorder=3)
        ax.grid(axis="x", color=grid, lw=.75)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#8d9eae")
        ax.tick_params(axis="x", colors=subtle, labelsize=10, length=3, pad=5)
        ax.tick_params(axis="y", colors=ink, labelsize=11, length=0, pad=12)

    axr.tick_params(labelleft=True)
    for label in axr.get_yticklabels():
        if label.get_text().startswith(("Custom head +", "BERT-Mini +", "Learned")):
            label.set_fontweight("bold")
    axr.set_xlim(-.035, 1.05)
    axr.xaxis.set_major_locator(FixedLocator([0, .25, .50, .75, 1]))
    axr.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axf.set_xlim(-.003, .047)
    axf.xaxis.set_major_locator(FixedLocator([0, .01, .02, .03, .04]))
    axf.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axf.axvline(.01, color="#758596", lw=1, linestyle=(0, (3, 3)), zorder=2)
    axt.set_xscale("log")
    axt.set_xlim(1, 1.6e6)
    axt.xaxis.set_major_locator(FixedLocator([1, 100, 10000, 1000000]))
    axt.xaxis.set_major_formatter(LogFormatterMathtext())
    axt.xaxis.set_minor_locator(NullLocator())
    for i, (method, _) in enumerate(MODELS):
        full = data[method, "all"]
        seen = float(data[method, "seen_scan"]["recall"])
        unseen = float(data[method, "held_release"]["recall"])
        axr.hlines(i-.14, 0, seen, color=teal, alpha=.32, linewidth=1.5)
        axr.hlines(i+.14, 0, unseen, color=blue, alpha=.32, linewidth=1.5)
        axr.scatter(seen, i-.14, s=35, c=teal, zorder=4)
        axr.scatter(unseen, i+.14, s=38, marker="D", facecolors="white",
                    edgecolors=blue, linewidths=1.4, zorder=4)
        fpr = float(full["fpr"])
        axf.hlines(i, 0, fpr, color=rust, alpha=.35, linewidth=1.5)
        axf.scatter(fpr, i, c=rust, s=38, zorder=4)
        speed = float(full["traces_per_cpu_s"])
        axt.hlines(i, 1, speed, color=purple, alpha=.35, linewidth=1.5)
        axt.scatter(speed, i, c=purple, marker="s", s=34, zorder=4)

    for ax, panel, heading, sub in (
        (axr, "A", "Detection rate", "Per fault slice"),
        (axf, "B", "False-alarm rate", "Normal test traces"),
        (axt, "C", "CPU throughput", "Traces per CPU second · log scale"),
    ):
        ax.text(0, 1.105, f"{panel}    {heading}", transform=ax.transAxes,
                color=ink, size=13, weight="bold", ha="left", va="bottom")
        ax.text(0, 1.017, sub, transform=ax.transAxes, color=subtle,
                size=9, ha="left", va="bottom")

    fig.text(.053, .955, "EQUIP TRIAGE   /   EXTERNAL MODEL COMPARISON", color=teal,
             size=10, weight="bold")
    fig.text(.053, .891, "Order-fault detection across models", color=ink, size=24, weight="bold")
    fig.text(.053, .85, "192 matched synthetic test pairs · seed 42 · identical event counts within each pair",
             color=subtle, size=11)
    handles = [
        Line2D([], [], marker="o", linestyle="None", color=teal, label="Familiar fault (96)"),
        Line2D([], [], marker="D", linestyle="None", color=blue, markerfacecolor="white",
               markeredgewidth=1.4, label="Withheld fault (96)"),
        Line2D([], [], linestyle="--", color="#758596", label="Validation false-alarm target (1%)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.044,.065),
               ncol=3, frameon=False, fontsize=10, labelcolor=ink, columnspacing=2.25)
    fig.text(.053, .045, "Shading: projected milestone inputs or a precedence rule. BERT is fine-tuned; SmolLM2 is frozen.",
             color=subtle, size=9.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with (HERE / "metrics.csv").open(newline="") as handle:
        data = {(r["method"], r["slice"]): r for r in csv.DictReader(handle) if r["seed"] == "42"}
    fig = make_figure(data)
    info = {"Title": "Equip Triage: order-fault model comparison"}
    if args.output.suffix.lower() == ".svg":
        info["Date"] = None
        info["Description"] = ("Aligned detection, false-alarm and CPU-throughput plots for 192 synthetic pairs. "
                               "Milestone-filtered custom and BERT-Mini models tie with learned precedence.")
    fig.savefig(args.output, dpi=135, facecolor="white", metadata=info)
    if args.output.suffix.lower() == ".svg":
        clean = "\n".join(line.rstrip() for line in args.output.read_text(encoding="utf-8").splitlines())
        args.output.write_text(clean + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
