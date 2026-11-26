"""Plot paired seed-level response contrasts from a completed partial-block run."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(folder):
    with (folder / "contrasts.csv").open() as f:
        rows = list(csv.DictReader(f))
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "partial-block", "font.size": 10})
    names = {"baseline_lif": "Baseline LIF", "selected_biophysics": "Stage 2", "selected_rebound": "Stage 4A"}
    colors = ["#324d69", "#05786f", "#b14d27"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.axvline(50, color="#c4cbd1", lw=1)
    ax.axhline(2, color="#8f9398", ls="--", lw=1, label="Decision margin (2 Hz)")
    for (key, label), color in zip(names.items(), colors):
        sub = [r for r in rows if r["model"] == key and r["measure"] == "h2_minus_h1_response"]
        x = [100 * float(r["block"]) for r in sub]
        y = [float(r["mean_hz"]) for r in sub]
        ax.fill_between(x, [float(r["low_hz"]) for r in sub], [float(r["high_hz"]) for r in sub],
                        color=color, alpha=.18)
        ax.plot(x, y, color=color, label=label, lw=2, marker="o", ms=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xlabel="Reduction in inhibitory input to GPi (%)", ylabel="H2 minus H1 firing-rate response (Hz)",
           xticks=[0, 25, 50, 75, 100])
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(folder / "response.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    plot(parser.parse_args().results)
