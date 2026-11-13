"""results/*.json -> results/summary.png"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
HI, LO, INK, MUTED, GRID = "#2563eb", "#b8bec9", "#1f2937", "#6b7280", "#e5e7eb"

tp = json.loads((R / "throughput.json").read_text())
chaos = json.loads((R / "chaos.json").read_text())

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})

order = ["cpp", "cpp_dense", "cpp_opencv", "py_tuned", "py_naive"]
names = {"cpp": "C++ mhprep (FFmpeg, skip non-ref)", "cpp_dense": "C++ (FFmpeg, every frame)",
         "cpp_opencv": "C++ (cv::VideoCapture)", "py_tuned": "Python, tuned", "py_naive": "Python, typical"}
c = tp["configs"]
vals = [c[k]["realtime_factor"] for k in order]
y = range(len(order))[::-1]
a1.barh(list(y), vals, color=[HI if k == "cpp" else LO for k in order], height=0.6)
a1.set_yticks(list(y), [names[k] for k in order])
for yy, v, k in zip(y, vals, order):
    a1.text(v + 0.4, yy, f"{v:.0f}x realtime  ({c[k]['wall_sec']:.0f} s)", va="center", color=INK, fontsize=9)
a1.set_xlim(0, max(vals) * 1.45)
a1.set_xlabel(f"Throughput, {tp['corpus']['video_minutes']:.0f} min of H.264 + {tp['corpus']['images']} images, "
              f"{tp['workers']} cores (higher is better)")
a1.set_title(f"Same algorithm, same accuracy: {c['cpp']['speedup_vs_py_tuned']:.1f}x tuned Python, "
             f"{c['cpp']['speedup_vs_py_naive']:.1f}x typical", loc="left", color=INK, fontsize=11)

v2 = [chaos["cpp"]["valid_completion_pct"], chaos["python"]["valid_completion_pct"]]
a2.barh([1, 0], v2, color=[HI, LO], height=0.5)
a2.set_yticks([1, 0], ["C++ mhprep", "Python ProcessPoolExecutor"])
for yy, v, k in zip([1, 0], v2, ["cpp", "python"]):
    a2.text(min(v, 100) + 1.5, yy, f"{v:.1f}%  ({chaos[k]['wall_sec']:.0f} s)", va="center", color=INK, fontsize=9)
a2.set_xlim(0, 135)
a2.set_xlabel("Valid jobs completed under injected faults (%)")
a2.set_title("Crashes, hangs, stalled URLs, bad files", loc="left", color=INK, fontsize=11)

for ax in (a1, a2):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(R / "summary.png", dpi=160, facecolor="white")
print("wrote", R / "summary.png")
