"""Three synthetic patients drawn as pictures: a pictogram, FibroScan stiffness against age, and a
follow-up lane with one tick per visit. Patient A returns on the serialization slide (section 4)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

RED, BLUE = "#d62728", "#2171b5"             # event / event-free, as on the next slide
W_IN, H_IN = 11.5, 4.85                      # figure size in inches; the layout below is in inches too


def pick(C):
    """A: the sickest hepatic-event patient with 5+ visits (serialized in section 4). B: a long,
    stable record. C: event-free but progressing, so still censored."""
    n = C.filter(regex=r"^Age_v\d+$").notna().sum(axis=1)
    free = ~C.hep_event & ~C.death_event
    return {"A": C[C.hep_event & (n >= 5)].sort_values("true_hep").iloc[-1],
            "B": C[free & (n >= 15)].sort_values("true_hep").iloc[0],
            "C": C[free & n.between(6, 10)].sort_values("true_hep").iloc[-1]}


def _record(r):
    ages = np.array([r.get(f"Age_v{v}", np.nan) for v in range(1, 23)], float)
    kpa = np.array([r.get(f"fibs_stiffness_med_BM_1_v{v}", np.nan) for v in range(1, 23)], float)
    seen = ~np.isnan(ages)
    return ages[seen], kpa[seen], r.Age_v1 + r.hep_time       # visit ages, FibroScan, end of follow-up


def draw(three, fs=1.0, colors=(RED, BLUE), short=False, out=None, dpi=100, font=None, note=True):
    """fs scales every font, short trims the captions to fit, out saves a PNG instead of showing,
    font sets the typeface, note=False drops the in-figure "synthetic" line (a slide footer can say it)."""
    prev = plt.rcParams["font.family"]
    if font:
        plt.rcParams["font.family"] = [font]
    red, blue = colors
    rec = {k: _record(r) for k, r in three.items()}
    ymax = max(np.nanmax(v[1]) for v in rec.values()) + 2      # one y scale, so patients compare honestly
    fig = plt.figure(figsize=(W_IN, H_IN))
    box = lambda x, y, w, h: [x, y / H_IN, w, h / H_IN]       # y and h in inches from the bottom
    for j, (k, r) in enumerate(three.items()):
        x0, w = (0.07, 0.395, 0.72)[j], 0.265
        ages, kpa, end = rec[k]
        ev = bool(r.hep_event); col = red if ev else blue
        ic = fig.add_axes(box(x0, 3.95, 0.8 / W_IN, 0.8)); ic.axis("off"); ic.set_aspect("equal")
        ic.set_xlim(0, 1); ic.set_ylim(0, 1)
        ic.add_patch(Circle((0.5, 0.75), 0.2, color=col)); ic.add_patch(Wedge((0.5, 0.0), 0.46, 0, 180, color=col))
        comorb = [n for n, f in (("type 2 diabetes", r.T2DM), ("hypertension", r.Hypertension),
                                 ("dyslipidaemia", r.Dyslipidaemia)) if f] or ["no comorbidity recorded"]
        sex = "man" if r.gender else "woman"
        who = f"{sex}, {r.Age_v1:.0f}" if short else f"{sex}, {r.Age_v1:.0f} at first visit"
        for y, text, kw in ((4.58, f"Patient {k}", dict(fontsize=17 * fs, weight="bold")),
                            (4.32, who, dict(fontsize=13 * fs)),
                            (4.08, ", ".join(comorb), dict(fontsize=13 * fs))):
            fig.text(x0 + 0.085, y / H_IN, text, va="center", **kw)
        ax = fig.add_axes(box(x0, 1.75, w, 1.85)); lane = fig.add_axes(box(x0, 0.95, w, 0.6), sharex=ax)
        m = ~np.isnan(kpa)
        ax.plot(ages[m], kpa[m], "-o", color=col, lw=2.5, ms=9, mec="white", mew=1.5)
        ax.axhline(12.5, ls="--", c="grey", lw=1.3)
        ax.set_title(f"{len(ages)} visits, FibroScan at {m.sum()}", loc="left", fontsize=13 * fs, pad=5)
        ax.set_ylim(0, ymax); ax.set_xlim(ages[0] - 0.4, end + 1.3)
        if j == 0:
            ax.text(ax.get_xlim()[1] - 0.1, 12.1, "cirrhosis, 12.5 kPa", ha="right", va="top", fontsize=12 * fs, color="#555")
        ax.tick_params(labelsize=12 * fs, labelbottom=False, labelleft=j == 0)
        if j == 0:
            ax.set_ylabel("liver stiffness (kPa)", fontsize=13 * fs)
        lane.hlines(0, ages[0], end, color="#d0d0d0", lw=8, capstyle="round")      # follow-up, a tick per visit
        lane.plot(ages, np.zeros_like(ages), "|", color="#333", ms=18, mew=2, zorder=4)   # above the event marker
        if ev:
            lane.plot(end, 0, "X", color=red, ms=17, mec="white", mew=1.5, zorder=3)
        else:
            lane.plot(end, 0, "o", mfc="white", mec=blue, mew=3, ms=14)
        lane.annotate("hepatic event" if ev else "still event-free", (end, 0), xytext=(0, 13),
                      textcoords="offset points", ha="right", fontsize=13 * fs)
        lane.set_ylim(-1, 1.6); lane.set_yticks([])
        for a, sides in ((ax, ("top", "right")), (lane, ("top", "right", "left"))):
            for side in sides:
                a.spines[side].set_visible(False)
        lane.xaxis.set_major_locator(plt.MaxNLocator(integer=True)); lane.tick_params(labelsize=12 * fs)
        lane.set_xlabel("age (years)", fontsize=13 * fs)
    if note:
        fig.text(0.5, 0.07 / H_IN, "Synthetic patients, not real people", ha="center", fontsize=12 * fs, color="#555")
    if out:
        fig.savefig(out, dpi=dpi, transparent=False, facecolor="white")
        plt.close(fig)
    else:
        plt.show()
    plt.rcParams["font.family"] = prev
