"""Data and images for the pptx (build_deck.js).

Reads the deck's synthetic cohort, the notebook's saved outputs and talk/data/*.csv, then writes
build/data.json and the PNGs the slides use. Every number comes from those sources, never typed
in here. A pattern that stops matching raises, so a changed notebook cannot slip through silently.

    python3 talk/pptx/figures.py
"""
import csv
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
TALK, ROOT = HERE.parent, HERE.parent.parent
OUT = HERE / "build"
OUT.mkdir(exist_ok=True)
for p in (ROOT / "scripts", TALK):
    sys.path.insert(0, str(p))
import patient_cards as PC                                   # noqa: E402
import synthetic as S                                        # noqa: E402
from annitia_helm_submit import _fib4, serialize_patient     # noqa: E402
from helm_gptoss_blend_v2 import _rank01                     # noqa: E402
from sksurv.metrics import concordance_index_censored        # noqa: E402

VERM, BLUE = "#D55E00", "#0072B2"                            # Okabe-Ito: event, event-free
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

nb = json.loads((TALK / "annitia_talk.ipynb").read_text())
TEXT = "\n".join("".join(o.get("text", [])) for c in nb["cells"] for o in c.get("outputs", []))
SRC = (TALK / "annitia_talk.py").read_text()


def grab(pattern, text=TEXT):
    m = re.search(pattern, text)
    assert m, f"no match in the notebook outputs: {pattern}"
    return m


def f(pattern, group=1):
    return float(grab(pattern).group(group))


def rows(name):
    with open(TALK / "data" / name, newline="") as fh:
        return list(csv.DictReader(fh))


def cidx(ev, t, s):
    return concordance_index_censored(np.asarray(ev, bool), np.asarray(t, float), s)[0]


# ---- numbers parsed from the notebook outputs
D = {"hook": [(r["submission"], float(r["public_score"])) for r in rows("leaderboard.csv")]}
D["objective"] = dict(score=f(r"C-index of a score\s+([\d.]+)"), exp=f(r"C-index of exp\(score\)\s+([\d.]+)"),
                      raw=f(r"raw 50/50 average\s+([\d.]+)"), rank=f(r"rank 50/50 average\s+([\d.]+)"))
oracle = r"hep\s+XGB [\d.]+\s+Cat [\d.]+\s+blend ([\d.]+)\s+true risk ([\d.]+)"
D["oracle"] = dict(whole=f(r"whole follow-up window:\n\s+" + oracle), truth=f(r"whole follow-up window:\n\s+" + oracle, 2),
                   baseline=f(r"baseline visit only:\n\s+" + oracle))
D["levels"] = [(int(a), int(b), float(c)) for a, b, c in re.findall(r"(\d+) levels:\s+(\d+) unique of \d+, LLM alone C = ([\d.]+)", TEXT)]
D["resolution"] = [(w, int(n), float(se), float(g)) for w, n, se, g in
                   re.findall(r"(public|private|whole test)\s+n = (\d+): SE ([\d.]+), credible gain ([\d.]+)", TEXT)]
D["null_gain"] = [(int(k), float(g)) for k, g in re.findall(r"K =\s+(\d+) tries on one sample -> typical best null gain ([\d.]+)", TEXT)]
D["mechanism"] = [(float(r), float(p), float(s)) for r, p, _, s in
                  re.findall(r"error corr ([\d.]+): predicted best gain ([+-][\d.]+) at w = ([\d.]+) \| simulated ([+-][\d.]+)", TEXT)]
D["observed"] = [(n, float(o), float(p), float(w)) for n, o, w, p in
                 re.findall(r"(\w[\w-]* into [\w -]+?): observed ([+-][\d.]+) \| predicted at w = ([\d.]+) with independent errors ([+-][\d.]+)", TEXT)]
D["leak"] = dict(cv_as_run=f(r"tabular base: 10-fold CV ([\d.]+)"), leaderboard=f(r"public leaderboard ([\d.]+)"),
                 window_free=f(r"same base, window-free features: CV ([\d.]+)"),
                 span_alone=f(r"leak probe, follow-up span alone: CV ([\d.]+)"), red_flag=float(grab(r"RED_FLAG = ([\d.]+)", SRC).group(1)))
D["replication"] = dict(base=f(r"\nbase\s+([\d.]+) \(hep"), gptoss_alone=f(r"GPT-OSS alone\s+([\d.]+)"),
                        blend=f(r"base \+ GPT-OSS\s+([\d.]+)"),
                        hep=[f(r"hepatic\s+blend wins (\d+)/10 folds"), f(r"hepatic\s+blend wins \d+/10 folds, mean ([+-][\d.]+)"),
                             f(r"hepatic\s+blend wins \d+/10 folds, mean [+-][\d.]+ ± ([\d.]+)")],
                        death=[f(r"death\s+blend wins (\d+)/10 folds"), f(r"death\s+blend wins \d+/10 folds, mean ([+-][\d.]+)"),
                               f(r"death\s+blend wins \d+/10 folds, mean [+-][\d.]+ ± ([\d.]+)")],
                        synthetic_wins=f(r"blend beats base in (\d+) of 10 folds"))
D["joint"] = [(n, float(a), float(g), float(e)) for n, a, g, e in
              re.findall(r"(GPT-OSS split|Nemotron joint|MedGemma split|MedGemma joint)\s+alone ([\d.]+) \| gain ([+-][\d.]+) ± ([\d.]+)", TEXT)]
D["xgb_vs_blend"] = dict(
    as_run=[f(r"CV as run\s+XGB ([\d.]+)"), f(r"CV as run\s+XGB [\d.]+ \| Cat ([\d.]+)"), f(r"CV as run\s+XGB [\d.]+ \| Cat [\d.]+ \| blend ([\d.]+)")],
    clean=[f(r"window-free CV\s+XGB ([\d.]+)"), f(r"window-free CV\s+XGB [\d.]+ \| Cat ([\d.]+)"), f(r"window-free CV\s+XGB [\d.]+ \| Cat [\d.]+ \| blend ([\d.]+)")],
    fold=[f(r"fold mean\s+([+-][\d.]+)"), f(r"fold mean\s+[+-][\d.]+ ± ([\d.]+)"), f(r"fold mean\s+[+-][\d.]+ ± [\d.]+ \(([\d.]+) SE")],
    boot=[f(r"patient bootstrap\s+([+-][\d.]+)"), f(r"patient bootstrap\s+[+-][\d.]+ ± ([\d.]+)"), f(r"patient bootstrap\s+[+-][\d.]+ ± [\d.]+ \(([\d.]+) SE")],
    hep_folds=f(r"XGB beats blend in (\d+)/10 hepatic folds"), death_folds=f(r"XGB beats blend in (\d+)/10 death folds"))
D["cohort"] = {r["endpoint"]: (int(r["n_patients"]), int(r["events"])) for r in rows("cohort_stats.csv")}

# ---- small synthetic computations, same code as the deck's cells
C = S.cohort(2000, seed=0)
visits = C.filter(regex=r"^Age_v\d+$").notna().sum(axis=1)
edges = list(range(1, 20))
D["visits"] = dict(free=np.histogram(visits[~C.hep_event], bins=edges)[0].tolist(),
                   event=np.histogram(visits[C.hep_event], bins=edges)[0].tolist())
truth, ev, t = C.true_hep.to_numpy(), C.hep_event, C.hep_time
W = np.linspace(0, 1, 21)
D["curves"] = {"w": [round(float(w), 2) for w in W], "series": {}}
for shared in (0.0, 0.3, 0.6):
    tab, llm = S.rankers(truth, seed=7, shared=shared)
    D["curves"]["series"][str(shared)] = [cidx(ev, t, (1 - w) * _rank01(tab) + w * _rank01(llm)) for w in W]
tab0, _ = S.rankers(truth, seed=7, shared=0.0)
D["curves"]["tabular"] = cidx(ev, t, tab0)
pairs = []
for shared in np.linspace(0, 0.95, 12):
    tab, llm = S.rankers(truth, seed=11, shared=shared)
    best = max(cidx(ev, t, (1 - w) * _rank01(tab) + w * _rank01(llm)) for w in W)
    pairs.append((float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(tab, llm)[0]), best - cidx(ev, t, tab)))
D["rho_gain"] = pairs

# ---- images
three = PC.pick(C)
PC.draw(three, fs=1.25, colors=(VERM, BLUE), short=True, out=OUT / "patients.png", dpi=200, font="Arial", note=False)
im = Image.open(OUT / "patients.png").convert("RGB")           # trim the empty strip left by dropping the note
edge = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()[3] + 24
im = im.crop((0, 0, im.width, min(im.height, edge)))
im.save(OUT / "patients.png")
D["patients_size"] = list(im.size)
for name, col in (("verm", VERM), ("blue", BLUE)):
    fig = plt.figure(figsize=(1, 1)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Circle((0.5, 0.75), 0.2, color=col)); ax.add_patch(Wedge((0.5, 0.0), 0.46, 0, 180, color=col))
    fig.savefig(OUT / f"pictogram_{name}.png", dpi=300, transparent=True); plt.close(fig)

# the public leaderboard as people: its patients, and the liver events expected at the training rate
n_pub = next(n for w, n, _, _ in D["resolution"] if w == "public")
k_hep = round(n_pub * D["cohort"]["hep"][1] / D["cohort"]["hep"][0])
ncol = 16
nrow = -(-n_pub // ncol)
fig = plt.figure(figsize=(ncol * 0.4, nrow * 0.46)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_aspect("equal")
ax.set_xlim(0, ncol); ax.set_ylim(0, nrow * 1.15)
hit = set(np.random.default_rng(3).choice(n_pub, k_hep, replace=False).tolist())    # where the events sit is arbitrary
for i in range(n_pub):
    x, y = i % ncol + 0.5, (nrow - 1 - i // ncol) * 1.15 + 0.05
    kw = dict(color=VERM) if i in hit else dict(facecolor="none", edgecolor="#8B97A3", lw=1.4)   # fill repeats the colour cue
    ax.add_patch(Circle((x, y + 0.75), 0.2, **kw)); ax.add_patch(Wedge((x, y), 0.42, 0, 180, **kw))
fig.savefig(OUT / "crowd.png", dpi=200, transparent=True); plt.close(fig)
D["crowd"] = [n_pub, k_hep]
D["crowd_size"] = list(Image.open(OUT / "crowd.png").size)

A = three["A"]
note = serialize_patient(A, _fib4(A)).splitlines()
pick = lambda prefix: next(l.strip() for l in note if l.strip().startswith(prefix))
lines = [note[0], note[1], pick("FibroScan"), pick("Platelets"), pick("ALT")]
esc = [html.escape(l) for l in lines]
esc[2] = esc[2].replace("F4/cirrhosis=12.5", "<b class=th>F4/cirrhosis=12.5</b>")
esc[3] = esc[3].replace("thrombocytopenia &lt;150", "<b class=th>thrombocytopenia &lt;150</b>")
esc[4] = esc[4].replace("ULN 40", "<b class=th>ULN 40</b>")
for i in (2, 3, 4):                           # keep a value with its unit, and the trend word with its value
    esc[i] = re.sub(r" (kPa|G/L|U/L)\b", r"&nbsp;\1", esc[i])
    esc[i] = re.sub(r",\s(stable|falling|rising)\b", r",&nbsp;\1", esc[i])
    esc[i] = re.sub(r"\b(stable|falling|rising)\b", r"<b class=dir>\1</b>", esc[i])
esc[1] = esc[1].replace("dyslipidaemia", "<b>dyslipidaemia</b>")
card = ("<html><body style='margin:0;background:#fff'><div style=\"width:800px;box-sizing:border-box;margin:20px;padding:22px 26px;"
        "background:#F3F6F8;border:2px solid #C9D2DA;border-radius:10px;font:26px/1.32 Arial,Helvetica,sans-serif;color:#1F2933;text-wrap:pretty\">"
        + "".join(f"<p style='margin:0 0 14px'>{l}</p>" for l in esc[:-1]) + f"<p style='margin:0'>{esc[-1]}</p></div>"
        "<style>b.th{color:#0072B2}b.dir{color:#D55E00}</style></body></html>")
(OUT / "note.html").write_text(card)
subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=2", "--window-size=860,900",
                f"--screenshot={OUT / 'note_raw.png'}", f"file://{OUT / 'note.html'}"], check=True, capture_output=True)
im = Image.open(OUT / "note_raw.png").convert("RGB")
im.crop(ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()).save(OUT / "note.png")
D["note"] = lines
D["note_size"] = list(Image.open(OUT / "note.png").size)

# Patient A's path through the first tree: the notebook's own figure and lines, so the slide shows the fitted model
cell = next(c for c in nb["cells"] if "TC.draw(" in "".join(c["source"]))
png = next(o["data"]["image/png"] for o in cell["outputs"] if "image/png" in o.get("data", {}))
(OUT / "tree.png").write_bytes(__import__("base64").b64decode(png))
im = Image.open(OUT / "tree.png")
D["tree_in"] = [round(v / im.info["dpi"][0], 3) for v in im.size]     # placed at this size, its 18 pt labels stay 18 pt
D["tree_path"] = grab(r"Patient A: (.*? risk)\n").group(1)
m = grab(r"XGBoost (\d+) trees \+ CatBoost (\d+) trees, rank blended: Patient A is riskier than (\d+)% of the (\d+) test patients")
D["forest"] = dict(xgb=int(m.group(1)), cat=int(m.group(2)), above=int(m.group(3)), n=int(m.group(4)))


# small icons for the boxes that stand for a model, so each model looks the same wherever it appears
def icon(name, draw):
    fig = plt.figure(figsize=(1, 1)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    draw(ax)
    fig.savefig(OUT / f"{name}.png", dpi=300, transparent=True); plt.close(fig)


def tree(col):                                               # the tree slide's dots and lines, in miniature
    def draw(ax):
        nodes = {0: (0.5, 0.84), 1: (0.27, 0.52), 2: (0.73, 0.52), 3: (0.14, 0.18), 4: (0.4, 0.18), 5: (0.6, 0.18), 6: (0.86, 0.18)}
        for p, c in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
            ax.plot(*zip(nodes[p], nodes[c]), color=col, lw=5, solid_capstyle="round")
        for x, y in nodes.values():
            ax.add_patch(Circle((x, y), 0.085, color=col))
    return draw


def note(ax):                                                # the clinical note the LLM reads, in miniature
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.1, 0.06), 0.8, 0.88, boxstyle="round,pad=0,rounding_size=0.08", fc="#F3F6F8", ec="#C9D2DA", lw=3))
    for y, segs in ((0.76, [(0.2, 0.8, "#8B97A3")]), (0.58, [(0.2, 0.45, "#8B97A3"), (0.49, 0.8, BLUE)]),
                    (0.4, [(0.2, 0.62, "#8B97A3"), (0.66, 0.8, VERM)]), (0.22, [(0.2, 0.7, "#8B97A3")])):
        for a, b, c in segs:
            ax.plot([a, b], [y, y], color=c, lw=7, solid_capstyle="round")


def tiles(ax):                                               # EfficientNet's input: the record as a 4 x 4 grid of image tiles
    shade = np.random.default_rng(5).uniform(0.25, 1, (4, 4))
    for i in range(4):
        for j in range(4):
            ax.add_patch(plt.Rectangle((0.06 + 0.225 * i, 0.06 + 0.225 * j), 0.2, 0.2, color=BLUE, alpha=shade[i, j], lw=0))


icon("tree_blue", tree(BLUE)); icon("tree_white", tree("#FFFFFF")); icon("note_icon", note); icon("tiles_icon", tiles)

(OUT / "data.json").write_text(json.dumps(D, indent=1))
print("wrote", OUT / "data.json", "and", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
