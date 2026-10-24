"""One boosted tree drawn as a flowchart, with one patient's path through it. Patient A, from the
data slide, walks the first tree of the honest model (first-visit features); the next slide shows
the same patient as the LLM reads her."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib.patches import Circle, Wedge

INK, GREY, VERM, BLUE = "#1F2933", "#8B97A3", "#D55E00", "#0072B2"
W_IN, H_IN = 7.6, 4.6                        # the size it takes on the slide, so 18 pt here is 18 pt there
NAMES = {"fibrotest_BM_2": ("FibroTest", "", 2), "fibs_stiffness_med_BM_1": ("FibroScan", " kPa", 1),
         "plt": ("Platelets", " G/L", 0), "ast": ("AST", " U/L", 0), "alt": ("ALT", " U/L", 0),
         "ggt": ("GGT", " U/L", 0), "bilirubin": ("Bilirubin", " µmol/L", 0), "gluc_fast": ("Glucose", " mmol/L", 1),
         "triglyc": ("Triglycerides", " mmol/L", 1), "BMI": ("BMI", "", 1), "Age": ("Age", "", 0), "fib4": ("FIB-4", "", 2)}
YESNO = {"T2DM": "Type 2 diabetes?", "Hypertension": "Hypertension?", "Dyslipidaemia": "Dyslipidaemia?",
         "gender": "Male?", "bariatric_surgery": "Bariatric surgery?"}


def question(feature, split):
    """XGBoost sends x < split left, so every question asks about the right-hand branch: "x ≥ split?"."""
    if feature in YESNO:
        return YESNO[feature]
    prefix, part = feature.rsplit("_", 1) if feature.endswith(("_first", "_last", "_slope")) else (feature, "first")
    name, unit, d = NAMES.get(prefix, (prefix, "", 2))
    return f"{name}{dict(first='', last=' (last visit)', slope=' slope')[part]} ≥ {split:.{d}f}{unit}?"


def path(booster, x, tree=0):
    """x's route through one tree, routed by XGBoost itself (pred_leaf), so ties and missing values go
    where the model sends them. Returns the tree's table, [(node, question, yes), ...] and the leaf."""
    t = booster.trees_to_dataframe().query("Tree == @tree").set_index("ID")
    dm = xgb.DMatrix(pd.DataFrame([x])[booster.feature_names], feature_names=booster.feature_names)
    leaf = f"{tree}-{int(booster.predict(dm, pred_leaf=True)[0][tree])}"
    parent = {c: (n, c == r.No) for n, r in t[t.Feature != "Leaf"].iterrows() for c in (r.Yes, r.No)}
    steps, node = [], leaf
    while node in parent:
        node, yes = parent[node]
        steps.append((node, question(t.Feature[node], t.Split[node]), yes))
    return t, steps[::-1], leaf


def _layout(t, node, depth=0, pos=None):
    """Leaves left to right in tree order; each question sits above the middle of its two branches."""
    pos = {} if pos is None else pos
    if t.Feature[node] == "Leaf":
        pos[node] = (sum(1 for v in pos.values() if v[2]), depth, True)
    else:
        _layout(t, t.Yes[node], depth + 1, pos); _layout(t, t.No[node], depth + 1, pos)
        pos[node] = ((pos[t.Yes[node]][0] + pos[t.No[node]][0]) / 2, depth, False)
    return pos


def draw(booster, x, tree=0, who="Patient A", out=None, dpi=200, font="Arial"):
    """Draw one tree, every question and end as a grey dot except those on x's path. x's end is
    coloured by its push: vermilion toward higher risk (a shorter predicted time than the tree's
    average end), blue toward lower. Only order matters to a C-index, so the push is relative.
    Returns x's path as one line of text; out saves a PNG instead of showing."""
    t, steps, leaf = path(booster, x, tree)
    pos = _layout(t, f"{tree}-0")
    leaves = t[t.Feature == "Leaf"]
    mid = np.average(leaves.Gain, weights=leaves.Cover)
    n_leaves, depth = 1 + max(p[0] for p in pos.values()), max(p[1] for p in pos.values())
    X = lambda n: 0.35 + pos[n][0] * (W_IN - 0.7) / (n_leaves - 1)
    Y = lambda n: H_IN - 0.45 - pos[n][1] * (H_IN - 1.0) / depth
    on = {n for n, _, _ in steps} | {leaf}

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_aspect("equal")
    ax.set_xlim(0, W_IN); ax.set_ylim(0, H_IN)
    for n, r in t[t.Feature != "Leaf"].iterrows():
        for c in (r.Yes, r.No):
            hot = n in on and c in on
            ax.plot([X(n), X(c)], [Y(n), Y(c)], color=VERM if hot else GREY, lw=3.5 if hot else 1.2, zorder=1,
                    solid_capstyle="round")
    push = "higher" if t.Gain[leaf] < mid else "lower"
    for n, (_, _, is_leaf) in pos.items():                    # grey: the questions and ends x never meets
        if n in on and not is_leaf:
            continue
        ax.add_patch(Circle((X(n), Y(n)), 0.16 if n == leaf else 0.09 if is_leaf else 0.07, zorder=2,
                            color=(VERM if push == "higher" else BLUE) if n == leaf else GREY))
    fig.canvas.draw()
    inches = lambda artist: artist.get_window_extent().transformed(ax.transData.inverted())

    def label(x0, y0, s, **kw):                               # text kept inside the figure, never cut at an edge
        txt = ax.text(x0, y0, s, ha="center", va="center", fontsize=18, fontweight="bold", family=font, zorder=3, **kw)
        w = inches(txt).width / 2 + 0.1
        txt.set_x(min(max(x0, w), W_IN - w))
        return txt

    boxes = [label(X(n), Y(n), q, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INK, lw=1.5))
             for n, q, _ in steps]                            # the questions x meets
    for (n, _, yes), c in zip(steps, [s[0] for s in steps[1:]] + [leaf]):   # each answer sits off its edge, on the upper side
        dx, dy = X(c) - X(n), Y(c) - Y(n)
        side = 0.2 / np.hypot(dx, dy) * (1 if dx > 0 else -1)
        ax.text((X(n) + X(c)) / 2 - dy * side, (Y(n) + Y(c)) / 2 + dx * side, "yes" if yes else "no",
                ha="left" if dx > 0 else "right", va="center", fontsize=18, family=font, color=INK, zorder=3)
    label(X(leaf), Y(leaf) - 0.36, f"this tree: {push} risk", color=VERM if push == "higher" else BLUE)
    root = inches(boxes[0].get_bbox_patch()) if boxes else None   # the patient enters beside the first question
    left = root is not None and root.x0 > W_IN - root.x1
    px = (root.x0 - 0.35 if left else root.x1 + 0.35) if root is not None else 0.45
    hy = Y(f"{tree}-0")
    ax.add_patch(Circle((px, hy + 0.12), 0.11, color=VERM, zorder=3))
    ax.add_patch(Wedge((px, hy - 0.22), 0.24, 0, 180, color=VERM, zorder=3))
    ax.text(px + (-0.32 if left else 0.32), hy, who, ha="right" if left else "left", va="center", fontsize=18,
            fontweight="bold", family=font, color=INK, zorder=3)
    if out:
        fig.savefig(out, dpi=dpi, transparent=True); plt.close(fig)
    else:
        plt.show()
    return f"{who}: " + " → ".join(f"{q} {'yes' if yes else 'no'}" for _, q, yes in steps) + f" → this tree: {push} risk"


if __name__ == "__main__":                                   # the path must agree with the splits it reports
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(300, 3)), columns=["plt_first", "BMI_first", "T2DM"])
    df["T2DM"] = (df.T2DM > 0).astype(float)
    b = xgb.train({"max_depth": 3, "verbosity": 0}, xgb.DMatrix(df, label=df.plt_first - df.T2DM + rng.normal(size=300)), 3)
    for i in range(20):
        t, steps, leaf = path(b, df.iloc[i], tree=1)
        assert steps[0][0] == "1-0" and t.Feature[leaf] == "Leaf", steps
        assert all((df.iloc[i][t.Feature[n]] >= t.Split[n]) == yes for n, _, yes in steps), steps
    assert question("fibrotest_BM_2_first", 0.6404) == "FibroTest ≥ 0.64?" and question("T2DM", 1.0) == "Type 2 diabetes?"
    print("tree_card ok")
