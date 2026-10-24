// Builds the ANNITIA PowerPoint deck for the GAIN webinar. Set PPTX_OUT to keep a named revision.
//
// Headlines and speaker notes come from ../speaker_track.md, numbers from build/data.json (written by
// figures.py), and the checklist and generalising table from ../annitia_talk.py. Nothing is typed twice.
//
//   npm install                     once, in talk/pptx
//   python3 figures.py              data.json and the images
//   PPTX_OUT=../annitia_talk_recap.pptx NODE_PATH=node_modules node build_deck.js
//                                                   (NODE_PATH lets the skill's theme step find jszip)
//
// The map slide shows thumbnails of three slides. After a first build, render the deck to a PDF and run
//   pdftoppm -jpeg -r 40 -f N -l N -singlefile deck.pdf build/thumbs/N     for N in 11 18 24
// then build again. Without thumbnails the map slide shows grey placeholders.
// Last, finish() repairs the charts for Keynote and gives each slide its click-to-reveal steps.
// SHOW_HIDDEN=1 keeps the backup slides visible (for previews). PPTX_SKILL points at the pptx skill.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const SKILL = process.env.PPTX_SKILL ||
  "/Users/ronnypolle/.claude/skills/synced/7944d5cf-7158-4a52-9f32-0d8da9f59824_92f3e563-993e-4f3d-a5b7-23d6cad70037/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
const BUILD = path.join(__dirname, "build");
const OUT = process.env.PPTX_OUT
  ? path.resolve(__dirname, process.env.PPTX_OUT)
  : path.join(__dirname, "..", "annitia_talk.pptx");
const D = JSON.parse(fs.readFileSync(path.join(BUILD, "data.json"), "utf8"));

// ---- the speaker track: headline, hook, script, timing and checks per slide
function parseTrack(file) {
  const text = fs.readFileSync(file, "utf8");
  const marks = [...text.matchAll(/^### Slide (\d+): (.*?) \((\d+) s\)$/gm)];
  const slides = {};
  marks.forEach((m, i) => {
    const end = i + 1 < marks.length ? marks[i + 1].index : text.length;
    const body = text.slice(m.index, end).split(/\n(?:---|## )/)[0];
    const get = (label) => (body.match(new RegExp(`^\\*\\*${label}:\\*\\* (.*)$`, "m")) || [])[1] || "";
    slides[+m[1]] = {
      header: m[2].replace(/\.$/, ""), secs: +m[3],
      hook: get("Hook \\(before the slide\\)"),
      script: get("Script").replace(/`(\[PAUSE[^`]*\])`/g, "$1"),
      extra: get("Extra time"), timeNote: get("Time note"), note: get("Note"), confirm: get("\\[CONFIRM\\]"),
    };
  });
  const backups = [...text.split("## Backup slides")[1].split("## Likely questions")[0].matchAll(/^- \*\*.*?\*\*\s*(.*)$/gm)]
    .map((m) => m[1]);
  return { slides, backups };
}
const TRACK = parseTrack(path.join(__dirname, "..", "speaker_track.md"));

function notesFor(n) {
  const s = TRACK.slides[n];
  const out = [`HOOK, before the slide: ${s.hook}`, "", s.script, "", `TIME: about ${s.secs} s${s.extra ? ` (${s.extra.replace(/\.$/, "")})` : ""}`];
  if (s.timeNote) out.push(`IF YOU RUN LONG: ${s.timeNote}`);
  if (s.note) out.push(`NOTE: ${s.note}`);
  if (s.confirm) out.push("", `CHECK BEFORE PRESENTING: ${s.confirm}`);
  return out.join("\n").replace(/`/g, "");
}

// ---- the deck's own text: the checklist and the generalising table
const deckSrc = fs.readFileSync(path.join(__dirname, "..", "annitia_talk.py"), "utf8");
const checklist = [...deckSrc.split("## The checklist")[1].matchAll(/^# - \[ \] (.*)$/gm)].map((m) => m[1].replace(/\*\*/g, ""));
const generalising = [...deckSrc.split("## Generalising the idea")[1].split("# %%")[0].matchAll(/^# \|(.*)\|$/gm)]
  .map((m) => m[1].split("|").map((c) => c.trim())).filter((r) => !/^-+$/.test(r[0]));

// ---- theme: Okabe-Ito colours, which stay distinct for colour-blind viewers
const THEME = {
  name: "ANNITIA talk", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "1F2933", lt1: "FFFFFF", dk2: "6B7785", lt2: "DDE3E9", accent1: "0072B2", accent2: "D55E00",
    accent3: "56B4E9", accent4: "E69F00", accent5: "009E73", accent6: "CC79A7", hlink: "0072B2", folHlink: "6B7785" },
};
const HEX = { ink: "1F2933", gray: "6B7785", blue: "0072B2", verm: "D55E00", sky: "56B4E9", grid: "DDE3E9", white: "FFFFFF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                      // 13.33 x 7.5 in
pres.title = "LLM Risk Rankers and Agentic Scaffolding for Liver Disease Survival Modeling";
pres.author = "Dr. Ronny Polle";
pres.subject = "GAIN Monthly Webinar";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const SH = pres.shapes;

// ---- layouts: one title frame, and one content frame per part so the footer names the part
const PART_NAMES = ["An LLM can help, if its mistakes differ", "Three tests show our gain was not real", "You can run the same checks yourself"];
const PARTS = ["", ...PART_NAMES.map((n, i) => `Part ${i + 1} · ${n}`), "Backup"];
PARTS.forEach((label, i) => pres.defineSlideMaster({
  title: `CONTENT_${i}`,
  background: { color: HEX.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 1.15, fontSize: 28, bold: true,
      color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
    ...(label ? [{ text: { text: label, options: { x: 6.2, y: 7.02, w: 5.9, h: 0.3, fontSize: 12, color: C.text2, align: "right", margin: 0 } } }] : []),
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.55, h: 0.3, fontSize: 12, color: C.text2, align: "right" },
}));
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: HEX.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.45, w: 7.5, h: 2.7, fontSize: 40, bold: true,
      color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.7, y: 4.3, w: 7.5, h: 0.8, fontSize: 22,
      color: C.text2, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "presenter", type: "body", x: 0.7, y: 5.45, w: 7.5, h: 1.3, fontSize: 18,
      color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});

// ---- helpers
// Keynote ignores valAxisMajorUnit and picks its own step (0.05 on a 0.5-1 axis, 0.125 on a small one), so those axes say
// "0.0##": every tick exact in Keynote (0.55, 0.625), and 0.5, 0.6 in PowerPoint. "0.0" printed 0.6 at 0.55 there.
const chart = (extra) => Object.assign({
  x: 0.6, y: 1.6, w: 12.1, h: 4.75,
  catAxisLabelFontSize: 14, catAxisLabelColor: HEX.ink, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontSize: 12, valAxisLabelColor: HEX.gray, valAxisLabelFontFace: "+mn-lt",
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 14, dataLabelColor: HEX.ink,
  legendFontFace: "+mn-lt", legendFontSize: 14, legendColor: HEX.ink,
  titleFontFace: "+mn-lt", titleFontSize: 16, titleColor: HEX.ink,
  valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
  valAxisTitleFontSize: 14, catAxisTitleFontSize: 14, valAxisTitleColor: HEX.gray, catAxisTitleColor: HEX.gray,
  valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  showLegend: false, showTitle: false,
}, extra);
const balance = (t) => {              // split a long headline near its middle, so no word sits alone on line two
  if (t.length <= 66) return t;
  const words = t.split(" ");
  let best = null;
  for (let i = 1; i < words.length; i++) {
    const a = words.slice(0, i).join(" "), b = words.slice(i).join(" ");
    if (a.length <= 66 && b.length <= 66 && (!best || Math.abs(a.length - b.length) < best[2])) best = [a, b, Math.abs(a.length - b.length)];
  }
  return best ? `${best[0]}\n${best[1]}` : t;
};
const FIG = (n, d = 4) => (Number.isFinite(+n) ? (+n).toFixed(d) : String(n));
const SIGNED = (n, d = 4) => (n >= 0 ? "+" : "-") + Math.abs(n).toFixed(d);

const STEPS = {};                      // slide number -> { bounds: object counts where a new click-to-reveal point starts, total } (see finish())
let CUR = 0;
function step(s) { (STEPS[CUR] = STEPS[CUR] || { bounds: [] }).bounds.push(s._slideObjects.length); }
function caption(s, text) {
  step(s);
  s.addText(text, { x: 0.6, y: 6.45, w: 12.1, h: 0.45, fontSize: 18, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
}
function dataLabel(s, text) {
  s.addText(text, { x: 0.6, y: 7.02, w: 5.6, h: 0.3, fontSize: 12, color: C.text2, margin: 0, isTextBox: true });
}
function box(s, runs, o) {
  s.addText(runs, Object.assign({ shape: SH.ROUNDED_RECTANGLE, rectRadius: 0.1, align: "center", valign: "middle", margin: 0.08, isTextBox: true,
    color: C.text1, fill: { color: C.background1 }, line: { color: HEX.gray, width: 1.5 } }, o));
}
const ICON_ALT = { tree_blue: "Tree icon", tree_white: "Tree icon", note_icon: "Clinical note icon", tiles_icon: "Image tiles icon" };
function iconBox(s, icon, runs, o, d = 0.8) {              // box(), with the model's icon beside its words
  box(s, "", o);
  s.addImage({ path: path.join(BUILD, `${icon}.png`), x: o.x + 0.2, y: o.y + (o.h - d) / 2, w: d, h: d, altText: ICON_ALT[icon] });
  s.addText(runs, { x: o.x + d + 0.35, y: o.y, w: o.w - d - 0.5, h: o.h, fontSize: o.fontSize || 18, color: o.color || C.text1,
    valign: "middle", margin: 0, isTextBox: true });
}
function arrow(s, x1, y1, x2, y2, color = HEX.gray) {
  s.addShape(SH.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 2.5, endArrowType: "triangle" } });
}
function badge(s, n, x, y, d = 0.54) {
  s.addText(String(n), { x, y, w: d, h: d, shape: SH.OVAL, fill: { color: C.accent1 }, line: { color: C.accent1 }, color: C.background1,
    bold: true, fontSize: 18, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function lane(s, x, y, len, ticks, end, color) {          // a follow-up lane: one tick per visit, an end marker
  s.addShape(SH.ROUNDED_RECTANGLE, { x, y, w: len, h: 0.16, rectRadius: 0.08, fill: { color: HEX.grid }, line: { color: HEX.grid, width: 0 } });
  ticks.forEach((t) => s.addShape(SH.LINE, { x: x + t, y: y - 0.12, w: 0, h: 0.4, line: { color: HEX.ink, width: 2 } }));
  const cx = x + len;
  if (end === "event") {
    s.addShape(SH.LINE, { x: cx - 0.17, y: y - 0.09, w: 0.34, h: 0.34, line: { color, width: 4 } });
    s.addShape(SH.LINE, { x: cx - 0.17, y: y - 0.09, w: 0.34, h: 0.34, flipV: true, line: { color, width: 4 } });
  } else {
    s.addShape(SH.OVAL, { x: cx - 0.17, y: y - 0.09, w: 0.34, h: 0.34, fill: { color: HEX.white }, line: { color, width: 3.5 } });
  }
}
function table(s, rows, o) {
  const border = { type: "solid", pt: 0.75, color: HEX.grid };
  s.addTable(rows, Object.assign({ fontSize: 18, color: HEX.ink, valign: "middle", border, margin: [0.05, 0.12, 0.05, 0.12] }, o));
}
const th = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true, fill: { color: HEX.grid }, color: HEX.ink }, o) });
function recap(s, items, accent = HEX.blue) {
  const y0 = 1.62, stride = 1.2, h = 0.92;
  items.forEach((text, i) => {
    const y = y0 + i * stride;
    s.addText(String(i + 1), { x: 0.72, y: y + 0.13, w: 0.5, h: 0.5, shape: SH.OVAL,
      fill: { color: HEX.grid }, line: { color: HEX.grid }, color: HEX.gray,
      bold: true, fontSize: 16, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(text, { x: 1.55, y, w: 10.75, h, fontSize: 23, color: "AEB8C2",
      valign: "middle", margin: 0, breakLine: false, isTextBox: true });
  });
  items.forEach((text, i) => {
    const y = y0 + i * stride;
    step(s);
    s.addShape(SH.RECTANGLE, { x: 0.58, y: y - 0.05, w: 12.1, h: h + 0.1,
      fill: { color: HEX.white }, line: { color: HEX.white, transparency: 100 } });
    s.addText(String(i + 1), { x: 0.72, y: y + 0.13, w: 0.5, h: 0.5, shape: SH.OVAL,
      fill: { color: accent }, line: { color: accent }, color: HEX.white,
      bold: true, fontSize: 16, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(text, { x: 1.55, y, w: 10.75, h, fontSize: 23, color: C.text1,
      valign: "middle", margin: 0, breakLine: false, isTextBox: true });
  });
}
const SHORT = {
  "Tabular base: 10-seed XGB AFT + CatBoost (alone)": "Tabular base (alone)",
  "Claude Haiku HeLM (alone)": "Claude Haiku LLM (alone)",
  "Tabular base + Haiku HeLM at weight 0.05": "Base + Haiku LLM, weight 0.05",
  "EfficientNet-B0 Cox fine-tune (alone)": "EfficientNet-B0 (alone)",
  "Tabular base + EfficientNet-B0 at weight 0.02": "Base + EfficientNet-B0, weight 0.02",
  "GPT-OSS 120B split-endpoint HeLM (alone)": "GPT-OSS 120B LLM (alone)",
  "EfficientNet base + GPT-OSS HeLM at weight 0.07": "EfficientNet base + GPT-OSS LLM, weight 0.07",
  "Final two-stage: base + (EfficientNet + GPT-OSS split)": "Final two-stage blend",
};

// ---- slides: n -> { layout, section, data footer, builder }
const DEFS = {};
const def = (n, layout, section, data, fn) => { DEFS[n] = { layout, section, data, fn }; };

// 1 Title
def(1, "TITLE", "Opening", "", (s) => {
  s.addText("LLM Risk Rankers and Agentic Scaffolding", { placeholder: "title" });
  s.addText("A case study for Liver Disease Survival Modeling", { placeholder: "subtitle" });
  s.addText([{ text: "Dr. Ronny Polle", options: { bold: true, breakLine: true } },
    { text: "AI Researcher", options: { breakLine: true } },
    { text: "GAIN Monthly Webinar" }], { placeholder: "presenter" });
  const rows = [["verm", 2.0, 1.6, [0.3, 0.9, 1.5], "event"], ["blue", 3.55, 2.8, [0.2, 0.65, 1.1, 1.5, 1.85, 2.3, 2.6], "free"], ["blue", 5.1, 2.1, [0.4, 0.95, 1.55], "free"]];
  rows.forEach(([c, y, len, ticks, end]) => {
    s.addImage({ path: path.join(BUILD, `pictogram_${c}.png`), x: 8.7, y: y - 0.45, w: 0.95, h: 0.95, altText: "Pictogram of a patient" });
    lane(s, 9.95, y, len, ticks, end, c === "verm" ? HEX.verm : HEX.blue);
  });
});

// 2 Map
def(2, "CONTENT_0", "Opening", "", (s) => {
  // The referenced content did not change; reuse the existing renders from before the two recap insertions.
  const thumbs = [[11, 11], [18, 17], [24, 22]];
  PART_NAMES.forEach((text, i) => {
    if (i) step(s);
    const [slideNo, thumbFile] = thumbs[i];
    const y = 1.75 + i * 1.6, img = path.join(BUILD, "thumbs", `${thumbFile}.jpg`);
    if (fs.existsSync(img)) {
      s.addImage({ path: img, x: 0.6, y, w: 2.4, h: 1.35, altText: `Preview of slide ${slideNo}, the key slide for part ${i + 1}` });
      s.addShape(SH.RECTANGLE, { x: 0.6, y, w: 2.4, h: 1.35, fill: { color: HEX.white, transparency: 100 }, line: { color: HEX.grid, width: 1 } });
    } else s.addShape(SH.RECTANGLE, { x: 0.6, y, w: 2.4, h: 1.35, fill: { color: HEX.grid }, line: { color: HEX.grid } });
    badge(s, i + 1, 3.35, y + 0.4);
    s.addText(text, { x: 4.15, y, w: 8.55, h: 1.35, fontSize: 28, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  });
});

// 3 Argument
def(3, "CONTENT_0", "Opening", "Schematic", (s) => {
  const t = (a, b) => [{ text: a, options: { bold: true, fontSize: 22, breakLine: !!b } }, ...(b ? [{ text: b, options: { fontSize: 18 } }] : [])];
  box(s, "", { x: 0.6, y: 3.0, w: 2.6, h: 1.5 });                       // a patient and her follow-up, as on the title slide
  s.addImage({ path: path.join(BUILD, "pictogram_verm.png"), x: 0.8, y: 3.12, w: 0.6, h: 0.6, altText: "Pictogram of a patient" });
  lane(s, 1.55, 3.34, 1.25, [0.2, 0.55, 0.9], "event", HEX.verm);
  s.addText(t("Patient record"), { x: 0.6, y: 3.8, w: 2.6, h: 0.55, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  step(s);
  iconBox(s, "tree_white", t("Tree models", "strong ranker"), { x: 4.3, y: 1.9, w: 3.2, h: 1.5, fill: { color: C.accent1 }, line: { color: C.accent1 }, color: C.background1 });
  arrow(s, 3.2, 3.75, 4.3, 2.65);
  step(s);
  iconBox(s, "note_icon", t("LLM", "weak, but different mistakes"), { x: 4.3, y: 4.1, w: 3.2, h: 1.5, line: { color: HEX.blue, width: 3 } });
  arrow(s, 3.2, 3.75, 4.3, 4.85);
  step(s);
  box(s, t("Rank blend"), { x: 8.3, y: 3.0, w: 2.1, h: 1.5 });
  arrow(s, 7.5, 2.65, 8.3, 3.5); arrow(s, 7.5, 4.85, 8.3, 4.0);
  step(s);
  box(s, t("Seemed to improve"), { x: 10.9, y: 3.0, w: 1.8, h: 1.5, fill: { color: C.accent2 }, line: { color: C.accent2 }, color: C.background1 });
  arrow(s, 10.4, 3.75, 10.9, 3.75);
  caption(s, "The LLM ranks worse alone, but its errors differ from the tree models'.");
});

// 4 Hook: the leaderboard
def(4, "CONTENT_0", "Opening", "Data: our leaderboard milestones", (s) => {
  const rows = D.hook.map(([label, v]) => ({ name: SHORT[label] || label, v, alone: /\(alone\)/.test(label) })).reverse();
  s.addChart(pres.charts.BAR, [
    { name: "Model alone", labels: rows.map((r) => r.name), values: rows.map((r) => (r.alone ? r.v : 0)) },
    { name: "Blend", labels: rows.map((r) => r.name), values: rows.map((r) => (r.alone ? 0 : r.v)) },
  ], chart({ barDir: "bar", barGrouping: "clustered", barOverlapPct: 100, chartColors: [HEX.blue, HEX.verm], valAxisMinVal: 0.5, valAxisMaxVal: 1.0,
    valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0##", showValue: true, dataLabelFormatCode: "0.000;;;", dataLabelPosition: "outEnd",
    barGapWidthPct: 35, showLegend: true, legendPos: "t", valGridLine: { style: "none" },
    showValAxisTitle: true, valAxisTitle: "weighted C-index (0.5 = random order)", h: 4.8,
    altText: "Eight bars of weighted C-index from our leaderboard milestones. The final two-stage blend, 0.904, scores highest." }));
  caption(s, "Blended at a small rank weight, GPT-OSS gave the pipeline's best public score.");
});

// 5 Objective
def(5, "CONTENT_1", PARTS[1], "Data: synthetic cohort", (s) => {
  const o = D.objective;
  s.addChart(pres.charts.BAR, [{ name: "C-index", labels: ["Tabular score", "Same score, exponentiated", "Raw 50/50 average", "Rank 50/50 average"],
    values: [o.score, o.exp, o.raw, o.rank] }],
  chart({ barDir: "col", chartColors: [HEX.blue, HEX.blue, HEX.verm, HEX.blue], valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
    valAxisLabelFormatCode: "0.0##", showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", barGapWidthPct: 70, valGridLine: { style: "none" },
    showValAxisTitle: true, valAxisTitle: "C-index (0.5 = random order)", catAxisLabelFontSize: 16,
    altText: "Four bars: a score and its exponential have the same C-index; a raw average is lower; a rank average is close to the score." }));
  caption(s, "Averaging raw scores lets the wider scale win. Averaging ranks weighs both fairly.");
});

// 6 Three patients
def(6, "CONTENT_1", PARTS[1], "Data: synthetic patients", (s) => {
  const [pw, ph] = D.patients_size, w = 11.5;
  s.addImage({ path: path.join(BUILD, "patients.png"), x: 0.9, y: 1.7, w, h: w * ph / pw,
    altText: "Three synthetic patients. Patient A, a woman of 41, has 5 visits, FibroScan at 2, above the cirrhosis line, then a hepatic event. " +
      "Patient B, a man of 55, has 15 visits, FibroScan at 12 with low readings, and is still event-free. " +
      "Patient C, a man of 70, has 6 visits, FibroScan at 4 above the cirrhosis line, and is still event-free." });
  caption(s, "Visits: 5, 15 and 6. FibroScan at 2, 12 and 4 of them. One event.");
});

// 7 Trees: Patient A's path through the first tree of the honest model, then both forests and their rank blend
def(7, "CONTENT_1", PARTS[1], `Data: synthetic cohort; the first of ${D.forest.xgb} XGBoost trees, first-visit features`, (s) => {
  const [tw, th] = D.tree_in, f = D.forest;
  s.addImage({ path: path.join(BUILD, "tree.png"), x: 0.5, y: 1.55, w: tw, h: th,
    altText: `The first tree of the XGBoost model as a flowchart, with Patient A's path: ${D.tree_path}.` });
  step(s);
  [["XGBoost", f.xgb, 8.7], ["CatBoost", f.cat, 10.85]].forEach(([name, n, x]) => {
    s.addText([{ text: name, options: { bold: true, breakLine: true } }, { text: `${n} trees`, options: { color: C.text2, fontSize: 18 } }],
      { x, y: 1.6, w: 1.85, h: 0.75, fontSize: 20, color: C.text1, valign: "top", margin: 0, isTextBox: true });
    for (let i = 0; i < 6; i++) {
      s.addImage({ path: path.join(BUILD, "tree_blue.png"), x: x + (i % 3) * 0.62, y: 2.45 + Math.floor(i / 3) * 0.62, w: 0.5, h: 0.5,
        altText: `${name} tree icon` });
    }
  });
  step(s);
  arrow(s, 9.6, 3.7, 10.4, 4.2); arrow(s, 11.75, 3.7, 10.95, 4.2);
  box(s, "Rank blend", { x: 9.45, y: 4.2, w: 2.45, h: 0.6, fontSize: 20, bold: true });
  s.addText([{ text: `${f.above}%`, options: { fontSize: 40, bold: true, color: C.accent2, breakLine: true } },
    { text: `of ${f.n} test patients rank below Patient\u00A0A`, options: { fontSize: 18, color: C.text1 } }],
  { x: 8.7, y: 4.95, w: 4.0, h: 1.2, valign: "top", margin: 0, isTextBox: true });
  caption(s, "Each tree asks a few yes/no questions. Added up, hundreds of them rank the patients.");
});

// 8 Leak
def(8, "CONTENT_1", PARTS[1], "Data: synthetic cohort", (s) => {
  const v = D.visits, cats = v.free.map((_, i) => String(i + 1));
  s.addChart(pres.charts.BAR, [{ name: "No event", labels: cats, values: v.free }, { name: "Hepatic event", labels: cats, values: v.event }],
    chart({ x: 0.6, w: 6.3, barDir: "col", barGrouping: "stacked", chartColors: [HEX.blue, HEX.verm], barGapWidthPct: 25, showLegend: true, legendPos: "r",   // on top, Keynote lays it over the title
      catAxisLabelFontSize: 12, showCatAxisTitle: true, catAxisTitle: "visits recorded", showValAxisTitle: true, valAxisTitle: "patients",
      showTitle: true, title: "Fewer visits where follow-up ends at the event",
      altText: "Stacked columns of visits per patient: patients with a hepatic event have fewer visits." }));
  step(s);
  const o = D.oracle;
  s.addChart(pres.charts.BAR, [{ name: "Hepatic C-index", labels: ["Baseline visit", "True risk", "Whole window"], values: [o.baseline, o.truth, o.whole] }],
    chart({ x: 7.2, w: 5.5, barDir: "col", chartColors: [HEX.blue, HEX.gray, HEX.verm], valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
      valAxisLabelFormatCode: "0.0##", showValue: true, dataLabelFormatCode: "0.000", dataLabelPosition: "outEnd", barGapWidthPct: 60, valGridLine: { style: "none" },
      showTitle: true, title: "Above the true risk: reading the answer", showCatAxisTitle: true, catAxisTitle: "what the score uses",
      showValAxisTitle: true, valAxisTitle: "hepatic C-index (0.5 = random order)",
      altText: "Three bars of hepatic C-index: baseline visit only 0.798, true risk 0.825, whole follow-up window 0.920." }));
  caption(s, "A score that sees the whole follow-up window beats the true risk, so it reads the outcome.");
});

// 9 Serialization
def(9, "CONTENT_1", PARTS[1], "Data: synthetic patient", (s) => {
  const [nw, nh] = D.note_size, w = 8.0;
  s.addText("Patient A, as the LLM reads her record", { x: 0.6, y: 1.6, w: 8.0, h: 0.4, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addImage({ path: path.join(BUILD, "note.png"), x: 0.6, y: 2.1, w, h: w * nh / nw,
    altText: "An excerpt of the clinical note the model reads: " + D.note.join(" ") });
  step(s);
  box(s, [{ text: "Hepatic prompt", options: { bold: true, breakLine: true } }, { text: "0–100 score", options: {} }], { x: 9.5, y: 2.4, w: 3.2, h: 1.3, fontSize: 20 });
  box(s, [{ text: "Death prompt", options: { bold: true, breakLine: true } }, { text: "0–100 score", options: {} }], { x: 9.5, y: 4.3, w: 3.2, h: 1.3, fontSize: 20 });
  arrow(s, 8.7, 3.9, 9.4, 3.05); arrow(s, 8.7, 4.1, 9.4, 4.95);
  caption(s, "HeLM idea: a language model reads the record as text and returns a risk ranking.");
});

// 10 Harness
def(10, "CONTENT_1", PARTS[1], "Schematic of the harness", (s) => {
  const items = [["Leak guard", "Drops follow-up length"], ["Prompt builder", "One prompt per endpoint"], ["Model call", "One prompt in, one JSON out"],
    ["Retry and rate limits", "Survives free-tier limits"], ["Score log", "Saves every score"]];
  const y = 1.75, h = 1.3, X = (i) => 0.6 + i * 2.5;
  items.forEach(([name, note], i) => {
    if (i) step(s);
    const x = X(i), o = { x, y, w: 2.1, h, fontSize: 20, bold: true };
    if (i === 2) box(s, name, Object.assign(o, { fill: { color: C.accent1 }, line: { color: C.accent1 }, color: C.background1 }));
    else if (i === 4) {                                   // a can for stored state; the label sits on the body, not the lid
      s.addShape(SH.CAN, { x, y, w: 2.1, h, fill: { color: C.background1 }, line: { color: HEX.gray, width: 1.5 } });
      s.addText(name, { x, y, w: 2.1, h, fontSize: 20, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    } else box(s, name, o);
    if (i < 4) arrow(s, x + 2.1, y + h / 2, x + 2.5, y + h / 2);
    const runs = i === 0 ? [{ text: "Drops " }, { text: "follow-up length", options: { strike: "sngStrike", color: HEX.verm } }] : note;
    s.addText(runs, { x: x - 0.1, y: y + h + 0.15, w: 2.3, h: 0.75, fontSize: 18, color: C.text2, align: "center", valign: "top", margin: 0, isTextBox: true });
  });
  // the difference: a guard written as a prompt line lets the column through to the model
  const gy = 4.75, gh = 1.1, dash = { color: HEX.gray, width: 1.5, dashType: "dash" };
  step(s);
  s.addText("If the guard were a prompt line:", { x: 0.6, y: 4.2, w: 12.1, h: 0.4, fontSize: 18, italic: true, color: C.text2, margin: 0, isTextBox: true });
  box(s, "follow-up length", { x: X(0), y: gy + 0.25, w: 2.1, h: 0.6, fontSize: 18, bold: true, color: HEX.white, fill: { color: HEX.verm }, line: { color: HEX.verm } });
  box(s, "Prompt: “ignore follow-up length”", { x: X(1), y: gy, w: 2.1, h: gh, fontSize: 18, line: dash });
  box(s, "Model", { x: X(2), y: gy, w: 2.1, h: gh, fontSize: 20, bold: true, line: dash });
  arrow(s, X(0) + 2.1, gy + gh / 2, X(1), gy + gh / 2, HEX.verm);
  arrow(s, X(1) + 2.1, gy + gh / 2, X(2), gy + gh / 2, HEX.verm);
  s.addText("still reaches the model", { x: X(3) - 0.1, y: gy, w: 4.8, h: gh, fontSize: 20, bold: true, color: HEX.verm,
    valign: "middle", margin: 0, isTextBox: true });
  caption(s, "The model picks no tools and no next steps.");
});

// 11 Error correlation
def(11, "CONTENT_1", PARTS[1], "Data: synthetic cohort, simulated rankers", (s) => {
  const cu = D.curves, keys = Object.keys(cu.series), cols = [HEX.blue, HEX.sky, HEX.verm];
  s.addChart(pres.charts.LINE, keys.map((k) => ({ name: `Error correlation ${k}`, labels: cu.w.map((w) => w.toFixed(2)), values: cu.series[k] })),
    chart({ w: 8.2, chartColors: cols, lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, valAxisMinVal: 0.75, valAxisMaxVal: 0.81,
      valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "0.00", catAxisLabelFontSize: 14, catAxisLabelFrequency: "2",
      showCatAxisTitle: true, catAxisTitle: "weight on the LLM ranks", showValAxisTitle: true, valAxisTitle: "C-index (hepatic), zoomed axis",
      altText: "Three lines of C-index against the weight on the LLM ranks. With independent mistakes the line rises above the tabular score before falling; with shared mistakes it only falls." }));
  step(s);
  // the legend as second opinions: how much of the two rankers' mistakes overlap (schematic)
  const d = 0.6, cx = 10.85;
  [["mistakes differ", 0.48], ["some shared", 0.33], ["mostly shared", 0.15]].forEach(([label, gap], i) => {
    const y = 1.8 + i * 1.5, x = cx - (gap + d) / 2, ring = { fill: { color: cols[i], transparency: 65 }, line: { color: cols[i], width: 2.5 } };
    s.addShape(SH.OVAL, Object.assign({ x, y, w: d, h: d }, ring));
    s.addShape(SH.OVAL, Object.assign({ x: x + gap, y, w: d, h: d }, ring));
    s.addText([{ text: label, options: { bold: true, breakLine: true } }, { text: `correlation ${keys[i]}`, options: { color: HEX.gray } }],
      { x: cx - 1.85, y: y + 0.65, w: 3.7, h: 0.7, fontSize: 18, color: C.text1, align: "center", valign: "top", margin: 0, isTextBox: true });
  });
  caption(s, "Independent mistakes: a weak model adds signal. Shared mistakes: it only adds noise.");
});

// 12 Pre-flight
def(12, "CONTENT_1", PARTS[1], "Data: synthetic cohort, simulated rankers", (s) => {
  const rho = D.rho_gain.map((p) => p[0]), gain = D.rho_gain.map((p) => p[1]);
  s.addChart(pres.charts.LINE, [{ name: "Best blend gain", labels: rho.map((r) => r.toFixed(2)), values: gain }],
    chart({ x: 0.6, w: 6.5, chartColors: [HEX.blue], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 8, valAxisMinVal: 0, valAxisMaxVal: 0.003,
      valAxisMajorUnit: 0.001, valAxisLabelFormatCode: "0.000", catAxisLabelFontSize: 12,
      showCatAxisTitle: true, catAxisTitle: "Spearman ρ, LLM vs tabular ranks", showValAxisTitle: true, valAxisTitle: "best blend gain in C-index",
      showTitle: true, title: "The lower the rank correlation, the bigger the gain",
      altText: "Best blend gain falls to zero as the rank correlation between the LLM and the tabular model rises." }));
  step(s);
  s.addChart(pres.charts.BAR, [{ name: "Unique scores", labels: D.levels.map((l) => `${l[0]} levels`), values: D.levels.map((l) => l[1]) }],
    chart({ x: 7.4, w: 5.3, barDir: "col", chartColors: [HEX.blue], valAxisMinVal: 0, valAxisMaxVal: 120, valAxisMajorUnit: 20, valGridLine: { style: "none" },
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", barGapWidthPct: 60, showValAxisTitle: true, valAxisTitle: "unique score values",
      showCatAxisTitle: true, catAxisTitle: "LLM score levels", showTitle: true, title: "Unique score values among 2000 patients",
      altText: "Unique score values for an LLM scored on 5, 20 and 100 levels: 6, 21 and 101." }));
  caption(s, "Lower rank correlation means more new information.");
});

// 13 Blend
def(13, "CONTENT_1", PARTS[1], "Schematic of the submitted blend", (s) => {
  const t = (a, b) => [{ text: a, options: { bold: true, breakLine: !!b } }, ...(b ? [{ text: b, options: { fontSize: 18 } }] : [])];
  iconBox(s, "tiles_icon", t("EfficientNet-B0", "image + tabular"), { x: 0.6, y: 1.9, w: 3.2, h: 1.1, fontSize: 18 }, 0.6);
  iconBox(s, "note_icon", t("GPT-OSS", "split-endpoint scores"), { x: 0.6, y: 3.3, w: 3.2, h: 1.1, fontSize: 18 }, 0.6);
  step(s);
  box(s, t("Stage 1", "50/50 rank blend"), { x: 4.7, y: 2.3, w: 2.5, h: 1.5, fontSize: 18, line: { color: HEX.blue, width: 3 } });
  arrow(s, 3.8, 2.45, 4.7, 2.85); arrow(s, 3.8, 3.85, 4.7, 3.25);
  step(s);
  iconBox(s, "tree_blue", t("Tabular base", "10 seeds"), { x: 4.5, y: 5.1, w: 2.9, h: 1.2, fontSize: 18 }, 0.6);
  step(s);
  box(s, t("Stage 2", "rank blend"), { x: 8.4, y: 3.5, w: 2.3, h: 1.5, fontSize: 18, line: { color: HEX.blue, width: 3 } });
  arrow(s, 7.2, 3.05, 8.4, 3.95); arrow(s, 7.4, 5.7, 8.4, 4.6);
  const onArrow = (txt, cx, cy) => s.addText(txt, { x: cx - 0.35, y: cy - 0.18, w: 0.7, h: 0.36, fontSize: 18, bold: true, color: C.text1, align: "center",
    valign: "middle", margin: 0, fill: { color: C.background1 }, isTextBox: true });
  onArrow("0.15", 7.8, 3.5); onArrow("0.85", 7.8, 5.15);
  step(s);
  box(s, t("Final score"), { x: 11.3, y: 3.75, w: 1.4, h: 1.0, fontSize: 18 });
  arrow(s, 10.7, 4.25, 11.3, 4.25);
  step(s);
  s.addText("0.075", { x: 8.4, y: 1.65, w: 4.3, h: 0.9, fontSize: 48, bold: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addText("effective weight of the LLM, close to the 0.07 we replicated", { x: 8.4, y: 2.6, w: 4.3, h: 0.4, fontSize: 18, color: C.text2, margin: 0, isTextBox: true });
});

// 14 Part 1 recap
def(14, "CONTENT_1", PARTS[1], "", (s) => {
  recap(s, [
    "MASLD prediction starts with irregular records that end at different times.",
    "When follow-up stops at an event, record length can reveal the outcome.",
    "The harness removes leaked time information before any model sees it.",
    "A weaker LLM can help only when its ranking errors differ.",
  ]);
});

// 15 Resolution
def(15, "CONTENT_2", PARTS[2], "Data: simulated from measured event rates; our leaderboard gain", (s) => {
  const r = Object.fromEntries(D.resolution.map(([w, n, se, g]) => [w, { n, g }]));
  const gain = D.observed[1][1], [nPub, kHep] = D.crowd, [cw, ch] = D.crowd_size, w = 4.6;
  const rate = Math.round(100 * D.cohort.hep[1] / D.cohort.hep[0]);
  s.addText([{ text: `Public board: ${nPub} patients`, options: { bold: true, breakLine: true } },
    { text: `about ${kHep} liver events, expected at the ${rate}% rate`, options: { color: HEX.verm } }],
  { x: 0.6, y: 1.65, w, h: 0.8, fontSize: 18, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addImage({ path: path.join(BUILD, "crowd.png"), x: 0.6, y: 2.6, w, h: w * ch / cw,
    altText: `${nPub} patient pictograms; ${kHep} filled ones mark the liver events expected at the measured rate` });
  step(s);
  s.addChart(pres.charts.BAR, [{ name: "Gain", labels: ["Our gain", `Public, n=${r.public.n}`, `Private, n=${r.private.n}`, `Whole test, n=${r["whole test"].n}`],
    values: [gain, r.public.g, r.private.g, r["whole test"].g] }],
  chart({ x: 5.5, w: 7.2, barDir: "col", chartColors: [HEX.verm, HEX.blue, HEX.blue, HEX.blue], valAxisMinVal: 0, valAxisMaxVal: 0.012, valAxisMajorUnit: 0.002,
    valAxisLabelFormatCode: "0.000", showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", barGapWidthPct: 60, valGridLine: { style: "none" },
    showTitle: true, title: "Our gain against the credible gain (two standard errors)",
    showValAxisTitle: true, valAxisTitle: "gain in weighted C-index", altText: "Our gain of 0.0056 against the credible gain on the public, private and whole test samples." }));
  caption(s, "Our gain was one standard error. A credible gain needed two.");
});

// 16 Multiplicity
def(16, "CONTENT_2", PARTS[2], "Data: simulated; our leaderboard gain", (s) => {
  const k = D.null_gain, gain = D.observed[1][1];
  s.addChart(pres.charts.BAR, [{ name: "Gain", labels: ["Our gain", ...k.map(([n]) => `Best of ${n} tries`)], values: [gain, ...k.map(([, g]) => g)] }],
    chart({ barDir: "col", chartColors: [HEX.verm, HEX.blue, HEX.blue, HEX.blue], valAxisMinVal: 0, valAxisMaxVal: 0.02, valAxisMajorUnit: 0.005,
      valAxisLabelFormatCode: "0.000", showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", barGapWidthPct: 60, valGridLine: { style: "none" },
      showValAxisTitle: true, valAxisTitle: "typical best gain by chance, weighted C-index",
      altText: "Our gain of 0.0056 against the typical best gain by chance for 5, 30 and 272 tries." }));
  caption(s, "The best of many tries looks good by construction.");
});

// 17 Mechanism
def(17, "CONTENT_2", PARTS[2], "Data: simulation, and leaderboard gains against the model's prediction", (s) => {
  s.addChart(pres.charts.BAR, [{ name: "Predicted best gain", labels: D.mechanism.map(([r]) => `Error correlation ${r.toFixed(1)}`), values: D.mechanism.map(([, p]) => p) }],
    chart({ x: 0.6, w: 5.6, barDir: "col", chartColors: [HEX.blue], valAxisMinVal: 0, valAxisMaxVal: 0.008, valAxisMajorUnit: 0.002, valAxisLabelFormatCode: "0.000",
      showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", barGapWidthPct: 60, catAxisLabelFontSize: 13, valGridLine: { style: "none" },
      showTitle: true, title: "Simulation: best gain at the best weight", showValAxisTitle: true, valAxisTitle: "gain in C-index",
      altText: "In simulation, the best blend gain falls from 0.0064 to zero as the error correlation rises from 0.0 to 0.6." }));
  step(s);
  const ob = D.observed, labels = ob.map((o) => o[0].replace(" into ", " into the "));
  s.addChart(pres.charts.BAR, [
    { name: "Best case at our weight, independent errors", labels, values: ob.map((o) => o[2]) },
    { name: "Observed", labels, values: ob.map((o) => o[1]) }],
  chart({ x: 6.5, w: 6.2, barDir: "col", chartColors: [HEX.sky, HEX.verm], valAxisMinVal: 0, valAxisMaxVal: 0.008, valAxisMajorUnit: 0.002, valAxisLabelFormatCode: "0.000",
    showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", barGapWidthPct: 50, catAxisLabelFontSize: 13, valGridLine: { style: "none" },
    showLegend: true, legendPos: "b", showTitle: true, title: "Our gains vs the best case at our weights", showValAxisTitle: true, valAxisTitle: "gain in C-index",
    altText: "For two blends, the observed gain is larger than the best case with independent errors at the blend's own weight." }));
  caption(s, "When reality beats the best case, suspect the measurement, not the model.");
});

// 18 CV leak
def(18, "CONTENT_2", PARTS[2], "Data: measured cross-validation and leaderboard", (s) => {
  const l = D.leak;
  const rows = [["Public leaderboard", l.leaderboard, HEX.blue], ["Window-free CV", l.window_free, HEX.blue], ["Red flag (a leak above this)", l.red_flag, HEX.gray],
    ["CV as run", l.cv_as_run, HEX.verm], ["Follow-up span only (no biology)", l.span_alone, HEX.verm]].reverse();
  s.addChart(pres.charts.BAR, [{ name: "Weighted C-index", labels: rows.map((r) => r[0]), values: rows.map((r) => r[1]) }],
    chart({ layout: { x: 0.3, y: 0.04, w: 0.66, h: 0.8 },             // with one colour per bar, Keynote's own layout cuts the labels
      barDir: "bar", chartColors: rows.map((r) => r[2]), valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0##",
      showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", barGapWidthPct: 45, catAxisLabelFontSize: 16, valGridLine: { style: "none" },
      showValAxisTitle: true, valAxisTitle: "weighted C-index (0.5 = random order)",
      altText: "Follow-up span alone scores 0.9614 and the CV as run 0.9502, above the 0.947 red flag. The window-free CV scores 0.9020 and the leaderboard 0.8938." }));
  caption(s, "Drop the 26 window features and the same base falls from 0.950 to 0.902.");
});

// 19 Replication
def(19, "CONTENT_2", PARTS[2], "Data: measured window-free 10-fold CV; synthetic contrast", (s) => {
  const r = D.replication;
  s.addChart(pres.charts.BAR, [{ name: "Weighted C-index", labels: ["Base", "GPT-OSS alone", "Base + GPT-OSS (0.07)"], values: [r.base, r.gptoss_alone, r.blend] }],
    chart({ x: 0.6, w: 6.6, barDir: "col", chartColors: [HEX.blue, HEX.gray, HEX.verm], valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
      valAxisLabelFormatCode: "0.0##", showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", barGapWidthPct: 55, catAxisLabelFontSize: 13, valGridLine: { style: "none" },
      showTitle: true, title: "Window-free cross-validation", showValAxisTitle: true, valAxisTitle: "weighted C-index (0.5 = random order)",
      altText: "Window-free CV: base 0.9020, GPT-OSS alone 0.7465, base plus GPT-OSS 0.8999." }));
  step(s);
  s.addChart(pres.charts.BAR, [{ name: "Folds won", labels: ["Hepatic", "Death", "Synthetic cohort"], values: [r.hep[0], r.death[0], r.synthetic_wins] }],
    chart({ x: 7.5, w: 5.2, barDir: "col", chartColors: [HEX.verm, HEX.verm, HEX.blue], valAxisMinVal: 0, valAxisMaxVal: 10, valAxisMajorUnit: 2, valGridLine: { style: "none" },
      showValue: true, dataLabelFormatCode: "0", dataLabelPosition: "outEnd", barGapWidthPct: 55, catAxisLabelFontSize: 13,
      showTitle: true, title: "Folds where the blend beat the base", showValAxisTitle: true, valAxisTitle: "folds won, of 10",
      altText: "The blend won 1 of 10 hepatic folds and 2 of 10 death folds. In a synthetic cohort with real signal it wins 9 of 10." }));
  caption(s, "Weight 0.07, fixed in advance: the blend loses in most folds. A synthetic cohort with real signal wins 9.");
});

// 20 Scorecard: what each LLM adds, whiskers at two standard errors (the talk's bar for a credible gain)
def(20, "CONTENT_2", PARTS[2], "Data: measured window-free 10-fold CV; whiskers span two standard errors", (s) => {
  const name = { "GPT-OSS split": "GPT-OSS, split", "Nemotron joint": "Nemotron, joint, reasoning", "MedGemma split": "MedGemma, split",
    "MedGemma joint": "MedGemma, joint" };                 // alone C-index and model versions are in the speaker notes
  const lo = -0.009, hi = 0.005, x0 = 4.6, x1 = 12.5, X = (v) => x0 + (v - lo) / (hi - lo) * (x1 - x0);
  const y0 = 1.7, rh = 0.8, yAxis = y0 + D.joint.length * rh;
  s.addShape(SH.LINE, { x: X(0), y: y0, w: 0, h: yAxis - y0, line: { color: HEX.gray, width: 1.5, dashType: "dash" } });
  s.addShape(SH.LINE, { x: x0, y: yAxis, w: x1 - x0, h: 0, line: { color: HEX.gray, width: 1 } });
  [-0.008, -0.004, 0, 0.004].forEach((v) => s.addText(v ? SIGNED(v, 3) : "0", { x: X(v) - 0.6, y: yAxis + 0.05, w: 1.2, h: 0.35,
    fontSize: 18, color: C.text2, align: "center", margin: 0, isTextBox: true }));
  s.addText("blend gain", { x: x0, y: yAxis + 0.4, w: x1 - x0, h: 0.35, fontSize: 18, color: C.text2,
    align: "center", margin: 0, isTextBox: true });
  D.joint.forEach(([n, alone, g, e], i) => {
    if (i) step(s);
    const y = y0 + i * rh + rh / 2, col = g > 0 ? HEX.verm : HEX.blue;
    s.addText(name[n] || n, { x: 0.6, y: y - rh / 2, w: 3.9, h: rh, fontSize: 20, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    s.addShape(SH.LINE, { x: X(g - 2 * e), y, w: X(g + 2 * e) - X(g - 2 * e), h: 0, line: { color: col, width: 3 } });
    s.addShape(SH.OVAL, { x: X(g) - 0.12, y: y - 0.12, w: 0.24, h: 0.24, fill: { color: col }, line: { color: col } });
    if (g > 0) s.addText(SIGNED(g), { x: X(g + 2 * e) + 0.1, y: y - 0.2, w: 1.0, h: 0.4, fontSize: 18, bold: true, color: col, margin: 0, isTextBox: true });
  });
  caption(s, "The LLMs agree with each other: rank correlation about 0.8.");
});

// 21 Part 2 recap
def(21, "CONTENT_2", PARTS[2], "", (s) => {
  recap(s, [
    "The 0.0056 public gain was one standard error. With 272 scored uploads, chance mattered more.",
    "The observed lift exceeded the best case predicted at the submitted weight.",
    "Follow-up span alone scored 0.9614, exposing leakage in cross-validation.",
    "Leak-free CV favored the base: 0.9020 versus 0.8999 for the blend.",
  ], HEX.verm);
});

// 22 Limits: two in words, the third as the score to quote
def(22, "CONTENT_3", PARTS[3], "Data: measured cross-validation and leaderboard", (s) => {
  const rows = [[th("Limit"), th("What it means")],
    ["Rankings, not risks", "The scores rank patients. They are not risks."],
    ["Cost and drift", "Two calls per patient. Store raw scores and the model tag."]];
  table(s, rows, { x: 0.6, y: 1.75, w: 5.9, colW: [2.3, 3.6], rowH: 1.15, fontSize: 20 });
  step(s);
  const l = D.leak, bars = [["CV as run (leaky)", l.cv_as_run, HEX.verm], ["Leak-free CV", l.window_free, HEX.blue], ["Public leaderboard", l.leaderboard, HEX.blue]].reverse();
  s.addChart(pres.charts.BAR, [{ name: "Weighted C-index", labels: bars.map((r) => r[0]), values: bars.map((r) => r[1]) }],
    chart({ x: 6.9, w: 5.8, h: 4.6, barDir: "bar", chartColors: bars.map((r) => r[2]), valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
      valAxisLabelFormatCode: "0.0##", showValue: true, dataLabelFormatCode: "0.000", dataLabelPosition: "outEnd", barGapWidthPct: 45, catAxisLabelFontSize: 16,
      valGridLine: { style: "none" }, showTitle: true, title: "Limit 3: which score to quote", showValAxisTitle: true, valAxisTitle: "weighted C-index (0.5 = random order)",
      altText: "CV as run scores 0.950 because it reads follow-up length; the leak-free CV scores 0.902, close to the leaderboard's 0.894." }));
  caption(s, "Quote the leak-free 0.902, not the leaky 0.950.");
});

// 23 Takeaways: three habits, each with its evidence beneath it
def(23, "CONTENT_3", PARTS[3], "Data: measured run time; synthetic rankers; measured CV", (s) => {
  const cols = [0.6, 4.75, 8.9], w = 3.8, top = 2.45, h = 2.9;
  [["Start local", "1,253 patients, on a laptop"], ["Test cheaply", "Three patients first"], ["Hunt for leaks", "Span alone beat the whole model"]].forEach(([habit, proof], i) => {
    badge(s, i + 1, cols[i], 1.72);
    s.addText(habit, { x: cols[i] + 0.7, y: 1.65, w: w - 0.7, h: 0.68, fontSize: 26, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
    s.addText(proof, { x: cols[i], y: top + h + 0.15, w, h: 0.8, fontSize: 18, color: C.text2, valign: "top", margin: 0, isTextBox: true });
  });
  step(s);
  s.addText([{ text: "35", options: { fontSize: 88, bold: true } }, { text: " min", options: { fontSize: 36, bold: true } }],
    { x: cols[0], y: top + 0.5, w, h: 1.6, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  step(s);
  const rho = D.rho_gain.map((r, i) => (i % 3 ? "" : r[0].toFixed(2))), gain = D.rho_gain.map((r) => r[1]);
  s.addChart(pres.charts.LINE, [{ name: "Best blend gain", labels: rho, values: gain }],
    chart({ x: cols[1], y: top, w, h, chartColors: [HEX.blue], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6, valAxisMinVal: 0, valAxisMaxVal: 0.003,
      valAxisMajorUnit: 0.001, valAxisLabelFormatCode: "0.000", catAxisLabelFontSize: 12, catAxisLabelRotate: 0, showTitle: true, title: "Lower correlation, bigger gain",
      showCatAxisTitle: true, catAxisTitle: "Spearman ρ, LLM vs base", altText: "Best blend gain falls to zero as the rank correlation with the base rises." }));
  step(s);
  const l = D.leak, bars = [["Span alone", l.span_alone, HEX.verm], ["CV as run", l.cv_as_run, HEX.verm], ["Leak-free CV", l.window_free, HEX.blue]].reverse();
  s.addChart(pres.charts.BAR, [{ name: "Weighted C-index", labels: bars.map((r) => r[0]), values: bars.map((r) => r[1]) }],
    chart({ x: cols[2], y: top, w, h, barDir: "bar", chartColors: bars.map((r) => r[2]), valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.25,
      valAxisLabelFormatCode: "0.0##", showValue: true, dataLabelFormatCode: "0.000", dataLabelPosition: "outEnd", barGapWidthPct: 40, catAxisLabelFontSize: 14,
      valGridLine: { style: "none" }, showTitle: true, title: "A leak probe beats the model",
      altText: "Follow-up span alone 0.961 beats the whole model's leaky CV 0.950; the leak-free CV is 0.902." }));
});

// 24 Close
def(24, "CONTENT_3", PARTS[3], "", (s) => {
  const mark = { text: "☐", options: { align: "center", fontSize: 22 } }, nb = (t) => t.replace(/ (\S+)$/, "\u00A0$1");
  ["Bigger than the noise?", "Too good to be true?", "Does it replicate?"].forEach((q, i) => {   // the three tests, above the full checklist
    const x = [0.6, 4.75, 8.9][i];
    if (i) step(s);
    badge(s, i + 1, x, 1.66);
    s.addText(q, { x: x + 0.65, y: 1.6, w: 3.4, h: 0.66, fontSize: 22, bold: true, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  });
  step(s);
  const half = Math.ceil(checklist.length / 2), rows = [];
  for (let i = 0; i < half; i++) {
    const j = i + half;
    rows.push([mark, { text: nb(checklist[i]) }, checklist[j] ? mark : { text: "" }, { text: checklist[j] ? nb(checklist[j]) : "" }]);
  }
  const rule = { type: "solid", pt: 0.75, color: HEX.grid }, none = { type: "none" };
  table(s, rows, { x: 0.6, y: 2.5, w: 12.1, colW: [0.55, 5.5, 0.55, 5.5], rowH: 0.74, fontSize: 17, border: [none, none, rule, none] });
  step(s);
  s.addText([{ text: "Questions or results: ", options: { color: HEX.gray } }, { text: "polleronny2@gmail.com", options: { bold: true, color: HEX.ink } }],
    { x: 0.6, y: 6.3, w: 12.1, h: 0.5, fontSize: 22, margin: 0, valign: "middle", isTextBox: true });
});

// ---- build
const order = Object.keys(DEFS).map(Number).sort((a, b) => a - b);
let lastSection = null;
order.forEach((n) => {
  const d = DEFS[n];
  if (d.section !== lastSection) { pres.addSection({ title: d.section }); lastSection = d.section; }
  const s = pres.addSlide({ masterName: d.layout, sectionTitle: d.section });
  if (n !== 1) s.addText(balance(TRACK.slides[n].header), { placeholder: "title" });
  CUR = n;
  d.fn(s);
  if (STEPS[n]) STEPS[n].total = s._slideObjects.length;   // the footer after this stays visible
  if (d.data) dataLabel(s, d.data);
  s.addNotes(notesFor(n));
});

// backup slides, hidden in the slideshow, for Q&A
function backup(headline, note, fn) {
  if (lastSection !== "Backup") { pres.addSection({ title: "Backup" }); lastSection = "Backup"; }
  const s = pres.addSlide({ masterName: `CONTENT_${PARTS.length - 1}`, sectionTitle: "Backup" });
  s.addText(balance(headline), { placeholder: "title" });
  fn(s);
  s.addNotes(note);
  if (!process.env.SHOW_HIDDEN) s.hidden = true;
}
backup("Without the leak, XGBoost alone looks better than the 50/50 blend, but not credibly",
  `BACKUP, for Q&A. ${TRACK.backups[0]}`, (s) => {
    const x = D.xgb_vs_blend;
    s.addChart(pres.charts.BAR, [
      { name: "XGBoost", labels: ["CV as run (leaky)", "Window-free CV"], values: [x.as_run[0], x.clean[0]] },
      { name: "CatBoost", labels: ["CV as run (leaky)", "Window-free CV"], values: [x.as_run[1], x.clean[1]] },
      { name: "50/50 blend", labels: ["CV as run (leaky)", "Window-free CV"], values: [x.as_run[2], x.clean[2]] }],
    chart({ barDir: "col", chartColors: [HEX.blue, HEX.sky, HEX.verm], valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0##",
      showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", barGapWidthPct: 60, showLegend: true, legendPos: "t", valGridLine: { style: "none" },
      showValAxisTitle: true, valAxisTitle: "weighted C-index (0.5 = random order)", altText: "XGBoost, CatBoost and the 50/50 blend in the leaky and the window-free CV." }));
    caption(s, `Fold mean ${SIGNED(x.fold[0])} ± ${FIG(x.fold[1])} (${x.fold[2]} SE) clears the bar. The patient bootstrap ${SIGNED(x.boot[0])} ± ${FIG(x.boot[1])} (${x.boot[2]} SE) does not.`);
    dataLabel(s, "Data: measured cross-validation");
  });
backup("The same checks apply wherever a strong base meets a weaker candidate ranker",
  `BACKUP, for Q&A. ${TRACK.backups[1]}`, (s) => {
    const rows = generalising.map((r, i) => r.map((c) => (i === 0 ? th(c) : { text: c })));
    table(s, rows, { x: 0.6, y: 1.8, w: 12.1, colW: [2.9, 3.2, 3.6, 2.4], rowH: 0.85, fontSize: 18 });
    dataLabel(s, "Schematic");
  });

// One click per group; a group's shapes fade in together. PowerPoint and Keynote have no blur entrance, so a fade is the nearest.
function timing(groups) {
  let id = 2;
  const fade = (spid, first) => `<p:par><p:cTn id="${++id}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" nodeType="${first ? "clickEffect" : "withEffect"}">` +
    `<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst><p:set><p:cBhvr><p:cTn id="${++id}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>` +
    `<p:tgtEl><p:spTgt spid="${spid}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>` +
    `<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="${++id}" dur="500"/><p:tgtEl><p:spTgt spid="${spid}"/></p:tgtEl></p:cBhvr></p:animEffect></p:childTnLst></p:cTn></p:par>`;
  const clicks = groups.map((ids) => {
    const a = ++id, b = ++id;
    return `<p:par><p:cTn id="${a}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst><p:childTnLst><p:par><p:cTn id="${b}" fill="hold">` +
      `<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>${ids.map((spid, k) => fade(spid, k === 0)).join("")}</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>`;
  }).join("");
  return `<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek">` +
    `<p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>${clicks}</p:childTnLst></p:cTn>` +
    `<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>` +
    `<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>`;
}

// pptxgenjs 4.0.1 writes each chart's embedded table range with a stray quote (ref="A1:B4'"). PowerPoint and LibreOffice
// shrug; Keynote rejects the workbook and drops the chart. applyTheme also consolidates the per-slide masters but leaves
// their content-type entries behind, so remove entries whose package parts no longer exist. Then each slide gets its steps:
// shapes before the first step stay put, and each step's shapes appear on one click. pptxgenjs writes shapes in the order
// they were added, so the k-th shape on a slide is the k-th <p:cNvPr> after the shape tree's own.
async function finish(file) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const contentTypesPart = "[Content_Types].xml";
  const contentTypes = await zip.file(contentTypesPart).async("string");
  zip.file(contentTypesPart, contentTypes.replace(/<Override PartName="([^"]+)" ContentType="[^"]+"\/>/g, (tag, part) =>
    zip.file(part.replace(/^\//, "")) ? tag : ""));
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/embeddings\/.+\.xlsx$/.test(n))) {
    const book = await JSZip.loadAsync(await zip.file(name).async("nodebuffer"));
    for (const t of Object.keys(book.files).filter((n) => /^xl\/tables\/.+\.xml$/.test(n))) {
      book.file(t, (await book.file(t).async("string")).replace(/ref="([A-Z]+\d+:[A-Z]+\d+)'"/, 'ref="$1"'));
    }
    zip.file(name, await book.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  }
  for (const [n, { bounds, total }] of Object.entries(STEPS)) {
    const part = `ppt/slides/slide${n}.xml`, xml = await zip.file(part).async("string");
    const ids = [...xml.matchAll(/<p:cNvPr id="(\d+)"/g)].slice(1).map((m) => m[1]);
    if (ids.length < total) throw new Error(`${part}: ${ids.length} shapes, the build counted ${total}`);
    const edges = [...new Set(bounds)].filter((b) => b > 0 && b < total).concat(total);
    const groups = edges.slice(0, -1).map((a, i) => ids.slice(a, edges[i + 1])).filter((g) => g.length);
    if (!groups.length) continue;
    if (!xml.includes("</p:clrMapOvr></p:sld>")) throw new Error(`${part}: no place for the timing`);
    zip.file(part, xml.replace("</p:clrMapOvr></p:sld>", `</p:clrMapOvr>${timing(groups)}</p:sld>`));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

pres.writeFile({ fileName: OUT }).then(() => applyTheme(OUT, THEME)).then(() => finish(OUT)).then(() => console.log("wrote", OUT));
