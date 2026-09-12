import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const { SKILL_DIR, TMP_DIR, FINAL_PPTX, WORKSPACE_DIR } = process.env;
if (!path.isAbsolute(SKILL_DIR ?? "") || !path.isAbsolute(TMP_DIR ?? "") || !path.isAbsolute(FINAL_PPTX ?? "")) {
  throw new Error("Set absolute SKILL_DIR, TMP_DIR, and FINAL_PPTX");
}

const workspaceDir = WORKSPACE_DIR || process.cwd();
const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
await fs.mkdir(path.join(workspaceDir, ".codex-finalizer"), { recursive: true });

const family = resolvePresentationFont();
const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});
const slide = presentation.slides.add();
slide.background.fill = "#FFFFFF";

const colors = {
  ink: "#172033",
  muted: "#5C667A",
  blue: "#1F5EA8",
  blueLight: "#EAF3FF",
  green: "#16824A",
  greenLight: "#EAF7EF",
  orange: "#B65C00",
  orangeLight: "#FFF2E0",
  gray: "#D8DEE8",
  grayLight: "#F7F9FC",
};

function addText(text, x, y, w, h, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: options.size ?? 18,
    bold: options.bold ?? false,
    color: options.color ?? colors.ink,
    autoFit: "shrinkText",
    alignment: options.align ?? "left",
  };
  return shape;
}

function addBox(text, x, y, w, h, options = {}) {
  const shape = slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill: { type: "solid", color: options.fill ?? colors.grayLight },
    line: { style: "solid", fill: options.line ?? colors.gray, width: options.lineWidth ?? 1.2 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: options.size ?? 17,
    bold: options.bold ?? false,
    color: options.color ?? colors.ink,
    autoFit: "shrinkText",
    alignment: "center",
  };
  return shape;
}

function connect(a, b, color = colors.muted) {
  const connector = slide.shapes.connect(a, b, {
    fromSide: "bottom",
    toSide: "top",
    kind: "straight",
    line: { style: "solid", fill: color, width: 2 },
    tail: { type: "arrow", width: "med", length: "med" },
  });
  connector.bringToFront();
  return connector;
}

addText("Pi0.5 Transcoder Feature Discovery", 48, 34, 760, 44, { size: 34, bold: true });
addText("Deliverables #4, #5, and #6", 48, 76, 520, 28, { size: 18, color: colors.muted });
addText("Pilot: 50 LIBERO episodes, 13,835 observations, 18 layers, 10 flow timesteps, 16,384 features per layer", 48, 112, 1120, 28, { size: 16, color: colors.muted });

const obs = addBox("LIBERO observations\nimage + state + task text", 60, 170, 260, 72, { fill: colors.blueLight, line: colors.blue, bold: true });
const model = addBox("Frozen Pi0.5 policy\ntrained transcoders in probe mode", 60, 278, 260, 78, { fill: colors.blueLight, line: colors.blue, bold: true });
const latent = addBox("Sparse activations\nz[l, tau, p, i]", 60, 394, 260, 72, { fill: colors.blueLight, line: colors.blue, bold: true });
const collapse = addBox("Collapse action position\nscore[l,tau,i] = max_p z[l,tau,p,i]\nretain p*", 60, 504, 260, 92, { fill: colors.orangeLight, line: colors.orange, bold: true, size: 16 });
connect(obs, model, colors.blue);
connect(model, latent, colors.blue);
connect(latent, collapse, colors.orange);

addText("Streaming summaries", 390, 158, 360, 32, { size: 24, bold: true });
const topk = addBox("#4 Top-K observations\nTop-20 examples for every\n(layer, tau, feature)", 380, 205, 350, 98, { fill: colors.greenLight, line: colors.green, bold: true, size: 17 });
const stats = addBox("#5 Global statistics\nmean, std, firing frequency,\ntop-M frequency", 380, 340, 350, 98, { fill: colors.greenLight, line: colors.green, bold: true, size: 17 });
const browser = addBox("#6 Feature browser\nrank candidates, filter rows,\ninspect Top-K images and tasks", 380, 475, 350, 105, { fill: colors.greenLight, line: colors.green, bold: true, size: 17 });

slide.shapes.connect(collapse, topk, {
  fromSide: "right",
  toSide: "left",
  kind: "elbow",
  line: { style: "solid", fill: colors.green, width: 2 },
  tail: { type: "arrow", width: "med", length: "med" },
}).bringToFront();
connect(topk, stats, colors.green);
connect(stats, browser, colors.green);

addText("Pseudo algorithm", 800, 158, 360, 32, { size: 24, bold: true });
const algorithmText = [
  "for each observation:",
  "  run Pi0.5 inference in probe mode",
  "  for each flow timestep tau:",
  "    for each action-expert layer l:",
  "      read z[l, tau, p, i]",
  "      score = max over action position p",
  "      update Top-K observations",
  "      update global statistics",
  "",
  "after collection:",
  "  rank (layer, tau, feature) cells",
  "  write candidate table",
  "  render HTML report with images",
].join("\n");
addBox(algorithmText, 790, 205, 420, 250, { fill: "#FFFFFF", line: colors.gray, size: 15 });

addText("How to read one row", 800, 486, 360, 30, { size: 24, bold: true });
addText("L12:tau1:F7584 means layer 12, flow timestep 1.0, feature 7584. If its Top-20 images and task text are coherent, treat it as a semantic hypothesis. Use later interventions before claiming causality.", 800, 526, 380, 92, { size: 17, color: colors.ink });

addText("Artifacts: feature_topk.pt, feature_stats.pt, feature_candidates.csv/json, feature_report.html, feature_report_with_images.html", 48, 660, 1184, 24, { size: 14, color: colors.muted, align: "center" });

slide.speakerNotes.textFrame.setText(
  "All numbers and artifact names come from the current Pi0.5 transcoder feature-discovery implementation and 50-episode pilot output in this repository. Interpretation labels are hypotheses until validated with image coherence and later intervention tests."
);

const outputStem = path.basename(FINAL_PPTX, ".pptx");
const candidatePath = path.join(workspaceDir, ".codex-finalizer", `${outputStem}_candidate.pptx`);
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const preview = await presentation.export({ slide, format: "png", scale: 2 });
const previewPath = path.join(path.dirname(FINAL_PPTX), `${outputStem}.png`);
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const result = await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: process.env.RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu",
    "12192000,6858000",
    "--validate-heading-fit",
  ],
  fontPolicy: { basis: "design", families: [family] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(workspaceDir, ".codex-finalizer", `${outputStem}.validation.json`),
});

await fs.writeFile(
  path.join(path.dirname(FINAL_PPTX), "pi05_feature_discovery_deliverables_manifest.json"),
  JSON.stringify({ finalPptx: FINAL_PPTX, previewPng: previewPath, validation: result }, null, 2),
);
