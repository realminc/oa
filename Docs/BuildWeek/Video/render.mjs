import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync, renameSync } from "node:fs";
import { join, resolve } from "node:path";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (value.startsWith("--")) pairs.push([value.slice(2), all[index + 1]]);
    return pairs;
  }, []),
);

const repo = resolve(args.repo);
const build = resolve(args.build);
const edgeTts = resolve(args["edge-tts"]);
const videoDir = join(repo, "Docs/BuildWeek/Video");
const narrationDir = join(videoDir, "narration");
const mediaDir = join(repo, "Docs/Media/BuildWeek");
const screencastsDir = resolve(args.screencasts ?? process.env.OA_SCREENCAST_DIR ?? join(repo, "var/build-week/screencasts"));
const landing = resolve(repo, "../landing/public");
const uiPublic = resolve(repo, "../ui/public");
const out = join(mediaDir, "OaBuildWeek2026Master1080p.mp4");
const poster = join(mediaDir, "OaBuildWeek2026Poster.webp");
const captions = join(mediaDir, "OaBuildWeek2026Captions.srt");
const mobileDemo = join(build, "OA-Mobile-Lab-Build-Week-Demo.mp4");
const fontSans = join(landing, "fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf");
const fontSansMedium = join(landing, "fonts/ibm-plex-sans/IBMPlexSans-Medium.ttf");
const fontMono = join(landing, "fonts/intel-one-mono/IntelOneMono-Regular.ttf");

for (const dir of [build, join(build, "audio"), join(build, "frames"), join(build, "scenes")]) {
  mkdirSync(dir, { recursive: true });
}

function run(command, commandArgs, options = {}) {
  const result = spawnSync(command, commandArgs, { stdio: "inherit", ...options });
  if (result.status !== 0) {
    throw new Error(`${command} failed with exit code ${result.status}`);
  }
}

function outputOf(command, commandArgs) {
  return execFileSync(command, commandArgs, { encoding: "utf8" }).trim();
}

function duration(path) {
  return Number(outputOf("ffprobe", [
    "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path,
  ]));
}

function fileReady(path) {
  return existsSync(path) && statSync(path).size > 1024;
}

function escapeXml(text) {
  return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function textLines(lines, x, y, size, color = "#ffffff", lineHeight = 1.18, weight = 400, family = "IBM Plex Sans") {
  return `<text x="${x}" y="${y}" fill="${color}" font-family="${family}" font-size="${size}" font-weight="${weight}">${lines.map((line, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : size * lineHeight}">${escapeXml(line)}</tspan>`).join("")}</text>`;
}

function codeLine(tokens, x, y, size = 24) {
  return `<text x="${x}" y="${y}" xml:space="preserve" font-family="Intel One Mono" font-size="${size}">${tokens.map(([token, color]) => `<tspan fill="${color}">${escapeXml(token)}</tspan>`).join("")}</text>`;
}

function terminalFrame(x, y, width, height, title) {
  return `<g>
    <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="18" fill="#0a0a0a" stroke="#333333"/>
    <line x1="${x}" y1="${y + 76}" x2="${x + width}" y2="${y + 76}" stroke="#242424"/>
    <g fill="none" stroke="#b3b3b3" stroke-width="2"><rect x="${x + 31}" y="${y + 28}" width="17" height="19" rx="2"/><path d="M${x + 37} ${y + 23} h16 a2 2 0 0 1 2 2 v16"/></g>
    <text x="${x + width / 2}" y="${y + 45}" text-anchor="middle" fill="#cccccc" font-family="IBM Plex Sans" font-size="21">${escapeXml(title)}</text>
    <circle cx="${x + width - 91}" cy="${y + 38}" r="9" fill="#ff5f57"/>
    <circle cx="${x + width - 61}" cy="${y + 38}" r="9" fill="#febc2e"/>
    <circle cx="${x + width - 31}" cy="${y + 38}" r="9" fill="#28c840"/>
  </g>`;
}

function imageData(path) {
  const extension = path.endsWith(".webp") ? "webp" : path.endsWith(".png") ? "png" : path.endsWith(".svg") ? "svg+xml" : "jpeg";
  return `data:image/${extension};base64,${readFileSync(path).toString("base64")}`;
}

function svgShell(content, defs = "") {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
  <defs>
    <linearGradient id="shade" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#050505" stop-opacity="0.98"/><stop offset="0.58" stop-color="#050505" stop-opacity="0.62"/><stop offset="1" stop-color="#050505" stop-opacity="0.18"/></linearGradient>
    <linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0a0a0a" stop-opacity="0"/><stop offset="1" stop-color="#0a0a0a" stop-opacity="0.92"/></linearGradient>
    <filter id="mono-white"><feColorMatrix values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 1 0"/></filter>
    ${defs}
  </defs>
  <rect width="1920" height="1080" fill="#0a0a0a"/>
  ${content}
  <text x="120" y="72" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="22" letter-spacing="2">OA / OPENAI BUILD WEEK 2026</text>
  <image href="${realmWordmark}" x="1632" y="31" width="168" height="44" preserveAspectRatio="xMidYMid meet"/>
</svg>`;
}

const blocks = [
  ["HPC", "matrices / kernels"], ["ML + AUTOGRAD", "training / inference"],
  ["VISION", "images / transforms"], ["AUDIO", "decode / features"],
  ["MEDIA", "video / codecs"], ["VIEWER", "plot / present"],
];
const blockSvg = blocks.map(([title, sub], index) => {
  const column = index % 3;
  const row = Math.floor(index / 3);
  const x = 120 + column * 570;
  const y = 370 + row * 210;
  return `<g><rect x="${x}" y="${y}" width="520" height="160" rx="20" fill="#181818" stroke="#ffffff" stroke-opacity="0.1"/>
    <circle cx="${x + 32}" cy="${y + 28}" r="5" fill="#ffffff" fill-opacity="0.32"/>
    <text x="${x + 28}" y="${y + 62}" fill="#ffffff" font-family="IBM Plex Sans" font-size="34" font-weight="500">${title}</text>
    <text x="${x + 28}" y="${y + 112}" fill="#a3a3a3" font-family="Intel One Mono" font-size="23">${sub}</text></g>`;
}).join("");

const coreImage = imageData(join(landing, "images/realm/core-engine.webp"));
const mlSpaceImage = imageData(join(landing, "images/realm/ml-space.webp"));
const audioPianoImage = imageData(join(landing, "images/realm/audio-piano.webp"));
const libraryOfficeImage = imageData(join(landing, "images/realm/library-office.webp"));
const visionInterviewImage = imageData(join(landing, "images/realm/vision-interview.webp"));
const mobileImage = imageData(join(mediaDir, "OaMobileLab1.webp"));
const realmWordmark = imageData(join(uiPublic, "logo/realm-wordmark-light.svg"));
const vendorNvidia = imageData(join(landing, "images/vendors/nvidia-wordmark.svg"));
const vendorAmd = imageData(join(landing, "images/vendors/amd.svg"));
const vendorIntel = imageData(join(landing, "images/vendors/intel.svg"));
const vendorQualcomm = imageData(join(landing, "images/vendors/qualcomm.svg"));

function roadmapHero(image, eyebrow, headline, detail, evidence, target) {
  return svgShell(`
    <image href="${image}" x="0" y="0" width="1920" height="1080" preserveAspectRatio="xMidYMid slice"/>
    <rect width="1920" height="1080" fill="url(#shade)"/>
    <text x="120" y="190" fill="#a3a3a3" font-family="Intel One Mono" font-size="21" letter-spacing="2">${escapeXml(eyebrow)}</text>
    ${textLines(headline, 120, 310, 68, "#ffffff", 1.08, 500)}
    <text x="120" y="520" fill="#d4d4d4" font-family="IBM Plex Sans" font-size="29">${escapeXml(detail)}</text>
    <rect x="120" y="615" width="1050" height="154" rx="24" fill="#0a0a0a" fill-opacity="0.78" stroke="#ffffff" stroke-opacity="0.14"/>
    <text x="158" y="674" fill="#ffffff" font-family="Intel One Mono" font-size="22">${escapeXml(evidence)}</text>
    <text x="158" y="724" fill="#a3a3a3" font-family="Intel One Mono" font-size="20">${escapeXml(target)}</text>
    <text x="120" y="928" fill="#ffffff" font-family="IBM Plex Sans" font-size="29">WORKING BASE → ARCHITECTURE CONVERGENCE → MEASURED BREADTH</text>`);
}

const slides = {
  "01-hook": svgShell(`
    <image href="${mlSpaceImage}" x="0" y="0" width="1920" height="1080" preserveAspectRatio="xMidYMid slice"/>
    <rect width="1920" height="1080" fill="url(#shade)"/>
    ${textLines(["This phone is training", "a Transformer. Locally."], 120, 245, 72, "#ffffff", 1.08, 500)}
    <text x="120" y="450" fill="#d4d4d4" font-family="IBM Plex Sans" font-size="30">One capability-driven Vulkan runtime.</text>
    <rect x="120" y="540" width="770" height="132" rx="22" fill="#0a0a0a" fill-opacity="0.72" stroke="#ffffff" stroke-opacity="0.12"/>
    <text x="158" y="594" fill="#ffffff" font-family="Intel One Mono" font-size="23">NVIDIA · AMD · INTEL · QUALCOMM</text>
    <text x="158" y="640" fill="#a3a3a3" font-family="Intel One Mono" font-size="21">DISCRETE · INTEGRATED · MOBILE</text>
    <rect x="1360" y="100" width="440" height="880" rx="30" fill="#0a0a0a" fill-opacity="0.84" stroke="#ffffff" stroke-opacity="0.16"/>
    <image href="${mobileImage}" x="1380" y="120" width="400" height="840" preserveAspectRatio="xMidYMid meet"/>
    <text x="120" y="820" fill="#ffffff" font-family="IBM Plex Sans" font-size="29">OA 0.7.5 · DEVELOPMENT PREVIEW</text>
    <text x="120" y="866" fill="#a3a3a3" font-family="Intel One Mono" font-size="21">REAL DEVICE PROOF / ADRENO 610 / FP32</text>`),

  "poster": svgShell(`
    <image href="${mlSpaceImage}" x="650" y="0" width="1270" height="1080" preserveAspectRatio="xMidYMid slice"/>
    <rect width="1920" height="1080" fill="url(#shade)"/>
    <text x="120" y="255" fill="#ffffff" font-family="IBM Plex Sans" font-size="168" font-weight="500">OA</text>
    ${textLines(["GPU-first architecture", "framework."], 120, 405, 68, "#ffffff", 1.08, 500)}
    <text x="120" y="650" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="31">HPC · ML · VISION · AUDIO · MEDIA · RENDER</text>
    <text x="120" y="704" fill="#d4d4d4" font-family="Intel One Mono" font-size="22">NVIDIA · AMD · INTEL · QUALCOMM · MOBILE</text>
    <rect x="120" y="750" width="560" height="82" rx="14" fill="#fafafa"/>
    <text x="400" y="803" text-anchor="middle" fill="#0a0a0a" font-family="Intel One Mono" font-size="24">OPENAI BUILD WEEK 2026</text>
    <text x="120" y="930" fill="#ffffff" font-family="IBM Plex Sans" font-size="30">realm.software</text>`),

  "02-problem": svgShell(`
    ${textLines(["One accelerator.", "Six disconnected stacks."], 120, 190, 70, "#ffffff", 1.06, 500)}
    <text x="120" y="322" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="30">Every boundary can add a copy, allocator, synchronization rule, and backend.</text>
    ${blockSvg}
    <line x1="120" y1="870" x2="1800" y2="870" stroke="#5e5e5e"/>
    <text x="120" y="930" fill="#ffffff" font-family="Intel One Mono" font-size="27">ONE RUNTIME  ·  ONE CAPABILITY MODEL  ·  EXPLICIT COMPLETION</text>`),

  "03-engine": svgShell(`
    <image href="${coreImage}" x="760" y="0" width="1160" height="1080" preserveAspectRatio="xMidYMid slice"/>
    <rect width="1920" height="1080" fill="url(#shade)"/>
    ${textLines(["Semantics above.", "Vulkan below."], 120, 220, 76, "#ffffff", 1.06, 500)}
    <text x="120" y="400" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="29">Values keep their meaning. One engine owns execution.</text>
    <g font-family="Intel One Mono" font-size="23">
      <rect x="120" y="510" width="245" height="86" rx="16" fill="#fafafa"/><text x="242" y="564" text-anchor="middle" fill="#0a0a0a">MATRIX</text>
      <rect x="385" y="510" width="245" height="86" rx="16" fill="#fafafa"/><text x="507" y="564" text-anchor="middle" fill="#0a0a0a">IMAGE</text>
      <rect x="120" y="616" width="245" height="86" rx="16" fill="#fafafa"/><text x="242" y="670" text-anchor="middle" fill="#0a0a0a">AUDIO</text>
      <rect x="385" y="616" width="245" height="86" rx="16" fill="#fafafa"/><text x="507" y="670" text-anchor="middle" fill="#0a0a0a">VIDEO</text>
      <rect x="120" y="758" width="510" height="108" rx="20" fill="#181818" stroke="#ffffff" stroke-opacity="0.28"/><text x="375" y="825" text-anchor="middle" fill="#ffffff">OA ENGINE</text>
    </g>
    <rect x="92" y="876" width="1736" height="92" rx="22" fill="#0a0a0a" fill-opacity="0.78" stroke="#ffffff" stroke-opacity="0.1"/>
    <image href="${vendorNvidia}" x="130" y="898" width="180" height="48" preserveAspectRatio="xMidYMid meet" filter="url(#mono-white)"/>
    <image href="${vendorAmd}" x="390" y="900" width="170" height="44" preserveAspectRatio="xMidYMid meet" filter="url(#mono-white)"/>
    <image href="${vendorIntel}" x="650" y="900" width="170" height="44" preserveAspectRatio="xMidYMid meet" filter="url(#mono-white)"/>
    <image href="${vendorQualcomm}" x="910" y="903" width="220" height="38" preserveAspectRatio="xMidYMid meet" filter="url(#mono-white)"/>
    <line x1="1210" y1="891" x2="1210" y2="953" stroke="#ffffff" stroke-opacity="0.16"/>
    <text x="1260" y="918" fill="#ffffff" font-family="Intel One Mono" font-size="20">DISCRETE · INTEGRATED</text>
    <text x="1260" y="947" fill="#a3a3a3" font-family="Intel One Mono" font-size="20">MOBILE · CAPABILITY-DRIVEN</text>`),

  "05-hard": svgShell(`
    ${textLines(["The hard part", "is the contract."], 120, 200, 76, "#ffffff", 1.06, 500)}
    <text x="120" y="370" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="29">Semantic intent becomes explicit executable work.</text>
    <g font-family="Intel One Mono" font-size="24">
      <rect x="120" y="480" width="290" height="108" rx="20" fill="#fafafa"/><text x="265" y="544" text-anchor="middle" fill="#0a0a0a">OPERATION</text>
      <path d="M410 534 H505" stroke="#ffffff" stroke-width="2"/><path d="M490 522 l18 12 -18 12" fill="#ffffff"/>
      <rect x="510" y="480" width="290" height="108" rx="20" fill="#181818" stroke="#ffffff" stroke-opacity="0.16"/><text x="655" y="544" text-anchor="middle" fill="#ffffff">LOWERING</text>
      <path d="M800 534 H895" stroke="#ffffff" stroke-width="2"/><path d="M880 522 l18 12 -18 12" fill="#ffffff"/>
      <rect x="900" y="480" width="290" height="108" rx="20" fill="#181818" stroke="#ffffff" stroke-opacity="0.16"/><text x="1045" y="544" text-anchor="middle" fill="#ffffff">DISPATCH</text>
      <path d="M1190 534 H1285" stroke="#ffffff" stroke-width="2"/><path d="M1270 522 l18 12 -18 12" fill="#ffffff"/>
      <rect x="1290" y="480" width="290" height="108" rx="20" fill="#181818" stroke="#ffffff" stroke-opacity="0.16"/><text x="1435" y="544" text-anchor="middle" fill="#ffffff">BARRIER</text>
      <path d="M1580 534 H1655" stroke="#ffffff" stroke-width="2"/><path d="M1640 522 l18 12 -18 12" fill="#ffffff"/>
      <rect x="1650" y="480" width="150" height="108" rx="20" fill="#fafafa"/><text x="1725" y="542" text-anchor="middle" fill="#0a0a0a">EVENT</text>
    </g>
    <g font-family="Intel One Mono" font-size="22" fill="#a3a3a3">
      <text x="120" y="730">shape</text><text x="340" y="730">lifetime</text><text x="590" y="730">descriptor</text>
      <text x="900" y="730">visibility</text><text x="1200" y="730">capability</text><text x="1500" y="730">completion</text>
    </g>
    <line x1="120" y1="775" x2="1800" y2="775" stroke="#404040"/>
    <text x="120" y="850" fill="#ffffff" font-family="IBM Plex Sans" font-size="35">Recorded work amortizes host overhead.</text>
    <text x="120" y="910" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="27">Kernel selection remains an internal lowering decision.</text>`),

  "07-api": svgShell(`
    ${textLines(["Same operations.", "C++ or Python."], 120, 185, 68, "#ffffff", 1.06, 500)}
    <text x="120" y="350" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="29">The high-level surface stays small. Ownership remains explicit.</text>
    ${terminalFrame(120, 430, 810, 438, "audio_to_model.cpp")}
    ${terminalFrame(970, 430, 830, 438, "audio_to_model.py")}
    ${codeLine([["auto", "#569cd6"], [" decoded ", "#9cdcfe"], ["= ", "#cccccc"], ["OaAudioDecoder", "#4ec9b0"], ["::", "#cccccc"], ["LoadFile", "#dcdcaa"], ["(\"speech.flac\");", "#ce9178"]], 158, 555, 22)}
    ${codeLine([["if", "#569cd6"], [" (decoded.", "#cccccc"], ["IsError", "#dcdcaa"], ["()) ", "#cccccc"], ["return", "#569cd6"], [" 1;", "#b5cea8"]], 158, 610, 22)}
    ${codeLine([["auto", "#569cd6"], [" clean ", "#9cdcfe"], ["= ", "#cccccc"], ["OaFnAudio", "#4ec9b0"], ["::", "#cccccc"], ["Normalize", "#dcdcaa"], ["(decoded->Buffer, ", "#cccccc"], ["-3.0F", "#b5cea8"], [");", "#cccccc"]], 158, 695, 22)}
    ${codeLine([["auto", "#569cd6"], [" mel ", "#9cdcfe"], ["= ", "#cccccc"], ["OaFnAudio", "#4ec9b0"], ["::", "#cccccc"], ["MelSpectrogram", "#dcdcaa"], ["(clean, decoded->", "#cccccc"], ["Meta", "#dcdcaa"], ["());", "#cccccc"]], 158, 750, 22)}
    ${codeLine([["from", "#569cd6"], [" oa ", "#cccccc"], ["import", "#569cd6"], [" audio", "#4ec9b0"]], 1008, 555, 22)}
    ${codeLine([["decoded ", "#9cdcfe"], ["= ", "#cccccc"], ["audio", "#4ec9b0"], [".", "#cccccc"], ["OaAudioDecoder", "#4ec9b0"], [".", "#cccccc"], ["LoadFile", "#dcdcaa"], ["(\"speech.flac\")", "#ce9178"]], 1008, 630, 22)}
    ${codeLine([["clean ", "#9cdcfe"], ["= ", "#cccccc"], ["audio", "#4ec9b0"], [".", "#cccccc"], ["Normalize", "#dcdcaa"], ["(decoded.Buffer, ", "#cccccc"], ["-3.0", "#b5cea8"], [", ", "#cccccc"], ["0", "#b5cea8"], [")", "#cccccc"]], 1008, 705, 22)}
    ${codeLine([["mel ", "#9cdcfe"], ["= ", "#cccccc"], ["audio", "#4ec9b0"], [".", "#cccccc"], ["MelSpectrogram", "#dcdcaa"], ["(clean, decoded.SampleRate)", "#cccccc"]], 1008, 760, 22)}
    <text x="120" y="396" fill="#737373" font-family="Intel One Mono" font-size="19">VS CODE DARK MODERN · INTEL ONE MONO · OA CODEBLOCK</text>`),

  "08-codex": svgShell(`
    ${textLines(["Codex + GPT-5.6.", "Evidence, not autocomplete."], 120, 180, 66, "#ffffff", 1.06, 500)}
    <text x="120" y="342" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="29">The engineering loop ends at the hardware gate.</text>
    <text x="1200" y="365" fill="#737373" font-family="Intel One Mono" font-size="18">NVIDIA · AMD · INTEL · QUALCOMM RUNTIME PATHS</text>
    ${terminalFrame(120, 410, 1030, 460, "build_week_gate.log")}
    ${codeLine([["OBSERVED", "#d7ba7d"], ["   phone-only Transformer corruption", "#cccccc"]], 162, 555, 23)}
    ${codeLine([["TRACED", "#d7ba7d"], ["     packed QKV descriptor selection", "#cccccc"]], 162, 625, 23)}
    ${codeLine([["FIXED", "#d7ba7d"], ["      uniform branch + explicit contract", "#cccccc"]], 162, 695, 23)}
    ${codeLine([["VERIFIED", "#28c840"], ["   Qualcomm mobile + Intel desktop", "#cccccc"]], 162, 765, 23)}
    <g>
      <rect x="1200" y="410" width="600" height="130" rx="20" fill="#181818" stroke="#ffffff" stroke-opacity="0.1"/>
      <text x="1240" y="462" fill="#ffffff" font-family="IBM Plex Sans" font-size="29">Audit</text>
      <text x="1240" y="505" fill="#a3a3a3" font-family="Intel One Mono" font-size="20">ownership · synchronization</text>
      <rect x="1200" y="565" width="600" height="130" rx="20" fill="#181818" stroke="#ffffff" stroke-opacity="0.1"/>
      <text x="1240" y="617" fill="#ffffff" font-family="IBM Plex Sans" font-size="29">Implement</text>
      <text x="1240" y="660" fill="#a3a3a3" font-family="Intel One Mono" font-size="20">smallest vertical correction</text>
      <rect x="1200" y="720" width="600" height="150" rx="20" fill="#fafafa"/>
      <text x="1240" y="775" fill="#0a0a0a" font-family="IBM Plex Sans" font-size="29">Prove</text>
      <text x="1240" y="818" fill="#525252" font-family="Intel One Mono" font-size="20">oracle · device · regression</text>
    </g>`),

  "09-close": svgShell(`
    <image href="${mobileImage}" x="1290" y="95" width="510" height="900" preserveAspectRatio="xMidYMid meet"/>
    ${textLines(["The accelerator", "you already own", "should be able to train."], 120, 240, 76, "#ffffff", 1.06, 500)}
    <text x="120" y="560" fill="#a3a3a3" font-family="IBM Plex Sans" font-size="30">OA 0.7.5 · DEVELOPMENT PREVIEW</text>
    <g font-family="Intel One Mono" font-size="24" fill="#ffffff">
      <text x="120" y="660">SIGNED ANDROID APP</text><text x="120" y="710">PYTHON WHEEL</text>
      <text x="120" y="760">LINUX PACKAGES</text><text x="120" y="810">SOURCE</text>
    </g>
    <text x="120" y="850" fill="#d4d4d4" font-family="Intel One Mono" font-size="22">NVIDIA · AMD · INTEL · QUALCOMM · MOBILE</text>
    <text x="120" y="887" fill="#a3a3a3" font-family="Intel One Mono" font-size="18">LINUX + ANDROID / NOW   ·   WINDOWS + MACOS / COMPUTE PLANNED</text>
    <text x="120" y="930" fill="#ffffff" font-family="IBM Plex Sans" font-size="30">realm.software</text>
    <text x="120" y="972" fill="#a3a3a3" font-family="Intel One Mono" font-size="22">github.com/realminc/oa/releases/tag/v0.7.6</text>`),

  "10-roadmap-ml": roadmapHero(
    mlSpaceImage,
    "WORKING BASE / ML + RL",
    ["Five mobile routes.", "CartPole PPO."],
    "Autograd · AdamW · evaluation · generation · checkpoint reload",
    "SHIPPED PREVIEW / COMPLETE TRAINING GATE",
    "ROADMAP / CORE + ML BASELINE ≤ 100 · CEILING 150",
  ),
  "10-roadmap-vision": roadmapHero(
    visionInterviewImage,
    "WORKING BASE / VISION",
    ["50 image operations.", "Video stays capability-gated."],
    "Transforms · evaluation · image and video viewer paths",
    "CURATED FLOAT32 NCHW COMMON PATH",
    "ROADMAP / BASELINE ≤ 80 · CEILING 100",
  ),
  "10-roadmap-audio": roadmapHero(
    audioPianoImage,
    "WORKING BASE / AUDIO",
    ["13 operation candidates.", "One zero-copy view."],
    "Decode · process · features · stream · encode",
    "RECORDED / 37 OF 37 NATIVE TESTS / NAMED IRIS XE PROFILE",
    "ROADMAP / 50 OFFLINE OPS · 18 REAL-TIME PROCESSORS",
  ),
  "10-roadmap-foundation": roadmapHero(
    libraryOfficeImage,
    "ARCHITECTURE CONVERGENCE",
    ["167 inventory entries.", "One generated contract."],
    "One engine · semantic values · explicit events · measured capabilities",
    "MIGRATION INVENTORY / NOT 167 SHIPPED OPERATIONS",
    "BOUNDED TARGETS / CORE + ML · VISION · AUDIO · 12 PLOT ARTISTS",
  ),
};

for (const [name, svg] of Object.entries(slides)) {
  const svgPath = join(build, "frames", `${name}.svg`);
  const pngPath = join(build, "frames", `${name}.png`);
  writeFileSync(svgPath, svg);
  run("magick", ["-background", "#0a0a0a", svgPath, pngPath]);
}

if (!fileReady(mobileDemo)) {
  run("curl", ["-fL", "--retry", "3", "--max-time", "180", "-o", mobileDemo,
    "https://github.com/realminc/oa/releases/download/v0.7.6/OA-Mobile-Lab-Build-Week-Demo.mp4"]);
}

const segmentNames = ["01-hook", "02-problem", "03-engine", "04-proof", "05-hard", "06-breadth", "07-api", "08-codex", "09-close", "10-roadmap"];
const gap = 0.38;
const audioDurations = [];

for (const name of segmentNames) {
  const textPath = join(narrationDir, `${name}.txt`);
  const raw = join(build, "audio", `${name}.raw.mp3`);
  const part = `${raw}.part`;
  const fast = join(build, "audio", `${name}.fast.wav`);
  const rawIsCurrent = fileReady(raw) && statSync(raw).mtimeMs >= statSync(textPath).mtimeMs;
  if (!rawIsCurrent) {
    let made = false;
    for (let attempt = 1; attempt <= 5 && !made; attempt += 1) {
      try {
        run(edgeTts, ["--voice", "en-US-AndrewMultilingualNeural", "--rate=+20%", "--pitch=-2Hz", "--file", textPath, "--write-media", part]);
        made = fileReady(part);
      } catch (error) {
        if (attempt === 5) throw error;
      }
    }
    if (!made) throw new Error(`TTS did not produce ${name}`);
    renameSync(part, raw);
  }
  run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-i", raw,
    "-filter:a", "atempo=1.5,aresample=48000", "-c:a", "pcm_s24le", fast]);
  audioDurations.push(duration(fast));
}

const sceneDurations = audioDurations.map((value) => value + gap);

function encodeStatic(name, seconds) {
  const input = join(build, "frames", `${name}.png`);
  const output = join(build, "scenes", `${name}.mp4`);
  const fadeOut = Math.max(0, seconds - 0.22).toFixed(3);
  run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-i", input,
    "-t", seconds.toFixed(3), "-vf",
    `zoompan=z='min(zoom+0.00015,1.025)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30,fade=t=in:st=0:d=0.22,fade=t=out:st=${fadeOut}:d=0.22,format=yuv420p`,
    "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", "30", output]);
}

function encodeMobile(name, seconds, start, label, evidence = "REAL DEVICE / RELEASE v0.7.5") {
  const output = join(build, "scenes", `${name}.mp4`);
  const fadeOut = Math.max(0, seconds - 0.18).toFixed(3);
  const vf = [
    "scale=1920:1080:force_original_aspect_ratio=increase", "crop=1920:1080", "setsar=1",
    "drawbox=x=0:y=0:w=iw:h=94:color=black@0.72:t=fill",
    `drawtext=fontfile=${fontSansMedium}:text='${label}':x=120:y=34:fontsize=24:fontcolor=white`,
    `drawtext=fontfile=${fontMono}:text='${evidence}':x=w-tw-120:y=35:fontsize=20:fontcolor=0xa3a3a3`,
    "fade=t=in:st=0:d=0.18", `fade=t=out:st=${fadeOut}:d=0.18`, "format=yuv420p",
  ].join(",");
  run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-ss", String(start), "-stream_loop", "-1", "-i", mobileDemo,
    "-t", seconds.toFixed(3), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", "30", output]);
}

function encodeBreadth(name, seconds) {
  const output = join(build, "scenes", `${name}.mp4`);
  const sources = {
    audio: join(screencastsDir, "ViewerAudio.mp4"),
    video: join(screencastsDir, "ViewerVideo1.mp4"),
    cartPole: join(screencastsDir, "TutorialMlRlCartPole.mp4"),
    nlp: join(screencastsDir, "TutorialNlpBpeTransformer.mp4"),
    skeleton: join(screencastsDir, "TutorialGen3dAnimAlm.mp4"),
  };
  for (const source of Object.values(sources)) {
    if (!fileReady(source)) throw new Error(`Missing montage source: ${source}`);
  }

  const totalFrames = Math.round(seconds * 30);
  const shots = [
    { source: sources.audio, start: 0.4, frames: 42, crop: "wide", label: "AUDIO / WAVEFORM VIEWER", flash: false },
    { source: sources.audio, start: 2.2, frames: 30, crop: "close", label: "AUDIO / WAVEFORM VIEWER" },
    { source: sources.video, start: 9.0, frames: 45, crop: "wide", label: "MEDIA / VIDEO VIEWER", flash: true },
    { source: sources.video, start: 20.1, frames: 36, crop: "close", label: "MEDIA / VIDEO VIEWER" },
    { source: sources.cartPole, start: 4.0, frames: 48, crop: "wide", label: "RL / CARTPOLE + LIVE METRICS", flash: true },
    { source: sources.cartPole, start: 16.2, frames: 42, crop: "tool", label: "RL / CARTPOLE + LIVE METRICS" },
    { source: sources.nlp, start: 1.6, frames: 48, crop: "wide", label: "ML / BPE TRANSFORMER TRAINING", flash: true },
    { source: sources.nlp, start: 5.2, frames: 36, crop: "terminal", label: "ML / BPE TRANSFORMER TRAINING" },
    { source: sources.skeleton, start: 50.5, frames: 54, crop: "wide", label: "GENERATIVE 3D / TEXT TO USD SKELETON", flash: true },
    { source: sources.skeleton, start: 54.0, frames: 60, crop: "tool", label: "GENERATIVE 3D / TEXT TO USD SKELETON" },
    { source: sources.skeleton, start: 58.5, frames: 60, crop: "close", label: "GENERATIVE 3D / TEXT TO USD SKELETON" },
    { source: sources.skeleton, start: 63.0, frames: 0, crop: "tool", label: "GENERATIVE 3D / TEXT TO USD SKELETON", outro: true },
  ];
  shots.at(-1).frames = totalFrames - shots.slice(0, -1).reduce((sum, shot) => sum + shot.frames, 0);
  if (shots.at(-1).frames < 1) throw new Error(`Breadth montage is too short: ${seconds.toFixed(3)} seconds`);

  const cropFilters = {
    wide: "scale=1920:-2,crop=1920:1080:0:(ih-oh)/2",
    close: "crop=3200:1800:(iw-ow)/2:(ih-oh)/2,scale=1920:1080",
    tool: "crop=2880:1620:(iw-ow)/2:(ih-oh)/2,scale=1920:1080",
    terminal: "crop=3200:1800:320:360,scale=1920:1080",
  };
  const inputs = [];
  const filters = [];
  for (const [index, shot] of shots.entries()) {
    inputs.push("-ss", shot.start.toFixed(3), "-i", shot.source);
    const effects = [
      "fps=30",
      `trim=end_frame=${shot.frames}`,
      "setpts=PTS-STARTPTS",
      cropFilters[shot.crop],
      "setsar=1",
      "drawbox=x=0:y=0:w=iw:h=92:color=black@0.68:t=fill",
      `drawtext=fontfile=${fontSansMedium}:text='${shot.label}':x=120:y=32:fontsize=24:fontcolor=white`,
      `drawtext=fontfile=${fontMono}:text='OA TOOLKIT / LIVE CAPTURE':x=w-tw-120:y=34:fontsize=19:fontcolor=0xa3a3a3`,
    ];
    if (shot.flash) effects.push("fade=t=in:st=0:d=0.08:color=white");
    if (index === 0) effects.push("fade=t=in:st=0:d=0.12:color=black");
    if (shot.outro) effects.push(`fade=t=out:st=${Math.max(0, shot.frames / 30 - 0.18).toFixed(3)}:d=0.18:color=black`);
    filters.push(`[${index}:v]${effects.join(",")}[v${index}]`);
  }
  filters.push(`${shots.map((_, index) => `[v${index}]`).join("")}concat=n=${shots.length}:v=1:a=0,format=yuv420p[out]`);

  run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...inputs,
    "-filter_complex", filters.join(";"), "-map", "[out]", "-an",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", "30", output]);
}

function encodeRoadmap(name, seconds) {
  const output = join(build, "scenes", `${name}.mp4`);
  const walkthrough = join(screencastsDir, "LandingDevDocsPageWalkthrough.mp4");
  if (!fileReady(walkthrough)) throw new Error(`Missing roadmap source: ${walkthrough}`);

  const totalFrames = Math.round(seconds * 30);
  const shots = [
    { type: "video", source: walkthrough, start: 26.2, frames: 45, crop: "wide", label: "DEVELOPER DOCS / MODULE MAP" },
    { type: "image", source: join(build, "frames", "10-roadmap-ml.png"), frames: 36, flash: true },
    { type: "image", source: join(build, "frames", "10-roadmap-vision.png"), frames: 36, flash: true },
    { type: "image", source: join(build, "frames", "10-roadmap-audio.png"), frames: 36, flash: true },
    { type: "image", source: join(build, "frames", "10-roadmap-foundation.png"), frames: 42, flash: true },
    { type: "video", source: walkthrough, start: 30.0, frames: 42, crop: "close", label: "DEVELOPER DOCS / ML + RL" },
    { type: "video", source: walkthrough, start: 34.0, frames: 42, crop: "tool", label: "DEVELOPER DOCS / EARLY TEXT TO USD" },
    { type: "video", source: walkthrough, start: 38.0, frames: 42, crop: "close", label: "DEVELOPER DOCS / DEVICE EVIDENCE" },
    { type: "video", source: walkthrough, start: 40.0, frames: 0, crop: "close", label: "GENERATED API / C++ + PYTHON", outro: true },
  ];
  shots.at(-1).frames = totalFrames - shots.slice(0, -1).reduce((sum, shot) => sum + shot.frames, 0);
  if (shots.at(-1).frames < 1) throw new Error(`Roadmap montage is too short: ${seconds.toFixed(3)} seconds`);

  const cropFilters = {
    wide: "scale=1920:-2,crop=1920:1080:0:(ih-oh)/2",
    close: "crop=3200:1800:(iw-ow)/2:(ih-oh)/2,scale=1920:1080",
    tool: "crop=2880:1620:(iw-ow)/2:(ih-oh)/2,scale=1920:1080",
  };
  const inputs = [];
  const filters = [];
  for (const [index, shot] of shots.entries()) {
    if (shot.type === "image") inputs.push("-loop", "1", "-i", shot.source);
    else inputs.push("-ss", shot.start.toFixed(3), "-i", shot.source);
    const effects = [
      "fps=30",
      `trim=end_frame=${shot.frames}`,
      "setpts=PTS-STARTPTS",
      shot.type === "image" ? "scale=1920:1080" : cropFilters[shot.crop],
      "setsar=1",
    ];
    if (shot.label) {
      effects.push(
        "drawbox=x=0:y=0:w=iw:h=92:color=black@0.68:t=fill",
        `drawtext=fontfile=${fontSansMedium}:text='${shot.label}':x=120:y=32:fontsize=24:fontcolor=white`,
        `drawtext=fontfile=${fontMono}:text='OA FRAMEWORK / LIVE DOCS':x=w-tw-120:y=34:fontsize=19:fontcolor=0xa3a3a3`,
      );
    }
    if (shot.flash) effects.push("fade=t=in:st=0:d=0.06:color=white");
    if (index === 0) effects.push("fade=t=in:st=0:d=0.14:color=black");
    if (shot.outro) {
      effects.push(
        "drawbox=x=0:y=h-124:w=iw:h=124:color=black@0.82:t=fill",
        `drawtext=fontfile=${fontSansMedium}:text='DEV.REALM.SOFTWARE':x=120:y=h-82:fontsize=27:fontcolor=white`,
        `drawtext=fontfile=${fontMono}:text='ARCHITECTURE · APIS · BENCHMARKS · ROADMAP':x=w-tw-120:y=h-78:fontsize=19:fontcolor=0xa3a3a3`,
        `fade=t=out:st=${Math.max(0, shot.frames / 30 - 0.22).toFixed(3)}:d=0.22:color=black`,
      );
    }
    filters.push(`[${index}:v]${effects.join(",")}[v${index}]`);
  }
  filters.push(`${shots.map((_, index) => `[v${index}]`).join("")}concat=n=${shots.length}:v=1:a=0,format=yuv420p[out]`);

  run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...inputs,
    "-filter_complex", filters.join(";"), "-map", "[out]", "-an",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", "30", output]);
}

encodeStatic("01-hook", sceneDurations[0]);
encodeStatic("02-problem", sceneDurations[1]);
encodeStatic("03-engine", sceneDurations[2]);
encodeMobile("04-proof", sceneDurations[3], 39, "FIVE TRAINING ROUTES / ONE CONTRACT", "ADRENO 610 / TURNIP 26.1.4 / FP32");
encodeStatic("05-hard", sceneDurations[4]);
encodeBreadth("06-breadth", sceneDurations[5]);
encodeStatic("07-api", sceneDurations[6]);
encodeStatic("08-codex", sceneDurations[7]);
encodeStatic("09-close", sceneDurations[8]);
encodeRoadmap("10-roadmap", sceneDurations[9]);

const sceneList = join(build, "scenes.txt");
writeFileSync(sceneList, segmentNames.map((name) => `file '${join(build, "scenes", `${name}.mp4`)}'`).join("\n") + "\n");
const visual = join(build, "visual.mp4");
run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", sceneList, "-c", "copy", visual]);

const paddedAudio = [];
for (const [index, name] of segmentNames.entries()) {
  const fast = join(build, "audio", `${name}.fast.wav`);
  const padded = join(build, "audio", `${name}.padded.wav`);
  run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-i", fast,
    "-af", `apad=pad_dur=${gap}`, "-t", sceneDurations[index].toFixed(3), "-c:a", "pcm_s24le", padded]);
  paddedAudio.push(padded);
}
const audioList = join(build, "audio.txt");
writeFileSync(audioList, paddedAudio.map((path) => `file '${path}'`).join("\n") + "\n");
const narration = join(build, "narration.wav");
run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", audioList, "-c", "copy", narration]);

function timecode(seconds) {
  const millis = Math.max(0, Math.round(seconds * 1000));
  const h = Math.floor(millis / 3600000);
  const m = Math.floor((millis % 3600000) / 60000);
  const s = Math.floor((millis % 60000) / 1000);
  const ms = millis % 1000;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}

function wrap(text, width = 76) {
  const words = text.split(/\s+/);
  const lines = [];
  let line = "";
  for (const word of words) {
    if (`${line} ${word}`.trim().length > width && line) {
      lines.push(line);
      line = word;
    } else line = `${line} ${word}`.trim();
  }
  if (line) lines.push(line);
  return lines.join("\n");
}

let cursor = 0;
let captionIndex = 1;
let srt = "";
for (const [segmentIndex, name] of segmentNames.entries()) {
  const text = readFileSync(join(narrationDir, `${name}.txt`), "utf8").trim();
  const sentences = text.match(/[^.!?]+[.!?]+/g) ?? [text];
  const captions = sentences.flatMap((sentence) => {
    const words = sentence.trim().split(/\s+/);
    const chunks = [];
    let chunk = [];
    for (const word of words) {
      const candidate = [...chunk, word].join(" ");
      if (chunk.length >= 12 || candidate.length > 68) {
        chunks.push(chunk.join(" "));
        chunk = [word];
      } else chunk.push(word);
    }
    if (chunk.length) chunks.push(chunk.join(" "));
    return chunks;
  });
  const weights = captions.map((caption) => caption.split(/\s+/).length);
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  let local = cursor;
  for (const [captionUnitIndex, caption] of captions.entries()) {
    const span = audioDurations[segmentIndex] * weights[captionUnitIndex] / totalWeight;
    srt += `${captionIndex}\n${timecode(local)} --> ${timecode(local + span)}\n${wrap(caption)}\n\n`;
    captionIndex += 1;
    local += span;
  }
  cursor += sceneDurations[segmentIndex];
}
writeFileSync(captions, srt);

const total = sceneDurations.reduce((sum, value) => sum + value, 0);
if (total >= 175) throw new Error(`Master is too long before encode: ${total.toFixed(3)} seconds`);
const audioFilter = `[1:a]loudnorm=I=-16:LRA=7:TP=-1.5[voice];aevalsrc='0.035*sin(2*PI*55*t)+0.018*sin(2*PI*82.41*t)+0.012*sin(2*PI*110*t)':s=48000:d=${total.toFixed(3)},lowpass=f=420,afade=t=in:st=0:d=2,afade=t=out:st=${Math.max(0, total - 3).toFixed(3)}:d=3[bed];[voice][bed]amix=inputs=2:duration=first:weights='1 0.45',alimiter=limit=0.92[aout]`;
run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-i", visual, "-i", narration,
  "-filter_complex", audioFilter, "-map", "0:v", "-map", "[aout]",
  "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
  "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", "-shortest", out]);

run("magick", [join(build, "frames", "poster.png"), "-resize", "1600x900", "-quality", "88", poster]);

console.log(`Rendered: ${out}`);
console.log(`Duration target: ${total.toFixed(3)} seconds`);
console.log(`Poster: ${poster}`);
console.log(`Captions: ${captions}`);
