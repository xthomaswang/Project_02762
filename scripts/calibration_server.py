#!/usr/bin/env python3
"""Web-based plate grid calibration tool.

Usage:
    conda activate webapp
    python scripts/calibration_server.py

Opens http://localhost:5050 with an interactive calibration UI.
"""

import json
import os
import platform
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CAL_DIR = ROOT / "configs" / "calibrations" / "cv_calibration"
CAL_PATH = CAL_DIR / "grid.json"
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"

app = Flask(__name__)


def _detect_backend():
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    elif system == "Linux":
        return cv2.CAP_V4L2
    elif system == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _camera_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
            return cfg.get("camera", {})
    return {}


# ── Routes ───────────────────────────────────────────────────────────


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/images")
def list_images():
    images = []
    search_dirs = [DATA_DIR, CAL_DIR / "source"]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in d.rglob(ext):
                if p.name.startswith("."):
                    continue
                images.append(str(p.relative_to(ROOT)))
    images.sort()
    return jsonify(images)


@app.route("/api/image/<path:filepath>")
def serve_image(filepath):
    full = ROOT / filepath
    if not full.exists():
        return "Not found", 404
    return send_file(full)


@app.route("/api/calibration", methods=["GET"])
def get_calibration():
    if CAL_PATH.exists():
        with open(CAL_PATH) as f:
            return jsonify(json.load(f))
    return jsonify(None)


@app.route("/api/calibration", methods=["POST"])
def save_calibration():
    data = request.json
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAL_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"ok": True, "path": str(CAL_PATH)})


@app.route("/api/cameras")
def list_cameras():
    cam_cfg = _camera_config()
    expected_id = cam_cfg.get("device_id", 0)
    backend = _detect_backend()
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append({
                "id": i, "width": w, "height": h,
                "selected": i == expected_id,
            })
            cap.release()
        else:
            cap.release()
    return jsonify(cameras)


@app.route("/api/capture", methods=["POST"])
def capture_image():
    data = request.json or {}
    camera_id = data.get("camera_id")

    cam_cfg = _camera_config()
    if camera_id is None:
        camera_id = cam_cfg.get("device_id", 0)

    width = cam_cfg.get("width", 1920)
    height = cam_cfg.get("height", 1080)
    warmup = cam_cfg.get("warmup_frames", 10)

    backend = _detect_backend()
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        return jsonify({"ok": False, "error": f"Cannot open camera {camera_id}"}), 500

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(warmup):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return jsonify({"ok": False, "error": "Capture failed"}), 500

    source_dir = CAL_DIR / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = source_dir / filename
    cv2.imwrite(str(filepath), frame)

    rel_path = str(filepath.relative_to(ROOT))
    return jsonify({"ok": True, "path": rel_path})


# ── Inline HTML/CSS/JS ──────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Plate Grid Calibration</title>
<style>
:root {
  --bg: #0f0f1a;
  --surface: #1a1a2e;
  --border: #2a2a4a;
  --accent: #6c63ff;
  --accent-hover: #7f78ff;
  --green: #44ff88;
  --red: #ff4466;
  --yellow: #ffaa33;
  --cyan: #33ddff;
  --text: #e8e8f0;
  --text-dim: #8888aa;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text);
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}

/* ── Toolbar ── */
.toolbar {
  display: flex; align-items: center; gap: 10px; padding: 10px 16px;
  background: var(--surface); border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}
.toolbar label { font-size: 13px; color: var(--text-dim); }
.toolbar select, .toolbar input[type=range] {
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; font-size: 13px; outline: none;
}
.toolbar select { max-width: 300px; }
.toolbar input[type=range] { width: 120px; }
.btn {
  padding: 7px 16px; border: none; border-radius: 6px;
  font-size: 13px; font-weight: 600; cursor: pointer; transition: all .15s;
}
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-mode {
  background: var(--bg); color: var(--text-dim); border: 1px solid var(--border);
}
.btn-mode.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.btn-save { background: var(--green); color: #111; }
.btn-save:hover:not(:disabled) { filter: brightness(1.1); }
.btn-reset { background: var(--red); color: #fff; }
.btn-reset:hover:not(:disabled) { filter: brightness(1.1); }
.btn-load { background: var(--accent); color: #fff; }
.btn-load:hover:not(:disabled) { background: var(--accent-hover); }
.btn-capture { background: var(--cyan); color: #111; }
.btn-capture:hover:not(:disabled) { filter: brightness(1.1); }
.separator { width: 1px; height: 28px; background: var(--border); }

/* ── Canvas ── */
.canvas-wrap {
  flex: 1; position: relative; overflow: hidden; background: #000;
}
canvas { display: block; cursor: crosshair; }

/* ── Status bar ── */
.status {
  display: flex; align-items: center; gap: 20px;
  padding: 8px 20px; background: var(--surface);
  border-top: 1px solid var(--border); font-size: 13px;
}
.status .coords { color: var(--text-dim); min-width: 200px; }
.status .message { color: var(--green); }

/* ── Toast ── */
.toast {
  position: fixed; top: 20px; right: 20px;
  padding: 12px 24px; border-radius: 8px;
  font-size: 14px; font-weight: 600;
  opacity: 0; transition: opacity .3s; pointer-events: none; z-index: 100;
}
.toast.show { opacity: 1; }
.toast.success { background: var(--green); color: #111; }
.toast.error { background: var(--red); color: #fff; }

/* ── Spinner ── */
.spinner {
  display: inline-block; width: 14px; height: 14px;
  border: 2px solid rgba(0,0,0,.2); border-top-color: #111;
  border-radius: 50%; animation: spin .6s linear infinite;
  vertical-align: middle; margin-right: 6px;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="toolbar">
  <label>Image:</label>
  <select id="image-select"><option value="">-- select plate image --</option></select>
  <button class="btn btn-capture" id="btn-capture" onclick="captureImage()">Capture</button>
  <div class="separator"></div>
  <button class="btn btn-mode active" id="btn-corners" onclick="setMode('corners')">4-Corner</button>
  <button class="btn btn-mode" id="btn-drag" onclick="setMode('drag')">Drag</button>
  <div class="separator"></div>
  <label>ROI:</label>
  <input type="range" id="roi-slider" min="5" max="80" value="34">
  <span id="roi-value">34</span>
  <div class="separator"></div>
  <button class="btn btn-load" onclick="loadCalibration()">Load Existing</button>
  <button class="btn btn-save" onclick="saveCalibration()">Save</button>
  <button class="btn btn-reset" onclick="resetAll()">Reset</button>
</div>

<div class="canvas-wrap">
  <canvas id="canvas"></canvas>
</div>

<div class="status">
  <span class="coords" id="coords">--</span>
  <span class="message" id="status-msg">Select a plate image or click Capture to take a photo</span>
</div>

<div class="toast" id="toast"></div>

<script>
// ── Constants ──
const ROWS = 'ABCDEFGH';
const N_ROWS = 8, N_COLS = 12;
const CORNER_KEYS = ['a1', 'a12', 'h1', 'h12'];
const CORNER_WELLS = ['A1', 'A12', 'H1', 'H12'];
const CORNER_HINTS = ['A1 (top-left)', 'A12 (top-right)', 'H1 (bottom-left)', 'H12 (bottom-right)'];

// ── State ──
let mode = 'corners';
let cornerStep = 0;
let corners = { a1: null, a12: null, h1: null, h12: null };
let wells = {};
let adjustedWells = new Set();
let roiRadius = 34;
let plateImage = null;
let imgW = 0, imgH = 0;
let scale = 1, offX = 0, offY = 0;
let dragTarget = null;
let hoverTarget = null;
let mouseImgX = 0, mouseImgY = 0;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const imageSelect = document.getElementById('image-select');
const roiSlider = document.getElementById('roi-slider');
const roiValue = document.getElementById('roi-value');
const coordsEl = document.getElementById('coords');
const statusEl = document.getElementById('status-msg');

// ── Bilinear interpolation ──
function bilinear(a1, a12, h1, h12, u, v) {
  return {
    x: (1-u)*(1-v)*a1.x + u*(1-v)*a12.x + (1-u)*v*h1.x + u*v*h12.x,
    y: (1-u)*(1-v)*a1.y + u*(1-v)*a12.y + (1-u)*v*h1.y + u*v*h12.y,
  };
}

function computeGrid() {
  const { a1, a12, h1, h12 } = corners;
  if (!a1 || !a12 || !h1 || !h12) return;
  for (let r = 0; r < N_ROWS; r++) {
    for (let c = 0; c < N_COLS; c++) {
      const label = ROWS[r] + (c + 1);
      if (adjustedWells.has(label)) continue;
      wells[label] = bilinear(a1, a12, h1, h12, c / 11, r / 7);
    }
  }
}

// ── Coordinate transforms ──
function imgToCanvas(ix, iy) { return { x: ix * scale + offX, y: iy * scale + offY }; }
function canvasToImg(cx, cy) { return { x: (cx - offX) / scale, y: (cy - offY) / scale }; }

function fitImage() {
  const wrap = canvas.parentElement;
  const ww = wrap.clientWidth, wh = wrap.clientHeight;
  canvas.width = ww; canvas.height = wh;
  if (!plateImage) return;
  scale = Math.min(ww / imgW, wh / imgH);
  offX = (ww - imgW * scale) / 2;
  offY = (wh - imgH * scale) / 2;
}

// ── Rendering ──
function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (plateImage) {
    ctx.drawImage(plateImage, offX, offY, imgW * scale, imgH * scale);
  }

  const r = roiRadius * scale;
  for (const [label, pos] of Object.entries(wells)) {
    const { x, y } = imgToCanvas(pos.x, pos.y);
    const isCorner = CORNER_WELLS.includes(label);
    const isAdj = adjustedWells.has(label);
    const isHover = label === hoverTarget;

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = isCorner ? '#ff4466' : isAdj ? '#ffaa33' : '#44ff88';
    ctx.lineWidth = isHover ? 2.5 : 1.2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, isHover ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = isCorner ? '#ff4466' : isAdj ? '#ffaa33' : '#44ff88';
    ctx.fill();

    const showLabel = isCorner || isHover || isAdj;
    if (showLabel) {
      const fs = Math.max(11, 13 * scale);
      ctx.font = `bold ${fs}px sans-serif`;
      ctx.fillStyle = '#fff';
      ctx.shadowColor = '#000';
      ctx.shadowBlur = 4;
      ctx.fillText(label, x + 8, y - r - 4);
      ctx.shadowBlur = 0;
    }
  }

  if (mode === 'corners') {
    for (let i = 0; i < cornerStep; i++) {
      const c = corners[CORNER_KEYS[i]];
      if (!c) continue;
      const { x, y } = imgToCanvas(c.x, c.y);
      ctx.strokeStyle = '#ff4466';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x - 15, y); ctx.lineTo(x + 15, y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x, y - 15); ctx.lineTo(x, y + 15); ctx.stroke();
      ctx.font = 'bold 14px sans-serif';
      ctx.fillStyle = '#ff4466';
      ctx.shadowColor = '#000'; ctx.shadowBlur = 4;
      ctx.fillText(CORNER_KEYS[i].toUpperCase(), x + 14, y - 8);
      ctx.shadowBlur = 0;
    }
  }

  requestAnimationFrame(render);
}

// ── Hit detection ──
function findNearestWell(imgX, imgY, maxDist) {
  let best = null, bestDist = maxDist;
  for (const [label, pos] of Object.entries(wells)) {
    const d = Math.hypot(pos.x - imgX, pos.y - imgY);
    if (d < bestDist) { bestDist = d; best = label; }
  }
  return best;
}

// ── Mouse events ──
canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const img = canvasToImg(cx, cy);
  mouseImgX = img.x; mouseImgY = img.y;
  coordsEl.textContent = `x: ${Math.round(img.x)}  y: ${Math.round(img.y)}`;

  if (mode === 'drag' && dragTarget) {
    const pos = { x: img.x, y: img.y };
    wells[dragTarget] = pos;
    const ci = CORNER_WELLS.indexOf(dragTarget);
    if (ci >= 0) {
      corners[CORNER_KEYS[ci]] = pos;
      adjustedWells.clear();
      computeGrid();
    } else {
      adjustedWells.add(dragTarget);
    }
    return;
  }

  const hitRadius = roiRadius * 1.2;
  hoverTarget = findNearestWell(img.x, img.y, hitRadius);
  canvas.style.cursor = (mode === 'drag' && hoverTarget) ? 'grab' : 'crosshair';
});

canvas.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const img = canvasToImg(cx, cy);

  if (mode === 'corners') {
    if (cornerStep < 4) {
      corners[CORNER_KEYS[cornerStep]] = { x: img.x, y: img.y };
      cornerStep++;
      if (cornerStep < 4) {
        statusEl.textContent = `Click well ${CORNER_HINTS[cornerStep]}`;
      } else {
        computeGrid();
        const topDist = Math.hypot(corners.a12.x - corners.a1.x, corners.a12.y - corners.a1.y) / 11;
        const leftDist = Math.hypot(corners.h1.x - corners.a1.x, corners.h1.y - corners.a1.y) / 7;
        roiRadius = Math.min(topDist, leftDist) * 0.35;
        roiSlider.value = Math.round(roiRadius);
        roiValue.textContent = Math.round(roiRadius);
        setMode('drag');
        statusEl.textContent = 'Grid computed! Drag wells to adjust, then Save.';
      }
    }
    return;
  }

  if (mode === 'drag') {
    const hit = findNearestWell(img.x, img.y, roiRadius * 1.2);
    if (hit) {
      dragTarget = hit;
      canvas.style.cursor = 'grabbing';
    }
  }
});

canvas.addEventListener('mouseup', () => {
  if (dragTarget) {
    dragTarget = null;
    canvas.style.cursor = 'grab';
  }
});

canvas.addEventListener('mouseleave', () => {
  hoverTarget = null;
  coordsEl.textContent = '--';
});

// ── ROI slider ──
roiSlider.addEventListener('input', () => {
  roiRadius = parseInt(roiSlider.value);
  roiValue.textContent = roiRadius;
});

// ── Mode switch ──
function setMode(m) {
  mode = m;
  document.getElementById('btn-corners').classList.toggle('active', m === 'corners');
  document.getElementById('btn-drag').classList.toggle('active', m === 'drag');
  if (m === 'corners' && cornerStep >= 4) {
    cornerStep = 0;
    corners = { a1: null, a12: null, h1: null, h12: null };
    wells = {};
    adjustedWells.clear();
    statusEl.textContent = `Click well ${CORNER_HINTS[0]}`;
  } else if (m === 'corners') {
    statusEl.textContent = `Click well ${CORNER_HINTS[cornerStep]}`;
  } else {
    statusEl.textContent = Object.keys(wells).length
      ? 'Drag any well to adjust. Corner drags recompute the grid.'
      : 'Set 4 corners first (switch to 4-Corner mode).';
  }
}

// ── Reset ──
function resetAll() {
  cornerStep = 0;
  corners = { a1: null, a12: null, h1: null, h12: null };
  wells = {};
  adjustedWells.clear();
  roiRadius = 34;
  roiSlider.value = 34;
  roiValue.textContent = '34';
  setMode('corners');
}

// ── Toast ──
function toast(msg, type = 'success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast show ${type}`;
  setTimeout(() => el.classList.remove('show'), 2500);
}

// ── Load image onto canvas ──
function loadImage(path) {
  const img = new Image();
  img.onload = () => {
    plateImage = img;
    imgW = img.naturalWidth; imgH = img.naturalHeight;
    fitImage();
    if (cornerStep === 0) {
      statusEl.textContent = `Image loaded (${imgW}x${imgH}). Click well ${CORNER_HINTS[0]}`;
    }
  };
  img.src = `/api/image/${path}`;
}

// ── Camera capture ──
async function captureImage() {
  const btn = document.getElementById('btn-capture');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Capturing...';
  statusEl.textContent = 'Capturing from camera...';

  try {
    const res = await fetch('/api/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const result = await res.json();

    if (!result.ok) {
      toast(result.error || 'Capture failed', 'error');
      statusEl.textContent = result.error || 'Capture failed';
      return;
    }

    // Add to dropdown and select it
    const opt = document.createElement('option');
    opt.value = result.path;
    opt.textContent = result.path;
    imageSelect.appendChild(opt);
    imageSelect.value = result.path;

    // Load onto canvas
    loadImage(result.path);
    toast('Photo captured!');
    statusEl.textContent = `Captured: ${result.path}. Click well ${CORNER_HINTS[cornerStep] || CORNER_HINTS[0]}`;
  } catch (err) {
    toast('Capture request failed', 'error');
    statusEl.textContent = 'Capture request failed — check server logs';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Capture';
  }
}

// ── API: save ──
async function saveCalibration() {
  const { a1, a12, h1, h12 } = corners;
  if (!a1 || !a12 || !h1 || !h12) {
    toast('Set all 4 corners first', 'error');
    return;
  }
  const data = {
    origin_x: a1.x,
    origin_y: a1.y,
    spacing_x: (a12.x - a1.x) / 11,
    spacing_y: (h1.y - a1.y) / 7,
    roi_radius: roiRadius,
    corners: {
      a1: [a1.x, a1.y],
      a12: [a12.x, a12.y],
      h1: [h1.x, h1.y],
      h12: [h12.x, h12.y],
    }
  };
  const res = await fetch('/api/calibration', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  const result = await res.json();
  if (result.ok) {
    toast('Calibration saved!');
    statusEl.textContent = `Saved to ${result.path}`;
  } else {
    toast('Save failed', 'error');
  }
}

// ── API: load existing ──
async function loadCalibration() {
  const res = await fetch('/api/calibration');
  const data = await res.json();
  if (!data) {
    toast('No existing calibration found', 'error');
    return;
  }
  roiRadius = data.roi_radius || 34;
  roiSlider.value = Math.round(roiRadius);
  roiValue.textContent = Math.round(roiRadius);

  if (data.corners) {
    corners.a1  = { x: data.corners.a1[0],  y: data.corners.a1[1] };
    corners.a12 = { x: data.corners.a12[0], y: data.corners.a12[1] };
    corners.h1  = { x: data.corners.h1[0],  y: data.corners.h1[1] };
    corners.h12 = { x: data.corners.h12[0], y: data.corners.h12[1] };
  } else {
    const ox = data.origin_x, oy = data.origin_y;
    const sx = data.spacing_x, sy = data.spacing_y;
    corners.a1  = { x: ox, y: oy };
    corners.a12 = { x: ox + 11 * sx, y: oy };
    corners.h1  = { x: ox, y: oy + 7 * sy };
    corners.h12 = { x: ox + 11 * sx, y: oy + 7 * sy };
  }

  cornerStep = 4;
  adjustedWells.clear();
  computeGrid();
  setMode('drag');
  toast('Calibration loaded');
  statusEl.textContent = 'Loaded existing calibration. Drag to adjust, then Save.';
}

// ── API: load images ──
async function loadImageList() {
  const res = await fetch('/api/images');
  const images = await res.json();
  imageSelect.innerHTML = '<option value="">-- select plate image --</option>';
  for (const img of images) {
    const opt = document.createElement('option');
    opt.value = img; opt.textContent = img;
    imageSelect.appendChild(opt);
  }
}

imageSelect.addEventListener('change', () => {
  const path = imageSelect.value;
  if (!path) return;
  loadImage(path);
});

// ── Resize ──
window.addEventListener('resize', () => { fitImage(); });

// ── Init ──
loadImageList();
fitImage();
requestAnimationFrame(render);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"\n  Calibration tool: http://localhost:{port}\n")
    webbrowser.open(f"http://localhost:{port}")
    app.run(port=port, debug=False)
