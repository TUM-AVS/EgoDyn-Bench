"""Local web viewer for CARLA clips: original, Cosmos-transferred, and QA ground truth.

Usage:
    python scripts/clip_viewer.py
    python scripts/clip_viewer.py --port 8080
    python scripts/clip_viewer.py --selected_clips selected_clips.json

Then open http://localhost:5050 in your browser.

Original CARLA videos use FMP4 codec (not browser-playable). The viewer
automatically transcodes them to H.264 on first access and caches the result
in output/carla_chunks_h264/ for instant replay on subsequent loads.
"""

import argparse
import io
import json
import os
import shutil
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults
DEFAULT_CARLA_CHUNKS = PROJECT_ROOT / "output" / "carla_chunks"
# Override with EGODYN_CARLA_TRANSFERRED_DIR or pass --transferred-dir explicitly.
DEFAULT_TRANSFERRED = Path(
    os.environ.get("EGODYN_CARLA_TRANSFERRED_DIR", "./data/carla/benchmark_transferred")
)
DEFAULT_CARLA_QA = PROJECT_ROOT / "output" / "carla_clips" / "qa.jsonl"
DEFAULT_CARLA_INDEX = PROJECT_ROOT / "output" / "carla_clips" / "clips_index.jsonl"
DEFAULT_SELECTED = PROJECT_ROOT / "selected_clips.json"
DEFAULT_H264_CACHE = PROJECT_ROOT / "output" / "carla_chunks_h264"


def load_data(args):
    """Load clips, QA, and build the viewer data structure."""
    # Load selected clip IDs (CARLA only)
    selected_ids = set()
    if args.selected_clips and Path(args.selected_clips).exists():
        with open(args.selected_clips) as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    if item.get("source") == "carla":
                        selected_ids.add(item["id"])
                else:
                    selected_ids.add(item)

    # Load clips index for features
    clips_index = {}
    idx_path = Path(args.carla_index)
    if idx_path.exists():
        with open(idx_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    clips_index[rec["clip_id"]] = rec

    # Load QA ground truth grouped by clip
    qa_by_clip: dict[str, list[dict]] = {}
    qa_path = Path(args.carla_qa)
    if qa_path.exists():
        with open(qa_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                qa = json.loads(line)
                qa_by_clip.setdefault(qa["clip_id"], []).append(qa)

    # Build clip list
    carla_chunks = Path(args.carla_chunks)
    transferred = Path(args.transferred)
    h264_cache = Path(args.h264_cache)

    clips = []
    clip_ids = sorted(selected_ids) if selected_ids else sorted(
        p.stem for p in carla_chunks.glob("*.mp4")
    )

    for clip_id in clip_ids:
        original = carla_chunks / f"{clip_id}.mp4"
        cosmos = transferred / f"{clip_id}.mp4"
        depth = transferred / f"{clip_id}_control_depth.mp4"

        if not original.exists() and not cosmos.exists():
            continue

        qa_items = qa_by_clip.get(clip_id, [])
        clip_rec = clips_index.get(clip_id, {})
        features = clip_rec.get("features", {})
        array_ref = clip_rec.get("array_ref")

        clips.append({
            "id": clip_id,
            "original": str(original) if original.exists() else None,
            "cosmos": str(cosmos) if cosmos.exists() else None,
            "depth": str(depth) if depth.exists() else None,
            "qa": qa_items,
            "features": features,
            "array_ref": str(idx_path.parent / array_ref) if array_ref else None,
        })

    return clips, h264_cache


def build_html(clips):
    """Generate the single-page HTML app."""
    # Serialize clip data (strip full paths for JSON, keep for serving)
    clips_json = json.dumps([
        {
            "id": c["id"],
            "has_original": c["original"] is not None,
            "has_cosmos": c["cosmos"] is not None,
            "has_depth": c["depth"] is not None,
            "qa": [
                {
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "answer": q["answer"],
                    "category": q.get("category", ""),
                    "choices": q.get("choices"),
                }
                for q in c["qa"]
            ],
            "features": c["features"],
        }
        for c in clips
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CARLA Clip Viewer — EgoDyn-Bench</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; }}
.container {{ display: flex; height: 100vh; }}

/* Sidebar */
.sidebar {{ width: 320px; min-width: 320px; background: #16213e; overflow-y: auto; border-right: 1px solid #333; }}
.sidebar h2 {{ padding: 16px; font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #333; position: sticky; top: 0; background: #16213e; z-index: 1; }}
.clip-item {{ padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #1a1a2e; font-size: 13px; font-family: monospace; transition: background 0.15s; }}
.clip-item:hover {{ background: #1a1a3e; }}
.clip-item.active {{ background: #0f3460; border-left: 3px solid #e37222; }}
.clip-item .badges {{ margin-top: 4px; }}
.clip-item .badge {{ display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 3px; margin-right: 4px; }}
.badge-orig {{ background: #0065bd; color: white; }}
.badge-cosmos {{ background: #e37222; color: white; }}
.badge-depth {{ background: #555; color: white; }}

/* Search */
.search-box {{ padding: 12px 16px; border-bottom: 1px solid #333; position: sticky; top: 42px; background: #16213e; z-index: 1; }}
.search-box input {{ width: 100%; padding: 8px 10px; background: #1a1a2e; border: 1px solid #333; color: #e0e0e0; border-radius: 4px; font-size: 13px; }}

/* Main content */
.main {{ flex: 1; overflow-y: auto; padding: 24px; }}
.clip-header {{ margin-bottom: 20px; }}
.clip-header h1 {{ font-size: 18px; font-family: monospace; color: #e37222; }}
.clip-header .nav {{ margin-top: 8px; }}
.clip-header .nav button {{ background: #0f3460; color: #e0e0e0; border: none; padding: 6px 14px; margin-right: 8px; cursor: pointer; border-radius: 4px; font-size: 13px; }}
.clip-header .nav button:hover {{ background: #1a5276; }}

/* Videos */
.videos {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
.video-box {{ flex: 1; min-width: 300px; }}
.video-box h3 {{ font-size: 13px; color: #888; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
.video-box video {{ width: 100%; border-radius: 6px; background: #000; }}

/* QA table */
.qa-section h2 {{ font-size: 15px; margin-bottom: 12px; color: #aaa; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; padding: 8px 12px; background: #16213e; color: #888; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #222; }}
tr:hover {{ background: #1a1a3e; }}
.answer {{ font-weight: 700; color: #e37222; }}
.category {{ color: #666; font-size: 11px; }}
.choices {{ color: #555; font-size: 11px; }}

/* Features */
.features {{ margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 8px; }}
.feat {{ background: #16213e; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-family: monospace; }}
.feat-name {{ color: #888; }}
.feat-val {{ color: #e37222; }}

.empty {{ color: #555; font-style: italic; padding: 40px; text-align: center; }}

/* Dynamics dashboard */
.dynamics-dashboard {{ margin-bottom: 24px; }}
.dynamics-dashboard h2 {{ font-size: 15px; margin-bottom: 12px; color: #aaa; }}
.chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.chart-box {{ background: #16213e; border-radius: 6px; padding: 12px; position: relative; }}
.chart-box h4 {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
.chart-box canvas {{ width: 100%; height: 140px; display: block; }}
.chart-loading {{ color: #555; font-size: 12px; font-style: italic; text-align: center; padding: 20px 0; }}
.chart-cursor {{ position: absolute; top: 28px; width: 2px; height: 140px; background: rgba(255,255,255,0.7); pointer-events: none; display: none; z-index: 2; transition: left 0.06s linear; }}
.chart-cursor-dot {{ position: absolute; top: -4px; left: -3px; width: 8px; height: 8px; border-radius: 50%; border: 1.5px solid rgba(255,255,255,0.9); }}
.chart-cursor-val {{ position: absolute; top: -18px; left: 4px; font-size: 10px; color: #fff; font-family: monospace; white-space: nowrap; text-shadow: 0 1px 3px rgba(0,0,0,0.8); }}
.video-time-bar {{ background: #16213e; border-radius: 6px; padding: 8px 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 12px; }}
.video-time-bar .time-label {{ font-size: 12px; color: #888; font-family: monospace; min-width: 80px; }}
.video-time-bar .time-track {{ flex: 1; height: 4px; background: #0d1b2a; border-radius: 2px; position: relative; cursor: pointer; }}
.video-time-bar .time-fill {{ height: 100%; background: #e37222; border-radius: 2px; width: 0%; transition: width 0.06s linear; }}
.video-time-bar .time-thumb {{ position: absolute; top: -5px; width: 14px; height: 14px; background: #e37222; border: 2px solid #fff; border-radius: 50%; transform: translateX(-50%); left: 0%; transition: left 0.06s linear; cursor: grab; }}
</style>
</head>
<body>
<div class="container">
  <div class="sidebar">
    <h2 id="sidebar-title">Clips (0)</h2>
    <div class="search-box"><input type="text" id="search" placeholder="Filter clips..." /></div>
    <div id="clip-list"></div>
  </div>
  <div class="main" id="main">
    <div class="empty">Select a clip from the sidebar</div>
  </div>
</div>

<script>
const CLIPS = {clips_json};
let currentIdx = -1;

const listEl = document.getElementById('clip-list');
const mainEl = document.getElementById('main');
const searchEl = document.getElementById('search');
const titleEl = document.getElementById('sidebar-title');

function renderList(filter) {{
  listEl.innerHTML = '';
  const f = (filter || '').toLowerCase();
  let count = 0;
  CLIPS.forEach((clip, idx) => {{
    if (f && !clip.id.toLowerCase().includes(f)) return;
    count++;
    const div = document.createElement('div');
    div.className = 'clip-item' + (idx === currentIdx ? ' active' : '');
    div.dataset.idx = idx;
    let badges = '';
    if (clip.has_original) badges += '<span class="badge badge-orig">CARLA</span>';
    if (clip.has_cosmos) badges += '<span class="badge badge-cosmos">Cosmos</span>';
    if (clip.has_depth) badges += '<span class="badge badge-depth">Depth</span>';
    div.innerHTML = clip.id + '<div class="badges">' + badges + '</div>';
    div.onclick = () => selectClip(idx);
    listEl.appendChild(div);
  }});
  titleEl.textContent = 'Clips (' + count + ')';
}}

function selectClip(idx) {{
  currentIdx = idx;
  const clip = CLIPS[idx];
  renderList(searchEl.value);

  const eid = encodeURIComponent(clip.id);
  function videoWithFallback(label, type) {{
    return `<div class="video-box"><h3>${{label}}</h3>
      <video controls loop muted autoplay src="/video/${{eid}}/${{type}}"
        onerror="this.nextElementSibling.style.display='block';"
      ></video>
      <div style="display:none; color:#888; font-size:12px; padding:8px;">Video codec not supported. Install ffmpeg for auto-transcoding.</div>
    </div>`;
  }}
  let videosHtml = '<div class="videos">';
  if (clip.has_original) videosHtml += videoWithFallback('Original CARLA', 'original');
  if (clip.has_cosmos) videosHtml += videoWithFallback('Cosmos Transferred', 'cosmos');
  if (clip.has_depth) videosHtml += videoWithFallback('Depth Control', 'depth');
  videosHtml += '</div>';

  // Features
  let featHtml = '';
  if (clip.features && Object.keys(clip.features).length > 0) {{
    featHtml = '<div class="features">';
    for (const [k, v] of Object.entries(clip.features)) {{
      const val = typeof v === 'number' ? v.toFixed(3) : v;
      featHtml += `<div class="feat"><span class="feat-name">${{k}}</span> <span class="feat-val">${{val}}</span></div>`;
    }}
    featHtml += '</div>';
  }}

  // QA table
  let qaHtml = '';
  if (clip.qa.length > 0) {{
    qaHtml = '<div class="qa-section"><h2>Ground Truth QA (' + clip.qa.length + ' questions)</h2>';
    qaHtml += '<table><tr><th>Question ID</th><th>Question</th><th>Answer</th><th>Choices</th><th>Category</th></tr>';
    for (const q of clip.qa) {{
      const choices = q.choices ? q.choices.join(' / ') : '';
      qaHtml += `<tr>
        <td style="font-family:monospace;font-size:12px">${{q.question_id}}</td>
        <td>${{q.question}}</td>
        <td class="answer">${{q.answer}}</td>
        <td class="choices">${{choices}}</td>
        <td class="category">${{q.category}}</td>
      </tr>`;
    }}
    qaHtml += '</table></div>';
  }}

  // Nav
  const navHtml = `<div class="nav">
    <button onclick="selectClip(${{Math.max(0, idx - 1)}})">← Previous</button>
    <button onclick="selectClip(${{Math.min(CLIPS.length - 1, idx + 1)}})">Next →</button>
    <span style="color:#666;font-size:12px;margin-left:8px">${{idx + 1}} / ${{CLIPS.length}}</span>
  </div>`;

  mainEl.innerHTML = `
    <div class="clip-header">
      <h1>${{clip.id}}</h1>
      ${{navHtml}}
    </div>
    ${{videosHtml}}
    <div class="video-time-bar" id="video-time-bar" style="display:none;">
      <span class="time-label" id="time-current">0.00s / 3.00s</span>
      <div class="time-track" id="time-track">
        <div class="time-fill" id="time-fill"></div>
        <div class="time-thumb" id="time-thumb"></div>
      </div>
    </div>
    <div class="dynamics-dashboard" id="dynamics-dashboard">
      <h2>Dynamics Overview</h2>
      <div class="chart-loading">Loading time-series data&hellip;</div>
    </div>
    ${{featHtml}}
    ${{qaHtml}}
  `;

  // Load time-series data and render charts
  loadTimeseries(clip.id);

  // Wire up video-to-chart synchronization
  initVideoSync();
}}

/* ---- Chart state for cursor sync ---- */
const chartState = {{ keys: [], data: null, tMin: 0, tMax: 3, canvasMap: {{}} }};

/* ---- Chart rendering ---- */
const CHART_COLORS = {{
  speed: '#4ecdc4',
  accel: '#e37222',
  yaw_rate: '#0065bd',
  jerk: '#ff6b6b',
  lateral_accel: '#a2ad00',
}};

const CHART_LABELS = {{
  speed: 'Speed (m/s)',
  accel: 'Longitudinal Accel. (m/s²)',
  yaw_rate: 'Yaw Rate (rad/s)',
  jerk: 'Jerk (m/s³)',
  lateral_accel: 'Lateral Accel. (m/s²)',
}};

function drawChart(canvas, timestamps, values, color, label) {{
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  const pad = {{ top: 8, right: 12, bottom: 22, left: 42 }};
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  // Compute range
  let yMin = Math.min(...values);
  let yMax = Math.max(...values);
  if (yMin === yMax) {{ yMin -= 1; yMax += 1; }}
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad; yMax += yPad;

  const tMin = timestamps[0], tMax = timestamps[timestamps.length - 1];

  function toX(t) {{ return pad.left + (t - tMin) / (tMax - tMin) * pw; }}
  function toY(v) {{ return pad.top + (1 - (v - yMin) / (yMax - yMin)) * ph; }}

  // Background
  ctx.fillStyle = '#0d1b2a';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#1e3045';
  ctx.lineWidth = 1;
  const nGridY = 4;
  for (let i = 0; i <= nGridY; i++) {{
    const v = yMin + (yMax - yMin) * i / nGridY;
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
  }}

  // Zero line if range spans zero
  if (yMin < 0 && yMax > 0) {{
    ctx.strokeStyle = '#334466';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    const y0 = toY(0);
    ctx.beginPath(); ctx.moveTo(pad.left, y0); ctx.lineTo(W - pad.right, y0); ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Data line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (let i = 0; i < timestamps.length; i++) {{
    const x = toX(timestamps[i]), y = toY(values[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  // Fill under curve
  ctx.globalAlpha = 0.1;
  ctx.fillStyle = color;
  ctx.lineTo(toX(timestamps[timestamps.length - 1]), toY(yMin));
  ctx.lineTo(toX(timestamps[0]), toY(yMin));
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1.0;

  // Axis labels
  ctx.fillStyle = '#667';
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= nGridY; i++) {{
    const v = yMin + (yMax - yMin) * i / nGridY;
    ctx.fillText(v.toFixed(1), pad.left - 4, toY(v) + 3);
  }}
  ctx.textAlign = 'center';
  ctx.fillText('0s', toX(tMin), H - 3);
  ctx.fillText(tMax.toFixed(1) + 's', toX(tMax), H - 3);
  if (tMax > 1.5) ctx.fillText((tMax / 2).toFixed(1) + 's', toX((tMin + tMax) / 2), H - 3);
}}

function loadTimeseries(clipId) {{
  const dashboard = document.getElementById('dynamics-dashboard');
  if (!dashboard) return;

  fetch('/timeseries/' + encodeURIComponent(clipId))
    .then(r => {{
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }})
    .then(data => {{
      if (!data.timestamps || data.timestamps.length === 0) {{
        dashboard.innerHTML = '<h2>Dynamics Overview</h2><div class="chart-loading">No time-series data available</div>';
        return;
      }}

      const keys = ['speed', 'accel', 'yaw_rate', 'jerk', 'lateral_accel'].filter(k => data[k]);
      // Use 3-column grid for 5 charts, 2-column for fewer
      const cols = keys.length >= 5 ? 3 : 2;
      let html = '<h2>Dynamics Overview</h2>';
      html += `<div class="chart-grid" style="grid-template-columns: repeat(${{cols}}, 1fr);">`;
      for (const key of keys) {{
        html += `<div class="chart-box" id="chartbox-${{key}}"><h4>${{CHART_LABELS[key] || key}}</h4><canvas id="chart-${{key}}"></canvas><div class="chart-cursor" id="cursor-${{key}}"><div class="chart-cursor-dot" style="background:${{CHART_COLORS[key] || '#888'}};"></div><div class="chart-cursor-val" id="cursorval-${{key}}"></div></div></div>`;
      }}
      html += '</div>';
      dashboard.innerHTML = html;

      // Store chart layout info for cursor positioning
      chartState.keys = keys;
      chartState.data = data;
      chartState.tMin = data.timestamps[0];
      chartState.tMax = data.timestamps[data.timestamps.length - 1];

      // Render charts after DOM update
      requestAnimationFrame(() => {{
        for (const key of keys) {{
          const canvas = document.getElementById('chart-' + key);
          if (canvas) {{
            drawChart(canvas, data.timestamps, data[key], CHART_COLORS[key] || '#888', key);
            // Store pixel mapping for cursor (pad.left and pad.right match drawChart)
            const rect = canvas.getBoundingClientRect();
            const padL = 42, padR = 12;
            chartState.canvasMap = chartState.canvasMap || {{}};
            chartState.canvasMap[key] = {{ left: padL, plotWidth: rect.width - padL - padR, cssWidth: rect.width }};
          }}
        }}
      }});
    }})
    .catch(() => {{
      dashboard.innerHTML = '<h2>Dynamics Overview</h2><div class="chart-loading">No time-series data available</div>';
    }});
}}

/* ---- Video-to-chart synchronization ---- */
function initVideoSync() {{
  const videos = mainEl.querySelectorAll('video');
  const timeBar = document.getElementById('video-time-bar');
  if (videos.length === 0) return;

  // Show the time bar
  if (timeBar) timeBar.style.display = 'flex';

  // Listen to timeupdate on all videos, use the first one that fires
  videos.forEach(v => {{
    v.addEventListener('timeupdate', () => {{
      const t = v.currentTime;
      const dur = v.duration || 3.0;
      updateCursors(t, dur);
      updateTimeBar(t, dur);
    }});
  }});

  // Scrubbing on the time bar track
  const track = document.getElementById('time-track');
  if (track) {{
    let dragging = false;

    function scrubTo(e) {{
      const rect = track.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      videos.forEach(v => {{
        if (v.duration) v.currentTime = frac * v.duration;
      }});
    }}

    track.addEventListener('mousedown', (e) => {{
      dragging = true;
      scrubTo(e);
    }});
    document.addEventListener('mousemove', (e) => {{
      if (dragging) scrubTo(e);
    }});
    document.addEventListener('mouseup', () => {{ dragging = false; }});
  }}
}}

function updateTimeBar(currentTime, duration) {{
  const pct = (currentTime / duration) * 100;
  const fill = document.getElementById('time-fill');
  const thumb = document.getElementById('time-thumb');
  const label = document.getElementById('time-current');
  if (fill) fill.style.width = pct + '%';
  if (thumb) thumb.style.left = pct + '%';
  if (label) label.textContent = currentTime.toFixed(2) + 's / ' + duration.toFixed(2) + 's';
}}

function updateCursors(currentTime, videoDuration) {{
  if (!chartState.data || chartState.keys.length === 0) return;

  // Map video time to data time range
  const frac = currentTime / videoDuration;
  const dataTime = chartState.tMin + frac * (chartState.tMax - chartState.tMin);

  for (const key of chartState.keys) {{
    const cursor = document.getElementById('cursor-' + key);
    const valEl = document.getElementById('cursorval-' + key);
    const map = chartState.canvasMap[key];
    if (!cursor || !map) continue;

    // Compute pixel offset within canvas (pad.left + fraction * plotWidth)
    const tFrac = (dataTime - chartState.tMin) / (chartState.tMax - chartState.tMin);
    const px = map.left + tFrac * map.plotWidth;

    // Position cursor div (relative to chart-box, account for 12px padding)
    cursor.style.left = (12 + px) + 'px';
    cursor.style.display = (tFrac >= 0 && tFrac <= 1) ? 'block' : 'none';

    // Interpolate value at current time
    if (valEl && chartState.data[key]) {{
      const ts = chartState.data.timestamps;
      const vals = chartState.data[key];
      // Find bracketing indices
      let val = vals[0];
      for (let i = 0; i < ts.length - 1; i++) {{
        if (dataTime >= ts[i] && dataTime <= ts[i + 1]) {{
          const f = (dataTime - ts[i]) / (ts[i + 1] - ts[i]);
          val = vals[i] + f * (vals[i + 1] - vals[i]);
          break;
        }}
      }}
      if (dataTime >= ts[ts.length - 1]) val = vals[vals.length - 1];
      valEl.textContent = val.toFixed(2);
    }}
  }}
}}

searchEl.addEventListener('input', () => renderList(searchEl.value));

// Keyboard navigation
document.addEventListener('keydown', (e) => {{
  if (e.target === searchEl) return;
  if (e.key === 'ArrowLeft' || e.key === 'k') selectClip(Math.max(0, currentIdx - 1));
  if (e.key === 'ArrowRight' || e.key === 'j') selectClip(Math.min(CLIPS.length - 1, currentIdx + 1));
}});

renderList('');
</script>
</body>
</html>"""


def _transcode_to_h264(src: Path, dst: Path) -> bool:
    """Transcode a video to H.264 MP4 using ffmpeg (preferred) or OpenCV fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")

    # Try ffmpeg first (better quality, preserves audio)
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(src),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-movflags", "+faststart",
                    str(tmp),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
                check=True,
            )
            tmp.rename(dst)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            tmp.unlink(missing_ok=True)

    # Fallback: OpenCV (no audio, but always available)
    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try H.264 (avc1), fall back to mp4v
        for fourcc_str in ["avc1", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
            if writer.isOpened():
                break
            writer.release()
        else:
            cap.release()
            return False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        cap.release()
        writer.release()

        if tmp.stat().st_size > 0:
            tmp.rename(dst)
            return True
        else:
            tmp.unlink(missing_ok=True)
            return False
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


class ViewerHandler(SimpleHTTPRequestHandler):
    """Serve the HTML app and video files."""

    def __init__(self, *args, html_content="", clip_lookup=None, h264_cache=None, **kwargs):
        self.html_content = html_content
        self.clip_lookup = clip_lookup or {}
        self.h264_cache = h264_cache
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path.startswith("/video/"):
            self._serve_video(path)
        elif path.startswith("/frames/"):
            self._serve_frames(path)
        elif path.startswith("/timeseries/"):
            self._serve_timeseries(path)
        else:
            self.send_error(404)

    def _serve_html(self):
        data = self.html_content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_video(self, path):
        # /video/{clip_id}/{type}
        parts = path.split("/")
        if len(parts) < 4:
            self.send_error(404)
            return

        clip_id = parts[2]
        video_type = parts[3]

        clip = self.clip_lookup.get(clip_id)
        if not clip:
            self.send_error(404, f"Clip not found: {clip_id}")
            return

        file_path = clip.get(video_type)
        if not file_path or not Path(file_path).exists():
            self.send_error(404, f"Video not found: {video_type}")
            return

        file_path = Path(file_path)

        # For "original" CARLA videos (FMP4 codec), transcode to H.264 on demand
        if video_type == "original" and self.h264_cache is not None:
            cached = self.h264_cache / f"{clip_id}.mp4"
            if not cached.exists():
                print(f"  Transcoding {clip_id} to H.264...")
                if not _transcode_to_h264(file_path, cached):
                    # Transcoding failed — fall through to serve original
                    print(f"  Transcoding failed for {clip_id}, serving original")
                else:
                    print(f"  Cached: {cached}")
            if cached.exists():
                file_path = cached

        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", file_path.stat().st_size)
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                self.wfile.write(chunk)

    def _serve_frames(self, path):
        """Extract frames from video and serve as a JPEG montage grid.

        URL: /frames/{clip_id}/{type}?cols=5&n=10
        """
        from urllib.parse import parse_qs

        parts = path.split("/")
        if len(parts) < 4:
            self.send_error(404)
            return

        clip_id = parts[2]
        video_type = parts[3].split("?")[0]
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        n_frames = int(qs.get("n", ["10"])[0])
        cols = int(qs.get("cols", ["5"])[0])

        clip = self.clip_lookup.get(clip_id)
        if not clip:
            self.send_error(404, f"Clip not found: {clip_id}")
            return

        file_path = clip.get(video_type)
        if not file_path or not Path(file_path).exists():
            self.send_error(404, f"Video not found: {video_type}")
            return

        # Extract frames with OpenCV
        cap = cv2.VideoCapture(file_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            self.send_error(500, "Cannot read video")
            return

        n_frames = min(n_frames, total)
        indices = [round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            self.send_error(500, "No frames extracted")
            return

        # Build montage grid
        h, w = frames[0].shape[:2]
        # Scale down for web
        scale = min(1.0, 400 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frames = [cv2.resize(f, (new_w, new_h)) for f in frames]

        rows = (len(frames) + cols - 1) // cols
        grid = np.zeros((rows * new_h, cols * new_w, 3), dtype=np.uint8)
        for i, frame in enumerate(frames):
            r, c = divmod(i, cols)
            grid[r * new_h:(r + 1) * new_h, c * new_w:(c + 1) * new_w] = frame

        # Encode as JPEG
        img = PILImage.fromarray(grid)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_timeseries(self, path):
        """Serve per-timestep dynamics arrays as JSON.

        URL: /timeseries/{clip_id}
        Returns: {"timestamps": [...], "speed": [...], "accel": [...],
                  "yaw_rate": [...], "jerk": [...]}
        """
        parts = path.split("/")
        if len(parts) < 3:
            self.send_error(404)
            return

        clip_id = parts[2]
        clip = self.clip_lookup.get(clip_id)
        if not clip:
            self.send_error(404, f"Clip not found: {clip_id}")
            return

        array_path = clip.get("array_ref")
        if not array_path or not Path(array_path).exists():
            # Fallback: look for arrays/{clip_id}.npz next to the QA file
            fallback = Path(self.clip_lookup.get("__qa_dir__", "")) / "arrays" / f"{clip_id}.npz"
            if fallback.exists():
                array_path = str(fallback)
            else:
                self.send_error(404, f"No array data for: {clip_id}")
                return

        arrays = np.load(array_path)
        result = {}
        for key in ("timestamps", "speed", "accel", "yaw_rate", "jerk"):
            if key in arrays:
                result[key] = [round(float(v), 4) for v in arrays[key]]

        # Also include lateral acceleration (speed * yaw_rate) if both exist
        if "speed" in arrays and "yaw_rate" in arrays:
            lat_accel = arrays["speed"] * arrays["yaw_rate"]
            result["lateral_accel"] = [round(float(v), 4) for v in lat_accel]

        data = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Suppress noisy per-request logs
        pass


def main():
    parser = argparse.ArgumentParser(description="CARLA Clip Viewer for EgoDyn-Bench")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--carla_chunks", type=str, default=str(DEFAULT_CARLA_CHUNKS))
    parser.add_argument("--transferred", type=str, default=str(DEFAULT_TRANSFERRED))
    parser.add_argument("--carla_qa", type=str, default=str(DEFAULT_CARLA_QA))
    parser.add_argument("--carla_index", type=str, default=str(DEFAULT_CARLA_INDEX))
    parser.add_argument("--selected_clips", type=str, default=str(DEFAULT_SELECTED))
    parser.add_argument("--h264_cache", type=str, default=str(DEFAULT_H264_CACHE),
                        help="Directory to cache H.264-transcoded original CARLA videos")
    args = parser.parse_args()

    print("Loading clip data...")
    clips, h264_cache = load_data(args)
    print(f"  {len(clips)} CARLA clips loaded")

    print(f"  Original CARLA videos will be transcoded to H.264 on first view")
    if shutil.which("ffmpeg"):
        print(f"  Using ffmpeg for transcoding")
    else:
        print(f"  ffmpeg not found — using OpenCV fallback (no audio)")
    print(f"  Cache dir: {h264_cache}")

    clip_lookup = {c["id"]: c for c in clips}
    clip_lookup["__qa_dir__"] = str(Path(args.carla_qa).parent)
    html = build_html(clips)

    def handler_factory(*handler_args, **handler_kwargs):
        return ViewerHandler(
            *handler_args,
            html_content=html,
            clip_lookup=clip_lookup,
            h264_cache=h264_cache,
            **handler_kwargs,
        )

    server = HTTPServer(("0.0.0.0", args.port), handler_factory)
    print(f"  Serving at http://localhost:{args.port}")
    print(f"  Navigate with ← → arrow keys or j/k")
    print(f"  Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
