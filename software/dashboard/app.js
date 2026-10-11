const $ = id => document.getElementById(id);

let state = null;
let photo = new Image();
let displayed = null;
let selecting = false;
let points = [];
let priorFrame = null;
let offline = false;
let receivedAt = 0;
let lastFrameTime = performance.now();
let renderFps = 60;
let fpsFrames = 0;
let fpsTimer = performance.now();
let radarAngle = -Math.PI * 0.9;
let radarDirection = 1;
let streamConnected = false;
const streamImg = $('stream-img');
if (streamImg) {
  streamImg.onload = () => { streamConnected = true; };
  streamImg.onerror = () => { streamConnected = false; };
}

const camera = $('camera');
const ctx = camera.getContext('2d');
const map = $('map');
const m = map.getContext('2d');

const cyan = '#00f5d4';
const cyanGlow = 'rgba(0, 245, 212, 0.4)';
const amber = '#f59e0b';
const amberGlow = 'rgba(245, 158, 11, 0.4)';
const red = '#ef4444';
const slate = '#64748b';
const edgeColor = '#1a2332';

const pointNames = ['Near-left', 'Near-right', 'Far-right', 'Far-left'];
$('points').innerHTML = pointNames.map((name, i) => `
  <div class="point">
    <strong>${i + 1} · ${name}</strong>
    <span id="p${i}" class="eyebrow">NOT SELECTED</span>
    <label>X (m) <input aria-label="${name} X meters" id="x${i}" type="number" step="0.1" value="${[-1, 1, 1, -1][i]}"></label>
    <label>Y (m) <input aria-label="${name} Y meters" id="y${i}" type="number" step="0.1" value="${[1, 1, 3, 3][i]}"></label>
  </div>
`).join('');

async function post(path, body) {
  const r = await fetch(path, {
    signal: AbortSignal.timeout(5000),
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (!r.ok) throw Error(j.error || 'Request failed');
  return j;
}

function message(text, isError = false) {
  $('notice').textContent = text;
  $('notice').classList.toggle('error', isError);
}

const violet = '#a855f7';
const violetGlow = 'rgba(168, 85, 247, 0.45)';
const emerald = '#10b981';

let currentRadarMaxRangeM = 7.0;

function computeAdaptiveRadarScale(detectedObjects, rememberedObjects) {
  const arcR = 340;
  const maxPlotR = arcR - 30; // 310 px usable radius inside the 340 px perimeter arc
  let targetRangeM = Number(
    (displayed && displayed.radar_max_range_m) ||
    state?.radar_max_range_m ||
    state?.scenario?.radar_max_range_m ||
    6.5
  );
  if (!Number.isFinite(targetRangeM) || targetRangeM < 5.5) targetRangeM = 6.5;

  const objs = [
    ...(detectedObjects || (displayed && displayed.objects) || state?.objects || []),
    ...(rememberedObjects || (displayed && displayed.remembered_objects) || state?.remembered_objects || [])
  ];
  for (const o of objs) {
    const gxy = o.ground_xy || o.depth_xy;
    if (Array.isArray(gxy) && gxy.length === 2) {
      const rM = Math.hypot(Number(gxy[0]) || 0, Number(gxy[1]) || 0);
      if (rM * 1.12 > targetRangeM) targetRangeM = rM * 1.12;
    } else if (o.depth_m != null) {
      const dM = Number(o.depth_m) || 0;
      if (dM * 1.12 > targetRangeM) targetRangeM = dM * 1.12;
    }
  }
  targetRangeM = Math.max(6.0, Math.min(14.0, targetRangeM));
  currentRadarMaxRangeM += (targetRangeM - currentRadarMaxRangeM) * 0.22;
  const sy = maxPlotR / currentRadarMaxRangeM;
  const sx = sy * 1.10;
  return { sx, sy, maxRangeM: currentRadarMaxRangeM, arcR, maxPlotR };
}

function clampToRadarArc(px, py, ox, oy, maxR = 312) {
  let dx = px - ox;
  let dy = Math.min(-14, py - oy); // Always in forward upper hemisphere
  const r = Math.hypot(dx, dy);
  if (r > maxR && r > 1e-4) {
    const scale = maxR / r;
    dx *= scale;
    dy *= scale;
  }
  return [ox + dx, oy + dy];
}

// -------------------------------------------------------------
// PERSISTENT OBJECT TRACKER (Smooth interpolation & BEV memory)
// -------------------------------------------------------------
class Tracker {
  constructor() {
    this.tracks = [];
    this.nextId = 1;
  }

  update(detectedObjects, rememberedObjects, mapWidth, mapHeight, isMetric, calib) {
    const now = performance.now();
    const ox = mapWidth / 2;
    const oy = 475;
    const { sx, sy, maxPlotR } = computeAdaptiveRadarScale(detectedObjects, rememberedObjects);
    const start = -Math.PI * 0.95;
    const end = -Math.PI * 0.05;

    const allInputs = [
      ...(detectedObjects || []).map(o => ({ ...o, remembered: false })),
      ...(rememberedObjects || []).map(o => ({ ...o, remembered: true }))
    ];

    // Convert live + remembered objects to map target coordinates
    const candidates = allInputs.map(obj => {
      let tx, ty, distStr;
      const frameW = displayed?.width || 640;
      const gxy = obj.ground_xy || obj.depth_xy;
      if (isMetric && gxy) {
        const rawX = ox + gxy[0] * sx;
        const rawY = oy - gxy[1] * sy;
        [tx, ty] = clampToRadarArc(rawX, rawY, ox, oy, maxPlotR);
        const dist = obj.depth_m != null ? obj.depth_m : Math.hypot(gxy[0], gxy[1]);
        distStr = `${dist.toFixed(1)} m`;
      } else {
        const u = Math.max(0, Math.min(1, (obj.xyxy[0] + obj.xyxy[2]) / 2 / frameW));
        const a = start + (end - start) * u;
        const distRadius = obj.depth_m != null ? Math.min(maxPlotR, Math.max(55, obj.depth_m * sy)) : 210;
        tx = ox + Math.cos(a) * distRadius;
        ty = oy + Math.sin(a) * distRadius;
        const zoneIdx = Math.floor(u * 6);
        const zoneNames = ['L3', 'L2', 'L1', 'R1', 'R2', 'R3'];
        distStr = obj.depth_m != null ? `${obj.depth_m.toFixed(1)}m` : zoneNames[Math.max(0, Math.min(5, zoneIdx))];
      }

      // Compute future radar points from 0.5s - 3.0s predicted_path
      const futureRadarPts = (obj.predicted_path || []).map(pt => {
        const ptGxy = pt.ground_xy || pt.depth_xy;
        if (isMetric && ptGxy) {
          const [fx, fy] = clampToRadarArc(
            ox + ptGxy[0] * sx,
            oy - ptGxy[1] * sy,
            ox,
            oy,
            maxPlotR + 8
          );
          return {
            dt: pt.dt,
            x: fx,
            y: fy
          };
        }
        const uFut = Math.max(0, Math.min(1, pt.cx / frameW));
        const aFut = start + (end - start) * uFut;
        const approachShift = obj.approaching ? Math.min(85, pt.dt * 26) : 0;
        const radFut = Math.max(65, 210 - approachShift);
        return {
          dt: pt.dt,
          x: ox + Math.cos(aFut) * radFut,
          y: oy + Math.sin(aFut) * radFut
        };
      });

      return {
        track_id: obj.track_id,
        class: obj.class,
        category: obj.category || (obj.class === 'person' ? 'pedestrian' : 'obstacle'),
        confidence: obj.confidence,
        targetX: tx,
        targetY: ty,
        ground_xy: gxy,
        depth_m: obj.depth_m,
        distStr,
        realistic: obj.realistic !== false,
        realism_reason: obj.realism_reason || 'confirmed',
        ttc_s: obj.ttc_s,
        approaching: !!obj.approaching,
        current_zone: obj.current_zone,
        predicted_zone_2s: obj.predicted_zone_2s,
        futureRadarPts,
        remembered: !!obj.remembered,
        memory_age_s: obj.memory_age_s || 0,
        memory_confidence: obj.memory_confidence ?? 1.0
      };
    });

    // Match by backend track_id first, then fallback to greedy spatial distance
    const matched = new Set();
    for (const c of candidates) {
      let bestTrack = null;
      if (c.track_id != null) {
        bestTrack = this.tracks.find(t => !matched.has(t.id) && t.backendTrackId === c.track_id);
      }
      if (!bestTrack) {
        let bestDist = 120;
        for (const t of this.tracks) {
          if (matched.has(t.id) || t.class !== c.class) continue;
          const d = Math.hypot(t.targetX - c.targetX, t.targetY - c.targetY);
          if (d < bestDist) {
            bestDist = d;
            bestTrack = t;
          }
        }
      }

      const desiredAlpha = c.remembered
        ? Math.max(0.28, Math.min(0.82, c.memory_confidence * 0.85))
        : 1.0;

      if (bestTrack) {
        bestTrack.backendTrackId = c.track_id ?? bestTrack.backendTrackId;
        bestTrack.category = c.category;
        bestTrack.targetX = c.targetX;
        bestTrack.targetY = c.targetY;
        bestTrack.ground_xy = c.ground_xy;
        bestTrack.depth_m = c.depth_m;
        bestTrack.distStr = c.distStr;
        bestTrack.confidence = c.confidence;
        bestTrack.realistic = c.realistic;
        bestTrack.realism_reason = c.realism_reason;
        bestTrack.ttc_s = c.ttc_s;
        bestTrack.approaching = c.approaching;
        bestTrack.current_zone = c.current_zone;
        bestTrack.predicted_zone_2s = c.predicted_zone_2s;
        bestTrack.futureRadarPts = c.futureRadarPts;
        bestTrack.remembered = c.remembered;
        bestTrack.memory_age_s = c.memory_age_s;
        bestTrack.memory_confidence = c.memory_confidence;
        bestTrack.targetAlpha = desiredAlpha;
        bestTrack.lastSeen = now;
        matched.add(bestTrack.id);
      } else {
        const newTrack = {
          id: this.nextId++,
          backendTrackId: c.track_id,
          class: c.class,
          category: c.category,
          confidence: c.confidence,
          currX: c.targetX,
          currY: c.targetY,
          targetX: c.targetX,
          targetY: c.targetY,
          alpha: 0,
          targetAlpha: desiredAlpha,
          ground_xy: c.ground_xy,
          depth_m: c.depth_m,
          distStr: c.distStr,
          realistic: c.realistic,
          realism_reason: c.realism_reason,
          ttc_s: c.ttc_s,
          approaching: c.approaching,
          current_zone: c.current_zone,
          predicted_zone_2s: c.predicted_zone_2s,
          futureRadarPts: c.futureRadarPts,
          remembered: c.remembered,
          memory_age_s: c.memory_age_s,
          memory_confidence: c.memory_confidence,
          lastSeen: now
        };
        this.tracks.push(newTrack);
        matched.add(newTrack.id);
      }
    }

    // Decay tracks no longer in either live or remembered lists
    for (const t of this.tracks) {
      if (!matched.has(t.id)) {
        if (now - t.lastSeen > 800) {
          t.targetAlpha = 0;
        }
      }
    }
  }

  step() {
    for (let i = this.tracks.length - 1; i >= 0; i--) {
      const t = this.tracks[i];
      // Smooth lerp
      t.currX += (t.targetX - t.currX) * 0.25;
      t.currY += (t.targetY - t.currY) * 0.25;
      t.alpha += (t.targetAlpha - t.alpha) * 0.15;
      if (t.alpha < 0.02 && t.targetAlpha === 0) {
        this.tracks.splice(i, 1);
      }
    }
  }
}

const tracker = new Tracker();

// -------------------------------------------------------------
// RENDER: CAMERA FEED (TESLA AUTOPILOT HUD BRACKETS)
// -------------------------------------------------------------
function drawCamera() {
  const useRifeStream = !selecting && state?.mode === 'live' && state?.status === 'live' && (state?.rife?.factor > 0) && streamConnected && streamImg && streamImg.naturalWidth > 0;
  const currentImg = useRifeStream ? streamImg : photo;
  const hasImage = currentImg && currentImg.complete && currentImg.naturalWidth > 0;

  if (!displayed || !hasImage) {
    if (camera.width !== 640 || camera.height !== 480) {
      camera.width = 640;
      camera.height = 480;
    }
    // Standby canvas rendering
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, camera.width, camera.height);
    ctx.strokeStyle = '#1a2332';
    ctx.lineWidth = 1;
    for (let i = 40; i < camera.width; i += 40) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, camera.height); ctx.stroke();
    }
    for (let j = 40; j < camera.height; j += 40) {
      ctx.beginPath(); ctx.moveTo(0, j); ctx.lineTo(camera.width, j); ctx.stroke();
    }
    ctx.fillStyle = cyan;
    ctx.font = '600 13px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('AWAITING FORWARD SENSOR FEED…', camera.width / 2, camera.height / 2);
    return;
  }

  const targetW = displayed.width || currentImg.naturalWidth || 640;
  const targetH = displayed.height || currentImg.naturalHeight || 480;
  if (camera.width !== targetW || camera.height !== targetH) {
    camera.width = targetW;
    camera.height = targetH;
  }
  ctx.drawImage(currentImg, 0, 0, camera.width, camera.height);

  // Sector vertical division guide lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  for (let i = 1; i < 6; i++) {
    const x = camera.width * i / 6;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, camera.height);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  const showForecast = $('forecast') ? $('forecast').checked : true;
  const dtSinceRecv = (!selecting && state?.status === 'live' && receivedAt > 0)
    ? Math.min(0.45, Math.max(0, (performance.now() - receivedAt) / 1000))
    : 0;

  // Draw Detections if toggled
  if ($('boxes').checked && displayed.objects) {
    for (const o of displayed.objects) {
      const isRealistic = o.realistic !== false;
      const vx = (isRealistic && o.velocity_px_s) ? o.velocity_px_s[0] * 0.65 : 0;
      const vy = (isRealistic && o.velocity_px_s) ? o.velocity_px_s[1] * 0.65 : 0;
      const dx = vx * dtSinceRecv;
      const dy = vy * dtSinceRecv;
      const x1 = Math.max(0, Math.min(camera.width - 8, o.xyxy[0] + dx));
      const y1 = Math.max(0, Math.min(camera.height - 8, o.xyxy[1] + dy));
      const x2 = Math.max(x1 + 8, Math.min(camera.width, o.xyxy[2] + dx));
      const y2 = Math.max(y1 + 8, Math.min(camera.height, o.xyxy[3] + dy));
      const w = x2 - x1;
      const h = y2 - y1;
      const color = !isRealistic ? slate : (o.class === 'person' ? cyan : amber);

      // 3-Second Future Ghost Box & Trajectory Rollout
      if (showForecast && isRealistic && o.predicted_boxes && o.predicted_path) {
        const box2s = o.predicted_boxes['2.0s'];
        if (box2s) {
          const [gx1, gy1, gx2, gy2] = box2s;
          const gw = gx2 - gx1;
          const gh = gy2 - gy1;
          if (Math.hypot(gx1 - x1, gy1 - y1) > 6) {
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = o.approaching ? red : '#38bdf8';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(gx1, gy1, gw, gh);
            ctx.fillStyle = o.approaching ? 'rgba(239, 68, 68, 0.06)' : 'rgba(56, 189, 248, 0.06)';
            ctx.fillRect(gx1, gy1, gw, gh);
            ctx.restore();
          }
        }

        // Dotted 3-second foot trajectory vector
        const footX = (x1 + x2) / 2;
        const footY = y2;
        ctx.save();
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = o.approaching ? red : '#38bdf8';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(footX, footY);
        for (const pt of o.predicted_path) {
          ctx.lineTo(pt.foot_x, pt.foot_y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        for (const pt of o.predicted_path) {
          if (Math.hypot(pt.foot_x - footX, pt.foot_y - footY) > 5) {
            ctx.fillStyle = o.approaching ? red : '#38bdf8';
            ctx.beginPath();
            ctx.arc(pt.foot_x, pt.foot_y, 3, 0, Math.PI * 2);
            ctx.fill();
            if (pt.dt === 1.0 || pt.dt === 3.0) {
              ctx.font = '600 9px -apple-system, sans-serif';
              ctx.fillText(`+${pt.dt}s`, pt.foot_x + 5, pt.foot_y - 3);
            }
          }
        }
        ctx.restore();
      }

      // Translucent box fill
      ctx.fillStyle = o.class === 'person' ? 'rgba(0, 245, 212, 0.08)' : 'rgba(245, 158, 11, 0.08)';
      ctx.fillRect(x1, y1, w, h);

      // Tesla style sleek corner brackets
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      const corner = Math.min(16, w / 4, h / 4);

      // Top-left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + corner);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + corner, y1);
      ctx.stroke();

      // Top-right
      ctx.beginPath();
      ctx.moveTo(x2 - corner, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + corner);
      ctx.stroke();

      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x1, y2 - corner);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1 + corner, y2);
      ctx.stroke();

      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x2 - corner, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - corner);
      ctx.stroke();

      // Subtle full perimeter border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, y1, w, h);

      // Ground contact point anchor dot
      const footX = (x1 + x2) / 2;
      const footY = y2;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(footX, footY, 3.5, 0, Math.PI * 2);
      ctx.fill();

      // High-tech HUD tag pill with Realism + Metric Depth + 2s Zone / TTC forecast
      const depthTag = o.depth_m != null ? ` · ${o.depth_m.toFixed(1)}m` : '';
      const extraTag = !isRealistic
        ? ' · FILTERED'
        : o.ttc_s != null
          ? `${depthTag} · TTC ${o.ttc_s.toFixed(1)}s`
          : (o.current_zone && o.predicted_zone_2s && o.current_zone !== o.predicted_zone_2s)
            ? `${depthTag} · ${o.current_zone}→${o.predicted_zone_2s}`
            : depthTag;
      const label = `${o.class.toUpperCase()} ${Math.round(o.confidence * 100)}%${extraTag}`;
      ctx.font = '600 11px -apple-system, sans-serif';
      ctx.textAlign = 'left';
      const tw = ctx.measureText(label).width;
      const tagH = 20;
      const tagY = Math.max(0, y1 - tagH - 4);

      ctx.fillStyle = 'rgba(7, 9, 12, 0.9)';
      ctx.beginPath();
      ctx.roundRect(x1, tagY, tw + 16, tagH, 4);
      ctx.fill();
      ctx.strokeStyle = o.approaching ? red : color;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = o.approaching ? '#fca5a5' : color;
      ctx.fillText(label, x1 + 8, tagY + 14);
    }
  }

  // Calibration points if active
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.fillStyle = amber;
    ctx.arc(p[0] * camera.width, p[1] * camera.height, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#07090c';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(String(i + 1), p[0] * camera.width, p[1] * camera.height + 4);
  });
}

// -------------------------------------------------------------
// RENDER: TOP-DOWN SURROUND VIEW (TESLA BEV + WAYMO OCCUPANCY FLOW)
// -------------------------------------------------------------
function drawMap() {
  const width = 700;
  const height = 540;
  m.clearRect(0, 0, width, height);

  const ox = width / 2;
  const oy = 475; // Ego wearer anchor point
  const { sx, sy, maxRangeM, maxPlotR } = computeAdaptiveRadarScale();
  const metric = !!state?.calibration || state?.map_mode === 'monocular_depth';
  const valid = (state?.valid || tracker.tracks.length > 0) && !offline;

  // 1. Radar background scan line
  radarAngle += 0.025 * radarDirection;
  if (radarAngle > -Math.PI * 0.05) radarDirection = -1;
  if (radarAngle < -Math.PI * 0.95) radarDirection = 1;

  // 2. Concentric Distance Radar Rings (dynamically scaled to target distance range)
  const stepM = maxRangeM > 8.8 ? 2 : 1;
  const ringDistances = [];
  for (let d = stepM; d <= Math.ceil(maxRangeM); d += stepM) {
    if (d * sy <= 336) ringDistances.push(d);
  }
  const ringRadii = metric
    ? ringDistances.map(d => d * sy)
    : [60, 120, 180, 240, 300, 340];

  const startArc = -Math.PI * 0.95;
  const endArc = -Math.PI * 0.05;

  // Forward visual perception illumination beam
  const beamGrad = m.createRadialGradient(ox, oy, 10, ox, oy, 380);
  beamGrad.addColorStop(0, 'rgba(0, 245, 212, 0.12)');
  beamGrad.addColorStop(0.5, 'rgba(0, 245, 212, 0.04)');
  beamGrad.addColorStop(1, 'rgba(0, 245, 212, 0)');
  m.fillStyle = beamGrad;
  m.beginPath();
  m.moveTo(ox, oy);
  m.arc(ox, oy, 380, startArc, endArc);
  m.closePath();
  m.fill();

  // 2B. TESLA FSD WALKABLE FREE-SPACE CORRIDOR (31-Ray Raycast Polygon)
  const freeSpaceContour = (displayed && displayed.free_space_contour) || state?.free_space_contour;
  if (metric && Array.isArray(freeSpaceContour) && freeSpaceContour.length >= 3) {
    m.save();
    const fsGrad = m.createRadialGradient(ox, oy, 12, ox, oy, 260);
    fsGrad.addColorStop(0, 'rgba(16, 185, 129, 0.14)');
    fsGrad.addColorStop(0.75, 'rgba(16, 185, 129, 0.05)');
    fsGrad.addColorStop(1, 'rgba(16, 185, 129, 0.01)');
    m.fillStyle = fsGrad;
    m.strokeStyle = 'rgba(16, 185, 129, 0.45)';
    m.lineWidth = 1.5;
    m.beginPath();
    m.moveTo(ox, oy);
    freeSpaceContour.forEach(pt => {
      const [px, py] = clampToRadarArc(
        ox + (pt.x ?? pt[0]) * sx,
        oy - (pt.y ?? pt[1]) * sy,
        ox,
        oy,
        maxPlotR + 14
      );
      m.lineTo(px, py);
    });
    m.closePath();
    m.fill();
    m.stroke();
    m.restore();
  }

  // Subtle moving radar sweep beam
  const sweepGrad = m.createRadialGradient(ox, oy, 0, ox, oy, 340);
  sweepGrad.addColorStop(0, 'rgba(0, 245, 212, 0.15)');
  sweepGrad.addColorStop(1, 'rgba(0, 245, 212, 0)');
  m.fillStyle = sweepGrad;
  m.beginPath();
  m.moveTo(ox, oy);
  m.arc(ox, oy, 340, radarAngle - 0.15, radarAngle + 0.15);
  m.closePath();
  m.fill();

  // Draw range rings
  ringRadii.forEach((rad, idx) => {
    m.strokeStyle = idx % 2 === 0 ? '#1a2332' : '#121924';
    m.lineWidth = 1;
    m.beginPath();
    m.arc(ox, oy, rad, startArc, endArc);
    m.stroke();

    // Distance label along the centerline
    const distM = ringDistances[idx];
    if (distM) {
      m.fillStyle = '#475569';
      m.font = '500 10px -apple-system, sans-serif';
      m.textAlign = 'left';
      m.fillText(`${distM}m`, ox + 6, oy - rad + 3);
    }
  });

  // Lateral metric grid if calibrated or using GX10 Monocular Depth
  if (metric) {
    const maxLatM = Math.min(7, Math.floor(320 / sx));
    for (let x = -maxLatM; x <= maxLatM; x++) {
      if (x === 0) continue;
      m.strokeStyle = '#121924';
      m.lineWidth = 1;
      m.beginPath();
      m.moveTo(ox + x * sx, 50);
      m.lineTo(ox + x * sx, oy);
      m.stroke();
      if (Math.abs(x) % 2 === 0) {
        m.fillStyle = '#475569';
        m.font = '500 10px -apple-system, sans-serif';
        m.textAlign = 'center';
        m.fillText(`${x}m`, ox + x * sx, 505);
      }
    }

    // Calibrated floor boundary polygon
    if (state?.calibration?.points) {
      m.fillStyle = 'rgba(0, 245, 212, 0.05)';
      m.strokeStyle = 'rgba(0, 245, 212, 0.5)';
      m.lineWidth = 1.5;
      m.beginPath();
      state.calibration.points.forEach((p, i) => {
        const px = ox + p[2] * sx;
        const py = oy - p[3] * sy;
        i ? m.lineTo(px, py) : m.moveTo(px, py);
      });
      m.closePath();
      m.fill();
      m.stroke();
    }
  }

  // 2C. TESLA / WAYMO BEV LOG-ODDS OCCUPANCY VOXELS & 2D FLOW VECTORS
  const occCells = (displayed && displayed.occupancy_cells) || state?.occupancy_cells || [];
  if (metric && Array.isArray(occCells) && occCells.length > 0) {
    for (const cell of occCells) {
      const [cx, cy] = clampToRadarArc(
        ox + cell.x_m * sx,
        oy - cell.y_m * sy,
        ox,
        oy,
        maxPlotR + 4
      );
      const cellW = Math.max(8, (cell.res_m || 0.35) * sx * 0.92);
      const cellH = Math.max(8, (cell.res_m || 0.35) * sy * 0.92);
      const prob = Math.max(0.2, Math.min(0.95, cell.occ_prob ?? 0.65));
      const isDyn = cell.dynamic || cell.class === 'person';

      m.save();
      if (cell.in_shadow) {
        // Waymo Raycast Occlusion Shadow preservation styling
        m.setLineDash([3, 2]);
        m.fillStyle = `rgba(168, 85, 247, ${(prob * 0.24).toFixed(2)})`;
        m.strokeStyle = `rgba(192, 132, 252, ${(prob * 0.65).toFixed(2)})`;
      } else if (isDyn) {
        m.fillStyle = `rgba(0, 245, 212, ${(prob * 0.22).toFixed(2)})`;
        m.strokeStyle = `rgba(0, 245, 212, ${(prob * 0.55).toFixed(2)})`;
      } else {
        m.fillStyle = `rgba(56, 189, 248, ${(prob * 0.2).toFixed(2)})`;
        m.strokeStyle = `rgba(56, 189, 248, ${(prob * 0.5).toFixed(2)})`;
      }
      m.lineWidth = 1;
      m.beginPath();
      m.roundRect(cx - cellW / 2, cy - cellH / 2, cellW, cellH, 3);
      m.fill();
      m.stroke();
      m.setLineDash([]);

      // Waymo Occupancy Flow 2D Velocity Vector (vx, vy in m/s)
      const fvx = cell.flow_vx || 0;
      const fvy = cell.flow_vy || 0;
      const speed = Math.hypot(fvx, fvy);
      if (speed > 0.12) {
        const endX = cx + fvx * sx * 0.9;
        const endY = cy - fvy * sy * 0.9;
        m.strokeStyle = cell.in_shadow ? '#c084fc' : '#38bdf8';
        m.lineWidth = 1.6;
        m.beginPath();
        m.moveTo(cx, cy);
        m.lineTo(endX, endY);
        m.stroke();
      }
      m.restore();
    }
  }

  // 3. TESLA SURROUND PROXIMITY ARCS (6 Zones: L3, L2, L1, R1, R2, R3)
  const arcR = 340;
  const zoneNames = ['L3', 'L2', 'L1', 'R1', 'R2', 'R3'];
  const showForecast = $('forecast') ? $('forecast').checked : true;
  const predZones2s = (displayed && displayed.predicted_zones_2s) ? displayed.predicted_zones_2s : state?.predicted_zones_2s;
  const memZones3 = (displayed && displayed.memory_zones_3) ? displayed.memory_zones_3 : state?.memory_zones_3;
  for (let i = 0; i < 6; i++) {
    const a = startArc + (endArc - startArc) * i / 6;
    const b = startArc + (endArc - startArc) * (i + 1) / 6;
    const mid = (a + b) / 2;
    const currentZones = (displayed && displayed.zones) ? displayed.zones : state?.zones;
    const isActive = valid && currentZones && currentZones[i] === 1;
    const isPredicted = valid && showForecast && !isActive && predZones2s && predZones2s[i] === 1;
    const isMemorySector = valid && !isActive && !isPredicted && Array.isArray(memZones3) && memZones3[Math.floor(i / 2)] === 1;

    // Radial sector divider lines
    m.strokeStyle = '#1e293b';
    m.lineWidth = 1;
    m.beginPath();
    m.moveTo(ox + Math.cos(a) * 40, oy + Math.sin(a) * 40);
    m.lineTo(ox + Math.cos(a) * (arcR + 10), oy + Math.sin(a) * (arcR + 10));
    m.stroke();

    if (isActive) {
      // Glow fill in sector
      const secGrad = m.createRadialGradient(ox, oy, 100, ox, oy, arcR);
      secGrad.addColorStop(0, 'rgba(0, 245, 212, 0.02)');
      secGrad.addColorStop(0.7, 'rgba(0, 245, 212, 0.15)');
      secGrad.addColorStop(1, 'rgba(0, 245, 212, 0.35)');
      m.fillStyle = secGrad;
      m.beginPath();
      m.moveTo(ox, oy);
      m.arc(ox, oy, arcR, a, b);
      m.closePath();
      m.fill();

      // Glowing outer proximity arc
      m.strokeStyle = cyan;
      m.lineWidth = 4;
      m.shadowColor = cyan;
      m.shadowBlur = 12;
      m.beginPath();
      m.arc(ox, oy, arcR, a, b);
      m.stroke();
      m.shadowBlur = 0;

      // Pulsing proximity radar ripple
      const ripple = (performance.now() / 600) % 1;
      m.strokeStyle = `rgba(0, 245, 212, ${0.8 * (1 - ripple)})`;
      m.lineWidth = 2;
      m.beginPath();
      m.arc(ox, oy, arcR + ripple * 22, a, b);
      m.stroke();
    } else if (isPredicted) {
      // Predicted +2s zone entry pre-alert arc
      m.save();
      m.setLineDash([6, 4]);
      m.strokeStyle = amber;
      m.lineWidth = 3;
      m.beginPath();
      m.arc(ox, oy, arcR, a, b);
      m.stroke();
      m.restore();
    } else if (isMemorySector) {
      // Persistent BEV spatial memory sector indicator
      m.save();
      m.setLineDash([3, 4]);
      m.strokeStyle = violet;
      m.lineWidth = 2.5;
      m.beginPath();
      m.arc(ox, oy, arcR, a, b);
      m.stroke();
      m.restore();
    } else {
      // Idle perimeter arc
      m.strokeStyle = '#1e293b';
      m.lineWidth = 2;
      m.beginPath();
      m.arc(ox, oy, arcR, a, b);
      m.stroke();
    }

    // Zone label (L3..R3)
    const labelR = arcR + 24;
    const lx = ox + Math.cos(mid) * labelR;
    const ly = oy + Math.sin(mid) * labelR;
    m.fillStyle = isActive ? cyan : isPredicted ? amber : isMemorySector ? '#c084fc' : '#64748b';
    m.font = `600 ${isActive || isPredicted || isMemorySector ? '12px' : '11px'} -apple-system, sans-serif`;
    m.textAlign = 'center';
    m.fillText(zoneNames[i], lx, ly);
  }

  // 4. DRAW TRACKED + REMEMBERED OBSTACLES & 3-SECOND PREDICTED RADAR TRAJECTORIES
  tracker.tracks.forEach(track => {
    if (track.alpha <= 0.01) return;
    m.save();
    m.globalAlpha = track.alpha;

    const [tx, ty] = clampToRadarArc(track.currX, track.currY, ox, oy, maxPlotR);
    const isPerson = track.class === 'person' || track.category === 'pedestrian';
    const isRemembered = !!track.remembered;
    const color = !track.realistic
      ? slate
      : isRemembered
        ? violet
        : (isPerson ? cyan : amber);
    const glow = isRemembered ? violetGlow : (isPerson ? cyanGlow : amberGlow);

    // Draw 3-second future path on radar if enabled and realistic
    if (showForecast && track.realistic && track.futureRadarPts && track.futureRadarPts.length > 0) {
      m.save();
      m.setLineDash([4, 4]);
      m.strokeStyle = track.approaching ? red : (isRemembered ? '#c084fc' : '#38bdf8');
      m.lineWidth = 2;
      m.beginPath();
      m.moveTo(tx, ty);
      track.futureRadarPts.forEach(pt => m.lineTo(pt.x, pt.y));
      m.stroke();
      m.setLineDash([]);

      track.futureRadarPts.forEach(pt => {
        if (Math.hypot(pt.x - tx, pt.y - ty) > 6) {
          m.fillStyle = track.approaching ? red : (isRemembered ? '#c084fc' : '#38bdf8');
          m.beginPath();
          m.arc(pt.x, pt.y, 3.5, 0, Math.PI * 2);
          m.fill();
          if (pt.dt === 2.0 || pt.dt === 3.0) {
            m.font = '600 9px -apple-system, sans-serif';
            m.textAlign = 'left';
            m.fillText(`+${pt.dt}s`, pt.x + 6, pt.y + 3);
          }
        }
      });
      m.restore();
    }

    // If remembered (occluded / out-of-FOV), draw a pulsing dashed BEV memory ring
    if (isRemembered) {
      m.save();
      m.setLineDash([4, 3]);
      m.strokeStyle = '#c084fc';
      m.lineWidth = 2;
      m.beginPath();
      m.arc(tx, ty, 22, 0, Math.PI * 2);
      m.stroke();
      m.restore();
    }

    if (isPerson) {
      // Tesla Pedestrian avatar
      m.fillStyle = 'rgba(0, 0, 0, 0.5)';
      m.beginPath();
      m.ellipse(tx, ty + 2, 14, 8, 0, 0, Math.PI * 2);
      m.fill();

      // Glowing safety halo
      m.strokeStyle = track.approaching ? 'rgba(239, 68, 68, 0.6)' : glow;
      m.lineWidth = 2;
      m.beginPath();
      m.arc(tx, ty, 18, 0, Math.PI * 2);
      m.stroke();

      // Pedestrian capsule body
      m.fillStyle = color;
      m.shadowColor = color;
      m.shadowBlur = 10;
      m.beginPath();
      m.arc(tx, ty, 6, 0, Math.PI * 2);
      m.fill();
      m.shadowBlur = 0;

      // Heading forward tick
      m.strokeStyle = color;
      m.lineWidth = 2;
      m.beginPath();
      m.moveTo(tx, ty - 6);
      m.lineTo(tx, ty - 12);
      m.stroke();
    } else {
      // Tesla Obstacle / Furniture / Hazard footprint
      m.fillStyle = 'rgba(0, 0, 0, 0.5)';
      m.beginPath();
      m.ellipse(tx, ty + 4, 16, 10, 0, 0, Math.PI * 2);
      m.fill();

      m.fillStyle = color;
      m.shadowColor = color;
      m.shadowBlur = 10;
      m.beginPath();
      m.roundRect(tx - 10, ty - 10, 20, 20, 4);
      m.fill();
      m.shadowBlur = 0;

      m.strokeStyle = isRemembered ? '#e9d5ff' : '#ffffff';
      m.lineWidth = 1;
      m.strokeRect(tx - 6, ty - 6, 12, 12);
    }

    // HUD Label tag with MEM indicator when retained in spatial memory
    const memTag = isRemembered ? `MEM ${track.memory_age_s.toFixed(1)}s · ` : '';
    const ttcNote = track.ttc_s != null ? ` · TTC ${track.ttc_s.toFixed(1)}s` : '';
    const tagText = `${memTag}${track.class.toUpperCase()} · ${track.distStr}${ttcNote}`;
    m.font = '600 11px -apple-system, sans-serif';
    const tw = m.measureText(tagText).width;
    const pillW = tw + 16;
    const pillH = 20;
    const pillX = Math.max(8, Math.min(width - pillW - 8, tx - pillW / 2));
    const pillY = Math.max(12, ty - 30);

    m.fillStyle = 'rgba(7, 9, 12, 0.9)';
    m.beginPath();
    m.roundRect(pillX, pillY, pillW, pillH, 5);
    m.fill();
    m.strokeStyle = track.approaching ? red : color;
    m.lineWidth = 1;
    m.stroke();

    m.fillStyle = track.approaching ? '#fca5a5' : (isRemembered ? '#e9d5ff' : color);
    m.textAlign = 'center';
    m.fillText(tagText, pillX + pillW / 2, pillY + 14);

    m.restore();
  });

  // 5. EGO VEHICLE / WEARER SILHOUETTE (Tesla model at bottom-center)
  const vW = 24;
  const vH = 46;
  const vx = ox - vW / 2;
  const vy = oy - vH / 2;

  // Ground shadow under vehicle
  m.fillStyle = 'rgba(0, 0, 0, 0.7)';
  m.beginPath();
  m.ellipse(ox, oy + 4, vW * 0.9, vH * 0.6, 0, 0, Math.PI * 2);
  m.fill();

  // Vehicle body
  m.fillStyle = '#17212b';
  m.strokeStyle = cyan;
  m.lineWidth = 1.5;
  m.beginPath();
  m.roundRect(vx, vy, vW, vH, 6);
  m.fill();
  m.stroke();

  // Glass roof / windshield
  m.fillStyle = '#0b0f14';
  m.beginPath();
  m.roundRect(vx + 3, vy + 9, vW - 6, vH - 18, 4);
  m.fill();

  // Forward headlights
  m.fillStyle = cyan;
  m.fillRect(vx + 3, vy + 2, 4, 3);
  m.fillRect(vx + vW - 7, vy + 2, 4, 3);

  // Ego label + dynamic radar range indicator
  m.fillStyle = '#94a3b8';
  m.font = '600 10px -apple-system, sans-serif';
  m.textAlign = 'center';
  m.fillText(`FORWARD SENSOR · ${maxRangeM.toFixed(1)}M RANGE`, ox, oy + 38);

  // 6. GX10 ASYNC VLM DEPTH REASONER OVERLAY PILL (when active)
  const vlmRep = (displayed && displayed.vlm_depth_report) || state?.vlm_depth_report || state?.scenario?.vlm_depth_report;
  if (vlmRep && vlmRep.active) {
    const modName = (vlmRep.model_name || 'gemma4:26b').toUpperCase();
    const scaleStr = Number(vlmRep.scale_factor || 1.0).toFixed(2);
    const reasonShort = (vlmRep.reasoning || 'Depth verified on BEV grid').slice(0, 68);
    const badgeStr = `GX10 VLM DEPTH DOUBLE-CHECK (${modName} · ×${scaleStr}): ${reasonShort}`;
    m.save();
    m.font = '600 10px -apple-system, sans-serif';
    const bw = Math.min(width - 24, m.measureText(badgeStr).width + 20);
    const bx = ox - bw / 2;
    const by = height - 26;
    m.fillStyle = 'rgba(15, 23, 42, 0.88)';
    m.strokeStyle = 'rgba(168, 85, 247, 0.65)';
    m.lineWidth = 1;
    m.beginPath();
    m.roundRect(bx, by, bw, 20, 5);
    m.fill();
    m.stroke();
    m.fillStyle = '#e9d5ff';
    m.textAlign = 'center';
    m.fillText(badgeStr, ox, by + 13.5);
    m.restore();
  }

  // Status banner if offline or no current observation
  if (!valid && tracker.tracks.length === 0) {
    m.fillStyle = 'rgba(7, 9, 12, 0.8)';
    m.beginPath();
    m.roundRect(ox - 140, 240, 280, 42, 8);
    m.fill();
    m.strokeStyle = amber;
    m.lineWidth = 1;
    m.stroke();

    m.fillStyle = amber;
    m.font = '600 12px -apple-system, sans-serif';
    m.textAlign = 'center';
    m.fillText(offline ? 'SENSOR STANDBY · RECONNECTING' : 'AWAITING SURROUND OBSERVATIONS', ox, 266);
  }
}

// -------------------------------------------------------------
// 60 FPS ANIMATION LOOP
// -------------------------------------------------------------
function animate() {
  fpsFrames++;
  const now = performance.now();
  if (now - fpsTimer >= 1000) {
    renderFps = fpsFrames;
    fpsFrames = 0;
    fpsTimer = now;
  }

  tracker.step();
  drawCamera();
  drawMap();
  requestAnimationFrame(animate);
}

// -------------------------------------------------------------
// DOM UPDATES FROM SERVER TELEMETRY
// -------------------------------------------------------------
function render() {
  const s = state;
  if (!s) return;

  const f = displayed || s.frame;
  const scen = f?.scenario || s.scenario;
  const rifeInfo = f?.rife || s.rife;

  $('observation-title').textContent = s.mode === 'live' ? 'Forward vision stream' : s.mode === 'replay' ? 'Recorded stream' : 'Simulated stream';
  $('source').textContent = s.mode === 'live' ? 'LIVE VISION' : s.mode === 'replay' ? 'RECORDED' : 'SIMULATION';
  $('connection').textContent = s.status === 'live' ? (s.mode === 'live' ? '● Receiving frames' : '● Playing preview') : s.status;
  $('pause').textContent = s.status === 'paused' ? 'Resume capture' : 'Pause capture';
  $('pause').disabled = selecting;

  if (s.error) {
    message(`Notice: ${s.error} — attempting seamless reconnection…`, true);
  } else if (s.status === 'stale') {
    message('Sensor stream paused / waiting for new exposure…', false);
  } else if (scen && scen.summary) {
    message(scen.summary, scen.risk_level === 'collision_warning');
  } else if (s.mode !== 'live') {
    message(s.mode === 'replay' ? 'Recorded replay mode active.' : 'Simulated demo mode active.');
  } else {
    message(s.calibration ? 'Calibrated metric ground plane active · Positions projected in meters.' : 'Tesla BEV occupancy memory + GX10 monocular metric depth active.');
  }

  if (s.mode && document.activeElement !== $('source-select')) {
    $('source-select').value = s.mode;
  }

  if (s.rotate !== undefined && document.activeElement !== $('rotate-select')) {
    $('rotate-select').value = String(s.rotate);
  }

  if (s.memory_ttl_s !== undefined && $('memory-select') && document.activeElement !== $('memory-select')) {
    const ttlStr = String(Math.round(s.memory_ttl_s));
    if ([...$('memory-select').options].some(o => o.value === ttlStr)) {
      $('memory-select').value = ttlStr;
    }
  }

  if (s.depth_mode && $('depth-select') && document.activeElement !== $('depth-select')) {
    if ([...$('depth-select').options].some(o => o.value === s.depth_mode)) {
      $('depth-select').value = s.depth_mode;
    }
  }

  if (s.model_spec && $('model-select') && document.activeElement !== $('model-select')) {
    if ([...$('model-select').options].some(o => o.value === s.model_spec)) {
      $('model-select').value = s.model_spec;
    }
  }

  if (s.classes_spec && $('classes-select') && document.activeElement !== $('classes-select')) {
    if ([...$('classes-select').options].some(o => o.value === s.classes_spec)) {
      $('classes-select').value = s.classes_spec;
    }
  }

  if (f) {
    const rotLabel = s.rotate ? ` · ${s.rotate}° CW` : '';
    const streamFps = s.stream_fps || s.capture_fps || 0;
    const rifeTag = rifeInfo?.rife_enabled ? ` · ${streamFps} FPS RIFE STREAM` : '';
    $('resolution').textContent = `${f.width} × ${f.height}${rotLabel} · ${s.capture_fps ?? 0} FPS INGEST${rifeTag} · ${renderFps} FPS UI`;
    $('frame-label').textContent = `${s.mode === 'live' ? 'FRAME' : s.mode === 'replay' ? 'REPLAY' : 'DEMO'} ${f.frame_id} / ${s.status.toUpperCase()}`;
  }

  const badge = $('scenario-badge');
  if (badge && scen) {
    badge.classList.remove('warn', 'danger');
    if (scen.risk_level === 'collision_warning') {
      badge.classList.add('danger');
      badge.textContent = `COLLISION RISK · TTC ${scen.min_ttc_s?.toFixed(1) ?? '<2'}s`;
    } else if (!scen.realistic) {
      badge.classList.add('warn');
      badge.textContent = 'REALISM FILTER: NOISE SUPPRESSED';
    } else if (scen.risk_level === 'approaching') {
      badge.classList.add('warn');
      badge.textContent = `3S FORECAST: APPROACHING (${Math.round(scen.realism_score * 100)}% REALISTIC)`;
    } else {
      const remNote = scen.remembered_count > 0 ? ` · ${scen.remembered_count} IN BEV MEMORY` : '';
      badge.textContent = `3S REALISM: ${Math.round(scen.realism_score * 100)}% VERIFIED${remNote}`;
    }
  }

  // Render actual requested strength independently of binary detection/prediction flags.
  const dec = s.decision || scen?.decision;
  const pred3 = dec?.recommended_pre_alert_zones_3 || f?.predicted_zones_3_2s || s.predicted_zones_3_2s || [0, 0, 0];
  const haptic = s.haptics;
  if ($('zones-3')) {
    $('zones-3').innerHTML = ['LEFT', 'CENTER', 'RIGHT'].map((name, i) => {
      const available = !offline && haptic?.valid && haptic.motor_valid_3?.[i]
        && performance.now() - receivedAt < haptic.ttl_ms;
      const strength = available ? Math.max(0, Math.min(100, haptic.motor_strength_3[i])) : 0;
      const confidence = haptic?.confidence_3?.[i];
      const distance = haptic?.distance_m_3?.[i];
      const score = Number.isFinite(confidence) ? `${Math.round(confidence * 100)}% confidence` : 'Confidence unavailable';
      const depth = Number.isFinite(distance) ? `${distance.toFixed(2)} m` : 'No distance';
      const modeTag = haptic?.override_active
        ? `Manual PWM override (${Math.round(strength * 2.55)}/255)`
        : (haptic?.preview_only || haptic?.source_mode !== 'live' ? 'Preview only' : 'Requested strength');
      return `<div class="zone ${!available ? 'unknown' : strength > 0 ? 'active' : ''}">
        <strong>${name}: ${available ? `${strength}%` : 'UNAVAILABLE'}</strong>
        <progress max="100" value="${strength}" aria-label="${name} requested motor strength"></progress>
        <small>${score} · ${depth}</small>
        <small>${modeTag}</small>
      </div>`;
    }).join('');
  }
  if ($('esp32-sensor-badge')) {
    const esp = s.esp32_sensor;
    if (esp && esp.connected) {
      const uText = esp.ultrasonic_cm != null ? `ULTRA ${esp.ultrasonic_cm.toFixed(1)} cm` : 'LINK ACTIVE';
      const rText = esp.reflex_active ? ' · ⚠️ REFLEX ON' : '';
      $('esp32-sensor-badge').textContent = `ESP32: ${uText}${rText}`;
    } else if (haptic?.override_active) {
      $('esp32-sensor-badge').textContent = `OVERRIDE ACTIVE: [${haptic.motor_strength_3.join('%, ')}%]`;
    } else {
      $('esp32-sensor-badge').textContent = 'ESP32 SENSOR: STANDBY';
    }
  }

  const activeBackend = s.decision_backend || dec?.requested_backend || dec?.backend || 'laya';
  if ($('decision-select') && document.activeElement !== $('decision-select')) {
    $('decision-select').value = activeBackend;
  }
  if ($('decision-toggle-group')) {
    for (const btn of $('decision-toggle-group').querySelectorAll('.decision-pill')) {
      btn.classList.toggle('active', btn.dataset.backend === activeBackend);
    }
  }

  if (dec) {
    if ($('decision-model-tag')) {
      const bMap = {
        'gemma4:26b': 'GX10 GEMMA4:26B VLM',
        'qwen3.6:35b': 'GX10 QWEN3.6:35B VLM',
        'nemotron3:33b': 'GX10 NEMOTRON3:33B VLM',
        'nemotron-3-super:latest': 'GX10 NEMOTRON-3-SUPER',
        'laya': 'LAYA LOCAL',
        'jev': 'JEV API',
        'kinematic': 'KINEMATIC'
      };
      const bLabel = bMap[dec.backend] || (dec.backend || 'GX10').toUpperCase();
      $('decision-model-tag').textContent = `${bLabel} · ${Math.round((dec.confidence ?? 0.9) * 100)}% (${dec.decision_ms ?? 0}ms)`;
    }
    if ($('decision-text')) {
      const note = dec.fallback_note ? ` [${dec.fallback_note}]` : '';
      const vlmDepthNote = dec.depth_reasoning ? ` · Depth Check: ${dec.depth_reasoning}` : '';
      $('decision-text').textContent = `${dec.predicted_next_3s}${vlmDepthNote}${note}`;
    }
    if ($('haptic-pattern-tag')) {
      $('haptic-pattern-tag').textContent = `FORECAST: ${(dec.haptic_pattern || 'silent').toUpperCase().replace(/_/g, ' ')}`;
    }
  } else if (scen && $('decision-text')) {
    $('decision-text').textContent = scen.summary || 'Evaluating 3-keyframe trajectory…';
  }

  // 6-Sector Sub-Grid - shows both active now (solid) and +2s predicted (dashed)
  const activeZones = (f && f.zones) ? f.zones : (s.zones || [-1, -1, -1, -1, -1, -1]);
  const predZones2s = (f && f.predicted_zones_2s) ? f.predicted_zones_2s : (s.predicted_zones_2s || [0, 0, 0, 0, 0, 0]);
  $('zones').innerHTML = activeZones.map((z, i) => {
    const isPred = z === 0 && predZones2s && predZones2s[i] === 1;
    const cls = z === 1 ? 'active' : isPred ? 'predicted' : z === -1 ? 'unknown' : '';
    const sym = z === 1 ? '●' : isPred ? '◎' : z === -1 ? '?' : '·';
    return `
      <div class="zone ${cls}" title="${isPred ? 'Predicted to enter within 2.0s' : ''}">
        ${['L3', 'L2', 'L1', 'R1', 'R2', 'R3'][i]} ${sym}
      </div>
    `;
  }).join('');

  $('count').textContent = tracker.tracks.length;
  if ($('realism-note') && scen) {
    const remCnt = scen.remembered_count ?? (s.remembered_objects?.length || 0);
    const voxCnt = (s.occupancy_cells || []).length;
    $('realism-note').textContent = `${scen.confirmed_tracks ?? 0} live · ${remCnt} remembered · ${voxCnt} BEV voxels`;
  }
  const procMs = f?.processing_ms ?? 0;
  const rifeMs = rifeInfo?.rife_ms ?? 0;
  const predMs = scen?.predict_ms ?? 0;
  const decMs = dec?.decision_ms ?? 0;
  const depthMs = f?.depth?.depth_ms ?? s.depth?.depth_ms ?? 0;
  $('latency').textContent = f?.processing_ms !== undefined ? `${Math.round(procMs)} ms` : `${(rifeMs + predMs + depthMs).toFixed(1)} ms`;
  $('timing-note').textContent = s.mode === 'live'
    ? `Cam ${Math.round(f?.capture_ms ?? 0)}ms · YOLO ${Math.round(procMs)}ms · Depth ${depthMs.toFixed(1)}ms`
    : `${dec ? `${dec.backend.toUpperCase()} ${decMs}ms · ` : ''}BEV+Pred ${predMs.toFixed(2)}ms · Depth ${depthMs.toFixed(2)}ms`;
  $('age-label').textContent = s.mode === 'live' ? 'Since exposure' : 'Since replay presentation';
  $('age').textContent = s.age_ms === null ? '—' : `${(s.age_ms / 1000).toFixed(1)} s`;

  const metric = !!s.calibration || s.map_mode === 'monocular_depth';
  const vlmRep = (f && f.vlm_depth_report) || s.vlm_depth_report || scen?.vlm_depth_report;
  $('map-title').textContent = s.calibration
    ? 'Calibrated floor map'
    : s.map_mode === 'monocular_depth'
      ? 'Tesla BEV + GX10 Metric Depth Map'
      : 'Surround radar view';
  $('map-badge').textContent = s.calibration
    ? '4-PT CALIBRATED'
    : s.map_mode === 'monocular_depth'
      ? (vlmRep?.active ? 'GX10 VLM-VERIFIED BEV' : 'GX10 METRIC BEV')
      : 'SURROUND RADAR';
  if (scen) {
    const zNames3 = ['LEFT', 'CENTER', 'RIGHT'];
    const p3List = (pred3 || []).map((v, idx) => v === 1 ? zNames3[idx] : null).filter(Boolean);
    $('mode-label').textContent = scen.realistic ? (scen.min_ttc_s ? `TTC ${scen.min_ttc_s.toFixed(1)}s` : 'Realistic') : 'Filtered';
    $('mode-note').textContent = p3List.length ? `3-Motor Pre-Alert: ${p3List.join(' + ')}` : '3 Motors Idle [0, 0, 0]';
  } else {
    $('mode-label').textContent = metric ? 'Metric BEV' : 'Surround radar';
    $('mode-note').textContent = metric ? 'Metric distance in meters' : '3-motor directional haptics';
  }

  // Push both live and remembered objects into tracker
  const activeObjects = (f && f.objects) ? f.objects : s.objects;
  const remObjects = (f && f.remembered_objects) ? f.remembered_objects : s.remembered_objects;
  if (s.valid && (activeObjects || remObjects)) {
    tracker.update(activeObjects || [], remObjects || [], 700, 540, metric, s.calibration);
  }
}

// -------------------------------------------------------------
// HTTP POLLER
// -------------------------------------------------------------
async function poll() {
  if (selecting) {
    setTimeout(poll, 100);
    return;
  }
  try {
    const url = '/api/state?after=' + encodeURIComponent(priorFrame ?? '');
    const r = await fetch(url, {
      cache: 'no-store',
      signal: AbortSignal.timeout(2500)
    });
    if (!r.ok) throw Error('Dashboard unavailable');
    const nextState = await r.json();

    if (state && nextState.mode !== state.mode) {
      photo = new Image();
      displayed = null;
      priorFrame = null;
    }

    if (nextState.frame && !selecting && nextState.frame.frame_id !== priorFrame) {
      if (nextState.frame.image) {
        const nextPhoto = new Image();
        nextPhoto.src = 'data:image/jpeg;base64,' + nextState.frame.image;
        await nextPhoto.decode().catch(() => {});
        photo = nextPhoto;
      }
      displayed = nextState.frame;
      priorFrame = displayed.frame_id;
    } else if (nextState.frame && displayed) {
      displayed = { ...displayed, ...nextState.frame };
    }

    state = nextState;
    receivedAt = performance.now();
    offline = false;
    render();
  } catch (e) {
    offline = true;
    $('connection').textContent = 'Sensor standby / Reconnecting';
    render();
  } finally {
    setTimeout(poll, 80);
  }
}

// -------------------------------------------------------------
// EVENT HANDLERS & CALIBRATION
// -------------------------------------------------------------
$('boxes').onchange = () => {};
if ($('forecast')) $('forecast').onchange = () => {};

if ($('camera-res-select')) {
  $('camera-res-select').onchange = async e => {
    const resolution = e.target.value;
    try {
      const res = await post('/api/camera', { resolution, stream_mode: 'auto' });
      priorFrame = null;
      message(`Configured ESP32 persistent stream to ${res.camera_resolution} (quality ${res.camera_quality}).`);
    } catch (err) {
      message(err.message, true);
    }
  };
}

if ($('rife-select')) {
  $('rife-select').onchange = async e => {
    const factor = parseInt(e.target.value, 10);
    try {
      await post('/api/rife', { factor });
      message(factor > 0 ? `RIFE optical-flow interpolation set to ${factor + 1}× smooth stream (${factor} intermediate frames).` : 'RIFE frame interpolation disabled (raw camera frame rate).');
    } catch (err) {
      message(err.message, true);
    }
  };
}

if ($('memory-select')) {
  $('memory-select').onchange = async e => {
    const memory_ttl_s = parseFloat(e.target.value);
    try {
      await post('/api/memory', { memory_ttl_s });
      message(memory_ttl_s > 0
        ? `Tesla/Waymo BEV spatial memory set to ${memory_ttl_s}s retention across occlusions and head turns.`
        : 'BEV spatial memory disabled (live frame detections only).');
    } catch (err) {
      message(err.message, true);
    }
  };
}

if ($('depth-select')) {
  $('depth-select').onchange = async e => {
    const mode = e.target.value;
    try {
      await post('/api/depth', { mode });
      message(mode === 'off'
        ? 'Monocular depth disabled — using 4-point manual floor calibration only.'
        : `GX10 Monocular Metric Depth set to '${mode}' — auto-projecting metric distances (m) without floor tape.`);
    } catch (err) {
      message(err.message, true);
    }
  };
}

if ($('model-select')) {
  $('model-select').onchange = async e => {
    const model = e.target.value;
    try {
      const res = await post('/api/model', { model });
      message(`Switched GX10 detector to '${model}' (resolved: ${res.resolved_model || model}).`);
    } catch (err) {
      message(err.message, true);
    }
  };
}

if ($('classes-select')) {
  $('classes-select').onchange = async e => {
    const classes = e.target.value;
    try {
      const res = await post('/api/classes', { classes });
      const count = Array.isArray(res.classes) ? res.classes.length : 0;
      message(`Updated obstacle detector preset to '${classes}' (${count} active hazard classes).`);
    } catch (err) {
      message(err.message, true);
    }
  };
}

async function setDecisionBackend(backend) {
  if (!backend) return;
  try {
    if ($('decision-select')) $('decision-select').value = backend;
    if ($('decision-toggle-group')) {
      for (const btn of $('decision-toggle-group').querySelectorAll('.decision-pill')) {
        btn.classList.toggle('active', btn.dataset.backend === backend);
      }
    }
    await post('/api/decision', { backend });
    message(`Switched 3s decision & 3-motor haptic model to ${backend.toUpperCase()}.`);
  } catch (err) {
    message(err.message, true);
  }
}

if ($('decision-select')) {
  $('decision-select').onchange = e => setDecisionBackend(e.target.value);
}

if ($('decision-toggle-group')) {
  for (const btn of $('decision-toggle-group').querySelectorAll('.decision-pill')) {
    btn.onclick = () => setDecisionBackend(btn.dataset.backend);
  }
}

$('source-select').onchange = async e => {
  const mode = e.target.value;
  try {
    photo = new Image();
    displayed = null;
    priorFrame = null;
    await post('/api/mode', { mode });
    message(`Switched feed source to ${mode === 'live' ? 'Live Camera' : mode === 'simulation' ? 'Synthetic Simulation' : 'Recorded Replay'}.`);
  } catch (err) {
    message(err.message, true);
  }
};

$('rotate-select').onchange = async e => {
  const rot = parseInt(e.target.value, 10);
  try {
    await post('/api/rotate', { rotate: rot });
    priorFrame = null;
    message(`Camera rotation adjusted to ${rot}° clockwise.`);
  } catch (err) {
    message(err.message, true);
  }
};

$('pause').onclick = async () => {
  try {
    await post('/api/control', { paused: state?.status !== 'paused' });
  } catch (e) {
    message(e.message, true);
  }
};

$('calibrate').onclick = async () => {
  if (!displayed) return message('Wait for a camera frame before calibrating.', true);
  try {
    await post('/api/control', { paused: true });
    selecting = true;
    points = [];
    $('calibration-form').hidden = false;
    $('calibration-message').textContent = 'Click corner 1: near-left';
    for (let i = 0; i < 4; i++) $('p' + i).textContent = 'NOT SELECTED';
  } catch (e) {
    message(e.message, true);
  }
};

camera.onclick = e => {
  if (!selecting || points.length >= 4) return;
  const r = camera.getBoundingClientRect();
  points.push([(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]);
  $('p' + (points.length - 1)).textContent = 'SELECTED';
  $('calibration-message').textContent = points.length < 4
    ? `Click corner ${points.length + 1}: ${pointNames[points.length]}`
    : 'Check measured coordinates, then apply.';
};

async function finish() {
  await post('/api/control', { paused: false });
  selecting = false;
  points = [];
  $('calibration-form').hidden = true;
  priorFrame = null;
}

$('save').onclick = async () => {
  try {
    if (points.length !== 4) throw Error('Select all four floor corners.');
    const p = points.map((p, i) => [...p, Number($('x' + i).value), Number($('y' + i).value)]);
    await post('/api/calibration', { width: displayed.width, height: displayed.height, points: p });
    await finish();
  } catch (e) {
    $('calibration-message').textContent = e.message;
  }
};

$('cancel').onclick = () => finish().catch(e => message(e.message, true));

$('clear').onclick = async () => {
  try {
    await post('/api/calibration', { clear: true });
  } catch (e) {
    message(e.message, true);
  }
};

let sweepTimers = [];
function clearSweepTimers() {
  for (const t of sweepTimers) clearTimeout(t);
  sweepTimers = [];
}

function getTestStrength() {
  const raw = Number($('haptic-test-strength')?.value ?? 35);
  return Math.max(5, Math.min(100, Number.isFinite(raw) ? Math.round(raw) : 35));
}

async function triggerHapticTest(strengths, duration_s = 1.5, label = 'manual_test', stop = false) {
  try {
    const res = await post('/api/haptic_test', { strengths, duration_s, label, stop });
    if (state && res.haptics) {
      state.haptics = { ...res.haptics, preview_only: false };
      receivedAt = performance.now();
      render();
    }
    if (stop || !strengths || strengths.every(v => v === 0)) {
      message('Haptic manual test stopped — returned to automatic obstacle haptics.');
    } else {
      const pwm = strengths.map(s => Math.round((s / 100) * 255));
      message(`Triggered ESP32 3-Motor PWM [L:42, C:21, R:47] = [${strengths.join('%, ')}%] (duty [${pwm.join(', ')}]/255 for ${duration_s}s).`);
    }
  } catch (err) {
    message(err.message, true);
  }
}

if ($('btn-haptic-left')) {
  $('btn-haptic-left').onclick = () => {
    clearSweepTimers();
    const s = getTestStrength();
    triggerHapticTest([s, 0, 0], 1.5, 'left_test');
  };
}
if ($('btn-haptic-center')) {
  $('btn-haptic-center').onclick = () => {
    clearSweepTimers();
    const s = getTestStrength();
    triggerHapticTest([0, s, 0], 1.5, 'center_test');
  };
}
if ($('btn-haptic-right')) {
  $('btn-haptic-right').onclick = () => {
    clearSweepTimers();
    const s = getTestStrength();
    triggerHapticTest([0, 0, s], 1.5, 'right_test');
  };
}
if ($('btn-haptic-all')) {
  $('btn-haptic-all').onclick = () => {
    clearSweepTimers();
    const s = getTestStrength();
    triggerHapticTest([s, s, s], 1.5, 'all_test');
  };
}
if ($('btn-haptic-Sweep') || $('btn-haptic-sweep')) {
  $('btn-haptic-sweep').onclick = () => {
    clearSweepTimers();
    const s = getTestStrength();
    triggerHapticTest([s, 0, 0], 1.0, 'sweep_left');
    sweepTimers.push(setTimeout(() => triggerHapticTest([0, s, 0], 1.0, 'sweep_center'), 950));
    sweepTimers.push(setTimeout(() => triggerHapticTest([0, 0, s], 1.0, 'sweep_right'), 1900));
  };
}
if ($('btn-haptic-stop')) {
  $('btn-haptic-stop').onclick = () => {
    clearSweepTimers();
    triggerHapticTest([0, 0, 0], 0, 'stop', true);
  };
}

// Start 60 FPS animation loop and poller
requestAnimationFrame(animate);
poll();
