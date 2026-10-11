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

// -------------------------------------------------------------
// PERSISTENT OBJECT TRACKER (Smooth interpolation & decay)
// -------------------------------------------------------------
class Tracker {
  constructor() {
    this.tracks = []; // array of {id, class, conf, currX, currY, targetX, targetY, alpha, targetAlpha, ground_xy, lastSeen}
    this.nextId = 1;
  }

  update(detectedObjects, mapWidth, mapHeight, isMetric, calib) {
    const now = performance.now();
    const ox = mapWidth / 2;
    const oy = 475;
    const sx = 58;
    const sy = 50;
    const r = 340;
    const start = -Math.PI * 0.95;
    const end = -Math.PI * 0.05;

    // Convert detected objects to map target coordinates
    const candidates = detectedObjects.map(obj => {
      let tx, ty, distStr;
      if (isMetric && obj.ground_xy) {
        tx = ox + obj.ground_xy[0] * sx;
        ty = oy - obj.ground_xy[1] * sy;
        const dist = Math.hypot(obj.ground_xy[0], obj.ground_xy[1]);
        distStr = `${dist.toFixed(1)} m`;
      } else {
        const u = (obj.xyxy[0] + obj.xyxy[2]) / 2 / (displayed?.width || 640);
        const a = start + (end - start) * u;
        const distRadius = 220; // default bearing arc
        tx = ox + Math.cos(a) * distRadius;
        ty = oy + Math.sin(a) * distRadius;
        const zoneIdx = Math.floor(u * 6);
        const zoneNames = ['L3', 'L2', 'L1', 'R1', 'R2', 'R3'];
        distStr = zoneNames[Math.max(0, Math.min(5, zoneIdx))];
      }
      return {
        class: obj.class,
        confidence: obj.confidence,
        targetX: tx,
        targetY: ty,
        ground_xy: obj.ground_xy,
        distStr
      };
    });

    // Greedy matching between tracks and candidates
    const matched = new Set();
    for (const c of candidates) {
      let bestTrack = null;
      let bestDist = 120; // max pixel distance to associate
      for (const t of this.tracks) {
        if (matched.has(t.id) || t.class !== c.class) continue;
        const d = Math.hypot(t.targetX - c.targetX, t.targetY - c.targetY);
        if (d < bestDist) {
          bestDist = d;
          bestTrack = t;
        }
      }

      if (bestTrack) {
        bestTrack.targetX = c.targetX;
        bestTrack.targetY = c.targetY;
        bestTrack.ground_xy = c.ground_xy;
        bestTrack.distStr = c.distStr;
        bestTrack.confidence = c.confidence;
        bestTrack.targetAlpha = 1;
        bestTrack.lastSeen = now;
        matched.add(bestTrack.id);
      } else {
        const newTrack = {
          id: this.nextId++,
          class: c.class,
          confidence: c.confidence,
          currX: c.targetX,
          currY: c.targetY,
          targetX: c.targetX,
          targetY: c.targetY,
          alpha: 0,
          targetAlpha: 1,
          ground_xy: c.ground_xy,
          distStr: c.distStr,
          lastSeen: now
        };
        this.tracks.push(newTrack);
        matched.add(newTrack.id);
      }
    }

    // Decay tracks not seen in this frame
    for (const t of this.tracks) {
      if (!matched.has(t.id)) {
        if (now - t.lastSeen > 1200) {
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
  const isLive = state?.mode === 'live';
  const hasStream = isLive && streamImg && streamImg.naturalWidth > 0;
  const currentImg = hasStream ? streamImg : photo;
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

  // Draw Detections if toggled
  if ($('boxes').checked && displayed.objects) {
    for (const o of displayed.objects) {
      const [x1, y1, x2, y2] = o.xyxy;
      const w = x2 - x1;
      const h = y2 - y1;
      const color = o.class === 'chair' ? amber : cyan;
      const glow = o.class === 'chair' ? amberGlow : cyanGlow;

      // Translucent box fill
      ctx.fillStyle = o.class === 'chair' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(0, 245, 212, 0.08)';
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

      // High-tech HUD tag pill
      const label = `${o.class.toUpperCase()} ${Math.round(o.confidence * 100)}%`;
      ctx.font = '600 11px -apple-system, sans-serif';
      const tw = ctx.measureText(label).width;
      const tagH = 20;
      const tagY = Math.max(0, y1 - tagH - 4);

      ctx.fillStyle = 'rgba(7, 9, 12, 0.9)';
      ctx.beginPath();
      ctx.roundRect(x1, tagY, tw + 16, tagH, 4);
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = color;
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
// RENDER: TOP-DOWN SURROUND VIEW (TESLA SURROUND RADAR)
// -------------------------------------------------------------
function drawMap() {
  const width = 700;
  const height = 540;
  m.clearRect(0, 0, width, height);

  const ox = width / 2;
  const oy = 475; // Ego vehicle anchor point
  const metric = !!state?.calibration;
  const valid = (state?.valid || tracker.tracks.length > 0) && !offline;

  // 1. Radar background scan line
  radarAngle += 0.025 * radarDirection;
  if (radarAngle > -Math.PI * 0.05) radarDirection = -1;
  if (radarAngle < -Math.PI * 0.95) radarDirection = 1;

  // 2. Concentric Distance Radar Rings
  const ringDistances = [1, 2, 3, 4, 5, 6, 7];
  const ringRadii = metric
    ? ringDistances.map(d => d * 50)
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

  // Lateral metric grid if calibrated
  if (metric) {
    const sx = 58;
    const sy = 50;
    for (let x = -5; x <= 5; x++) {
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
    if (state.calibration?.points) {
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

  // 3. TESLA SURROUND PROXIMITY ARCS (6 Zones: L3, L2, L1, R1, R2, R3)
  const arcR = 340;
  const zoneNames = ['L3', 'L2', 'L1', 'R1', 'R2', 'R3'];
  for (let i = 0; i < 6; i++) {
    const a = startArc + (endArc - startArc) * i / 6;
    const b = startArc + (endArc - startArc) * (i + 1) / 6;
    const mid = (a + b) / 2;
    const currentZones = (displayed && displayed.zones) ? displayed.zones : state?.zones;
    const isActive = valid && currentZones && currentZones[i] === 1;

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
    m.fillStyle = isActive ? cyan : '#64748b';
    m.font = `600 ${isActive ? '12px' : '11px'} -apple-system, sans-serif`;
    m.textAlign = 'center';
    m.fillText(zoneNames[i], lx, ly);
  }

  // 4. DRAW TRACKED OBSTACLES (Smooth interpolation & decay)
  tracker.tracks.forEach(track => {
    if (track.alpha <= 0.01) return;
    m.save();
    m.globalAlpha = track.alpha;

    const tx = track.currX;
    const ty = track.currY;
    const isPerson = track.class === 'person';
    const color = isPerson ? cyan : amber;
    const glow = isPerson ? cyanGlow : amberGlow;

    if (isPerson) {
      // Tesla Pedestrian avatar
      // Ground shadow
      m.fillStyle = 'rgba(0, 0, 0, 0.5)';
      m.beginPath();
      m.ellipse(tx, ty + 2, 14, 8, 0, 0, Math.PI * 2);
      m.fill();

      // Glowing safety halo
      m.strokeStyle = glow;
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
      // Tesla Obstacle / Chair box
      // Ground shadow
      m.fillStyle = 'rgba(0, 0, 0, 0.5)';
      m.beginPath();
      m.ellipse(tx, ty + 4, 16, 10, 0, 0, Math.PI * 2);
      m.fill();

      // 3D obstacle footprint
      m.fillStyle = color;
      m.shadowColor = color;
      m.shadowBlur = 10;
      m.beginPath();
      m.roundRect(tx - 10, ty - 10, 20, 20, 4);
      m.fill();
      m.shadowBlur = 0;

      m.strokeStyle = '#ffffff';
      m.lineWidth = 1;
      m.strokeRect(tx - 6, ty - 6, 12, 12);
    }

    // HUD Label tag
    const tagText = `${track.class.toUpperCase()} · ${track.distStr}`;
    m.font = '600 11px -apple-system, sans-serif';
    const tw = m.measureText(tagText).width;
    const pillW = tw + 16;
    const pillH = 20;
    const pillX = tx - pillW / 2;
    const pillY = ty - 28;

    m.fillStyle = 'rgba(7, 9, 12, 0.88)';
    m.beginPath();
    m.roundRect(pillX, pillY, pillW, pillH, 5);
    m.fill();
    m.strokeStyle = color;
    m.lineWidth = 1;
    m.stroke();

    m.fillStyle = color;
    m.textAlign = 'center';
    m.fillText(tagText, tx, pillY + 14);

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

  // Ego label
  m.fillStyle = '#94a3b8';
  m.font = '600 10px -apple-system, sans-serif';
  m.textAlign = 'center';
  m.fillText('FORWARD SENSOR', ox, oy + 38);

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

  $('observation-title').textContent = s.mode === 'live' ? 'Forward vision stream' : s.mode === 'replay' ? 'Recorded stream' : 'Simulated stream';
  $('source').textContent = s.mode === 'live' ? 'LIVE VISION' : s.mode === 'replay' ? 'RECORDED' : 'SIMULATION';
  $('connection').textContent = s.status === 'live' ? (s.mode === 'live' ? '● Receiving frames' : '● Playing preview') : s.status;
  $('pause').textContent = s.status === 'paused' ? 'Resume capture' : 'Pause capture';
  $('pause').disabled = selecting;

  if (s.error) {
    message(`Notice: ${s.error} — attempting seamless reconnection…`, true);
  } else if (s.status === 'stale') {
    message('Sensor stream paused / waiting for new exposure…', false);
  } else if (s.mode !== 'live') {
    message(s.mode === 'replay' ? 'Recorded replay mode active.' : 'Simulated demo mode active.');
  } else {
    message(s.calibration ? 'Calibrated metric ground plane active · Positions projected in meters.' : 'Surround directional radar active · 6-zone real-time bearing awareness.');
  }

  if (s.mode && document.activeElement !== $('source-select')) {
    $('source-select').value = s.mode;
  }

  if (s.rotate !== undefined && document.activeElement !== $('rotate-select')) {
    $('rotate-select').value = String(s.rotate);
  }

  const f = displayed || s.frame;
  if (f) {
    const rotLabel = s.rotate ? ` · ${s.rotate}° CW` : '';
    $('resolution').textContent = `${f.width} × ${f.height}${rotLabel} · ${s.capture_fps ?? 0} FPS INGEST · ${renderFps} FPS UI`;
    $('frame-label').textContent = `${s.mode === 'live' ? 'FRAME' : s.mode === 'replay' ? 'REPLAY' : 'DEMO'} ${f.frame_id} / ${s.status.toUpperCase()}`;
  }

  // Directional Zones Pill - strictly synchronized to displayed frame
  const activeZones = (f && f.zones) ? f.zones : (s.zones || [-1, -1, -1, -1, -1, -1]);
  $('zones').innerHTML = activeZones.map((z, i) => `
    <div class="zone ${z === 1 ? 'active' : z === -1 ? 'unknown' : ''}">
      ${['L3', 'L2', 'L1', 'R1', 'R2', 'R3'][i]} ${z === 1 ? '●' : z === -1 ? '?' : '·'}
    </div>
  `).join('');

  $('count').textContent = tracker.tracks.length;
  $('latency').textContent = f?.processing_ms !== undefined ? `${Math.round(f.processing_ms)} ms` : '—';
  $('timing-note').textContent = s.mode === 'replay' ? 'Recorded YOLO processing' : s.mode === 'simulation' ? 'Simulation latency' : `Camera ${Math.round(f?.capture_ms ?? 0)} ms · GPU ${Math.round(f?.processing_ms ?? 0)} ms`;
  $('age-label').textContent = s.mode === 'live' ? 'Since exposure' : 'Since replay presentation';
  $('age').textContent = s.age_ms === null ? '—' : `${(s.age_ms / 1000).toFixed(1)} s`;

  const metric = !!s.calibration;
  $('map-title').textContent = metric ? 'Calibrated floor map' : 'Surround radar view';
  $('map-badge').textContent = metric ? 'METRIC ESTIMATE' : 'SURROUND RADAR';
  $('mode-label').textContent = metric ? 'Floor estimate' : 'Surround radar';
  $('mode-note').textContent = metric ? 'Metric distance in meters' : '6-zone directional bearings';

  // Push detected objects into tracker matching displayed frame
  const activeObjects = (f && f.objects) ? f.objects : s.objects;
  if (s.valid && activeObjects) {
    tracker.update(activeObjects, 700, 540, metric, s.calibration);
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
    const isLive = (state?.mode === 'live');
    const hasStream = isLive && (streamConnected || (streamImg && streamImg.naturalWidth > 0));
    const url = '/api/state?after=' + encodeURIComponent(priorFrame ?? '') + (hasStream ? '&meta_only=1' : '');
    const r = await fetch(url, {
      cache: 'no-store',
      signal: AbortSignal.timeout(2500)
    });
    if (!r.ok) throw Error('Dashboard unavailable');
    const nextState = await r.json();

    if (nextState.frame && !selecting && nextState.frame.frame_id !== priorFrame) {
      if (nextState.frame.image) {
        const nextPhoto = new Image();
        nextPhoto.src = 'data:image/jpeg;base64,' + nextState.frame.image;
        await nextPhoto.decode().catch(() => {});
        photo = nextPhoto;
      }
      displayed = nextState.frame;
      priorFrame = displayed.frame_id;
    } else if (nextState.frame) {
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

$('source-select').onchange = async e => {
  const mode = e.target.value;
  try {
    await post('/api/mode', { mode });
    priorFrame = null;
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

// Start 60 FPS animation loop and poller
requestAnimationFrame(animate);
poll();
