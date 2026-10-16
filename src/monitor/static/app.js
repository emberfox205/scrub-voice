/**
 * Scrub-Voice Intraoperative Monitor Client
 * Adheres to IBM Carbon g100 dark theme and OHIF Viewer DICOM standards.
 * Pure JavaScript, 100% offline, zero CDN dependencies.
 */

(function () {
  'use strict';

  // Global UI State Cache
  let monitorState = {
    active_scan: null,
    current_slice: 1,
    zoom_factor: 1.0,
    held_tool: null,
    recent_milestones: [],
    emergency_active: false,
    last_command: null,
    telemetry: {
      status: "STANDBY / OFFLINE",
      x_mm: 0.0,
      y_mm: 0.0,
      z_mm: 0.0,
      rx_deg: 0.0,
      ry_deg: 0.0,
      rz_deg: 0.0,
      gripper_status: "STANDBY / OFFLINE",
      gripper_force_n: 0.0,
      gripper_width_mm: 0.0
    }
  };

  // In-memory Voice Audit Log Records
  const auditRecords = [];
  const MAX_AUDIT_RECORDS = 50;

  // DOM Elements - Voice Status Bar
  const transcriptText = document.getElementById('transcriptText');
  const confidencePill = document.getElementById('confidencePill');
  const latencyPill = document.getElementById('latencyPill');

  // DOM Elements - DICOM Viewport
  const canvas = document.getElementById('dicomCanvas');
  const ctx = canvas.getContext('2d');
  const hudScanType = document.getElementById('hudScanType');
  const statSlice = document.getElementById('statSlice');
  const statZoom = document.getElementById('statZoom');

  // DOM Elements - Audit Log
  const auditLogTableBody = document.getElementById('auditLogTableBody');

  // DOM Elements - Subsystem Telemetry
  const telemRobotStatus = document.getElementById('telemRobotStatus');
  const telemPose = document.getElementById('telemPose');
  const telemOrientation = document.getElementById('telemOrientation');
  const telemHeldTool = document.getElementById('telemHeldTool');
  const telemGripperStatus = document.getElementById('telemGripperStatus');
  const telemGripperForce = document.getElementById('telemGripperForce');
  const telemGripperWidth = document.getElementById('telemGripperWidth');
  const telemForceMeter = document.getElementById('telemForceMeter');
  const telemWidthMeter = document.getElementById('telemWidthMeter');

  // Resize canvas for crisp rendering (retina-ready)
  function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    renderViewport();
  }
  window.addEventListener('resize', resizeCanvas);

  // WebSocket Connection with Resilient Auto-Reconnect
  let ws = null;
  let reconnectInterval = 1000;

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
      reconnectInterval = 1000;
    };

    ws.onmessage = function (event) {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'snapshot') {
          monitorState = msg.state;
          if (monitorState.recent_milestones && monitorState.recent_milestones.length > 0) {
            syncMilestonesToAuditLog(monitorState.recent_milestones);
          }
          updateUI(true);
        } else if (msg.type === 'command' || msg.type === 'state_update') {
          if (msg.state) {
            monitorState = msg.state;
          }
          if (msg.command) {
            updateSpeechTicker(msg.command);
            appendAuditRecord(msg.command);
          }
          updateUI(false);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onclose = function () {
      setTimeout(connectWebSocket, Math.min(reconnectInterval, 5000));
      reconnectInterval *= 1.5;
    };

    ws.onerror = function () {
      ws.close();
    };
  }

  // Update Voice Command Status Bar
  function updateSpeechTicker(cmd) {
    if (!cmd) return;

    if (transcriptText) {
      transcriptText.textContent = `"${cmd.raw_transcript || '---'}"`;
    }

    // Confidence Metric Badge (Color matched with Latency metric)
    const conf = typeof cmd.confidence === 'number' ? cmd.confidence : 1.0;
    const confPct = Math.round(conf * 100);
    if (confidencePill) {
      confidencePill.textContent = `Confidence: ${confPct}%`;
    }

    // Processing Latency Metric
    const latency = cmd.latency_ms || 0.0;
    if (latencyPill) {
      latencyPill.textContent = `Latency: ${latency.toFixed(1)} ms`;
    }
  }

  // HTML Sanitizer
  function escapeHtml(str) {
    if (str == null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Append incoming voice command to audit record buffer
  function appendAuditRecord(cmd) {
    if (!cmd) return;
    const ts = cmd.timestamp ? cmd.timestamp : (Date.now() / 1000);
    const trigger = cmd.raw_transcript ? `"${cmd.raw_transcript}"` : '---';
    const intent = cmd.intent || 'unknown';
    const status = (cmd.status || 'EXECUTE').toUpperCase();

    auditRecords.unshift({
      timestamp: ts,
      voice_trigger: trigger,
      intent: intent,
      status: status
    });

    if (auditRecords.length > MAX_AUDIT_RECORDS) {
      auditRecords.pop();
    }
  }

  // Synchronize milestones from state snapshot
  function syncMilestonesToAuditLog(milestones) {
    if (!milestones || milestones.length === 0) return;
    for (let i = 0; i < milestones.length; i++) {
      const m = milestones[i];
      const commentStr = m.comment ? ` — ${m.comment}` : '';
      const trigger = `"${m.event_name}${commentStr}"`;
      auditRecords.unshift({
        timestamp: m.timestamp || (Date.now() / 1000),
        voice_trigger: trigger,
        intent: 'log_event',
        status: 'LOGGED'
      });
    }
    if (auditRecords.length > MAX_AUDIT_RECORDS) {
      auditRecords.length = MAX_AUDIT_RECORDS;
    }
  }

  // Render EMR Voice Audit Log Terminal Table (4 Columns Whitelist)
  function updateAuditLog() {
    if (!auditLogTableBody) return;

    if (auditRecords.length === 0) {
      const milestones = monitorState.recent_milestones || [];
      if (milestones.length === 0) {
        auditLogTableBody.innerHTML = '<tr><td colspan="4" class="empty-log-msg">No intraoperative voice events recorded yet.</td></tr>';
        return;
      }
      syncMilestonesToAuditLog(milestones);
    }

    let rowsHtml = '';
    for (let i = 0; i < auditRecords.length; i++) {
      const item = auditRecords[i];
      const d = new Date(item.timestamp * 1000);
      const timeStr = d.toTimeString().split(' ')[0] + '.' + String(Math.floor((item.timestamp % 1) * 1000)).padStart(3, '0').slice(0, 2);

      let statusTagClass = 'status-standby';
      if (item.status === 'EXECUTE' || item.status === 'LOGGED') {
        statusTagClass = 'status-normal';
      } else if (item.status === 'AMBIGUOUS' || item.status === 'CLEARING' || item.status === 'EVACUATING') {
        statusTagClass = 'status-warning';
      } else if (item.status === 'HALT' || item.status === 'EMERGENCY' || item.status === 'CRITICAL') {
        statusTagClass = 'status-critical';
      }

      rowsHtml += `
        <tr>
          <td class="col-time">${escapeHtml(timeStr)}</td>
          <td class="col-trigger">${escapeHtml(item.voice_trigger)}</td>
          <td class="col-intent">${escapeHtml(item.intent)}</td>
          <td class="col-status"><span class="table-status-tag ${statusTagClass}">${escapeHtml(item.status)}</span></td>
        </tr>
      `;
    }

    auditLogTableBody.innerHTML = rowsHtml;
  }

  // Update Subsystems Telemetry Grid
  function updateTelemetry() {
    const telem = monitorState.telemetry || {};

    if (telemRobotStatus) {
      const st = (telem.status || 'STANDBY / OFFLINE').toUpperCase();
      telemRobotStatus.textContent = st;
      if (st.includes('EMERGENCY') || st.includes('HALT') || st.includes('LOCKED')) {
        telemRobotStatus.className = 'status-tag critical';
      } else if (st.includes('EVACUATING') || st.includes('CLEARING')) {
        telemRobotStatus.className = 'status-tag warning';
      } else if (st.includes('ONLINE') || st.includes('OPERATIONAL')) {
        telemRobotStatus.className = 'status-tag normal';
      } else {
        telemRobotStatus.className = 'status-tag standby';
      }
    }

    if (telemPose) {
      const x = (telem.x_mm || 0.0).toFixed(1);
      const y = (telem.y_mm || 0.0).toFixed(1);
      const z = (telem.z_mm || 0.0).toFixed(1);
      telemPose.textContent = `X: ${x} mm | Y: ${y} mm | Z: ${z} mm`;
    }

    if (telemOrientation) {
      const rx = (telem.rx_deg || 0.0).toFixed(1);
      const ry = (telem.ry_deg || 0.0).toFixed(1);
      const rz = (telem.rz_deg || 0.0).toFixed(1);
      telemOrientation.textContent = `Rx: ${rx}° | Ry: ${ry}° | Rz: ${rz}°`;
    }

    if (telemHeldTool) {
      telemHeldTool.textContent = monitorState.held_tool ? monitorState.held_tool.toUpperCase() : 'NONE / PARKED';
    }

    if (telemGripperStatus) {
      telemGripperStatus.textContent = telem.gripper_status || 'STANDBY / OFFLINE';
    }

    const forceN = telem.gripper_force_n || 0.0;
    if (telemGripperForce) {
      telemGripperForce.textContent = `${forceN.toFixed(1)} N`;
    }
    if (telemForceMeter) {
      const pct = Math.min(100, Math.max(0, (forceN / 40.0) * 100));
      telemForceMeter.style.width = `${pct}%`;
    }

    const widthMm = telem.gripper_width_mm || 0.0;
    if (telemGripperWidth) {
      telemGripperWidth.textContent = `${widthMm.toFixed(1)} mm`;
    }
    if (telemWidthMeter) {
      const pct = Math.min(100, Math.max(0, (widthMm / 110.0) * 100));
      telemWidthMeter.style.width = `${pct}%`;
    }
  }

  // Render Synthetic DICOM / Medical Viewport (Pure Black, OHIF Standard)
  function renderViewport() {
    if (!canvas || !ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    // Pure pitch black viewport background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);

    const activeScan = monitorState.active_scan;
    const slice = monitorState.current_slice || 1;
    const zoom = monitorState.zoom_factor || 1.0;

    // Update HUD text displays
    if (hudScanType) hudScanType.textContent = activeScan ? activeScan.toUpperCase() : 'STANDBY';
    if (statSlice) statSlice.textContent = `${slice}`;
    if (statZoom) statZoom.textContent = `${zoom.toFixed(2)}x`;

    if (!activeScan) {
      drawStandbyScreen(w, h);
      return;
    }

    ctx.save();
    // Center-point scaling based on zoom_factor
    ctx.translate(w / 2, h / 2);
    ctx.scale(zoom, zoom);
    ctx.translate(-w / 2, -h / 2);

    if (activeScan && scanImages[activeScan]) {
      renderScanImage(w, h, activeScan);
    }

    ctx.restore();

    // Render Overlay Crosshairs and Orientation Markings in screen space
    drawHUDOverlays(w, h, activeScan, slice);
  }

  // Preloaded Open Source Clinical Scans (100% Offline Static Assets)
  const scanImages = {
    ct: new Image(),
    mri: new Image(),
    xray: new Image()
  };
  scanImages.ct.src = '/static/scans/ct.jpg';
  scanImages.mri.src = '/static/scans/mri.jpg';
  scanImages.xray.src = '/static/scans/xray.jpg';

  // Redraw when any scan image finishes loading
  Object.keys(scanImages).forEach(function (k) {
    scanImages[k].onload = function () {
      renderViewport();
    };
  });

  // Render Real Open Source Clinical Scan Image (Aspect-Ratio Fitted)
  function renderScanImage(w, h, scanType) {
    const img = scanImages[scanType];
    if (!img || !img.complete || img.naturalWidth === 0) {
      ctx.fillStyle = '#161616';
      ctx.fillRect(w * 0.1, h * 0.1, w * 0.8, h * 0.8);
      ctx.fillStyle = '#8d8d8d';
      ctx.font = '600 12px "IBM Plex Sans", system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('LOADING CLINICAL DICOM MODALITY: ' + scanType.toUpperCase(), w / 2, h / 2);
      return;
    }

    const imgW = img.naturalWidth;
    const imgH = img.naturalHeight;

    // Maintain aspect ratio, fit cleanly within 88% of viewport
    const maxW = w * 0.88;
    const maxH = h * 0.88;
    const scale = Math.min(maxW / imgW, maxH / imgH);

    const drawW = imgW * scale;
    const drawH = imgH * scale;
    const drawX = (w - drawW) / 2;
    const drawY = (h - drawH) / 2;

    ctx.drawImage(img, drawX, drawY, drawW, drawH);
  }

  // Clean Standby Screen (Strictly Sans-Serif, Zero Monospace)
  function drawStandbyScreen(w, h) {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillStyle = '#8d8d8d';
    ctx.font = '600 13px "IBM Plex Sans", system-ui, -apple-system, sans-serif';
    ctx.fillText('NO ACTIVE SCAN MODALITY', w / 2, h / 2 - 10);

    ctx.fillStyle = '#6f6f6f';
    ctx.font = '11px "IBM Plex Sans", system-ui, -apple-system, sans-serif';
    ctx.fillText('Awaiting DICOM modality ("Open CT scan", "Open MRI", "Open X-ray")', w / 2, h / 2 + 12);
  }

  // Draw HUD Overlays: Clean Muted Gray Crosshairs & Sans-Serif Orientation
  function drawHUDOverlays(w, h, activeScan, slice) {
    // Center reticle
    ctx.strokeStyle = 'rgba(142, 150, 160, 0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w / 2 - 12, h / 2);
    ctx.lineTo(w / 2 + 12, h / 2);
    ctx.moveTo(w / 2, h / 2 - 12);
    ctx.lineTo(w / 2, h / 2 + 12);
    ctx.stroke();

    // Patient orientation labels (A, P, R, L) — strictly sans-serif
    ctx.fillStyle = '#c6c6c6';
    ctx.font = '600 11px "IBM Plex Sans", system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('A', w / 2, 22);
    ctx.fillText('P', w / 2, h - 14);
    ctx.textAlign = 'left';
    ctx.fillText('R', 14, h / 2);
    ctx.textAlign = 'right';
    ctx.fillText('L', w - 14, h / 2);

    // Scale bar in bottom right
    const barLen = 50;
    const barX = w - barLen - 16;
    const barY = h - 22;
    ctx.strokeStyle = '#6f6f6f';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(barX, barY);
    ctx.lineTo(barX + barLen, barY);
    ctx.moveTo(barX, barY - 3);
    ctx.lineTo(barX, barY + 3);
    ctx.moveTo(barX + barLen, barY - 3);
    ctx.lineTo(barX + barLen, barY + 3);
    ctx.stroke();

    ctx.fillStyle = '#8d8d8d';
    ctx.font = '10px "IBM Plex Sans", system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('5 cm', barX + barLen / 2, barY - 5);
  }

  // Master UI Update
  function updateUI(isInitial) {
    if (isInitial && monitorState.last_command) {
      updateSpeechTicker(monitorState.last_command);
    }
    updateAuditLog();
    updateTelemetry();
    renderViewport();
  }

  // Start initialization on DOM Ready
  window.addEventListener('DOMContentLoaded', function () {
    resizeCanvas();
    connectWebSocket();
  });
})();
