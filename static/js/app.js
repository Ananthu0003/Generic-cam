/**
 * Main Application Controller for Generic CAM Studio
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize 3D Viewer
  const viewer = new Cam3DViewer('canvas-3d');

  // Application State
  const state = {
    currentStep: 1,
    model: null,
    stock: {
      margin_x: 5.0,
      margin_y: 5.0,
      margin_z_top: 1.0,
      margin_z_bottom: 3.0,
      material: 'aluminum_6061',
    },
    features: [],
    tools: [],
    plannedOps: [],
    toolpaths: [],
    gcodeResult: null,
  };

  // DOM Elements
  const headerModelName = document.getElementById('header-model-name');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const btnLoadDemo = document.getElementById('btn-load-demo');
  const btnExportGcode = document.getElementById('btn-export-gcode');

  // Stepper Elements
  const stepBtns = document.querySelectorAll('.step-btn');
  const stepPanels = document.querySelectorAll('.step-panel');
  const nextStepBtns = document.querySelectorAll('.next-step-btn');

  // Step 1: Model Stats
  const statDimX = document.getElementById('stat-dim-x');
  const statDimY = document.getElementById('stat-dim-y');
  const statDimZ = document.getElementById('stat-dim-z');
  const statMeshCount = document.getElementById('stat-mesh-count');

  // Step 2: Stock Inputs
  const inputMarginXY = document.getElementById('stock-margin-xy');
  const inputMarginZTop = document.getElementById('stock-margin-z-top');
  const inputMarginZBot = document.getElementById('stock-margin-z-bot');
  const selectMaterial = document.getElementById('stock-material');

  // Step 3: Features
  const featureListContainer = document.getElementById('feature-list-container');
  const btnRunFeatureRec = document.getElementById('btn-run-feature-rec');

  // Step 4: Tools & Ops
  const toolListContainer = document.getElementById('tool-list-container');
  const opListContainer = document.getElementById('op-list-container');

  // Step 5: Toolpaths & Simulation
  const btnGenerateToolpaths = document.getElementById('btn-generate-toolpaths');
  const statCutLen = document.getElementById('stat-cut-len');
  const statRapidLen = document.getElementById('stat-rapid-len');
  const statEstTime = document.getElementById('stat-est-time');
  const statSegCount = document.getElementById('stat-seg-count');
  const simBtnPlay = document.getElementById('sim-btn-play');
  const simBtnReset = document.getElementById('sim-btn-reset');
  const simSpeedSelect = document.getElementById('sim-speed-select');
  const simScrubber = document.getElementById('sim-scrubber');
  const simCurrTime = document.getElementById('sim-curr-time');
  const simTotalTime = document.getElementById('sim-total-time');

  // Step 6: G-Code
  const postControllerSelect = document.getElementById('post-controller-select');
  const gcodeLineCounter = document.getElementById('gcode-line-counter');
  const gcodeOutput = document.getElementById('gcode-output');
  const btnCopyGcode = document.getElementById('btn-copy-gcode');
  const btnDownloadFile = document.getElementById('btn-download-file');

  // Viewport Toolbar Buttons
  const togglePartBtn = document.getElementById('toggle-part');
  const toggleStockBtn = document.getElementById('toggle-stock');
  const toggleToolpathBtn = document.getElementById('toggle-toolpath');
  const toggleWireframeBtn = document.getElementById('toggle-wireframe');
  const viewIsoBtn = document.getElementById('view-iso');
  const viewTopBtn = document.getElementById('view-top');
  const viewFrontBtn = document.getElementById('view-front');
  const viewResetBtn = document.getElementById('view-reset');

  // HUD Readouts
  const hudSpindle = document.getElementById('hud-spindle');
  const hudFeed = document.getElementById('hud-feed');
  const hudTool = document.getElementById('hud-tool');
  const hudXyz = document.getElementById('hud-xyz');

  // ------------------------------------------------------------- Stepper Logic
  function setStep(stepNum) {
    state.currentStep = stepNum;
    stepBtns.forEach((btn) => {
      btn.classList.toggle('active', parseInt(btn.dataset.step) === stepNum);
    });
    stepPanels.forEach((panel) => {
      panel.classList.toggle('active', panel.id === `panel-step-${stepNum}`);
    });

    // Auto-trigger appropriate step workflow if needed
    if (stepNum === 3 && state.features.length === 0) {
      loadFeatures();
    } else if (stepNum === 4 && state.plannedOps.length === 0) {
      loadOperations();
    } else if (stepNum === 5 && state.toolpaths.length === 0) {
      generateToolpaths();
    } else if (stepNum === 6 && !state.gcodeResult) {
      generateGcode();
    }
  }

  stepBtns.forEach((btn) => {
    btn.addEventListener('click', () => setStep(parseInt(btn.dataset.step)));
  });

  nextStepBtns.forEach((btn) => {
    btn.addEventListener('click', () => setStep(parseInt(btn.dataset.next)));
  });

  // ----------------------------------------------------------- Telemetry Hooks
  viewer.onTelemetryUpdate = (data) => {
    hudXyz.textContent = `X ${data.x.toFixed(2)}  Y ${data.y.toFixed(2)}  Z ${data.z.toFixed(2)}`;
    hudFeed.textContent = `${Math.round(data.feed)} mm/min`;
    hudSpindle.textContent = `${Math.round(data.spindle)} RPM`;
    if (data.tool_id) hudTool.textContent = data.tool_id;

    if (!simScrubber.matches(':active')) {
      simScrubber.value = (data.progress * 100).toFixed(1);
    }

    const totalSeconds = state.gcodeResult ? state.gcodeResult.cycle_time_seconds : 60;
    const curSec = Math.round(data.progress * totalSeconds);
    simCurrTime.textContent = formatTime(curSec);
    simTotalTime.textContent = formatTime(Math.round(totalSeconds));
  };

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }

  // ----------------------------------------------------------- API Interactions
  async function loadDemoModel() {
    try {
      const res = await fetch('/api/model/demo');
      const data = await res.json();
      applyLoadedModel(data);
    } catch (err) {
      console.error('Failed to load demo model:', err);
    }
  }

  function applyLoadedModel(data) {
    state.model = data;
    headerModelName.textContent = data.name;

    const bbox = data.mesh.bounding_box;
    statDimX.textContent = `${bbox.size[0].toFixed(2)} mm`;
    statDimY.textContent = `${bbox.size[1].toFixed(2)} mm`;
    statDimZ.textContent = `${bbox.size[2].toFixed(2)} mm`;
    statMeshCount.textContent = `${data.mesh.triangle_count} / ${data.mesh.face_count}`;

    viewer.loadModelMesh(data.mesh);
    updateStockBounds();

    // Reset dependent state
    state.features = [];
    state.plannedOps = [];
    state.toolpaths = [];
    state.gcodeResult = null;
  }

  function updateStockBounds() {
    if (!state.model) return;
    const bbox = state.model.mesh.bounding_box;
    const marginXY = parseFloat(inputMarginXY.value) || 5.0;
    const marginZTop = parseFloat(inputMarginZTop.value) || 1.0;
    const marginZBot = parseFloat(inputMarginZBot.value) || 3.0;

    state.stock.margin_x = marginXY;
    state.stock.margin_y = marginXY;
    state.stock.margin_z_top = marginZTop;
    state.stock.margin_z_bottom = marginZBot;

    const stockBounds = {
      min: [bbox.min[0] - marginXY, bbox.min[1] - marginXY, bbox.min[2] - marginZBot],
      max: [bbox.max[0] + marginXY, bbox.max[1] + marginXY, bbox.max[2] + marginZTop],
    };
    viewer.updateStock(stockBounds);
  }

  [inputMarginXY, inputMarginZTop, inputMarginZBot].forEach((input) => {
    input.addEventListener('input', updateStockBounds);
  });

  // Upload Step File
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
      await handleFileUpload(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      await handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    try {
      headerModelName.textContent = 'Parsing CAD Geometry...';
      const res = await fetch('/api/model/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`Error importing file: ${err.detail || 'Invalid STEP file'}`);
        headerModelName.textContent = state.model ? state.model.name : 'Ready';
        return;
      }
      const data = await res.json();
      applyLoadedModel(data);
    } catch (err) {
      console.error('File upload failed:', err);
      alert('Failed to upload file.');
    }
  }

  // ------------------------------------------------------------- Features Logic
  async function loadFeatures() {
    try {
      const res = await fetch('/api/recognize-features', { method: 'POST' });
      const data = await res.json();
      state.features = data.features;
      renderFeaturesList(data.features);
    } catch (err) {
      console.error('Failed to recognize features:', err);
    }
  }

  function renderFeaturesList(features) {
    featureListContainer.innerHTML = '';
    if (features.length === 0) {
      featureListContainer.innerHTML = '<div class="text-dim p-2">No features recognized.</div>';
      return;
    }

    features.forEach((f) => {
      const item = document.createElement('div');
      item.className = 'feature-item';
      item.dataset.featureId = f.id;

      const title = f.id.replace('feat_', '').replace('_', ' ');
      const dimStr = f.diameter ? `Ø${f.diameter.toFixed(1)}mm` : `Depth: ${f.depth.toFixed(1)}mm`;

      item.innerHTML = `
        <div>
          <div style="font-weight: 500; font-size: 0.85rem; text-transform: capitalize;">${title}</div>
          <div style="font-size: 0.72rem; color: var(--text-dim);">${dimStr}</div>
        </div>
        <span class="feature-tag ${f.type}">${f.type}</span>
      `;

      item.addEventListener('mouseenter', () => viewer.highlightFeature(f));
      item.addEventListener('mouseleave', () => viewer.highlightFeature(null));
      item.addEventListener('click', () => {
        document.querySelectorAll('.feature-item').forEach((el) => el.classList.remove('selected'));
        item.classList.add('selected');
        viewer.highlightFeature(f);
      });

      featureListContainer.appendChild(item);
    });
  }

  btnRunFeatureRec.addEventListener('click', loadFeatures);

  // ----------------------------------------------------------- Tooling & Operations
  async function loadToolsAndOps() {
    try {
      const resTools = await fetch('/api/tools-and-machines');
      const dataTools = await resTools.json();
      state.tools = dataTools.tools;
      renderToolsList(dataTools.tools);
    } catch (err) {
      console.error('Failed to load tooling:', err);
    }
  }

  function renderToolsList(tools) {
    toolListContainer.innerHTML = '';
    tools.forEach((t) => {
      const item = document.createElement('div');
      item.className = 'tool-card-item';
      item.innerHTML = `
        <div>
          <div style="font-weight: 600; font-size: 0.85rem;" class="mono text-cyan">${t.id}: ${t.name}</div>
          <div style="font-size: 0.72rem; color: var(--text-dim);">Max ${t.max_rpm} RPM | ${t.flutes} Flutes | Ø${t.diameter}mm</div>
        </div>
      `;
      toolListContainer.appendChild(item);
    });
  }

  async function loadOperations() {
    try {
      const res = await fetch('/api/plan-operations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stock: state.stock }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`Operation Planning Notice:\n${err.detail || 'Failed to plan operations'}`);
        return;
      }
      const data = await res.json();
      state.plannedOps = data.operations;
      renderOpsList(data.operations);
    } catch (err) {
      console.error('Failed to plan operations:', err);
    }
  }

  function renderOpsList(ops) {
    opListContainer.innerHTML = '';
    ops.forEach((op, index) => {
      const item = document.createElement('div');
      item.className = 'op-card-item';
      item.innerHTML = `
        <div>
          <div style="font-weight: 600; font-size: 0.85rem;">
            ${index + 1}. <span style="text-transform: uppercase;">${op.purpose}</span> (${op.feature_id})
          </div>
          <div style="font-size: 0.72rem; color: var(--text-dim);">
            ${op.tool_id} | Feed: ${op.feed_rate} mm/min | ${op.spindle_rpm} RPM
          </div>
        </div>
        <span class="feature-tag ${op.purpose}">${op.purpose}</span>
      `;
      opListContainer.appendChild(item);
    });
  }

  // ------------------------------------------------------------- Toolpath Generation
  async function generateToolpaths() {
    btnGenerateToolpaths.disabled = true;
    btnGenerateToolpaths.innerHTML = 'Calculating Paths...';

    try {
      const res = await fetch('/api/generate-toolpaths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stock: state.stock }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`Toolpath Generation Notice:\n${err.detail || 'Failed to generate toolpaths'}`);
        return;
      }
      const data = await res.json();
      state.toolpaths = data.toolpaths;

      // Update metrics
      statCutLen.textContent = `${data.total_cutting_length_mm.toFixed(1)} mm`;
      statRapidLen.textContent = `${data.total_rapid_length_mm.toFixed(1)} mm`;
      statEstTime.textContent = formatTime(Math.round(data.estimated_time_seconds));
      statSegCount.textContent = data.toolpaths.reduce((acc, tp) => acc + tp.segment_count, 0);

      viewer.renderToolpaths(data.toolpaths);
    } catch (err) {
      console.error('Failed to generate toolpaths:', err);
    } finally {
      btnGenerateToolpaths.disabled = false;
      btnGenerateToolpaths.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Generate Toolpaths
      `;
    }
  }

  btnGenerateToolpaths.addEventListener('click', generateToolpaths);

  // ------------------------------------------------------------- 3D Simulation Controls
  simBtnPlay.addEventListener('click', () => {
    viewer.isPlaying = !viewer.isPlaying;
    const icon = document.getElementById('sim-play-icon');
    if (viewer.isPlaying) {
      icon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
    } else {
      icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
    }
  });

  simBtnReset.addEventListener('click', () => {
    viewer.isPlaying = false;
    viewer.setSimulationProgress(0);
    simScrubber.value = 0;
    const icon = document.getElementById('sim-play-icon');
    icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  });

  simSpeedSelect.addEventListener('change', (e) => {
    viewer.simSpeed = parseFloat(e.target.value);
  });

  simScrubber.addEventListener('input', (e) => {
    viewer.isPlaying = false;
    viewer.setSimulationProgress(parseFloat(e.target.value) / 100);
  });

  // ------------------------------------------------------------- G-Code Generation
  async function generateGcode() {
    const controller = postControllerSelect.value;
    try {
      const res = await fetch('/api/generate-gcode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ controller: controller, program_name: 'PART_01' }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`G-Code Generation Notice:\n${err.detail || 'Failed to generate G-Code'}`);
        return;
      }
      const data = await res.json();
      state.gcodeResult = data;
      gcodeOutput.textContent = data.gcode;
      gcodeLineCounter.textContent = `${data.line_count} lines (${(data.cycle_time_seconds / 60).toFixed(1)} min)`;
    } catch (err) {
      console.error('Failed to post-process G-Code:', err);
    }
  }

  postControllerSelect.addEventListener('change', generateGcode);
  btnExportGcode.addEventListener('click', () => {
    setStep(6);
    generateGcode();
  });

  btnCopyGcode.addEventListener('click', () => {
    if (state.gcodeResult) {
      navigator.clipboard.writeText(state.gcodeResult.gcode);
      btnCopyGcode.textContent = 'Copied!';
      setTimeout(() => (btnCopyGcode.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy
      `), 2000);
    }
  });

  btnDownloadFile.addEventListener('click', () => {
    if (!state.gcodeResult) return;
    const blob = new Blob([state.gcodeResult.gcode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PART_01_${state.gcodeResult.controller}.nc`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // ------------------------------------------------------------- Viewport Toolbar
  let partVisible = true;
  let stockVisible = true;
  let toolpathVisible = true;
  let wireframeEnabled = false;

  togglePartBtn.addEventListener('click', () => {
    partVisible = !partVisible;
    viewer.togglePart(partVisible);
    togglePartBtn.classList.toggle('active', partVisible);
  });

  toggleStockBtn.addEventListener('click', () => {
    stockVisible = !stockVisible;
    viewer.toggleStock(stockVisible);
    toggleStockBtn.classList.toggle('active', stockVisible);
  });

  toggleToolpathBtn.addEventListener('click', () => {
    toolpathVisible = !toolpathVisible;
    viewer.toggleToolpath(toolpathVisible);
    toggleToolpathBtn.classList.toggle('active', toolpathVisible);
  });

  toggleWireframeBtn.addEventListener('click', () => {
    wireframeEnabled = !wireframeEnabled;
    viewer.toggleWireframe(wireframeEnabled);
    toggleWireframeBtn.classList.toggle('active', wireframeEnabled);
  });

  viewIsoBtn.addEventListener('click', () => viewer.setCameraView('iso'));
  viewTopBtn.addEventListener('click', () => viewer.setCameraView('top'));
  viewFrontBtn.addEventListener('click', () => viewer.setCameraView('front'));
  viewResetBtn.addEventListener('click', () => viewer.setCameraView('reset'));

  btnLoadDemo.addEventListener('click', () => {
    loadDemoModel();
    setStep(1);
  });

  // ------------------------------------------------------------- Initial Boot
  loadToolsAndOps();
  loadDemoModel();
});
