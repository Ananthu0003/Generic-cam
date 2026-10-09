/**
 * Main Application Controller for VexCAM Studio
 * Supports CAD Model Orientation Alignment & Multi-Setup (OP10/OP20/OP30) Planning
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
      offset_x: 0.0,
      offset_y: 0.0,
      offset_z: 0.0,
      material: 'aluminum_6061',
      clamp_type: 'vise_jaws',
      clamp_height: 3.0,
      clamp_width: 14.0,
    },
    setups: [],
    activeSetupId: 'setup_001',
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

  // Step 1: Model Stats & Orientation
  const statDimX = document.getElementById('stat-dim-x');
  const statDimY = document.getElementById('stat-dim-y');
  const statDimZ = document.getElementById('stat-dim-z');
  const statMeshCount = document.getElementById('stat-mesh-count');
  const btnOrientRx90 = document.getElementById('btn-orient-rx-90');
  const btnOrientRy90 = document.getElementById('btn-orient-ry-90');
  const btnOrientRz90 = document.getElementById('btn-orient-rz-90');
  const btnOrientFlipX = document.getElementById('btn-orient-flip-x');
  const btnOrientFlipY = document.getElementById('btn-orient-flip-y');

  // Step 2: Setups & Stock Inputs
  const setupTabsContainer = document.getElementById('setup-tabs-container');
  const btnOpenAddSetup = document.getElementById('btn-open-add-setup');
  const btnAutoDetectSetups = document.getElementById('btn-auto-detect-setups');
  const addSetupForm = document.getElementById('add-setup-form');
  const newSetupName = document.getElementById('new-setup-name');
  const newSetupPreset = document.getElementById('new-setup-preset');
  const newSetupWorkoffset = document.getElementById('new-setup-workoffset');
  const btnConfirmAddSetup = document.getElementById('btn-confirm-add-setup');
  const btnCancelAddSetup = document.getElementById('btn-cancel-add-setup');
  const selectStockMode = document.getElementById('stock-mode');
  const stockPanelRelativeBox = document.getElementById('stock-panel-relative-box');
  const stockPanelFixedBox = document.getElementById('stock-panel-fixed-box');
  const stockPanelRelativeCyl = document.getElementById('stock-panel-relative-cylinder');
  const stockPanelFixedCyl = document.getElementById('stock-panel-fixed-cylinder');

  // Relative Box Inputs
  const inputMarginXY = document.getElementById('stock-margin-xy');
  const inputMarginZTop = document.getElementById('stock-margin-z-top');
  const inputMarginZBot = document.getElementById('stock-margin-z-bot');
  const inputOffsetX = document.getElementById('stock-offset-x');
  const inputOffsetY = document.getElementById('stock-offset-y');

  // Fixed Box Inputs
  const inputFixedX = document.getElementById('stock-fixed-x');
  const inputFixedY = document.getElementById('stock-fixed-y');
  const inputFixedZ = document.getElementById('stock-fixed-z');

  // Relative Cylinder Inputs
  const inputRelCylRadial = document.getElementById('stock-rel-cyl-radial');
  const inputRelCylTop = document.getElementById('stock-rel-cyl-top');
  const inputRelCylBot = document.getElementById('stock-rel-cyl-bot');
  const selectRelCylAxis = document.getElementById('stock-rel-cyl-axis');

  // Fixed Cylinder Inputs
  const inputFixedCylDia = document.getElementById('stock-fixed-cyl-dia');
  const inputFixedCylLen = document.getElementById('stock-fixed-cyl-len');
  const inputFixedCylTop = document.getElementById('stock-fixed-cyl-top');
  const selectFixedCylAxis = document.getElementById('stock-fixed-cyl-axis');

  const selectMaterial = document.getElementById('stock-material');
  const selectClampType = document.getElementById('stock-clamp-type');
  const inputClampHeight = document.getElementById('stock-clamp-height');
  const inputClampWidth = document.getElementById('stock-clamp-width');

  // Step 3: Features
  const featureListContainer = document.getElementById('feature-list-container');
  const btnRunFeatureRec = document.getElementById('btn-run-feature-rec');

  // Step 4: Tools & Ops
  const toolListContainer = document.getElementById('tool-list-container');
  const opListContainer = document.getElementById('op-list-container');

  // Step 5: Toolpaths & Simulation
  const btnGenerateToolpaths = document.getElementById('btn-generate-toolpaths');
  const toolpathsSetupTabsContainer = document.getElementById('toolpaths-setup-tabs-container');
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
  const gcodeSetupBadge = document.getElementById('gcode-setup-badge');
  const gcodeOutput = document.getElementById('gcode-output');
  const gcodeViewerSingle = document.getElementById('gcode-viewer-single');
  const gcodeSetupTabsContainer = document.getElementById('gcode-setup-tabs-container');
  const btnCopyGcode = document.getElementById('btn-copy-gcode');
  const btnDownloadFile = document.getElementById('btn-download-file');
  const btnDownloadLabel = document.getElementById('btn-download-label');
  const btnExportPackage = document.getElementById('btn-export-package');
  const btnViewSetupSheet = document.getElementById('btn-view-setup-sheet');

  // Setup Sheet Modal Elements
  const setupSheetModal = document.getElementById('setup-sheet-modal');
  const btnCloseSetupSheet = document.getElementById('btn-close-setup-sheet');
  const setupSheetTabs = document.getElementById('setup-sheet-tabs');
  const setupSheetContent = document.getElementById('setup-sheet-content');
  const setupSheetMeta = document.getElementById('setup-sheet-meta');
  const btnCopySetupSheet = document.getElementById('btn-copy-setup-sheet');
  const btnPrintSetupSheet = document.getElementById('btn-print-setup-sheet');



  // Viewport Toolbar Buttons
  const togglePartBtn = document.getElementById('toggle-part');
  const toggleStockBtn = document.getElementById('toggle-stock');
  const toggleFixtureBtn = document.getElementById('toggle-fixture');
  const toggleToolpathBtn = document.getElementById('toggle-toolpath');
  const toggleToolBtn = document.getElementById('toggle-tool');
  const toggleWireframeBtn = document.getElementById('toggle-wireframe');
  const viewIsoBtn = document.getElementById('view-iso');
  const viewTopBtn = document.getElementById('view-top');
  const viewFrontBtn = document.getElementById('view-front');
  const viewResetBtn = document.getElementById('view-reset');

  // HUD Readouts
  const hudSetup = document.getElementById('hud-setup');
  const hudSpindle = document.getElementById('hud-spindle');
  const hudFeed = document.getElementById('hud-feed');
  const hudTool = document.getElementById('hud-tool');
  const hudXyz = document.getElementById('hud-xyz');

  // ------------------------------------------------------------- Stepper Logic
  function setStep(stepNum) {
    state.currentStep = stepNum;

    // Clear any active 3D feature highlight when transitioning between workflow steps
    viewer.clearFeatureHighlight();

    stepBtns.forEach((btn) => {
      const bStep = parseInt(btn.dataset.step);
      btn.classList.toggle('active', bStep === stepNum);
      btn.classList.toggle('completed', bStep < stepNum);
    });
    stepPanels.forEach((panel) => {
      panel.classList.toggle('active', panel.id === `panel-step-${stepNum}`);
    });

    if (stepNum === 2) {
      loadSetups();
    } else if (stepNum === 3 && state.features.length === 0) {
      loadFeatures();
    } else if (stepNum === 4) {
      loadToolsAndOps();
      if (state.plannedOps.length === 0) {
        loadOperations();
      }
    } else if (stepNum === 5) {
      if (state.toolpaths.length === 0) {
        generateToolpaths();
      } else {
        renderToolpathsSetupTabs();
        applyActiveToolpathView();
      }
      viewer.toggleTool(true);
      toolVisible = true;
      if (toggleToolBtn) toggleToolBtn.classList.add('active');
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

  // Sync simulation telemetry with HUD
  viewer.onProgressUpdate = (info) => {
    if (hudSetup && (info.setup_name || info.work_offset)) {
      hudSetup.textContent = info.work_offset ? `${info.work_offset} (${info.setup_name || ''})` : (info.setup_name || 'Setup 1');
    }
    hudSpindle.textContent = `${Math.round(info.spindle)} RPM`;
    hudFeed.textContent = `${Math.round(info.feed)} mm/min`;
    hudTool.textContent = info.tool || '--';

    if (info.isClampBreach) {
      hudXyz.innerHTML = `<span style="color: #ef4444; font-weight: bold;">⚠️ CLAMP ZONE (Z=${info.pos.z.toFixed(2)})</span>`;
    } else {
      hudXyz.textContent = `X ${info.pos.x.toFixed(2)} Y ${info.pos.y.toFixed(2)} Z ${info.pos.z.toFixed(2)}`;
    }

    simCurrTime.textContent = formatTime(Math.round(info.elapsedSec));
    simTotalTime.textContent = formatTime(Math.round(info.totalSec));
    simScrubber.value = (info.progress * 100).toFixed(1);
  };

  viewer.onPlaybackEnd = () => {
    const icon = document.getElementById('sim-play-icon');
    if (icon) {
      icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
    }
  };

  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  // ------------------------------------------------------------- Model Loading & CAD Orientation
  async function loadDemoModel() {
    try {
      headerModelName.textContent = 'Loading Demo Model...';
      const res = await fetch('/api/model/demo');
      const data = await res.json();
      applyLoadedModel(data);
    } catch (err) {
      console.error('Failed to load demo model:', err);
      headerModelName.textContent = 'Error Loading Model';
    }
  }

  function applyLoadedModel(data) {
    state.model = data;
    headerModelName.textContent = data.name;

    const size = data.mesh.bounding_box.size;
    statDimX.textContent = `${size[0].toFixed(2)} mm`;
    statDimY.textContent = `${size[1].toFixed(2)} mm`;
    statDimZ.textContent = `${size[2].toFixed(2)} mm`;
    statMeshCount.textContent = `${data.mesh.triangle_count} / ${data.mesh.face_count}`;

    viewer.loadModelMesh(data.mesh);
    initStockInputsForModel(data.mesh.bounding_box);
    syncStockModeUI();
    updateStockBounds();

    state.features = [];
    state.plannedOps = [];
    state.toolpaths = [];
    state.gcodeResult = null;
    loadSetups();
    loadToolsAndOps();
  }

  async function orientModel(axis, angleDeg) {
    try {
      headerModelName.textContent = `Rotating Model (${axis} ${angleDeg}°)...`;
      const res = await fetch('/api/model/orient', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ axis: axis, angle_deg: angleDeg }),
      });
      if (!res.ok) {
        alert('Failed to reorient model');
        return;
      }
      const data = await res.json();
      applyLoadedModel(data);
    } catch (err) {
      console.error('Failed to reorient model:', err);
    }
  }

  if (btnOrientRx90) btnOrientRx90.addEventListener('click', () => orientModel('X', 90.0));
  if (btnOrientRy90) btnOrientRy90.addEventListener('click', () => orientModel('Y', 90.0));
  if (btnOrientRz90) btnOrientRz90.addEventListener('click', () => orientModel('Z', 90.0));
  if (btnOrientFlipX) btnOrientFlipX.addEventListener('click', () => orientModel('X', 180.0));
  if (btnOrientFlipY) btnOrientFlipY.addEventListener('click', () => orientModel('Y', 180.0));

  // ------------------------------------------------------------- Multi-Setup Manager
  async function loadSetups() {
    try {
      const res = await fetch('/api/setups');
      const data = await res.json();
      state.setups = data.setups;
      state.activeSetupId = data.active_setup_id;
      renderSetupsList(data.setups, data.active_setup_id);

      const activeSetup = data.setups.find(s => s.id === data.active_setup_id) || data.setups[0];
      if (activeSetup && activeSetup.stock_bounds) {
        viewer.setSetupOrientation(activeSetup.rotation_deg || [0, 0, 0], activeSetup.stock_bounds, activeSetup.stock || state.stock);
      }
    } catch (err) {
      console.error('Failed to load setups:', err);
    }
  }

  function renderSetupsList(setups, activeId) {
    if (!setupTabsContainer) return;
    setupTabsContainer.innerHTML = '';

    setups.forEach((s, idx) => {
      const item = document.createElement('div');
      item.className = `setup-tab-item ${s.id === activeId ? 'active' : ''}`;
      const badgeClass = (s.work_offset || '').toLowerCase().replace('.', '-');
      const rot = s.rotation_deg || [0, 0, 0];
      const rotStr = (rot[0] !== 0 || rot[1] !== 0 || rot[2] !== 0) 
        ? `${rot[0] ? `X ${rot[0]}° ` : ''}${rot[1] ? `Y ${rot[1]}° ` : ''}${rot[2] ? `Z ${rot[2]}°` : ''}`.trim()
        : 'Top Orientation (0°)';

      item.innerHTML = `
        <div class="setup-tab-info">
          <span class="setup-badge ${badgeClass}">${s.work_offset || 'G54'}</span>
          <div>
            <div class="setup-tab-title">${s.name}</div>
            <div style="font-size: 0.70rem; color: var(--text-dim); display:flex; gap:6px; align-items:center;">
              <span>${rotStr}</span>
              <span>•</span>
              <span class="text-cyan">${s.feature_ids ? s.feature_ids.length : 0} feats</span>
            </div>
          </div>
        </div>
        <div class="setup-tab-actions">
          ${s.id === activeId ? '<span class="status-dot" title="Active Setup" style="width:6px; height:6px;"></span>' : ''}
          ${setups.length > 1 && idx > 0 ? `<button class="btn-icon-del" data-setup-id="${s.id}" title="Delete setup">&times;</button>` : ''}
        </div>
      `;

      item.addEventListener('click', async (e) => {
        if (e.target.classList.contains('btn-icon-del')) return;
        await setActiveSetup(s.id);
      });

      const delBtn = item.querySelector('.btn-icon-del');
      if (delBtn) {
        delBtn.addEventListener('click', async (e) => {
          e.stopPropagation();
          await deleteSetup(s.id);
        });
      }

      setupTabsContainer.appendChild(item);
    });
  }

  async function setActiveSetup(setupId) {
    try {
      const res = await fetch(`/api/setups/active/${setupId}`, { method: 'POST' });
      const data = await res.json();
      state.setups = data.setups;
      state.activeSetupId = data.active_setup_id;
      renderSetupsList(data.setups, data.active_setup_id);

      const activeSetup = data.setups.find(s => s.id === data.active_setup_id);
      if (activeSetup) {
        viewer.setSetupOrientation(activeSetup.rotation_deg || [0, 0, 0], activeSetup.stock_bounds || null, activeSetup.stock || state.stock);
      }
    } catch (err) {
      console.error('Failed to set active setup:', err);
    }
  }

  async function deleteSetup(setupId) {
    try {
      const res = await fetch(`/api/setups/${setupId}`, { method: 'DELETE' });
      const data = await res.json();
      state.setups = data.setups;
      state.activeSetupId = data.active_setup_id;
      renderSetupsList(data.setups, data.active_setup_id);

      const activeSetup = data.setups.find(s => s.id === data.active_setup_id) || data.setups[0];
      if (activeSetup) {
        viewer.setSetupOrientation(activeSetup.rotation_deg || [0, 0, 0], activeSetup.stock_bounds || null, activeSetup.stock || state.stock);
      }
    } catch (err) {
      console.error('Failed to delete setup:', err);
    }
  }

  if (btnOpenAddSetup) {
    btnOpenAddSetup.addEventListener('click', () => {
      addSetupForm.style.display = 'block';
      const nextNum = state.setups.length + 1;
      newSetupName.value = nextNum === 2 ? 'Setup 2 - Bottom Flip' : `Setup ${nextNum} - Multi-Axis`;
      newSetupWorkoffset.value = nextNum === 2 ? 'G55' : (nextNum === 3 ? 'G56' : 'G57');
    });
  }

  if (btnCancelAddSetup) {
    btnCancelAddSetup.addEventListener('click', () => {
      addSetupForm.style.display = 'none';
    });
  }

  if (btnConfirmAddSetup) {
    btnConfirmAddSetup.addEventListener('click', async () => {
      const payload = {
        name: newSetupName.value,
        work_offset: newSetupWorkoffset.value,
        preset: newSetupPreset.value,
        stock: state.stock,
      };
      try {
        const res = await fetch('/api/setups/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        state.setups = data.setups;
        state.activeSetupId = data.active_setup_id;
        renderSetupsList(data.setups, data.active_setup_id);
        addSetupForm.style.display = 'none';
        // Invalidate cached features
        state.features = [];
        state.plannedOps = [];
        state.toolpaths = [];
      } catch (err) {
        console.error('Failed to add setup:', err);
      }
    });
  }

  if (btnAutoDetectSetups) {
    btnAutoDetectSetups.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/setups/auto-generate', { method: 'POST' });
        const data = await res.json();
        state.setups = data.setups;
        state.activeSetupId = data.active_setup_id;
        renderSetupsList(data.setups, data.active_setup_id);

        const activeSetup = data.setups.find(s => s.id === data.active_setup_id) || data.setups[0];
        if (activeSetup && activeSetup.stock_bounds) {
          viewer.setSetupOrientation(activeSetup.rotation_deg || [0, 0, 0], activeSetup.stock_bounds, activeSetup.stock || state.stock);
        }

        state.features = [];
        state.plannedOps = [];
        state.toolpaths = [];
      } catch (err) {
        console.error('Failed to auto-generate setups:', err);
      }
    });
  }

  // ------------------------------------------------------------- Stock & Clamping Updates
  function initStockInputsForModel(b) {
    if (!b) return;
    const sx = b.max[0] - b.min[0];
    const sy = b.max[1] - b.min[1];
    const sz = b.max[2] - b.min[2];
    const partR = Math.hypot(sx / 2, sy / 2);
    const maxSpan = Math.max(sx, sy);

    if (maxSpan < 8.0) {
      // Micro / miniature component defaults
      const padXY = Math.max(maxSpan * 0.2, 0.4);
      const padZ = Math.max(sz * 0.2, 0.4);
      if (inputFixedX) inputFixedX.value = (Math.ceil((sx + padXY * 2) * 10) / 10).toFixed(1);
      if (inputFixedY) inputFixedY.value = (Math.ceil((sy + padXY * 2) * 10) / 10).toFixed(1);
      if (inputFixedZ) inputFixedZ.value = (Math.ceil((sz + padZ * 2) * 10) / 10).toFixed(1);
      if (inputFixedCylDia) inputFixedCylDia.value = (Math.ceil((partR * 2 + padXY * 2) * 10) / 10).toFixed(1);
      if (inputFixedCylLen) inputFixedCylLen.value = (Math.ceil((sz + padZ * 2) * 10) / 10).toFixed(1);
    } else {
      // Standard workpiece envelope defaults
      if (inputFixedX) inputFixedX.value = (Math.ceil((sx + 10) / 5) * 5).toFixed(1);
      if (inputFixedY) inputFixedY.value = (Math.ceil((sy + 10) / 5) * 5).toFixed(1);
      if (inputFixedZ) inputFixedZ.value = (Math.ceil((sz + 4) / 5) * 5).toFixed(1);
      if (inputFixedCylDia) inputFixedCylDia.value = (Math.ceil((partR * 2 + 6) / 5) * 5).toFixed(1);
      if (inputFixedCylLen) inputFixedCylLen.value = (Math.ceil((sz + 16) / 5) * 5).toFixed(1);
    }
  }

  function syncStockModeUI() {
    const mode = selectStockMode ? selectStockMode.value : 'relative_box';
    state.stock.stock_mode = mode;
    if (stockPanelRelativeBox) stockPanelRelativeBox.style.display = mode === 'relative_box' ? 'block' : 'none';
    if (stockPanelFixedBox) stockPanelFixedBox.style.display = mode === 'fixed_box' ? 'block' : 'none';
    if (stockPanelRelativeCyl) stockPanelRelativeCyl.style.display = mode === 'relative_cylinder' ? 'block' : 'none';
    if (stockPanelFixedCyl) stockPanelFixedCyl.style.display = mode === 'fixed_cylinder' ? 'block' : 'none';
  }

  if (selectStockMode) {
    selectStockMode.addEventListener('change', () => {
      syncStockModeUI();
      updateStockBounds();
    });
  }

  function updateStockBounds() {
    if (!state.model) return;
    const b = state.model.mesh.bounding_box;
    const mode = state.stock.stock_mode || 'relative_box';
    const offX = state.stock.offset_x || 0;
    const offY = state.stock.offset_y || 0;
    const offZ = state.stock.offset_z || 0;

    let min, max;
    if (mode === 'relative_cylinder') {
      const cx = (b.min[0] + b.max[0]) / 2 + offX;
      const cy = (b.min[1] + b.max[1]) / 2 + offY;
      const rx = (b.max[0] - b.min[0]) / 2;
      const ry = (b.max[1] - b.min[1]) / 2;
      const partR = Math.hypot(rx, ry);
      const radMargin = parseFloat(inputRelCylRadial?.value) || 2.5;
      const dia = 2 * (partR + radMargin);
      const radius = dia / 2;
      const topMargin = parseFloat(inputRelCylTop?.value) || 1.0;
      const botMargin = parseFloat(inputRelCylBot?.value) || 15.0;
      const zTop = b.max[2] + topMargin + offZ;
      const zBot = b.min[2] - botMargin + offZ;
      const len = zTop - zBot;

      state.stock.cylinder_diameter = dia;
      state.stock.cylinder_length = len;
      state.stock.cylinder_axis = selectRelCylAxis ? selectRelCylAxis.value : 'Z';
      state.stock.cylinder_margin_radial = radMargin;
      state.stock.cylinder_margin_axial_top = topMargin;
      state.stock.cylinder_margin_axial_bot = botMargin;

      min = [cx - radius, cy - radius, zBot];
      max = [cx + radius, cy + radius, zTop];
    } else if (mode === 'fixed_cylinder') {
      const cx = (b.min[0] + b.max[0]) / 2 + offX;
      const cy = (b.min[1] + b.max[1]) / 2 + offY;
      const rx = (b.max[0] - b.min[0]) / 2;
      const ry = (b.max[1] - b.min[1]) / 2;
      const partR = Math.hypot(rx, ry);
      const dia = parseFloat(inputFixedCylDia?.value) || (Math.ceil((partR * 2 + 6) / 5) * 5);
      const radius = dia / 2;
      const len = parseFloat(inputFixedCylLen?.value) || 50.0;
      const topMargin = parseFloat(inputFixedCylTop?.value) || 1.0;
      const zTop = b.max[2] + topMargin + offZ;
      const zBot = zTop - len;

      state.stock.cylinder_diameter = dia;
      state.stock.cylinder_length = len;
      state.stock.cylinder_axis = selectFixedCylAxis ? selectFixedCylAxis.value : 'Z';
      state.stock.cylinder_margin_axial_top = topMargin;

      min = [cx - radius, cy - radius, zBot];
      max = [cx + radius, cy + radius, zTop];
    } else if (mode === 'fixed_box') {
      const partSx = b.max[0] - b.min[0];
      const partSy = b.max[1] - b.min[1];
      const partSz = b.max[2] - b.min[2];
      const fx = parseFloat(inputFixedX?.value) || (partSx + 10.0);
      const fy = parseFloat(inputFixedY?.value) || (partSy + 10.0);
      const fz = parseFloat(inputFixedZ?.value) || (partSz + 4.0);
      const cx = (b.min[0] + b.max[0]) / 2 + offX;
      const cy = (b.min[1] + b.max[1]) / 2 + offY;
      const zTop = b.max[2] + (state.stock.margin_z_top || 1.0) + offZ;
      const zBot = zTop - fz;

      state.stock.fixed_size_x = fx;
      state.stock.fixed_size_y = fy;
      state.stock.fixed_size_z = fz;

      min = [cx - fx / 2, cy - fy / 2, zBot];
      max = [cx + fx / 2, cy + fy / 2, zTop];
    } else {
      // relative_box
      min = [
        b.min[0] - state.stock.margin_x + offX,
        b.min[1] - state.stock.margin_y + offY,
        b.min[2] - state.stock.margin_z_bottom + offZ,
      ];
      max = [
        b.max[0] + state.stock.margin_x + offX,
        b.max[1] + state.stock.margin_y + offY,
        b.max[2] + state.stock.margin_z_top + offZ,
      ];
    }

    viewer.updateStockBounds(min, max, state.stock);
  }

  [
    inputMarginXY, inputMarginZTop, inputMarginZBot, inputOffsetX, inputOffsetY,
    inputFixedX, inputFixedY, inputFixedZ,
    inputRelCylRadial, inputRelCylTop, inputRelCylBot,
    inputFixedCylDia, inputFixedCylLen, inputFixedCylTop,
    inputClampHeight, inputClampWidth
  ].forEach((input) => {
    if (!input) return;
    input.addEventListener('input', () => {
      if (inputMarginXY) state.stock.margin_x = parseFloat(inputMarginXY.value) || 0;
      if (inputMarginXY) state.stock.margin_y = parseFloat(inputMarginXY.value) || 0;
      if (inputMarginZTop) state.stock.margin_z_top = parseFloat(inputMarginZTop.value) || 0;
      if (inputMarginZBot) state.stock.margin_z_bottom = parseFloat(inputMarginZBot.value) || 0;
      if (inputOffsetX) state.stock.offset_x = parseFloat(inputOffsetX.value) || 0;
      if (inputOffsetY) state.stock.offset_y = parseFloat(inputOffsetY.value) || 0;
      if (inputFixedX) state.stock.fixed_size_x = parseFloat(inputFixedX.value) || 0;
      if (inputFixedY) state.stock.fixed_size_y = parseFloat(inputFixedY.value) || 0;
      if (inputFixedZ) state.stock.fixed_size_z = parseFloat(inputFixedZ.value) || 0;
      if (inputRelCylRadial) state.stock.cylinder_margin_radial = parseFloat(inputRelCylRadial.value) || 2.5;
      if (inputRelCylTop) state.stock.cylinder_margin_axial_top = parseFloat(inputRelCylTop.value) || 1.0;
      if (inputRelCylBot) state.stock.cylinder_margin_axial_bot = parseFloat(inputRelCylBot.value) || 15.0;
      if (inputFixedCylDia) state.stock.cylinder_diameter = parseFloat(inputFixedCylDia.value) || 0;
      if (inputFixedCylLen) state.stock.cylinder_length = parseFloat(inputFixedCylLen.value) || 0;
      if (inputFixedCylTop) state.stock.cylinder_margin_axial_top = parseFloat(inputFixedCylTop.value) || 1.0;
      if (inputClampHeight) state.stock.clamp_height = parseFloat(inputClampHeight.value) || 3.0;
      if (inputClampWidth) state.stock.clamp_width = parseFloat(inputClampWidth.value) || 14.0;
      updateStockBounds();
    });
  });

  if (selectRelCylAxis) {
    selectRelCylAxis.addEventListener('change', () => {
      state.stock.cylinder_axis = selectRelCylAxis.value;
      updateStockBounds();
    });
  }

  if (selectFixedCylAxis) {
    selectFixedCylAxis.addEventListener('change', () => {
      state.stock.cylinder_axis = selectFixedCylAxis.value;
      updateStockBounds();
    });
  }

  if (selectMaterial) {
    selectMaterial.addEventListener('change', () => {
      state.stock.material = selectMaterial.value;
    });
    state.stock.material = selectMaterial.value;
  }

  if (selectClampType) {
    selectClampType.addEventListener('change', () => {
      state.stock.clamp_type = selectClampType.value;
      updateStockBounds();
    });
    state.stock.clamp_type = selectClampType.value;
  }

  const inputThreadPitch = document.getElementById('thread-pitch');
  const inputGrooveWidth = document.getElementById('groove-width');
  if (inputThreadPitch) {
    state.stock.thread_pitch = parseFloat(inputThreadPitch.value) || 1.0;
    inputThreadPitch.addEventListener('input', () => {
      state.stock.thread_pitch = parseFloat(inputThreadPitch.value) || 1.0;
    });
  }
  if (inputGrooveWidth) {
    state.stock.groove_width = parseFloat(inputGrooveWidth.value) || 3.0;
    inputGrooveWidth.addEventListener('input', () => {
      state.stock.groove_width = parseFloat(inputGrooveWidth.value) || 3.0;
    });
  }

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
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      headerModelName.textContent = `Parsing CAD Geometry (${file.name})...`;
      const res = await fetch('/api/model/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        let errMsg = 'Invalid CAD file format or unreadable geometry.';
        try {
          const err = await res.json();
          errMsg = err.detail || JSON.stringify(err);
        } catch (e) {
          errMsg = (await res.text()) || res.statusText || 'Server Error';
        }
        alert(`Error importing ${file.name}:\n\n${errMsg}`);
        headerModelName.textContent = state.model ? state.model.name : 'Ready';
        return;
      }
      const data = await res.json();
      applyLoadedModel(data);
    } catch (err) {
      console.error('File upload failed:', err);
      alert(`Failed to upload ${file.name}:\n\n${err.message || err}`);
      headerModelName.textContent = state.model ? state.model.name : 'Ready';
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
    if (!features || features.length === 0) {
      featureListContainer.innerHTML = '<div class="text-dim p-2" style="font-size:0.8rem; text-align:center;">No machinable features recognized yet. Click "Scan Features" to analyze 3D B-Rep topology.</div>';
      return;
    }

    features.forEach((f) => {
      const item = document.createElement('div');
      item.className = 'feature-item';
      item.dataset.featureId = f.id;

      const title = f.id.replace('feat_', '').replace('_', ' ');
      const dimStr = f.diameter ? `Ø${f.diameter.toFixed(1)}mm` : `Depth ${f.depth.toFixed(1)}mm`;
      const setupName = f.notes && f.notes.setup_name ? f.notes.setup_name : '';
      const workOffset = f.notes && f.notes.work_offset ? f.notes.work_offset : '';

      item.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:2px;">
          <div style="font-weight: 600; font-size: 0.825rem; text-transform: capitalize; display:flex; align-items:center; gap:6px;">
            <span>${title}</span>
            ${workOffset ? `<span class="setup-badge ${workOffset === 'G55' ? 'g55' : ''}" style="font-size:0.6rem; padding:1px 5px;">${workOffset}</span>` : ''}
          </div>
          <div style="font-size: 0.70rem; color: var(--text-dim); display:flex; gap:6px; align-items:center;">
            <span class="mono text-cyan">${dimStr}</span>
            ${f.depth !== undefined && f.diameter ? `<span class="mono">Depth: ${f.depth.toFixed(1)}mm</span>` : ''}
            ${setupName ? `<span>• <span class="text-muted">${setupName}</span></span>` : ''}
          </div>
        </div>
        <span class="feature-tag ${f.type}">${f.type.replace('_', ' ')}</span>
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
  const btnShowAddTool = document.getElementById('btn-show-add-tool');
  const addToolFormCard = document.getElementById('add-tool-form-card');
  const btnSaveNewTool = document.getElementById('btn-save-new-tool');
  const btnCancelNewTool = document.getElementById('btn-cancel-new-tool');
  const opPlanningWarnings = document.getElementById('op-planning-warnings');

  if (btnShowAddTool) {
    btnShowAddTool.addEventListener('click', () => {
      addToolFormCard.style.display = addToolFormCard.style.display === 'none' ? 'block' : 'none';
    });
  }
  if (btnCancelNewTool) {
    btnCancelNewTool.addEventListener('click', () => {
      addToolFormCard.style.display = 'none';
    });
  }
  if (btnSaveNewTool) {
    btnSaveNewTool.addEventListener('click', async () => {
      const toolType = document.getElementById('new-tool-type').value;
      const diameter = parseFloat(document.getElementById('new-tool-dia').value);
      const flute = parseFloat(document.getElementById('new-tool-flute').value);
      const flutes = parseInt(document.getElementById('new-tool-flutes').value, 10);

      if (!diameter || diameter <= 0) {
        alert('Please enter a valid tool diameter (> 0)');
        return;
      }

      try {
        const res = await fetch('/api/tools/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            tool_type: toolType,
            diameter: diameter,
            flute_length: flute || 15.0,
            overall_length: (flute ? flute * 2.5 : 50.0),
            flutes: flutes || 2,
            material: 'carbide'
          }),
        });
        if (res.ok) {
          const data = await res.json();
          state.tools = data.tools;
          viewer.setToolCatalog(data.tools);
          renderToolsList(data.tools);
          addToolFormCard.style.display = 'none';
          // Re-plan operations with newly added tool
          await loadOperations();
        }
      } catch (err) {
        console.error('Failed to add custom tool:', err);
      }
    });
  }

  async function loadToolsAndOps() {
    try {
      const resTools = await fetch('/api/tools-and-machines');
      const dataTools = await resTools.json();
      state.tools = dataTools.tools;
      viewer.setToolCatalog(dataTools.tools);
      renderToolsList(dataTools.tools);
    } catch (err) {
      console.error('Failed to load tooling:', err);
    }
  }

  function renderToolsList(tools) {
    if (!toolListContainer) return;
    toolListContainer.innerHTML = '';
    if (!tools || !Array.isArray(tools) || tools.length === 0) {
      toolListContainer.innerHTML = `
        <div style="padding: 16px; text-align: center; color: var(--text-dim); font-size: 0.85rem; border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px;">
          No active tools. <button id="btn-reload-default-tools" class="btn btn-sm btn-primary" style="margin-left: 8px; font-size:0.75rem; padding:3px 10px;">Load Defaults</button>
        </div>
      `;
      const reloadBtn = toolListContainer.querySelector('#btn-reload-default-tools');
      if (reloadBtn) {
        reloadBtn.addEventListener('click', loadToolsAndOps);
      }
      return;
    }
    tools.forEach((t) => {
      const item = document.createElement('div');
      item.className = 'tool-card-item';
      item.style.display = 'flex';
      item.style.justifyContent = 'space-between';
      item.style.alignItems = 'center';
      item.style.cursor = 'pointer';
      item.title = 'Click to inspect tool geometry & cutting parameters';

      item.innerHTML = `
        <div style="display:flex; flex-direction:column; gap:2px; flex:1;">
          <div style="font-weight: 700; font-size: 0.825rem; display:flex; align-items:center; gap:6px;" class="mono">
            <span class="text-cyan">${t.id}</span>
            <span style="color:var(--text-main); font-weight:600;">${t.name}</span>
          </div>
          <div style="font-size: 0.70rem; color: var(--text-dim); display:flex; gap:6px; align-items:center;">
            <span class="mono text-emerald">Ø${t.diameter}mm</span>
            <span>•</span>
            <span class="mono">${t.flutes}F</span>
            <span>•</span>
            <span>Flute: ${t.flute_length || 15}mm</span>
            <span>•</span>
            <span>${t.max_rpm || 10000} RPM</span>
          </div>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <button class="btn btn-sm btn-outline inspect-tool-btn" data-id="${t.id}" title="Inspect Tool Specs" style="padding:2px 7px; font-size:0.7rem; color:var(--accent-cyan); border-color:rgba(0,240,255,0.3);">Specs</button>
          <button class="btn btn-sm btn-outline delete-tool-btn" data-id="${t.id}" title="Remove Tool" style="padding:2px 7px; font-size:0.75rem; color:#f87171; border-color:rgba(248,113,113,0.3);">✕</button>
        </div>
      `;

      item.addEventListener('click', (e) => {
        if (e.target.closest('.delete-tool-btn')) return;
        openToolInspector(t.id);
      });

      toolListContainer.appendChild(item);
    });

    // Wire up delete tool buttons
    toolListContainer.querySelectorAll('.delete-tool-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const tid = e.currentTarget.getAttribute('data-id');
        try {
          const res = await fetch(`/api/tools/${tid}`, { method: 'DELETE' });
          if (res.ok) {
            const data = await res.json();
            state.tools = data.tools;
            viewer.setToolCatalog(data.tools);
            renderToolsList(data.tools);
            await loadOperations();
          }
        } catch (err) {
          console.error('Failed to delete tool:', err);
        }
      });
    });
  }

  async function loadOperations() {
    try {
      if (!state.features || state.features.length === 0) {
        try {
          const featRes = await fetch('/api/recognize-features', { method: 'POST' });
          if (featRes.ok) {
            const featData = await featRes.json();
            state.features = featData.features;
          }
        } catch (e) {
          console.warn('Could not auto-fetch features for op highlighting:', e);
        }
      }

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

      // Render non-blocking warnings and smart tool auto-add actions
      if (opPlanningWarnings) {
        if (data.warnings && data.warnings.length > 0) {
          opPlanningWarnings.style.display = 'block';
          const validSuggestions = data.warnings.filter(w => w.suggested_tool);
          
          let headerHtml = '';
          if (validSuggestions.length > 1) {
            headerHtml = `
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; padding:4px 2px;">
                <span style="font-size:0.8rem; font-weight:600; color:var(--text-bright);">Unmachined Features (${data.warnings.length})</span>
                <button id="auto-add-all-tools-btn" class="btn btn-sm btn-primary" style="font-size:0.75rem; padding:4px 10px;">
                  ⚡ Auto-Add All Missing Tools (${validSuggestions.length})
                </button>
              </div>
            `;
          }

          const warningsHtml = data.warnings.map(w => `
            <div class="card p-3 mb-2" style="background:rgba(234, 179, 8, 0.08); border:1px solid rgba(234, 179, 8, 0.3); border-radius:8px;">
              <div style="font-weight:600; font-size:0.82rem; color:#facc15; margin-bottom:4px;">
                ⚠️ Notice: ${w.message}
              </div>
              ${w.suggested_tool ? `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
                  <span style="font-size:0.75rem; color:var(--text-dim);">Recommended: Ø${w.suggested_tool.diameter}mm ${w.suggested_tool.type.replace('_', ' ')}</span>
                  <button class="btn btn-sm btn-primary auto-add-tool-btn" data-feature="${w.feature_id}" data-type="${w.suggested_tool.type}" data-dia="${w.suggested_tool.diameter}" style="font-size:0.78rem; padding:4px 10px;">
                    ⚡ Auto-Add Ø${w.suggested_tool.diameter}mm Tool &amp; Re-Plan
                  </button>
                </div>
              ` : ''}
            </div>
          `).join('');

          opPlanningWarnings.innerHTML = headerHtml + warningsHtml;

          // Wire up auto-add all tools button
          const autoAddAllBtn = opPlanningWarnings.querySelector('#auto-add-all-tools-btn');
          if (autoAddAllBtn) {
            autoAddAllBtn.addEventListener('click', async () => {
              try {
                autoAddAllBtn.disabled = true;
                autoAddAllBtn.textContent = 'Adding All Tools...';
                
                for (const w of validSuggestions) {
                  const fid = w.feature_id;
                  const ttype = w.suggested_tool.type;
                  const dia = w.suggested_tool.diameter;
                  const res = await fetch('/api/tools/auto-suggest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ feature_id: fid, tool_type: ttype, diameter: dia }),
                  });
                  if (res.ok) {
                    const d = await res.json();
                    state.tools = d.tools;
                    viewer.setToolCatalog(d.tools);
                    renderToolsList(d.tools);
                  }
                }
                await loadOperations();
              } catch (err) {
                console.error('Failed to auto-add all tools:', err);
              }
            });
          }

          // Wire up auto-add tool buttons
          opPlanningWarnings.querySelectorAll('.auto-add-tool-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
              const fid = e.currentTarget.getAttribute('data-feature');
              const ttype = e.currentTarget.getAttribute('data-type');
              const dia = parseFloat(e.currentTarget.getAttribute('data-dia'));
              try {
                btn.disabled = true;
                btn.textContent = 'Adding Tool...';
                const res = await fetch('/api/tools/auto-suggest', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ feature_id: fid, tool_type: ttype, diameter: dia }),
                });
                if (res.ok) {
                  const d = await res.json();
                  state.tools = d.tools;
                  viewer.setToolCatalog(d.tools);
                  renderToolsList(d.tools);
                  await loadOperations();
                }
              } catch (err) {
                console.error('Failed to auto-add tool:', err);
              }
            });
          });
        } else {
          opPlanningWarnings.style.display = 'none';
          opPlanningWarnings.innerHTML = '';
        }
      }
    } catch (err) {
      console.error('Failed to plan operations:', err);
    }
  }

  let activeOpsSetupTab = 'all';

  function renderOpsSetupFilter() {
    const filterContainer = document.getElementById('ops-setup-filter-container');
    if (!filterContainer) return;
    const setups = state.setups || [];
    if (setups.length <= 1) {
      filterContainer.style.display = 'none';
      return;
    }

    filterContainer.style.display = 'grid';
    filterContainer.style.gridTemplateColumns = `repeat(${setups.length + 1}, 1fr)`;
    filterContainer.style.gap = '4px';
    filterContainer.innerHTML = '';

    const allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.className = `gcode-setup-tab ${activeOpsSetupTab === 'all' ? 'active' : ''}`;
    allBtn.style.padding = '4px 8px';
    allBtn.innerHTML = `
      <span class="tab-title" style="font-size:0.75rem;">ALL</span>
      <span class="tab-chip" style="font-size:0.65rem;">${(state.plannedOps || []).length}</span>
    `;
    allBtn.addEventListener('click', (e) => {
      e.preventDefault();
      activeOpsSetupTab = 'all';
      renderOpsSetupFilter();
      renderOpsList(state.plannedOps);
    });
    filterContainer.appendChild(allBtn);

    setups.forEach((s, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `gcode-setup-tab ${activeOpsSetupTab === s.id ? 'active' : ''}`;
      btn.style.padding = '4px 8px';
      const opCount = (state.plannedOps || []).filter(op => op.notes?.setup_id === s.id).length;
      btn.innerHTML = `
        <span class="tab-title" style="font-size:0.75rem;">OP${(idx + 1) * 10}</span>
        <span class="tab-chip" style="font-size:0.65rem;">${opCount}</span>
      `;
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        activeOpsSetupTab = s.id;
        renderOpsSetupFilter();
        renderOpsList(state.plannedOps);
      });
      filterContainer.appendChild(btn);
    });
  }

  function renderOpsList(ops) {
    const opCountBadge = document.getElementById('op-count-badge');
    if (!ops || ops.length === 0) {
      if (opCountBadge) opCountBadge.textContent = '0 Ops';
      opListContainer.innerHTML = `
        <div style="padding: 24px 16px; text-align: center; color: var(--text-dim); font-size: 0.82rem; border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px; background: rgba(0,0,0,0.2);">
          <div style="margin-bottom: 6px; font-size: 1.1rem;">⚙️</div>
          <div>No planned operations generated yet.</div>
          <button id="btn-initial-plan-ops" class="btn btn-sm btn-primary" style="margin-top: 10px; font-size: 0.78rem;">Generate Sequence</button>
        </div>
      `;
      const planBtn = opListContainer.querySelector('#btn-initial-plan-ops');
      if (planBtn) planBtn.addEventListener('click', loadOperations);
      return;
    }

    renderOpsSetupFilter();

    let displayOps = ops;
    if (activeOpsSetupTab !== 'all') {
      displayOps = ops.filter(op => op.notes?.setup_id === activeOpsSetupTab);
    }

    if (opCountBadge) {
      opCountBadge.textContent = `${displayOps.length} / ${ops.length} Ops`;
    }

    opListContainer.innerHTML = '';

    displayOps.forEach((op, index) => {
      const origIndex = ops.indexOf(op);
      const item = document.createElement('div');
      item.className = 'op-card-item';
      item.dataset.purpose = (op.purpose || 'operation').toLowerCase();
      item.dataset.opId = op.id;

      const setupName = op.notes && op.notes.setup_name ? op.notes.setup_name : '';
      const workOffset = op.notes && op.notes.work_offset ? op.notes.work_offset : 'G54';
      
      // Clean purpose string
      let rawPurpose = op.purpose || 'Operation';
      let purposeStr = rawPurpose.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

      const tool = (state.tools || []).find(t => t.id === op.tool_id);
      const toolName = tool ? tool.name : (op.tool_id || 'Tool');
      const toolDia = tool ? `Ø${tool.diameter}mm` : '';
      
      // Clean numeric parameters
      const feedVal = op.feed_rate ? Math.round(Number(op.feed_rate)) : 0;
      const rpmVal = op.spindle_rpm ? Math.round(Number(op.spindle_rpm)) : 0;
      const apVal = (op.cut_depth !== undefined && op.cut_depth !== null) ? Number(op.cut_depth).toFixed(1) : null;
      const aeVal = (op.stepover !== undefined && op.stepover !== null) ? Number(op.stepover).toFixed(1) : null;

      item.innerHTML = `
        <!-- Top Row: # Seq + Operation Name + Setup -->
        <div class="op-row-top">
          <div class="op-title-wrap">
            <span class="op-seq-pill">#${origIndex + 1}</span>
            <span class="op-title-text">${purposeStr}</span>
          </div>
          <span class="setup-badge ${workOffset === 'G55' ? 'g55' : (workOffset === 'G56' ? 'g56' : '')}">
            ${workOffset}
          </span>
        </div>

        <!-- Target CAD Feature Link -->
        <div class="op-row-feature">
          <span class="op-feat-link" title="Target CAD Feature: ${op.feature_id || ''}">⚲ ${op.feature_id || 'Feature'}</span>
        </div>
        
        <!-- Tool Row: Clear Descriptive Tool Dropdown + Specs Button -->
        <div class="op-row-tool">
          <div class="op-tool-label-prefix">
            <span class="op-tool-icon">🔧</span>
            <span class="op-tool-text-lbl">Tool:</span>
          </div>
          <select class="op-tool-select" data-op-id="${op.id}" data-op-orig-idx="${origIndex}" title="Switch cutting tool for this operation">
            ${(state.tools || []).map(t => {
              const cleanType = (t.name || t.type || 'Tool').replace(/_/g, ' ');
              return `
                <option value="${t.id}" ${t.id === op.tool_id ? 'selected' : ''}>
                  ${t.id}: ${cleanType} (Ø${t.diameter}mm)
                </option>
              `;
            }).join('')}
          </select>
          <button type="button" class="btn-inspect-tool inspect-op-tool-btn" data-tool-id="${op.tool_id}" title="Inspect Tool Geometry & Specs">
            🔍 Specs
          </button>
        </div>

        <!-- Feeds & Speeds Parameters -->
        <div class="op-row-stats">
          <span class="op-stat-pill" title="Cutting Feed Rate">
            <span class="lbl">F</span>
            <b>${feedVal}</b>
            <span class="unit">mm/min</span>
          </span>
          <span class="op-stat-pill" title="Spindle Speed">
            <span class="lbl">S</span>
            <b>${rpmVal}</b>
            <span class="unit">RPM</span>
          </span>
          ${apVal !== null ? `
            <span class="op-stat-pill" title="Depth of Cut">
              <span class="lbl">ap</span>
              <b>${apVal}</b>
              <span class="unit">mm</span>
            </span>
          ` : ''}
          <span class="op-stat-pill op-stat-safe" title="Clearance Z Verified">✓ Safe Z</span>
        </div>
      `;

      // Click and hover card to highlight target feature in 3D
      item.addEventListener('mouseenter', () => {
        const matchedFeature = (state.features || []).find(f => f.id === op.feature_id || f.canonical_id === op.feature_id || f.id === op.notes?.feature_id);
        if (matchedFeature) {
          viewer.highlightFeature(matchedFeature);
        }
      });

      item.addEventListener('click', (e) => {
        if (e.target.closest('.op-tool-select') || e.target.closest('.inspect-op-tool-btn')) return;
        opListContainer.querySelectorAll('.op-card-item').forEach(c => c.classList.remove('selected'));
        item.classList.add('selected');
        
        const matchedFeature = (state.features || []).find(f => f.id === op.feature_id || f.canonical_id === op.feature_id || f.id === op.notes?.feature_id);
        if (matchedFeature) {
          viewer.highlightFeature(matchedFeature);
        }
      });

      opListContainer.appendChild(item);
    });

    // Wire up tool selection changes per operation
    opListContainer.querySelectorAll('.op-tool-select').forEach(sel => {
      sel.addEventListener('change', (e) => {
        const origIdx = parseInt(e.target.dataset.opOrigIdx);
        const newToolId = e.target.value;
        if (state.plannedOps && state.plannedOps[origIdx]) {
          state.plannedOps[origIdx].tool_id = newToolId;
          const matchedTool = (state.tools || []).find(t => t.id === newToolId);
          if (matchedTool && matchedTool.max_rpm) {
            state.plannedOps[origIdx].spindle_rpm = Math.min(state.plannedOps[origIdx].spindle_rpm || 6000, matchedTool.max_rpm);
          }
          showToast(`Assigned ${newToolId} to Operation #${origIdx + 1}`, 'success', 2000);
          renderOpsList(state.plannedOps);
        }
      });
    });

    // Wire up clicking tool badge to open Tool Inspector Modal
    opListContainer.querySelectorAll('.inspect-op-tool-btn').forEach(badge => {
      badge.addEventListener('click', (e) => {
        e.stopPropagation();
        const tid = badge.dataset.toolId;
        openToolInspector(tid);
      });
    });
  }

  const btnReplanOps = document.getElementById('btn-replan-ops');
  if (btnReplanOps) {
    btnReplanOps.addEventListener('click', async () => {
      showToast('Re-planning machining operations...', 'info', 1500);
      await loadOperations();
    });
  }


  // ------------------------------------------------------------- Toolpath Generation & Multi-Setup Filtering
  let activeToolpathTab = 'all';

  function renderToolpathsSetupTabs() {
    if (!toolpathsSetupTabsContainer) return;
    const setups = state.setups || [];
    if (setups.length <= 1) {
      toolpathsSetupTabsContainer.style.display = 'none';
      return;
    }

    toolpathsSetupTabsContainer.style.display = 'grid';
    toolpathsSetupTabsContainer.innerHTML = '';

    const allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.className = `gcode-setup-tab ${activeToolpathTab === 'all' ? 'active' : ''}`;
    allBtn.innerHTML = `
      <span class="tab-title">Master</span>
      <span class="tab-chip">ALL</span>
    `;
    allBtn.addEventListener('click', (e) => {
      e.preventDefault();
      activeToolpathTab = 'all';
      renderToolpathsSetupTabs();
      applyActiveToolpathView();
    });
    toolpathsSetupTabsContainer.appendChild(allBtn);

    setups.forEach((s, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `gcode-setup-tab ${activeToolpathTab === s.id ? 'active' : ''}`;
      const opLabel = `OP${(idx + 1) * 10}`;
      const wo = s.work_offset || `G${54 + idx}`;
      btn.innerHTML = `
        <span class="tab-title">${opLabel}</span>
        <span class="tab-chip">${wo}</span>
      `;
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        activeToolpathTab = s.id;
        renderToolpathsSetupTabs();
        applyActiveToolpathView();
      });
      toolpathsSetupTabsContainer.appendChild(btn);
    });
  }

  function applyActiveToolpathView() {
    if (!state.toolpaths || state.toolpaths.length === 0) return;

    // Clear feature highlights when inspecting toolpaths
    viewer.clearFeatureHighlight();

    let paths = state.toolpaths;
    let setupObj = null;

    if (activeToolpathTab !== 'all') {
      const filtered = state.toolpaths.filter(tp => (
        tp.setup_id === activeToolpathTab ||
        tp.metadata?.setup_id === activeToolpathTab ||
        tp.notes?.setup_id === activeToolpathTab
      ));
      if (filtered.length > 0) {
        paths = filtered;
      }
      setupObj = state.setups?.find(s => s.id === activeToolpathTab);
    } else {
      setupObj = state.setups?.[0] || null;
    }

    // Update Analytics for the active setup
    const totalCut = paths.reduce((acc, tp) => acc + (tp.cutting_length_mm || 0), 0);
    const totalRapid = paths.reduce((acc, tp) => acc + (tp.rapid_length_mm || 0), 0);
    const totalSegs = paths.reduce((acc, tp) => acc + (tp.segment_count || tp.segments?.length || 0), 0);
    
    // Estimate cycle time:
    let estSec = 0;
    paths.forEach(tp => {
      tp.segments?.forEach(seg => {
        const s = seg.start;
        const e = seg.end;
        const len = Math.hypot(e[0] - s[0], e[1] - s[1], e[2] - s[2]);
        const feed = seg.feed || (seg.motion_type === 'rapid' ? 5000 : 1000);
        estSec += len > 0 && feed > 0 ? (len / feed) * 60.0 : 0;
      });
    });

    statCutLen.textContent = `${totalCut.toFixed(1)} mm`;
    statRapidLen.textContent = `${totalRapid.toFixed(1)} mm`;
    statEstTime.textContent = formatTime(Math.round(estSec));
    statSegCount.textContent = totalSegs;

    // Orient the 3D viewport model and stock according to setup
    if (setupObj) {
      const rot = setupObj.rotation_deg || [0, 0, 0];
      const stockBounds = setupObj.stock_bounds || null;
      viewer.setSetupOrientation(rot, stockBounds, setupObj.stock || setupObj.stock_cfg || state.stock);
    } else {
      viewer.setSetupOrientation([0, 0, 0], null, state.stock);
    }

    // Render toolpaths and reset simulation
    viewer.setToolCatalog(state.tools);
    viewer.renderToolpaths(paths, activeToolpathTab, state.setups, state.toolpaths);
    simScrubber.value = 0;
    viewer.setSimulationProgress(0);

    // Render Active Tooling Breakdown in Step 5
    const toolpathToolsCard = document.getElementById('toolpath-tools-card');
    const toolpathToolsList = document.getElementById('toolpath-tools-list');
    const toolpathToolsCount = document.getElementById('toolpath-tools-count');

    if (toolpathToolsCard && toolpathToolsList) {
      const toolMap = {};
      paths.forEach(tp => {
        const tid = tp.tool_id || 'T01';
        if (!toolMap[tid]) {
          const tInfo = (state.tools || []).find(t => t.id === tid);
          toolMap[tid] = {
            id: tid,
            name: tInfo ? tInfo.name : tid,
            diameter: tInfo ? tInfo.diameter : 0,
            type: tInfo ? (tInfo.type || 'endmill').replace(/_/g, ' ') : 'endmill',
            cutLen: 0,
            rapidLen: 0,
            segments: 0,
          };
        }
        toolMap[tid].cutLen += (tp.cutting_length_mm || 0);
        toolMap[tid].rapidLen += (tp.rapid_length_mm || 0);
        toolMap[tid].segments += (tp.segment_count || tp.segments?.length || 0);
      });

      const uniqueTools = Object.values(toolMap);
      if (uniqueTools.length > 0) {
        toolpathToolsCard.style.display = 'block';
        if (toolpathToolsCount) toolpathToolsCount.textContent = `${uniqueTools.length} Tool${uniqueTools.length > 1 ? 's' : ''}`;
        toolpathToolsList.innerHTML = uniqueTools.map(t => `
          <div class="tool-card-item" data-tool-id="${t.id}" style="cursor:pointer;" title="Click to view tool specs">
            <div style="display:flex; flex-direction:column; gap:2px;">
              <div style="font-weight:700; font-size:0.8rem; display:flex; align-items:center; gap:6px;">
                <span class="mono text-cyan">${t.id}</span>
                <span style="color:var(--text-main); font-weight:600;">${t.name}</span>
                <span class="mono text-emerald" style="font-size:0.72rem;">Ø${t.diameter}mm</span>
              </div>
              <div style="font-size:0.70rem; color:var(--text-dim); display:flex; gap:6px; align-items:center;">
                <span>Cut: <span class="mono text-cyan">${t.cutLen.toFixed(1)}mm</span></span>
                <span>•</span>
                <span>${t.segments} Segs</span>
                <span>•</span>
                <span style="text-transform:capitalize;">${t.type}</span>
              </div>
            </div>
            <button class="btn btn-sm btn-outline" style="padding:2px 7px; font-size:0.7rem; color:var(--accent-cyan); border-color:rgba(0,240,255,0.3);" title="View Specs">Specs</button>
          </div>
        `).join('');

        toolpathToolsList.querySelectorAll('.tool-card-item').forEach(item => {
          item.addEventListener('click', () => {
            const tid = item.dataset.toolId;
            openToolInspector(tid);
          });
        });
      } else {
        toolpathToolsCard.style.display = 'none';
      }
    }
  }

  const chkHoldingTabs = document.getElementById('chk-holding-tabs');
  const chkDogboneCorners = document.getElementById('chk-dogbone-corners');
  const holdingTabsOpts = document.getElementById('holding-tabs-opts');
  const tabCountInput = document.getElementById('tab-count');
  const tabHeightInput = document.getElementById('tab-height');

  if (chkHoldingTabs && holdingTabsOpts) {
    chkHoldingTabs.addEventListener('change', () => {
      holdingTabsOpts.style.display = chkHoldingTabs.checked ? 'grid' : 'none';
    });
  }

  async function generateToolpaths() {
    btnGenerateToolpaths.disabled = true;
    btnGenerateToolpaths.innerHTML = 'Calculating Multi-Setup Paths...';

    const dressups = {
      holding_tabs: Boolean(chkHoldingTabs?.checked),
      tab_count: parseInt(tabCountInput?.value || '4'),
      tab_height: parseFloat(tabHeightInput?.value || '2.0'),
      dogbone: Boolean(chkDogboneCorners?.checked),
    };

    try {
      const res = await fetch('/api/generate-toolpaths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stock: state.stock, all_setups: true, dressups: dressups }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`Toolpath Generation Notice:\n${err.detail || 'Failed to generate toolpaths'}`);
        return;
      }
      const data = await res.json();
      state.toolpaths = data.toolpaths;

      renderToolpathsSetupTabs();
      applyActiveToolpathView();
      showToast(`Generated ${data.toolpaths.length} toolpath operations`, 'success', 2500);
    } catch (err) {
      console.error('Failed to generate toolpaths:', err);
    } finally {
      btnGenerateToolpaths.disabled = false;
      btnGenerateToolpaths.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Generate Toolpaths (All Setups)
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

  // ------------------------------------------------------------- G-Code Generation & Post-Processing
  let gcodeGenerating = false;
  let activeGcodeTab = 'all';
  let cachedPerSetupGcode = {};

  function triggerBlobDownload(content, filename, mimeType = 'text/plain') {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function apiErrorText(res, fallback) {
    try {
      const err = await res.json();
      let msg = err.detail || fallback;
      const fails = (err.context && err.context.blocking_failures) || [];
      if (fails.length) {
        for (const f of fails) msg += `\n- ${f.operation_id || '?'} [${f.check}] ${f.message || ''}`;
      }
      return msg;
    } catch (e) {
      return fallback;
    }
  }

  async function generateGcode() {
    if (gcodeGenerating) return;
    gcodeGenerating = true;
    const controller = postControllerSelect.value;
    try {
      const res = await fetch('/api/generate-gcode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          controller: controller, 
          program_name: (state.model?.name || 'PART_01').replace(/[^a-zA-Z0-9_-]/g, '_'), 
          all_setups: true,
          separate_files: false
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        const fails = (err.context && err.context.blocking_failures) || [];
        let msg = err.detail || 'Failed to generate G-Code';
        if (fails.length) {
          msg += '\n\nFailed check(s):';
          for (const f of fails) {
            msg += `\n- ${f.operation_id || '?'} [${f.check}] ${f.message || ''}`;
          }
        }
        alert(`${fails.length ? 'G-Code Blocked' : 'G-Code Generation Notice'}:\n${msg}`);
        return;
      }
      const data = await res.json();
      state.gcodeResult = data;
      cachedPerSetupGcode = data.per_setup || {};

      renderGcodeSetupTabs();
      renderActiveGcodeView();
    } catch (err) {
      console.error('Failed to post-process G-Code:', err);
    } finally {
      gcodeGenerating = false;
    }
  }

  function renderGcodeSetupTabs() {
    if (!gcodeSetupTabsContainer) return;
    const setups = state.setups || [];
    const perSetupKeys = Object.keys(cachedPerSetupGcode);
    if (setups.length <= 1 && perSetupKeys.length <= 1) {
      gcodeSetupTabsContainer.style.display = 'none';
      return;
    }

    gcodeSetupTabsContainer.style.display = 'grid';
    gcodeSetupTabsContainer.innerHTML = '';

    const allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.className = `gcode-setup-tab ${activeGcodeTab === 'all' ? 'active' : ''}`;
    allBtn.innerHTML = `
      <span class="tab-title">Master</span>
      <span class="tab-chip">ALL</span>
    `;
    allBtn.addEventListener('click', (e) => {
      e.preventDefault();
      activeGcodeTab = 'all';
      renderGcodeSetupTabs();
      renderActiveGcodeView();
    });
    gcodeSetupTabsContainer.appendChild(allBtn);

    setups.forEach((s, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `gcode-setup-tab ${activeGcodeTab === s.id ? 'active' : ''}`;
      const opLabel = `OP${(idx + 1) * 10}`;
      const wo = s.work_offset || `G${54 + idx}`;
      btn.innerHTML = `
        <span class="tab-title">${opLabel}</span>
        <span class="tab-chip">${wo}</span>
      `;
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        activeGcodeTab = s.id;
        renderGcodeSetupTabs();
        renderActiveGcodeView();
      });
      gcodeSetupTabsContainer.appendChild(btn);
    });
  }


  function highlightGcode(rawGcode) {
    if (!rawGcode) return '';
    const escaped = rawGcode
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    return escaped
      .replace(/(\([^\)]*\))/g, '<span class="gcode-comment">$1</span>')
      .replace(/\b([GM][0-9]+(?:\.[0-9]+)?)\b/g, '<span class="gcode-code">$1</span>')
      .replace(/\b([XYZIJKABCR][+-]?[0-9]+(?:\.[0-9]+)?)\b/g, '<span class="gcode-coord">$1</span>')
      .replace(/\b([FS][0-9]+(?:\.[0-9]+)?)\b/g, '<span class="gcode-feed">$1</span>')
      .replace(/\b(T[0-9]+)\b/g, '<span class="gcode-tool">$1</span>');
  }

  function renderActiveGcodeView() {
    if (!state.gcodeResult) return;

    if (activeGcodeTab !== 'all' && cachedPerSetupGcode && cachedPerSetupGcode[activeGcodeTab]) {
      const setupData = cachedPerSetupGcode[activeGcodeTab];
      gcodeOutput.innerHTML = `<code>${highlightGcode(setupData.gcode)}</code>`;
      if (gcodeLineCounter) {
        gcodeLineCounter.textContent = `${setupData.line_count.toLocaleString()} lines (${(setupData.cycle_time_seconds / 60).toFixed(1)} min)`;
      }
      if (gcodeSetupBadge) {
        gcodeSetupBadge.textContent = `${setupData.name} • ${setupData.work_offset}`;
      }
      if (btnDownloadLabel) {
        btnDownloadLabel.textContent = `Download ${setupData.work_offset} .NC`;
      }
    } else {
      gcodeOutput.innerHTML = `<code>${highlightGcode(state.gcodeResult.gcode)}</code>`;
      const count = Object.keys(cachedPerSetupGcode).length || (state.setups ? state.setups.length : 1);
      if (gcodeLineCounter) {
        gcodeLineCounter.textContent = `${state.gcodeResult.line_count.toLocaleString()} lines (${(state.gcodeResult.cycle_time_seconds / 60).toFixed(1)} min)`;
      }
      if (gcodeSetupBadge) {
        gcodeSetupBadge.textContent = count > 1 ? `Master Program (${count} Setups Combined)` : `Setup 1 • ${state.setups?.[0]?.work_offset || 'G54'}`;
      }
      if (btnDownloadLabel) {
        btnDownloadLabel.textContent = 'Download Master .NC';
      }
    }
  }

  postControllerSelect.addEventListener('change', generateGcode);
  btnExportGcode.addEventListener('click', () => {
    setStep(6);
    generateGcode();
  });

  btnCopyGcode.addEventListener('click', () => {
    let rawText = '';
    if (activeGcodeTab !== 'all' && cachedPerSetupGcode && cachedPerSetupGcode[activeGcodeTab]) {
      rawText = cachedPerSetupGcode[activeGcodeTab].gcode;
    } else if (state.gcodeResult) {
      rawText = state.gcodeResult.gcode;
    }
    if (rawText) {
      navigator.clipboard.writeText(rawText);
      btnCopyGcode.textContent = 'Copied!';
      setTimeout(() => (btnCopyGcode.innerHTML = `
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy
      `), 2000);
    }
  });

  btnDownloadFile.addEventListener('click', async () => {
    const controller = postControllerSelect.value;
    const partName = (state.modelName || 'PART_01').replace(/[^a-zA-Z0-9_-]/g, '_');

    if (activeGcodeTab !== 'all') {
      if (cachedPerSetupGcode[activeGcodeTab]) {
        const sData = cachedPerSetupGcode[activeGcodeTab];
        const safeName = `${partName}_${sData.name.replace(/[^a-zA-Z0-9_-]/g, '_')}_${sData.work_offset}.nc`;
        triggerBlobDownload(sData.gcode, safeName, 'text/plain');
      } else {
        try {
          const res = await fetch(`/api/gcode/download/${activeGcodeTab}?controller=${controller}`);
          if (!res.ok) throw new Error(await apiErrorText(res, 'Failed to download setup G-code'));
          const text = await res.text();
          triggerBlobDownload(text, `${partName}_${activeGcodeTab}_${controller.toUpperCase()}.nc`, 'text/plain');
        } catch (e) {
          alert(`Download Error: ${e.message}`);
        }
      }
    } else if (state.gcodeResult && state.gcodeResult.gcode) {
      triggerBlobDownload(state.gcodeResult.gcode, `${partName}_MASTER_${controller.toUpperCase()}.nc`, 'text/plain');
    }
  });

  if (btnExportPackage) {
    btnExportPackage.addEventListener('click', async () => {
      const controller = postControllerSelect.value;
      const partName = (state.modelName || 'PART_01').replace(/[^a-zA-Z0-9_-]/g, '_');
      const origHTML = btnExportPackage.innerHTML;
      try {
        btnExportPackage.disabled = true;
        btnExportPackage.textContent = 'Packaging...';
        const res = await fetch(`/api/export-package?controller=${controller}`);
        if (!res.ok) throw new Error(await apiErrorText(res, 'Failed to export CAM production package'));
        const blob = await res.blob();
        triggerBlobDownload(blob, `${partName}_CAM_Package_${controller.toUpperCase()}.zip`, 'application/zip');
      } catch (e) {
        alert(`Export Package Error: ${e.message}`);
      } finally {
        btnExportPackage.disabled = false;
        btnExportPackage.innerHTML = origHTML;
      }
    });
  }

  // ------------------------------------------------------------- Setup Sheet Modal
  let activeSetupSheetTab = 'master';

  async function loadSetupSheet(tabId = 'master') {
    activeSetupSheetTab = tabId;
    if (setupSheetContent) {
      setupSheetContent.textContent = 'Generating shop floor setup sheet...';
    }

    try {
      let endpoint = '/api/setup-sheet-master';
      if (tabId !== 'master') {
        endpoint = `/api/setup-sheet/${tabId}`;
      }
      const res = await fetch(endpoint);
      if (!res.ok) {
        throw new Error(await apiErrorText(res, 'Failed to generate setup sheet'));
      }
      const data = await res.json();
      if (setupSheetContent) {
        setupSheetContent.textContent = data.text;
      }
      if (setupSheetMeta) {
        const timeSec = data.sheet?.total_cycle_time || data.sheet?.estimated_cycle_time || 0;
        setupSheetMeta.textContent = `Est. Time: ${(timeSec/60).toFixed(1)} min | Machine: ${data.sheet?.machine || '--'}`;
      }
    } catch (err) {
      if (setupSheetContent) {
        setupSheetContent.textContent = `Error loading setup sheet:\n${err.message}`;
      }
    }
  }

  function renderSetupSheetTabs() {
    if (!setupSheetTabs) return;
    setupSheetTabs.innerHTML = '';

    const masterBtn = document.createElement('button');
    masterBtn.type = 'button';
    masterBtn.className = `gcode-setup-tab ${activeSetupSheetTab === 'master' ? 'active' : ''}`;
    masterBtn.innerHTML = `<span class="tab-title">Master Routing</span><span class="tab-chip">ALL</span>`;
    masterBtn.addEventListener('click', () => {
      activeSetupSheetTab = 'master';
      renderSetupSheetTabs();
      loadSetupSheet('master');
    });
    setupSheetTabs.appendChild(masterBtn);

    const setups = state.setups || [];
    setups.forEach((s, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `gcode-setup-tab ${activeSetupSheetTab === s.id ? 'active' : ''}`;
      const opLabel = `OP${(idx + 1) * 10}`;
      const wo = s.work_offset || `G${54 + idx}`;
      btn.innerHTML = `<span class="tab-title">${opLabel}</span><span class="tab-chip">${wo}</span>`;
      btn.addEventListener('click', () => {
        activeSetupSheetTab = s.id;
        renderSetupSheetTabs();
        loadSetupSheet(s.id);
      });
      setupSheetTabs.appendChild(btn);
    });
  }

  if (btnViewSetupSheet) {
    btnViewSetupSheet.addEventListener('click', () => {
      if (setupSheetModal) {
        setupSheetModal.classList.add('active');
        activeSetupSheetTab = 'master';
        renderSetupSheetTabs();
        loadSetupSheet('master');
      }
    });
  }

  if (btnCloseSetupSheet) {
    btnCloseSetupSheet.addEventListener('click', () => {
      if (setupSheetModal) {
        setupSheetModal.classList.remove('active');
      }
    });
  }

  if (setupSheetModal) {
    setupSheetModal.addEventListener('click', (e) => {
      if (e.target === setupSheetModal) {
        setupSheetModal.classList.remove('active');
      }
    });
  }

  if (btnCopySetupSheet) {
    btnCopySetupSheet.addEventListener('click', () => {
      if (setupSheetContent && setupSheetContent.textContent) {
        navigator.clipboard.writeText(setupSheetContent.textContent);
        btnCopySetupSheet.textContent = 'Copied!';
        setTimeout(() => (btnCopySetupSheet.innerHTML = `
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          Copy Report
        `), 2000);
      }
    });
  }

  if (btnPrintSetupSheet) {
    btnPrintSetupSheet.addEventListener('click', () => {
      if (!setupSheetContent) return;
      const printWin = window.open('', '_blank');
      if (!printWin) return;
      printWin.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>Shop Floor Setup Sheet</title>
          <style>
            body { font-family: 'Courier New', monospace; font-size: 11pt; padding: 20px; line-height: 1.4; color: #111; }
            pre { white-space: pre-wrap; word-wrap: break-word; }
          </style>
        </head>
        <body>
          <pre>${setupSheetContent.textContent}</pre>
        </body>
        </html>
      `);
      printWin.document.close();
      printWin.focus();
      setTimeout(() => {
        printWin.print();
        printWin.close();
      }, 300);
    });
  }


  // ------------------------------------------------------------- Viewport Toolbar

  let partVisible = true;
  let stockVisible = true;
  let fixtureVisible = true;
  let toolpathVisible = true;
  let toolVisible = false;
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

  if (toggleFixtureBtn) {
    toggleFixtureBtn.addEventListener('click', () => {
      fixtureVisible = !fixtureVisible;
      viewer.toggleFixture(fixtureVisible);
      toggleFixtureBtn.classList.toggle('active', fixtureVisible);
    });
  }

  toggleToolpathBtn.addEventListener('click', () => {
    toolpathVisible = !toolpathVisible;
    viewer.toggleToolpath(toolpathVisible);
    toggleToolpathBtn.classList.toggle('active', toolpathVisible);
  });

  if (toggleToolBtn) {
    toggleToolBtn.addEventListener('click', () => {
      toolVisible = !toolVisible;
      viewer.toggleTool(toolVisible);
      toggleToolBtn.classList.toggle('active', toolVisible);
    });
  }

  toggleWireframeBtn.addEventListener('click', () => {
    wireframeEnabled = !wireframeEnabled;
    viewer.toggleWireframe(wireframeEnabled);
    toggleWireframeBtn.classList.toggle('active', wireframeEnabled);
  });

  const viewRightBtn = document.getElementById('view-right');
  const viewBottomBtn = document.getElementById('view-bottom');
  const stockOpacitySlider = document.getElementById('stock-opacity-slider');

  viewIsoBtn.addEventListener('click', () => viewer.setCameraView('iso'));
  viewTopBtn.addEventListener('click', () => viewer.setCameraView('top'));
  viewFrontBtn.addEventListener('click', () => viewer.setCameraView('front'));
  if (viewRightBtn) viewRightBtn.addEventListener('click', () => viewer.setCameraView('right'));
  if (viewBottomBtn) viewBottomBtn.addEventListener('click', () => viewer.setCameraView('bottom'));
  viewResetBtn.addEventListener('click', () => viewer.setCameraView('reset'));

  if (stockOpacitySlider) {
    stockOpacitySlider.addEventListener('input', (e) => {
      viewer.setStockOpacity(e.target.value);
    });
  }

  // Interactive 3D ViewCube Face Clicks
  const viewCubeFaces = document.querySelectorAll('.cube-face');
  viewCubeFaces.forEach((face) => {
    face.addEventListener('click', (e) => {
      e.stopPropagation();
      const v = face.dataset.view;
      if (v) viewer.setCameraView(v, true);
    });
  });

  // ------------------------------------------------------------- Toast Notification Engine
  function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    let iconSvg = '';
    if (type === 'success') {
      iconSvg = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" class="text-emerald"><polyline points="20 6 9 17 4 12"/></svg>';
    } else if (type === 'warning') {
      iconSvg = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" class="text-amber"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    } else if (type === 'error') {
      iconSvg = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" class="text-red"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
    } else {
      iconSvg = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" class="text-cyan"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
    }
    toast.innerHTML = `${iconSvg}<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.25s ease';
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  // ------------------------------------------------------------- Keyboard Shortcuts Engine & Modal
  const btnShortcuts = document.getElementById('btn-shortcuts');
  const shortcutsModal = document.getElementById('shortcuts-modal');
  const btnCloseShortcuts = document.getElementById('btn-close-shortcuts');

  function openShortcutsModal() {
    if (shortcutsModal) shortcutsModal.classList.add('active');
  }

  function closeShortcutsModal() {
    if (shortcutsModal) shortcutsModal.classList.remove('active');
  }

  if (btnShortcuts) btnShortcuts.addEventListener('click', openShortcutsModal);
  if (btnCloseShortcuts) btnCloseShortcuts.addEventListener('click', closeShortcutsModal);
  if (shortcutsModal) {
    shortcutsModal.addEventListener('click', (e) => {
      if (e.target === shortcutsModal) closeShortcutsModal();
    });
  }

  window.addEventListener('keydown', (e) => {
    // Ignore keypresses if user is typing into an input or select
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      if (e.key === 'Escape') {
        document.activeElement.blur();
      }
      return;
    }

    if (e.key >= '1' && e.key <= '6') {
      const step = parseInt(e.key, 10);
      setStep(step);
      showToast(`Switched to Step ${step}`, 'info', 1500);
    } else if (e.code === 'Space') {
      e.preventDefault();
      simBtnPlay.click();
    } else if (e.key === 'r' || e.key === 'R') {
      simBtnReset.click();
      viewer.setCameraView('reset');
      showToast('Simulation & View Reset', 'info', 1500);
    } else if (e.key === 't' || e.key === 'T') {
      viewer.setCameraView('top');
    } else if (e.key === 'f' || e.key === 'F') {
      viewer.setCameraView('front');
    } else if (e.key === 'i' || e.key === 'I') {
      viewer.setCameraView('iso');
    } else if (e.key === 'w' || e.key === 'W') {
      toggleWireframeBtn.click();
    } else if (e.key === 'p' || e.key === 'P') {
      togglePartBtn.click();
    } else if (e.key === 's' || e.key === 'S') {
      toggleStockBtn.click();
    } else if (e.key === '?' || (e.shiftKey && e.key === '/')) {
      if (shortcutsModal?.classList.contains('active')) {
        closeShortcutsModal();
      } else {
        openShortcutsModal();
      }
    } else if (e.key === 'Escape') {
      closeShortcutsModal();
      if (setupSheetModal) setupSheetModal.classList.remove('active');
      if (toolInspectorModal) toolInspectorModal.style.display = 'none';
    }
  });

  // ------------------------------------------------------------- Tool Geometry & Parameters Inspector
  const toolInspectorModal = document.getElementById('tool-inspector-modal');
  const btnCloseToolInspector = document.getElementById('btn-close-tool-inspector');
  const toolInspectorTitle = document.getElementById('tool-inspector-title');
  const toolInspectorBody = document.getElementById('tool-inspector-body');

  function openToolInspector(toolId) {
    if (!toolInspectorModal || !toolInspectorBody) return;
    const tool = (state.tools || []).find(t => t.id === toolId);
    if (!tool) {
      showToast(`Tool ${toolId} not found in library`, 'warning', 2500);
      return;
    }

    if (toolInspectorTitle) {
      toolInspectorTitle.textContent = `${tool.id} - ${tool.name}`;
    }

    const usingOps = (state.plannedOps || []).filter(op => op.tool_id === tool.id);

    toolInspectorBody.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; background:rgba(0, 240, 255, 0.06); padding:10px 14px; border-radius:6px; border:1px solid rgba(0, 240, 255, 0.2);">
        <div>
          <div style="font-size:0.75rem; color:var(--text-dim); text-transform:uppercase; font-weight:600;">Tool Classification</div>
          <div style="font-size:1.0rem; font-weight:700; color:var(--accent-cyan); text-transform:capitalize;">${(tool.type || 'Tool').replace(/_/g, ' ')}</div>
        </div>
        <span class="mono text-emerald" style="font-size:1.15rem; font-weight:700;">Ø${tool.diameter} mm</span>
      </div>

      <div class="tool-inspector-grid">
        <div class="tool-inspector-stat">
          <span class="label">Cutting Diameter (ØD)</span>
          <span class="val text-cyan">Ø${tool.diameter} mm</span>
        </div>
        <div class="tool-inspector-stat">
          <span class="label">Flute Length (Lc)</span>
          <span class="val">${tool.flute_length || 15.0} mm</span>
        </div>
        <div class="tool-inspector-stat">
          <span class="label">Flute Count (z)</span>
          <span class="val">${tool.flutes || 2} Flutes</span>
        </div>
        <div class="tool-inspector-stat">
          <span class="label">Max Spindle Speed</span>
          <span class="val text-amber">${tool.max_rpm || 10000} RPM</span>
        </div>
        <div class="tool-inspector-stat">
          <span class="label">Corner Radius (r)</span>
          <span class="val">${tool.corner_radius !== undefined ? `${tool.corner_radius} mm` : '0.0 mm (Square)'}</span>
        </div>
        <div class="tool-inspector-stat">
          <span class="label">Overall Length (L)</span>
          <span class="val">${tool.overall_length || 50.0} mm</span>
        </div>
      </div>

      <div style="margin-top:14px; border-top:1px solid rgba(255,255,255,0.08); padding-top:12px;">
        <div style="font-size:0.75rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; margin-bottom:8px; display:flex; justify-content:space-between;">
          <span>Assigned Operations (${usingOps.length})</span>
          <span class="text-cyan">${usingOps.length > 0 ? 'Active in Program' : 'Idle in Carousel'}</span>
        </div>
        ${usingOps.length > 0 ? `
          <div style="display:flex; flex-direction:column; gap:5px; max-height:140px; overflow-y:auto;">
            ${usingOps.map((op, idx) => `
              <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(0,0,0,0.3); padding:6px 10px; border-radius:4px; font-size:0.75rem; border:1px solid rgba(255,255,255,0.04);">
                <div style="display:flex; align-items:center; gap:6px;">
                  <span class="mono text-cyan" style="font-weight:700;">#${idx + 1}</span>
                  <span style="font-weight:600; text-transform:uppercase;">${(op.purpose || 'Op').replace(/_/g, ' ')}</span>
                  <span style="color:var(--text-dim);">(${op.feature_id})</span>
                </div>
                <div style="display:flex; gap:8px; align-items:center;">
                  <span class="mono text-emerald">${op.feed_rate} mm/min</span>
                  <span class="mono text-muted">${op.spindle_rpm} RPM</span>
                </div>
              </div>
            `).join('')}
          </div>
        ` : `
          <div style="color:var(--text-dim); font-size:0.78rem; font-style:italic; padding:6px 0;">No operations are currently assigned to this tool.</div>
        `}
      </div>

      <div style="margin-top:16px; display:flex; justify-content:flex-end; gap:8px;">
        <button id="btn-close-tool-inspector-footer" class="btn btn-secondary btn-sm" style="padding:6px 14px; font-size:0.8rem;">Close</button>
      </div>
    `;

    toolInspectorModal.style.display = 'flex';

    const closeFooter = toolInspectorBody.querySelector('#btn-close-tool-inspector-footer');
    if (closeFooter) {
      closeFooter.addEventListener('click', () => {
        toolInspectorModal.style.display = 'none';
      });
    }
  }

  // Expose globally for inline onclick handlers if needed
  window.openToolInspector = openToolInspector;

  if (btnCloseToolInspector) {
    btnCloseToolInspector.addEventListener('click', () => {
      if (toolInspectorModal) toolInspectorModal.style.display = 'none';
    });
  }
  if (toolInspectorModal) {
    toolInspectorModal.addEventListener('click', (e) => {
      if (e.target === toolInspectorModal) toolInspectorModal.style.display = 'none';
    });
  }

  btnLoadDemo.addEventListener('click', () => {
    loadDemoModel();
    setStep(1);
    showToast('Demo model loaded', 'success', 2000);
  });

  // ------------------------------------------------------------- Initial Boot
  loadToolsAndOps();
  loadDemoModel();
});
