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

    if (stepNum === 2) {
      loadSetups();
    } else if (stepNum === 3 && state.features.length === 0) {
      loadFeatures();
    } else if (stepNum === 4 && state.plannedOps.length === 0) {
      loadOperations();
    } else if (stepNum === 5) {
      if (state.toolpaths.length === 0) {
        generateToolpaths();
      } else {
        renderToolpathsSetupTabs();
        applyActiveToolpathView();
      }
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
      const badgeClass = s.work_offset === 'G54' ? '' : (s.work_offset === 'G55' ? 'g55' : 'g56');

      item.innerHTML = `
        <div class="setup-tab-info">
          <span class="setup-badge ${badgeClass}">${s.work_offset}</span>
          <span class="setup-tab-title">${s.name}</span>
          <span style="font-size: 0.72rem; color: var(--text-dim);">(${s.feature_ids ? s.feature_ids.length : 0} feats)</span>
        </div>
        <div class="setup-tab-actions">
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
    if (inputFixedX) inputFixedX.value = (Math.ceil((sx + 10) / 5) * 5).toFixed(1);
    if (inputFixedY) inputFixedY.value = (Math.ceil((sy + 10) / 5) * 5).toFixed(1);
    if (inputFixedZ) inputFixedZ.value = (Math.ceil((sz + 4) / 5) * 5).toFixed(1);
    if (inputFixedCylDia) inputFixedCylDia.value = (Math.ceil((partR * 2 + 6) / 5) * 5).toFixed(1);
    if (inputFixedCylLen) inputFixedCylLen.value = (Math.ceil((sz + 16) / 5) * 5).toFixed(1);
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
      const setupName = f.notes && f.notes.setup_name ? f.notes.setup_name : '';

      item.innerHTML = `
        <div>
          <div style="font-weight: 500; font-size: 0.85rem; text-transform: capitalize;">${title}</div>
          <div style="font-size: 0.72rem; color: var(--text-dim);">${dimStr} ${setupName ? `| <span class="text-cyan">${setupName}</span>` : ''}</div>
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
    toolListContainer.innerHTML = '';
    tools.forEach((t) => {
      const item = document.createElement('div');
      item.className = 'tool-card-item';
      item.style.display = 'flex';
      item.style.justifyContent = 'space-between';
      item.style.alignItems = 'center';
      item.innerHTML = `
        <div>
          <div style="font-weight: 600; font-size: 0.85rem;" class="mono text-cyan">${t.id}: ${t.name}</div>
          <div style="font-size: 0.72rem; color: var(--text-dim);">Max ${t.max_rpm} RPM | ${t.flutes} Flutes | Ø${t.diameter}mm</div>
        </div>
        <button class="btn btn-sm btn-outline delete-tool-btn" data-id="${t.id}" title="Remove Tool" style="padding:2px 8px; font-size:0.75rem; color:#f87171; border-color:rgba(248,113,113,0.3);">✕</button>
      `;
      toolListContainer.appendChild(item);
    });

    // Wire up delete tool buttons
    toolListContainer.querySelectorAll('.delete-tool-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
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

  function renderOpsList(ops) {
    opListContainer.innerHTML = '';
    ops.forEach((op, index) => {
      const item = document.createElement('div');
      item.className = 'op-card-item';
      const setupName = op.notes && op.notes.setup_name ? op.notes.setup_name : '';
      const workOffset = op.notes && op.notes.work_offset ? op.notes.work_offset : '';

      item.innerHTML = `
        <div>
          <div style="font-weight: 600; font-size: 0.85rem;">
            ${index + 1}. <span style="text-transform: uppercase;">${op.purpose}</span> (${op.feature_id})
            ${workOffset ? `<span class="setup-badge ${workOffset === 'G55' ? 'g55' : ''}">${workOffset}</span>` : ''}
          </div>
          <div style="font-size: 0.72rem; color: var(--text-dim);">
            ${op.tool_id} | Feed: ${op.feed_rate} mm/min | ${op.spindle_rpm} RPM ${setupName ? `| ${setupName}` : ''}
          </div>
        </div>
        <span class="feature-tag ${op.purpose}">${op.purpose}</span>
      `;
      opListContainer.appendChild(item);
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
  }

  async function generateToolpaths() {
    btnGenerateToolpaths.disabled = true;
    btnGenerateToolpaths.innerHTML = 'Calculating Multi-Setup Paths...';

    try {
      const res = await fetch('/api/generate-toolpaths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stock: state.stock, all_setups: true }),
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
        alert(`G-Code Generation Notice:\n${err.detail || 'Failed to generate G-Code'}`);
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


  function renderActiveGcodeView() {
    if (!state.gcodeResult) return;

    if (activeGcodeTab !== 'all' && cachedPerSetupGcode && cachedPerSetupGcode[activeGcodeTab]) {
      const setupData = cachedPerSetupGcode[activeGcodeTab];
      gcodeOutput.textContent = setupData.gcode;
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
      gcodeOutput.textContent = state.gcodeResult.gcode;
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
    const textToCopy = gcodeOutput.textContent;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
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
          if (!res.ok) throw new Error('Failed to download setup G-code');
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
        if (!res.ok) throw new Error('Failed to export CAM production package');
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
        throw new Error('Failed to generate setup sheet');
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
  let toolVisible = true;
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
