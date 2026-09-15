/**
 * Main Application Controller for Generic CAM Studio
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
  const addSetupForm = document.getElementById('add-setup-form');
  const newSetupName = document.getElementById('new-setup-name');
  const newSetupPreset = document.getElementById('new-setup-preset');
  const newSetupWorkoffset = document.getElementById('new-setup-workoffset');
  const btnConfirmAddSetup = document.getElementById('btn-confirm-add-setup');
  const btnCancelAddSetup = document.getElementById('btn-cancel-add-setup');

  const inputMarginXY = document.getElementById('stock-margin-xy');
  const inputMarginZTop = document.getElementById('stock-margin-z-top');
  const inputMarginZBot = document.getElementById('stock-margin-z-bot');
  const inputOffsetX = document.getElementById('stock-offset-x');
  const inputOffsetY = document.getElementById('stock-offset-y');
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
  const toggleFixtureBtn = document.getElementById('toggle-fixture');
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

    if (stepNum === 2) {
      loadSetups();
    } else if (stepNum === 3 && state.features.length === 0) {
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

  // ------------------------------------------------------------- Stock & Clamping Updates
  function updateStockBounds() {
    if (!state.model) return;
    const b = state.model.mesh.bounding_box;
    const offX = state.stock.offset_x || 0;
    const offY = state.stock.offset_y || 0;
    const offZ = state.stock.offset_z || 0;
    const min = [
      b.min[0] - state.stock.margin_x + offX,
      b.min[1] - state.stock.margin_y + offY,
      b.min[2] - state.stock.margin_z_bottom + offZ,
    ];
    const max = [
      b.max[0] + state.stock.margin_x + offX,
      b.max[1] + state.stock.margin_y + offY,
      b.max[2] + state.stock.margin_z_top + offZ,
    ];
    viewer.updateStockBounds(min, max, state.stock);
  }

  [inputMarginXY, inputMarginZTop, inputMarginZBot, inputOffsetX, inputOffsetY, inputClampHeight, inputClampWidth].forEach((input) => {
    if (!input) return;
    input.addEventListener('input', () => {
      state.stock.margin_x = parseFloat(inputMarginXY.value) || 0;
      state.stock.margin_y = parseFloat(inputMarginXY.value) || 0;
      state.stock.margin_z_top = parseFloat(inputMarginZTop.value) || 0;
      state.stock.margin_z_bottom = parseFloat(inputMarginZBot.value) || 0;
      state.stock.offset_x = parseFloat(inputOffsetX.value) || 0;
      state.stock.offset_y = parseFloat(inputOffsetY.value) || 0;
      state.stock.clamp_height = parseFloat(inputClampHeight.value) || 3.0;
      state.stock.clamp_width = parseFloat(inputClampWidth.value) || 14.0;
      updateStockBounds();
    });
  });

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
          opPlanningWarnings.innerHTML = data.warnings.map(w => `
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


  // ------------------------------------------------------------- Toolpath Generation
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

      statCutLen.textContent = `${data.total_cutting_length_mm.toFixed(1)} mm`;
      statRapidLen.textContent = `${data.total_rapid_length_mm.toFixed(1)} mm`;
      statEstTime.textContent = formatTime(Math.round(data.estimated_time_seconds));
      statSegCount.textContent = data.toolpaths.reduce((acc, tp) => acc + tp.segment_count, 0);

      viewer.setToolCatalog(state.tools);
      viewer.renderToolpaths(data.toolpaths);
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

  // ------------------------------------------------------------- G-Code Generation
  let gcodeGenerating = false;
  async function generateGcode() {
    if (gcodeGenerating) return;
    gcodeGenerating = true;
    const controller = postControllerSelect.value;
    try {
      const res = await fetch('/api/generate-gcode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ controller: controller, program_name: 'PART_01', all_setups: true }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`G-Code Generation Notice:\n${err.detail || 'Failed to generate G-Code'}`);
        return;
      }
      const data = await res.json();
      state.gcodeResult = data;
      gcodeOutput.textContent = data.gcode;
      const setupsStr = data.setups && data.setups.length ? ` | Setups: ${data.setups.join(', ')}` : '';
      gcodeLineCounter.textContent = `${data.line_count} lines (${(data.cycle_time_seconds / 60).toFixed(1)} min)${setupsStr}`;
    } catch (err) {
      console.error('Failed to post-process G-Code:', err);
    } finally {
      gcodeGenerating = false;
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
  let fixtureVisible = true;
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
