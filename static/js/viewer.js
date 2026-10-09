/**
 * 3D WebGL CAM Viewport powered by Three.js
 */

class Cam3DViewer {
  constructor(containerId) {
    console.log('Cam3DViewer: constructor called for', containerId);
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    
    // Scene objects
    this.partGroup = new THREE.Group();
    this.partMesh = null;
    this.partEdges = null;
    this.partVisible = true;
    this.stockMesh = null;
    this.stockVisible = true;
    this.wcsMarker = null;
    this.toolpathGroup = new THREE.Group();
    this.toolpathVisible = true;
    this.featureHighlightGroup = new THREE.Group();
    this.toolGroup = new THREE.Group();
    this.toolVisible = false;
    this.fixtureGroup = new THREE.Group();
    this.fixtureVisible = true;
    this.clampZoneGroup = new THREE.Group();
    this.wireframeEnabled = false;
    this.grid = null;
    this.partCenter = new THREE.Vector3(0, 0, 0);
    this.rawMeshData = null;
    
    // Tool simulation state
    this.toolpaths = [];
    this.animSegments = [];
    this.totalAnimDistance = 0;
    this.currentAnimDistance = 0;
    this.lastCarvedDistance = 0;
    this.isPlaying = false;
    this.simSpeed = 2.0; // mm per frame multiplier
    this.onTelemetryUpdate = null;
    this.onProgressUpdate = null;
    this.clampingConfig = null;
    this.stockBounds = null;
    this.toolCatalog = {};
    this.currentSimToolId = null;

    // Dynamic stock heightfield grid (optimized 64x64 for fast real-time 60fps carving)
    this.gridNX = 64;
    this.gridNY = 64;
    this.stockHeights = null;
    this.initialStockHeights = null;
    this.stockGeom = null;

    this.init();
  }

  setToolCatalog(tools) {
    if (!tools || !Array.isArray(tools)) return;
    tools.forEach((t) => {
      this.toolCatalog[t.id] = t;
    });
  }

  init() {
    console.log('Cam3DViewer: init() called');
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 600;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0e17);

    // Camera (Z-up orientation for CNC / CAM standard)
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    this.camera.position.set(120, -160, 140);
    this.camera.up.set(0, 0, 1);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(50, 30, 12.5);

    // Lighting
    this.setupLighting();

    // Environment & Grid
    this.setupEnvironment();

    // Add object groups to scene
    this.scene.add(this.partGroup);
    this.scene.add(this.toolpathGroup);
    this.scene.add(this.featureHighlightGroup);
    this.scene.add(this.toolGroup);
    this.scene.add(this.fixtureGroup);
    this.scene.add(this.clampZoneGroup);

    // Build Cutting Tool
    this.buildSimTool();

    // Event listeners
    window.addEventListener('resize', () => this.onWindowResize());

    // Animation Loop
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  setupLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(200, 200, 300);
    dirLight1.castShadow = true;
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
    dirLight2.position.set(-200, -200, -100);
    this.scene.add(dirLight2);
  }

  setupEnvironment() {
    // 3D Grid on XY Plane (Z=0)
    const gridSize = 400;
    const gridDivisions = 40;
    this.grid = new THREE.GridHelper(gridSize, gridDivisions, 0x00f0ff, 0x1a243b);
    this.grid.rotateX(Math.PI / 2); // Rotate to lie on XY plane (Z-up)
    this.grid.position.set(50, 30, 0);
    this.scene.add(this.grid);

    // Build Coordinate Origin Datum Triad
    this.buildWcsOriginMarker('G54');
  }

  buildWcsOriginMarker(workOffset = 'G54') {
    if (this.wcsMarker) {
      this.scene.remove(this.wcsMarker);
      this.wcsMarker = null;
    }
    const group = new THREE.Group();
    group.name = 'wcs_triad';
    const axisLen = 25;
    const headLen = 6;
    const headWidth = 2.5;

    // X Axis (Red)
    const arrowX = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), axisLen, 0xef4444, headLen, headWidth);
    arrowX.renderOrder = 999;
    group.add(arrowX);

    // Y Axis (Green)
    const arrowY = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), axisLen, 0x10b981, headLen, headWidth);
    arrowY.renderOrder = 999;
    group.add(arrowY);

    // Z Axis (Cyan/Blue)
    const arrowZ = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), axisLen, 0x00f0ff, headLen, headWidth);
    arrowZ.renderOrder = 999;
    group.add(arrowZ);

    // Origin Center Sphere
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.6, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    group.add(sphere);

    this.wcsMarker = group;
    this.scene.add(this.wcsMarker);
  }

  buildSimTool(diameter = 10, fluteLength = 35) {
    this.toolGroup.clear();
    const d = Math.max(2, Math.min(diameter, 25));
    const radius = d / 2;
    const length = Math.max(15, Math.min(fluteLength, 45));

    // Cutting Flute Cylinder (Cyan carbide tip)
    const geomFlute = new THREE.CylinderGeometry(radius, radius, length, 32);
    geomFlute.rotateX(Math.PI / 2);
    geomFlute.translate(0, 0, length / 2);
    const matFlute = new THREE.MeshStandardMaterial({
      color: 0x00f0ff,
      metalness: 0.85,
      roughness: 0.15,
      emissive: 0x002233,
    });
    const meshFlute = new THREE.Mesh(geomFlute, matFlute);
    this.toolGroup.add(meshFlute);

    // Tool Shank Cylinder (Steel)
    const shankRadius = Math.max(radius, 4.0);
    const geomShank = new THREE.CylinderGeometry(shankRadius, shankRadius, 25, 32);
    geomShank.rotateX(Math.PI / 2);
    geomShank.translate(0, 0, length + 12.5);
    const matShank = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      metalness: 0.8,
      roughness: 0.25,
    });
    const meshShank = new THREE.Mesh(geomShank, matShank);
    this.toolGroup.add(meshShank);

    // ER Collet Nut & Toolholder Taper
    const geomSpindle = new THREE.CylinderGeometry(shankRadius * 1.8, shankRadius * 1.2, 22, 32);
    geomSpindle.rotateX(Math.PI / 2);
    geomSpindle.translate(0, 0, length + 25 + 11);
    const matSpindle = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.7,
      roughness: 0.35,
    });
    const meshSpindle = new THREE.Mesh(geomSpindle, matSpindle);
    this.toolGroup.add(meshSpindle);

    this.toolGroup.position.set(0, 0, 50);
    this.toolGroup.visible = this.toolVisible;
  }

  loadModelMesh(meshData) {
    this.rawMeshData = meshData;
    this.partGroup.clear();

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(meshData.vertices, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(meshData.normals, 3));
    geometry.setIndex(meshData.indices);

    // Material with subtle CAD metallic finish
    const material = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      roughness: 0.35,
      metalness: 0.6,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });

    this.partMesh = new THREE.Mesh(geometry, material);
    this.partMesh.castShadow = true;
    this.partMesh.receiveShadow = true;

    // Outline Edges for CAD wireframe appearance
    const edgesGeom = new THREE.EdgesGeometry(geometry, 24);
    this.partEdges = new THREE.LineSegments(
      edgesGeom,
      new THREE.LineBasicMaterial({ color: 0x94a3b8, linewidth: 1 })
    );

    const bbox = meshData.bounding_box;
    const cx = (bbox.min[0] + bbox.max[0]) / 2;
    const cy = (bbox.min[1] + bbox.max[1]) / 2;
    const cz = (bbox.min[2] + bbox.max[2]) / 2;
    // Position mesh and edges at (0, 0, 0) inside partGroup (also at 0, 0, 0)
    // so rotations pivot around (0,0,0), matching OpenCascade shape.transformed()
    this.partMesh.position.set(0, 0, 0);
    this.partEdges.position.set(0, 0, 0);

    this.partMesh.visible = (this.partVisible !== undefined ? this.partVisible : true);
    this.partEdges.visible = (this.partVisible !== undefined ? this.partVisible : true);
    if (this.wireframeEnabled) {
      this.partMesh.material.wireframe = true;
    }

    this.partGroup.position.set(0, 0, 0);
    this.partGroup.rotation.set(0, 0, 0);
    this.partGroup.add(this.partMesh);
    this.partGroup.add(this.partEdges);

    // Recenter camera target to part center
    this.controls.target.set(cx, cy, cz);
    this.grid.position.set(cx, cy, bbox.min[2] - 0.1);
  }

  setSetupOrientation(rotationDeg = [0, 0, 0], setupStockBounds = null, stockConfig = null) {
    this.currentSetupRotation = rotationDeg || [0, 0, 0];
    if (this.partGroup) {
      const rx = THREE.MathUtils.degToRad(rotationDeg[0] || 0);
      const ry = THREE.MathUtils.degToRad(rotationDeg[1] || 0);
      const rz = THREE.MathUtils.degToRad(rotationDeg[2] || 0);
      this.partGroup.rotation.set(rx, ry, rz);
    }
    if (setupStockBounds && setupStockBounds.min && setupStockBounds.max) {
      this.updateFixtures(setupStockBounds, stockConfig);
      const scx = (setupStockBounds.min[0] + setupStockBounds.max[0]) / 2;
      const scy = (setupStockBounds.min[1] + setupStockBounds.max[1]) / 2;
      const scz = (setupStockBounds.min[2] + setupStockBounds.max[2]) / 2;
      this.controls.target.set(scx, scy, scz);
      this.grid.position.set(scx, scy, setupStockBounds.min[2] - 0.1);
    }
  }

  updateStockBounds(min, max, stockConfig = null) {
    this.partStockBounds = { min: min, max: max };
    this.updateStock(this.partStockBounds, stockConfig);
  }

  updateStock(stockBounds, stockConfig = null) {
    if (!stockBounds || !stockBounds.min || !stockBounds.max) return;
    this.rawStockBounds = stockBounds;
    this.partStockBounds = stockBounds;
    if (stockConfig) this.clampingConfig = stockConfig;

    // Reset setup snapshots when raw stock dimensions change
    this.setupSnapshots = {};

    const min = stockBounds.min;
    const max = stockBounds.max;
    const sx = Math.max(max[0] - min[0], 1);
    const sy = Math.max(max[1] - min[1], 1);
    const sz = Math.max(max[2] - min[2], 1);

    if (this.stockMesh) {
      this.partGroup.remove(this.stockMesh);
      if (this.stockMesh.geometry) this.stockMesh.geometry.dispose();
      this.stockMesh = null;
    }

    const isCylinder = Boolean(stockConfig && (
      stockConfig.stock_mode === 'cylinder' ||
      stockConfig.stock_mode === 'relative_cylinder' ||
      stockConfig.stock_mode === 'fixed_cylinder'
    ));
    this.isCylinderStock = isCylinder;

    // Dual-Surface Solid Stock Mesh for 3D In-Process Workpiece (IPW)
    const NX = this.gridNX;
    const NY = this.gridNY;
    const numVerts = NX * NY;
    const dx = sx / (NX - 1);
    const dy = sy / (NY - 1);

    this.initialTopHeights = new Float32Array(numVerts);
    this.initialBotHeights = new Float32Array(numVerts);

    if (isCylinder) {
      const cx = min[0] + sx / 2;
      const cy = min[1] + sy / 2;
      const radius = Math.max(sx, sy) / 2;
      const rSq = radius * radius;

      for (let j = 0; j < NY; j++) {
        for (let i = 0; i < NX; i++) {
          const k = j * NX + i;
          const gx = min[0] + i * dx;
          const gy = min[1] + j * dy;
          const dSq = (gx - cx) * (gx - cx) + (gy - cy) * (gy - cy);
          if (dSq <= rSq) {
            this.initialTopHeights[k] = max[2];
            this.initialBotHeights[k] = min[2];
          } else {
            this.initialTopHeights[k] = min[2];
            this.initialBotHeights[k] = min[2];
          }
        }
      }
    } else {
      this.initialTopHeights.fill(max[2]);
      this.initialBotHeights.fill(min[2]);
    }

    this.currentTopHeights = new Float32Array(this.initialTopHeights);
    this.currentBotHeights = new Float32Array(this.initialBotHeights);
    this.lastCarvedDistance = 0;

    // Layout:
    // [0 .. numVerts-1]: Top Face
    // [numVerts .. 2*numVerts-1]: Bottom Face
    // [2*numVerts .. 2*numVerts + 2*NX - 1]: Front Wall (Top & Bot)
    // [+ 2*NX]: Back Wall (Top & Bot)
    // [+ 2*NY]: Left Wall (Top & Bot)
    // [+ 2*NY]: Right Wall (Top & Bot)
    const totalVertsCount = 2 * numVerts + 4 * NX + 4 * NY;
    const positions = new Float32Array(totalVertsCount * 3);
    const indices = [];

    // 1. Top Face Vertices [0 .. numVerts-1]
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const idx = j * NX + i;
        positions[idx * 3] = min[0] + i * dx;
        positions[idx * 3 + 1] = min[1] + j * dy;
        positions[idx * 3 + 2] = this.currentTopHeights[idx];
      }
    }

    // 2. Bottom Face Vertices [numVerts .. 2*numVerts-1]
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const idx = numVerts + j * NX + i;
        const k = j * NX + i;
        positions[idx * 3] = min[0] + i * dx;
        positions[idx * 3 + 1] = min[1] + j * dy;
        positions[idx * 3 + 2] = this.currentBotHeights[k];
      }
    }

    // Top Face Triangles (Normal +Z)
    for (let j = 0; j < NY - 1; j++) {
      for (let i = 0; i < NX - 1; i++) {
        const v00 = j * NX + i;
        const v10 = j * NX + i + 1;
        const v01 = (j + 1) * NX + i;
        const v11 = (j + 1) * NX + i + 1;
        indices.push(v00, v10, v11);
        indices.push(v00, v11, v01);
      }
    }

    // Bottom Face Triangles (Normal -Z)
    for (let j = 0; j < NY - 1; j++) {
      for (let i = 0; i < NX - 1; i++) {
        const b00 = numVerts + j * NX + i;
        const b10 = numVerts + j * NX + i + 1;
        const b01 = numVerts + (j + 1) * NX + i;
        const b11 = numVerts + (j + 1) * NX + i + 1;
        indices.push(b00, b11, b10);
        indices.push(b00, b01, b11);
      }
    }

    let off = numVerts * 2;

    // Front Wall (j = 0)
    const frontTopStart = off;
    const frontBotStart = off + NX;
    for (let i = 0; i < NX; i++) {
      positions[(frontTopStart + i) * 3] = min[0] + i * dx;
      positions[(frontTopStart + i) * 3 + 1] = min[1];
      positions[(frontTopStart + i) * 3 + 2] = this.currentTopHeights[i];

      positions[(frontBotStart + i) * 3] = min[0] + i * dx;
      positions[(frontBotStart + i) * 3 + 1] = min[1];
      positions[(frontBotStart + i) * 3 + 2] = this.currentBotHeights[i];
    }
    for (let i = 0; i < NX - 1; i++) {
      const t0 = frontTopStart + i;
      const t1 = frontTopStart + i + 1;
      const b0 = frontBotStart + i;
      const b1 = frontBotStart + i + 1;
      indices.push(t0, b0, b1);
      indices.push(t0, b1, t1);
    }
    off += NX * 2;

    // Back Wall (j = NY - 1)
    const backTopStart = off;
    const backBotStart = off + NX;
    const backRow = (NY - 1) * NX;
    for (let i = 0; i < NX; i++) {
      positions[(backTopStart + i) * 3] = min[0] + i * dx;
      positions[(backTopStart + i) * 3 + 1] = max[1];
      positions[(backTopStart + i) * 3 + 2] = this.currentTopHeights[backRow + i];

      positions[(backBotStart + i) * 3] = min[0] + i * dx;
      positions[(backBotStart + i) * 3 + 1] = max[1];
      positions[(backBotStart + i) * 3 + 2] = this.currentBotHeights[backRow + i];
    }
    for (let i = 0; i < NX - 1; i++) {
      const t0 = backTopStart + i;
      const t1 = backTopStart + i + 1;
      const b0 = backBotStart + i;
      const b1 = backBotStart + i + 1;
      indices.push(t0, b1, b0);
      indices.push(t0, t1, b1);
    }
    off += NX * 2;

    // Left Wall (i = 0)
    const leftTopStart = off;
    const leftBotStart = off + NY;
    for (let j = 0; j < NY; j++) {
      positions[(leftTopStart + j) * 3] = min[0];
      positions[(leftTopStart + j) * 3 + 1] = min[1] + j * dy;
      positions[(leftTopStart + j) * 3 + 2] = this.currentTopHeights[j * NX];

      positions[(leftBotStart + j) * 3] = min[0];
      positions[(leftBotStart + j) * 3 + 1] = min[1] + j * dy;
      positions[(leftBotStart + j) * 3 + 2] = this.currentBotHeights[j * NX];
    }
    for (let j = 0; j < NY - 1; j++) {
      const t0 = leftTopStart + j;
      const t1 = leftTopStart + j + 1;
      const b0 = leftBotStart + j;
      const b1 = leftBotStart + j + 1;
      indices.push(t0, b1, b0);
      indices.push(t0, t1, b1);
    }
    off += NY * 2;

    // Right Wall (i = NX - 1)
    const rightTopStart = off;
    const rightBotStart = off + NY;
    for (let j = 0; j < NY; j++) {
      positions[(rightTopStart + j) * 3] = max[0];
      positions[(rightTopStart + j) * 3 + 1] = min[1] + j * dy;
      positions[(rightTopStart + j) * 3 + 2] = this.currentTopHeights[j * NX + NX - 1];

      positions[(rightBotStart + j) * 3] = max[0];
      positions[(rightBotStart + j) * 3 + 1] = min[1] + j * dy;
      positions[(rightBotStart + j) * 3 + 2] = this.currentBotHeights[j * NX + NX - 1];
    }
    for (let j = 0; j < NY - 1; j++) {
      const t0 = rightTopStart + j;
      const t1 = rightTopStart + j + 1;
      const b0 = rightBotStart + j;
      const b1 = rightBotStart + j + 1;
      indices.push(t0, b0, b1);
      indices.push(t0, b1, t1);
    }

    const stockGeom = new THREE.BufferGeometry();
    stockGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    stockGeom.setIndex(indices);
    stockGeom.computeVertexNormals();
    this.stockGeom = stockGeom;

    const stockMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      roughness: 0.35,
      metalness: 0.25,
      transparent: true,
      opacity: 0.60,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    this.stockMesh = new THREE.Mesh(stockGeom, stockMat);
    this.stockMesh.castShadow = true;
    this.stockMesh.receiveShadow = true;
    this.stockMesh.renderOrder = 2;
    this.stockMesh.visible = (this.stockVisible !== undefined ? this.stockVisible : true);

    // Bounding wireframe outline in part local coordinates
    if (isCylinder) {
      const radius = Math.max(sx, sy) / 2;
      const height = sz;
      const cylinderGeom = new THREE.CylinderGeometry(radius, radius, height, 48, 1, false);
      const axis = (stockConfig.cylinder_axis || 'Z').toUpperCase();
      if (axis === 'Z') {
        cylinderGeom.rotateX(Math.PI / 2);
      } else if (axis === 'X') {
        cylinderGeom.rotateZ(Math.PI / 2);
      }
      const edges = new THREE.EdgesGeometry(cylinderGeom, 30);
      const cylWire = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.6 })
      );
      cylWire.position.set(min[0] + sx / 2, min[1] + sy / 2, min[2] + sz / 2);
      this.stockMesh.add(cylWire);
    } else {
      const boxGeom = new THREE.BoxGeometry(sx, sy, sz);
      const edges = new THREE.EdgesGeometry(boxGeom);
      const boxWire = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.4 })
      );
      boxWire.position.set(min[0] + sx / 2, min[1] + sy / 2, min[2] + sz / 2);
      this.stockMesh.add(boxWire);
    }

    // Add stockMesh directly to partGroup so it flips/rotates synchronously with the part!
    this.partGroup.add(this.stockMesh);
    this.updateFixtures(stockBounds, stockConfig);
  }

  updateStockMeshGeometry() {
    if (!this.stockGeom || !this.currentTopHeights || !this.currentBotHeights || !this.rawStockBounds) return;
    const posAttr = this.stockGeom.attributes.position;
    if (!posAttr) return;
    const positions = posAttr.array;
    const NX = this.gridNX;
    const NY = this.gridNY;
    const numVerts = NX * NY;

    // 1. Top face
    for (let k = 0; k < numVerts; k++) {
      positions[k * 3 + 2] = this.currentTopHeights[k];
    }
    // 2. Bottom face
    for (let k = 0; k < numVerts; k++) {
      positions[(numVerts + k) * 3 + 2] = this.currentBotHeights[k];
    }

    // 3. Side walls
    let off = numVerts * 2;
    // Front wall
    for (let i = 0; i < NX; i++) {
      positions[(off + i) * 3 + 2] = this.currentTopHeights[i];
      positions[(off + NX + i) * 3 + 2] = this.currentBotHeights[i];
    }
    off += NX * 2;

    // Back wall
    const backRow = (NY - 1) * NX;
    for (let i = 0; i < NX; i++) {
      positions[(off + i) * 3 + 2] = this.currentTopHeights[backRow + i];
      positions[(off + NX + i) * 3 + 2] = this.currentBotHeights[backRow + i];
    }
    off += NX * 2;

    // Left wall
    for (let j = 0; j < NY; j++) {
      positions[(off + j) * 3 + 2] = this.currentTopHeights[j * NX];
      positions[(off + NY + j) * 3 + 2] = this.currentBotHeights[j * NX];
    }
    off += NY * 2;

    // Right wall
    for (let j = 0; j < NY; j++) {
      positions[(off + j) * 3 + 2] = this.currentTopHeights[j * NX + NX - 1];
      positions[(off + NY + j) * 3 + 2] = this.currentBotHeights[j * NX + NX - 1];
    }

    posAttr.needsUpdate = true;
    this.stockGeom.computeVertexNormals();
    this.stockGeom.computeBoundingBox();
    this.stockGeom.computeBoundingSphere();
  }

  updateFixtures(stockBounds, stockConfig = null) {
    if (!stockBounds || !stockBounds.min || !stockBounds.max) return;
    const min = stockBounds.min;
    const max = stockBounds.max;
    const sx = Math.max(max[0] - min[0], 1);
    const sy = Math.max(max[1] - min[1], 1);

    if (this.wcsMarker) {
      this.scene.remove(this.wcsMarker);
      this.wcsMarker = null;
    }
    this.fixtureGroup.clear();
    this.clampZoneGroup.clear();

    // 3. Clamping / Fixture Geometry in world coordinates
    const cfg = stockConfig || this.clampingConfig || {};
    const clampType = cfg.clamp_type || 'vise_jaws';
    const clampH = Math.max(parseFloat(cfg.clamp_height !== undefined ? cfg.clamp_height : (cfg.margin_z_bottom || 3.0)) || 3.0, 0.5);
    const clampW = Math.max(parseFloat(cfg.clamp_width || 14.0) || 14.0, 5.0);

    if (clampType !== 'none') {
      const zBot = min[2];
      const zClampTop = zBot + clampH;
      const jawLength = sx + 20.0;
      const jawHeight = clampH + 15.0;

      const jawMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        metalness: 0.85,
        roughness: 0.25,
      });
      const jawWireMat = new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.7 });

      if (clampType === 'vise_jaws') {
        // Front Vise Jaw (-Y)
        const jawGeom = new THREE.BoxGeometry(jawLength, clampW, jawHeight);
        const jawFront = new THREE.Mesh(jawGeom, jawMat);
        jawFront.position.set(min[0] + sx / 2, min[1] - clampW / 2, zClampTop - jawHeight / 2);
        const jawFrontEdges = new THREE.LineSegments(new THREE.EdgesGeometry(jawGeom), jawWireMat);
        jawFront.add(jawFrontEdges);

        // Rear Vise Jaw (+Y)
        const jawRear = new THREE.Mesh(jawGeom, jawMat);
        jawRear.position.set(min[0] + sx / 2, max[1] + clampW / 2, zClampTop - jawHeight / 2);
        const jawRearEdges = new THREE.LineSegments(new THREE.EdgesGeometry(jawGeom), jawWireMat);
        jawRear.add(jawRearEdges);

        // Bolt fasteners on jaws
        const boltGeom = new THREE.CylinderGeometry(2.5, 2.5, 3, 16);
        boltGeom.rotateX(Math.PI / 2);
        const boltMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9, roughness: 0.2 });
        [-jawLength * 0.3, jawLength * 0.3].forEach(bx => {
          const b1 = new THREE.Mesh(boltGeom, boltMat);
          b1.position.set(min[0] + sx / 2 + bx, min[1] - clampW / 2, zClampTop - jawHeight * 0.4);
          const b2 = new THREE.Mesh(boltGeom, boltMat);
          b2.position.set(min[0] + sx / 2 + bx, max[1] + clampW / 2, zClampTop - jawHeight * 0.4);
          this.fixtureGroup.add(b1);
          this.fixtureGroup.add(b2);
        });

        this.fixtureGroup.add(jawFront);
        this.fixtureGroup.add(jawRear);
      } else if (clampType === 'toe_clamps') {
        const toeGeom = new THREE.BoxGeometry(16, 20, clampH + 6);
        const toeMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.3 });
        const corners = [
          [min[0] - 4, min[1] - 4],
          [max[0] + 4, min[1] - 4],
          [min[0] - 4, max[1] + 4],
          [max[0] + 4, max[1] + 4],
        ];
        corners.forEach(([cx, cy]) => {
          const toeMesh = new THREE.Mesh(toeGeom, toeMat);
          toeMesh.position.set(cx, cy, zClampTop - (clampH + 6) / 2);
          this.fixtureGroup.add(toeMesh);
        });
      }

      // Clamping Safe Clearance Boundary Plane
      const planeGeom = new THREE.PlaneGeometry(sx, sy);
      const planeMat = new THREE.MeshBasicMaterial({
        color: 0xef4444,
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
      });
      const clampPlane = new THREE.Mesh(planeGeom, planeMat);
      clampPlane.position.set(min[0] + sx / 2, min[1] + sy / 2, zClampTop);
      this.clampZoneGroup.add(clampPlane);

      const planeBorderGeom = new THREE.EdgesGeometry(new THREE.BoxGeometry(sx, sy, 0.1));
      const planeBorder = new THREE.LineSegments(
        planeBorderGeom,
        new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 2 })
      );
      planeBorder.position.set(min[0] + sx / 2, min[1] + sy / 2, zClampTop);
      this.clampZoneGroup.add(planeBorder);
    }

    if (this.fixtureGroup) this.fixtureGroup.visible = (this.fixtureVisible !== undefined ? this.fixtureVisible : true);
    if (this.clampZoneGroup) this.clampZoneGroup.visible = (this.fixtureVisible !== undefined ? this.fixtureVisible : true);
  }

  carveSegmentIntoArrays(p0_setup, p1_setup, toolRadius, invRotMatrix, topHeights, botHeights, precomputedApproach = null) {
    if (!this.rawStockBounds || !topHeights || !botHeights) return;

    const min = this.rawStockBounds.min;
    const max = this.rawStockBounds.max;
    const sx = Math.max(max[0] - min[0], 1);
    const sy = Math.max(max[1] - min[1], 1);
    const NX = this.gridNX;
    const NY = this.gridNY;
    const dx = sx / (NX - 1);
    const dy = sy / (NY - 1);

    // Direct matrix transform without heap object allocation
    const m = invRotMatrix.elements;
    const x0 = p0_setup.x !== undefined ? p0_setup.x : p0_setup[0];
    const y0 = p0_setup.y !== undefined ? p0_setup.y : p0_setup[1];
    const z0 = p0_setup.z !== undefined ? p0_setup.z : p0_setup[2];
    const p0x = m[0] * x0 + m[4] * y0 + m[8] * z0 + m[12];
    const p0y = m[1] * x0 + m[5] * y0 + m[9] * z0 + m[13];
    const p0z = m[2] * x0 + m[6] * y0 + m[10] * z0 + m[14];

    const x1 = p1_setup.x !== undefined ? p1_setup.x : p1_setup[0];
    const y1 = p1_setup.y !== undefined ? p1_setup.y : p1_setup[1];
    const z1 = p1_setup.z !== undefined ? p1_setup.z : p1_setup[2];
    const p1x = m[0] * x1 + m[4] * y1 + m[8] * z1 + m[12];
    const p1y = m[1] * x1 + m[5] * y1 + m[9] * z1 + m[13];
    const p1z = m[2] * x1 + m[6] * y1 + m[10] * z1 + m[14];

    let isTopApproach = true;
    let isBotApproach = false;
    if (precomputedApproach) {
      isTopApproach = precomputedApproach.isTop;
      isBotApproach = precomputedApproach.isBot;
    } else {
      const spz = m[10]; // spindle (0, 0, 1) transformed z component (m33 in Three.js column-major Matrix4)
      isTopApproach = spz > 0.4;
      isBotApproach = spz < -0.4;
    }

    const minCutX = Math.min(p0x, p1x) - toolRadius;
    const maxCutX = Math.max(p0x, p1x) + toolRadius;
    const minCutY = Math.min(p0y, p1y) - toolRadius;
    const maxCutY = Math.max(p0y, p1y) + toolRadius;

    const iMin = Math.max(0, Math.floor((minCutX - min[0]) / dx));
    const iMax = Math.min(NX - 1, Math.ceil((maxCutX - min[0]) / dx));
    const jMin = Math.max(0, Math.floor((minCutY - min[1]) / dy));
    const jMax = Math.min(NY - 1, Math.ceil((maxCutY - min[1]) / dy));

    if (iMin > iMax || jMin > jMax) return;

    const segVecX = p1x - p0x;
    const segVecY = p1y - p0y;
    const segLenSq = segVecX * segVecX + segVecY * segVecY;
    const rSq = toolRadius * toolRadius;
    const invSegLenSq = segLenSq > 1e-6 ? 1.0 / segLenSq : 0;

    for (let j = jMin; j <= jMax; j++) {
      const gy = min[1] + j * dy;
      const jRow = j * NX;
      for (let i = iMin; i <= iMax; i++) {
        const gx = min[0] + i * dx;
        let t = 0;
        if (invSegLenSq > 0) {
          t = ((gx - p0x) * segVecX + (gy - p0y) * segVecY) * invSegLenSq;
          if (t < 0) t = 0;
          else if (t > 1) t = 1;
        }
        const nx = p0x + t * segVecX;
        const ny = p0y + t * segVecY;
        const dSq = (gx - nx) * (gx - nx) + (gy - ny) * (gy - ny);

        if (dSq <= rSq) {
          const zTool = invSegLenSq === 0 ? Math.min(p0z, p1z) : (p0z + t * (p1z - p0z));
          const gridIdx = jRow + i;

          if (isTopApproach) {
            if (zTool < topHeights[gridIdx]) {
              topHeights[gridIdx] = zTool > min[2] ? zTool : min[2];
              if (topHeights[gridIdx] < botHeights[gridIdx]) {
                botHeights[gridIdx] = topHeights[gridIdx];
              }
            }
          } else if (isBotApproach) {
            if (zTool > botHeights[gridIdx]) {
              botHeights[gridIdx] = zTool < max[2] ? zTool : max[2];
              if (botHeights[gridIdx] > topHeights[gridIdx]) {
                topHeights[gridIdx] = botHeights[gridIdx];
              }
            }
          } else {
            if (zTool < topHeights[gridIdx] && zTool > botHeights[gridIdx]) {
              topHeights[gridIdx] = zTool > min[2] ? zTool : min[2];
            }
          }
        }
      }
    }
  }

  computeAllSetupSnapshots(setups = [], allToolpaths = []) {
    if (!this.rawStockBounds || !this.initialTopHeights || !this.initialBotHeights) return;
    this.setupSnapshots = {};
    this.allSetups = setups;
    this.allSetupsMap = {};
    setups.forEach(s => { this.allSetupsMap[s.id] = s; });

    let currentTop = new Float32Array(this.initialTopHeights);
    let currentBot = new Float32Array(this.initialBotHeights);

    const cuttingTypes = new Set([
      'cut', 'plunge', 'ramp', 'helix',
      'arc_cw', 'arc_ccw', 'entry', 'exit', 'dwell'
    ]);

    setups.forEach((setup) => {
      const startTop = new Float32Array(currentTop);
      const startBot = new Float32Array(currentBot);

      const setupPaths = allToolpaths.filter(tp => (
        tp.setup_id === setup.id ||
        tp.metadata?.setup_id === setup.id ||
        tp.notes?.setup_id === setup.id
      ));

      const rotDeg = setup.rotation_deg || [0, 0, 0];
      const rx = THREE.MathUtils.degToRad(rotDeg[0] || 0);
      const ry = THREE.MathUtils.degToRad(rotDeg[1] || 0);
      const rz = THREE.MathUtils.degToRad(rotDeg[2] || 0);
      const rotMatrix = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz, 'XYZ'));
      const invRotMatrix = rotMatrix.clone().invert();
      const m = invRotMatrix.elements;
      const spz = m[10]; // m33 in Three.js column-major Matrix4
      const precomputedApproach = {
        isTop: spz > 0.4,
        isBot: spz < -0.4,
      };

      setupPaths.forEach((tp) => {
        let toolRadius = 5.0;
        if (tp.tool_id && this.toolCatalog[tp.tool_id]) {
          const dia = this.toolCatalog[tp.tool_id].diameter;
          if (dia && dia > 0) toolRadius = dia / 2.0;
        }

        tp.segments?.forEach((seg) => {
          if (cuttingTypes.has(seg.motion_type)) {
            let r = toolRadius;
            if (seg.tool_id && this.toolCatalog[seg.tool_id]) {
              const dia = this.toolCatalog[seg.tool_id].diameter;
              if (dia && dia > 0) r = dia / 2.0;
            }
            this.carveSegmentIntoArrays(seg.start, seg.end, r, invRotMatrix, currentTop, currentBot, precomputedApproach);
          }
        });
      });

      this.setupSnapshots[setup.id] = {
        startTop: startTop,
        startBot: startBot,
        endTop: new Float32Array(currentTop),
        endBot: new Float32Array(currentBot),
      };
    });
  }

  carveStockToDistance(targetDistance) {
    if (!this.currentTopHeights || !this.currentBotHeights || !this.rawStockBounds || !this.animSegments || this.animSegments.length === 0) {
      return;
    }

    const cuttingTypes = new Set([
      'cut', 'plunge', 'ramp', 'helix',
      'arc_cw', 'arc_ccw', 'entry', 'exit', 'dwell'
    ]);

    // If reset to start or scrubbed backwards, reset to active setup base stock state
    if (targetDistance <= 1e-4 || targetDistance < this.lastCarvedDistance - 1e-3) {
      if (this.activeSetupId && this.activeSetupId !== 'all' && this.setupSnapshots[this.activeSetupId]) {
        this.currentTopHeights.set(this.setupSnapshots[this.activeSetupId].startTop);
        this.currentBotHeights.set(this.setupSnapshots[this.activeSetupId].startBot);
      } else {
        this.currentTopHeights.set(this.initialTopHeights);
        this.currentBotHeights.set(this.initialBotHeights);
      }
      this.lastCarvedDistance = 0;
    }

    let accumulatedDist = 0;

    for (let sIdx = 0; sIdx < this.animSegments.length; sIdx++) {
      const seg = this.animSegments[sIdx];
      const segStartDist = accumulatedDist;
      const segEndDist = accumulatedDist + seg.length;

      if (segEndDist > this.lastCarvedDistance && segStartDist < targetDistance) {
        if (cuttingTypes.has(seg.type)) {
          let toolRadius = 5.0;
          if (seg.tool_id && this.toolCatalog[seg.tool_id]) {
            const dia = this.toolCatalog[seg.tool_id].diameter;
            if (dia && dia > 0) toolRadius = dia / 2.0;
          }
          toolRadius = Math.max(0.1, Math.min(toolRadius, 100.0));

          const sliceStartFrac = seg.length > 0 ? Math.max(0, Math.min(1, (this.lastCarvedDistance - segStartDist) / seg.length)) : 0;
          const sliceEndFrac = seg.length > 0 ? Math.max(0, Math.min(1, (targetDistance - segStartDist) / seg.length)) : 1;

          const p0 = new THREE.Vector3().lerpVectors(seg.start, seg.end, sliceStartFrac);
          const p1 = new THREE.Vector3().lerpVectors(seg.start, seg.end, sliceEndFrac);

          const invRotMatrix = seg.inv_rot_matrix || this.defaultInvRotMatrix || new THREE.Matrix4();
          this.carveSegmentIntoArrays(p0, p1, toolRadius, invRotMatrix, this.currentTopHeights, this.currentBotHeights);
        }
      }

      accumulatedDist += seg.length;
      if (accumulatedDist >= targetDistance) break;
    }

    this.lastCarvedDistance = targetDistance;
    this.updateStockMeshGeometry();
  }

  showSetupToolpaths(setupId) {
    if (!this.setupToolpathGroups) return;
    Object.keys(this.setupToolpathGroups).forEach(sId => {
      const grp = this.setupToolpathGroups[sId];
      if (grp) {
        grp.visible = (setupId === 'all' || sId === setupId);
      }
    });
  }

  renderToolpaths(toolpathList, activeSetupId = 'all', allSetups = [], fullToolpathList = []) {
    this.toolpathGroup.clear();
    this.toolpaths = toolpathList;
    this.activeSetupId = activeSetupId;
    this.animSegments = [];
    this.totalAnimDistance = 0;
    this.currentAnimDistance = 0;
    this.lastCarvedDistance = 0;
    this.totalEstSeconds = 0;
    this.currentSimSetupId = null;

    this.allSetups = allSetups || [];
    this.allSetupsMap = {};
    this.allSetups.forEach(s => { this.allSetupsMap[s.id] = s; });

    // Precompute / ensure setup snapshots exist across all setups
    if (allSetups && allSetups.length > 0) {
      const fullList = (fullToolpathList && fullToolpathList.length > 0) ? fullToolpathList : toolpathList;
      this.computeAllSetupSnapshots(allSetups, fullList);
    }

    // Set active starting stock heights
    if (this.currentTopHeights && this.currentBotHeights) {
      if (activeSetupId !== 'all' && this.setupSnapshots[activeSetupId]) {
        this.currentTopHeights.set(this.setupSnapshots[activeSetupId].startTop);
        this.currentBotHeights.set(this.setupSnapshots[activeSetupId].startBot);
      } else if (this.initialTopHeights && this.initialBotHeights) {
        this.currentTopHeights.set(this.initialTopHeights);
        this.currentBotHeights.set(this.initialBotHeights);
      }
      this.updateStockMeshGeometry();
    }

    this.setupToolpathGroups = {};
    const perSetupVerts = {};

    toolpathList.forEach((tp) => {
      const sId = tp.setup_id || tp.metadata?.setup_id || tp.notes?.setup_id || (allSetups?.[0]?.id || 'setup_001');
      const setupObj = this.allSetupsMap[sId] || allSetups?.[0] || { rotation_deg: [0, 0, 0] };
      const rotDeg = setupObj.rotation_deg || [0, 0, 0];
      const rx = THREE.MathUtils.degToRad(rotDeg[0] || 0);
      const ry = THREE.MathUtils.degToRad(rotDeg[1] || 0);
      const rz = THREE.MathUtils.degToRad(rotDeg[2] || 0);
      const rotMatrix = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz, 'XYZ'));
      const invRotMatrix = rotMatrix.clone().invert();

      if (!perSetupVerts[sId]) {
        perSetupVerts[sId] = { cut: [], rapid: [], plunge: [] };
      }

      tp.segments.forEach((seg) => {
        const s = seg.start;
        const e = seg.end;
        const dx = e[0] - s[0];
        const dy = e[1] - s[1];
        const dz = e[2] - s[2];
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const feed = seg.feed || (seg.motion_type === 'rapid' ? 5000 : 1000);
        const segSec = len > 0 && feed > 0 ? (len / feed) * 60.0 : 0.0;
        this.totalEstSeconds += segSec;

        this.animSegments.push({
          segment_id: seg.segment_id,
          setup_id: sId,
          setup_obj: setupObj,
          rot_deg: rotDeg,
          inv_rot_matrix: invRotMatrix,
          rot_matrix: rotMatrix,
          operation_id: seg.operation_id || tp.operation_id,
          feature_id: seg.feature_id || tp.feature_id,
          source_face_ids: seg.source_face_ids || tp.source_face_ids || [],
          source_edge_ids: seg.source_edge_ids || tp.source_edge_ids || [],
          geometry_mode: seg.geometry_mode || tp.geometry_mode,
          target_surface: seg.target_surface || tp.target_surface,
          start: new THREE.Vector3(s[0], s[1], s[2]),
          end: new THREE.Vector3(e[0], e[1], e[2]),
          length: len,
          type: seg.motion_type,
          feed: feed,
          spindle: seg.spindle || 6000,
          tool_id: seg.tool_id || tp.tool_id,
          durationSec: segSec,
        });
        this.totalAnimDistance += len;

        if (seg.motion_type === 'rapid' || seg.motion_type === 'link' || seg.motion_type === 'retract') {
          perSetupVerts[sId].rapid.push(s[0], s[1], s[2], e[0], e[1], e[2]);
        } else if (seg.motion_type === 'plunge' || seg.motion_type === 'ramp') {
          perSetupVerts[sId].plunge.push(s[0], s[1], s[2], e[0], e[1], e[2]);
        } else {
          perSetupVerts[sId].cut.push(s[0], s[1], s[2], e[0], e[1], e[2]);
        }
      });
    });

    // Build sub-groups per setup
    Object.keys(perSetupVerts).forEach((sId) => {
      const sGrp = new THREE.Group();
      sGrp.name = `toolpaths_${sId}`;
      const verts = perSetupVerts[sId];

      if (verts.cut.length > 0) {
        const cutGeom = new THREE.BufferGeometry();
        cutGeom.setAttribute('position', new THREE.Float32BufferAttribute(verts.cut, 3));
        const cutMat = new THREE.LineBasicMaterial({
          color: 0x00f0ff,
          linewidth: 2,
          transparent: true,
          opacity: 0.95,
          depthTest: false,
        });
        const lines = new THREE.LineSegments(cutGeom, cutMat);
        lines.renderOrder = 100;
        sGrp.add(lines);
      }

      if (verts.rapid.length > 0) {
        const rapidGeom = new THREE.BufferGeometry();
        rapidGeom.setAttribute('position', new THREE.Float32BufferAttribute(verts.rapid, 3));
        const rapidMat = new THREE.LineBasicMaterial({
          color: 0xf59e0b,
          transparent: true,
          opacity: 0.85,
          depthTest: false,
        });
        const lines = new THREE.LineSegments(rapidGeom, rapidMat);
        lines.renderOrder = 100;
        sGrp.add(lines);
      }

      if (verts.plunge.length > 0) {
        const plungeGeom = new THREE.BufferGeometry();
        plungeGeom.setAttribute('position', new THREE.Float32BufferAttribute(verts.plunge, 3));
        const plungeMat = new THREE.LineBasicMaterial({
          color: 0xeab308,
          linewidth: 2,
          transparent: true,
          opacity: 0.95,
          depthTest: false,
        });
        const lines = new THREE.LineSegments(plungeGeom, plungeMat);
        lines.renderOrder = 100;
        sGrp.add(lines);
      }

      this.setupToolpathGroups[sId] = sGrp;
      this.toolpathGroup.add(sGrp);
    });

    // In 'all' mode, initial setup is first segment's setup or first setup
    const initialSetupId = this.animSegments[0]?.setup_id || activeSetupId;
    this.showSetupToolpaths(activeSetupId === 'all' ? initialSetupId : activeSetupId);
    this.toolpathGroup.visible = (this.toolpathVisible !== undefined ? this.toolpathVisible : true);

    // Position tool at start of first segment
    if (this.animSegments.length > 0) {
      const p0 = this.animSegments[0].start;
      this.toolGroup.position.set(p0.x, p0.y, p0.z);
      this.toolGroup.visible = this.toolVisible;
      const firstToolId = this.animSegments[0].tool_id;
      if (firstToolId && this.toolCatalog[firstToolId]) {
        const tInfo = this.toolCatalog[firstToolId];
        this.buildSimTool(tInfo.diameter || 10, tInfo.flute_length || 35);
        this.currentSimToolId = firstToolId;
      }
    }
  }

  highlightFeature(feature) {
    if (!this.featureHighlightGroup) return;
    this.featureHighlightGroup.clear();
    if (!feature) return;

    let min = feature.bounds_min || (feature.bounding_box && feature.bounding_box.min) || (feature.bounds && feature.bounds.min);
    let max = feature.bounds_max || (feature.bounding_box && feature.bounding_box.max) || (feature.bounds && feature.bounds.max);

    if (!min || !max) {
      if (feature.center) {
        const r = (feature.diameter || 10) / 2;
        const d = feature.depth || 10;
        min = [feature.center[0] - r, feature.center[1] - r, feature.center[2] - d];
        max = [feature.center[0] + r, feature.center[1] + r, feature.center[2] + d];
      } else {
        return;
      }
    }

    const sx = Math.max(max[0] - min[0], 1.5);
    const sy = Math.max(max[1] - min[1], 1.5);
    const sz = Math.max(max[2] - min[2], 1.5);

    const boxGeom = new THREE.BoxGeometry(sx, sy, sz);
    const boxMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.25,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const boxMesh = new THREE.Mesh(boxGeom, boxMat);
    boxMesh.position.set(min[0] + sx / 2, min[1] + sy / 2, min[2] + sz / 2);

    const edgeGeom = new THREE.EdgesGeometry(boxGeom);
    const edgeLine = new THREE.LineSegments(
      edgeGeom,
      new THREE.LineBasicMaterial({ color: 0x00f0ff, linewidth: 2, transparent: true, opacity: 0.9 })
    );
    boxMesh.add(edgeLine);

    this.featureHighlightGroup.add(boxMesh);
  }

  clearFeatureHighlight() {
    if (this.featureHighlightGroup) {
      this.featureHighlightGroup.clear();
    }
  }

  setSimulationProgress(fraction) {
    if (this.totalAnimDistance <= 0 || this.animSegments.length === 0) return;
    this.currentAnimDistance = Math.max(0, Math.min(1, fraction)) * this.totalAnimDistance;
    this.updateToolPosition();
  }

  updateToolPosition() {
    if (!this.animSegments || this.animSegments.length === 0) return;
    let accumulatedDist = 0;
    let accumulatedSec = 0;

    for (let i = 0; i < this.animSegments.length; i++) {
      const seg = this.animSegments[i];
      if (accumulatedDist + seg.length >= this.currentAnimDistance || i === this.animSegments.length - 1) {
        const segDist = this.currentAnimDistance - accumulatedDist;
        const alpha = seg.length > 0 ? Math.min(1, Math.max(0, segDist / seg.length)) : 0;
        
        const pos = new THREE.Vector3().lerpVectors(seg.start, seg.end, alpha);
        this.toolGroup.position.copy(pos);

        // Dynamically switch active setup orientation in Master/All mode
        if (this.activeSetupId === 'all' && seg.setup_id && seg.setup_id !== this.currentSimSetupId) {
          this.currentSimSetupId = seg.setup_id;
          const sObj = seg.setup_obj || this.allSetupsMap?.[seg.setup_id];
          if (sObj) {
            const rot = sObj.rotation_deg || [0, 0, 0];
            const stockBounds = sObj.stock_bounds || null;
            this.setSetupOrientation(rot, stockBounds, sObj.stock || sObj.stock_cfg || this.clampingConfig);
          }
          this.showSetupToolpaths(seg.setup_id);
        }

        // Dynamically update tool model geometry if tool changed
        if (seg.tool_id && seg.tool_id !== this.currentSimToolId && this.toolCatalog[seg.tool_id]) {
          const tInfo = this.toolCatalog[seg.tool_id];
          this.buildSimTool(tInfo.diameter || 10, tInfo.flute_length || 35);
          this.currentSimToolId = seg.tool_id;
        }

        // Real-time stock carving update
        this.carveStockToDistance(this.currentAnimDistance);

        const currentElapsed = accumulatedSec + (alpha * (seg.durationSec || 0));
        const prog = this.totalAnimDistance > 0 ? this.currentAnimDistance / this.totalAnimDistance : 0;

        // Check for clamping / fixture breach
        let isClampBreach = false;
        const sObj = seg.setup_obj || this.allSetupsMap?.[seg.setup_id];
        const cfg = sObj?.stock || sObj?.stock_cfg || this.clampingConfig;
        const stkBounds = sObj?.stock_bounds || this.stockBounds;

        if (stkBounds && cfg && cfg.clamp_type !== 'none') {
          const clampH = Math.max(parseFloat(cfg.clamp_height !== undefined ? cfg.clamp_height : (cfg.margin_z_bottom || 3.0)) || 3.0, 0.5);
          const zClampTop = stkBounds.min[2] + clampH;
          if (pos.z <= zClampTop + 1e-4) {
            isClampBreach = true;
          }
        }

        const tInfo = seg.tool_id ? this.toolCatalog[seg.tool_id] : null;
        const toolDisplay = tInfo ? `${tInfo.id}: ${tInfo.name} (Ø${tInfo.diameter}mm)` : (seg.tool_id || '--');

        const info = {
          pos: { x: pos.x, y: pos.y, z: pos.z },
          spindle: seg.spindle || 0,
          feed: seg.feed || 0,
          tool: toolDisplay,
          tool_id: seg.tool_id || '--',
          tool_info: tInfo,
          setup_id: seg.setup_id || '--',
          setup_name: sObj?.name || (seg.setup_id === 'setup_002' ? 'Setup 2 (G55)' : 'Setup 1 (G54)'),
          work_offset: sObj?.work_offset || (seg.setup_id === 'setup_002' ? 'G55' : 'G54'),
          progress: prog,
          elapsedSec: currentElapsed,
          totalSec: this.totalEstSeconds || (this.totalAnimDistance / 1500) * 60,
          isClampBreach: isClampBreach,
        };

        if (this.onProgressUpdate) {
          this.onProgressUpdate(info);
        }
        if (this.onTelemetryUpdate) {
          this.onTelemetryUpdate(info);
        }
        break;
      }
      accumulatedDist += seg.length;
      accumulatedSec += (seg.durationSec || 0);
    }
  }

  setCameraView(viewType, smooth = true) {
    const target = this.controls.target;
    const dist = 180;
    const newPos = new THREE.Vector3();

    if (viewType === 'iso') {
      newPos.set(target.x + 120, target.y - 120, target.z + 100);
    } else if (viewType === 'top') {
      newPos.set(target.x, target.y, target.z + dist);
    } else if (viewType === 'front') {
      newPos.set(target.x, target.y - dist, target.z);
    } else if (viewType === 'right') {
      newPos.set(target.x + dist, target.y, target.z);
    } else if (viewType === 'left') {
      newPos.set(target.x - dist, target.y, target.z);
    } else if (viewType === 'back') {
      newPos.set(target.x, target.y + dist, target.z);
    } else if (viewType === 'bottom') {
      newPos.set(target.x, target.y, target.z - dist);
    } else if (viewType === 'reset') {
      newPos.set(target.x + 100, target.y - 150, target.z + 120);
    } else {
      newPos.set(target.x + 120, target.y - 120, target.z + 100);
    }

    if (smooth) {
      this.animateCameraTo(newPos, target, 350);
    } else {
      this.camera.position.copy(newPos);
      this.camera.lookAt(target);
      this.controls.update();
    }
  }

  animateCameraTo(targetPos, targetLookAt, durationMs = 350) {
    const startPos = this.camera.position.clone();
    const startTarget = this.controls.target.clone();
    const startTime = performance.now();

    const step = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      const ease = 1 - Math.pow(1 - progress, 3);

      this.camera.position.lerpVectors(startPos, targetPos, ease);
      this.controls.target.lerpVectors(startTarget, targetLookAt, ease);
      this.controls.update();

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    };
    requestAnimationFrame(step);
  }

  setStockOpacity(val) {
    const opacity = Math.max(0.05, Math.min(1.0, (parseFloat(val) || 60) / 100));
    if (this.stockMesh && this.stockMesh.material) {
      this.stockMesh.material.opacity = opacity;
      this.stockMesh.material.needsUpdate = true;
    }
  }

  updateViewCube() {
    const cubeEl = document.getElementById('viewcube');
    if (!cubeEl || !this.camera || !this.controls) return;

    const v = new THREE.Vector3().subVectors(this.camera.position, this.controls.target).normalize();
    const yawDeg = THREE.MathUtils.radToDeg(Math.atan2(v.x, -v.y));
    const pitchDeg = THREE.MathUtils.radToDeg(Math.asin(Math.max(-1, Math.min(1, v.z))));

    cubeEl.style.transform = `rotateX(${-pitchDeg}deg) rotateY(${-yawDeg}deg)`;
  }

  togglePart(visible) {
    this.partVisible = visible !== undefined ? visible : !this.partVisible;
    if (this.partMesh) this.partMesh.visible = this.partVisible;
    if (this.partEdges) this.partEdges.visible = this.partVisible;
  }

  toggleStock(visible) {
    this.stockVisible = visible !== undefined ? visible : !this.stockVisible;
    if (this.stockMesh) this.stockMesh.visible = this.stockVisible;
  }

  toggleFixture(visible) {
    this.fixtureVisible = visible !== undefined ? visible : !this.fixtureVisible;
    if (this.fixtureGroup) this.fixtureGroup.visible = this.fixtureVisible;
    if (this.clampZoneGroup) this.clampZoneGroup.visible = this.fixtureVisible;
  }

  toggleToolpath(visible) {
    this.toolpathVisible = visible !== undefined ? visible : !this.toolpathVisible;
    if (this.toolpathGroup) {
      this.toolpathGroup.visible = this.toolpathVisible;
    }
  }

  toggleTool(visible) {
    this.toolVisible = visible !== undefined ? visible : !this.toolVisible;
    if (this.toolGroup) {
      this.toolGroup.visible = this.toolVisible;
    }
  }

  toggleWireframe(enabled) {
    this.wireframeEnabled = enabled !== undefined ? enabled : !this.wireframeEnabled;
    if (this.partMesh) this.partMesh.material.wireframe = this.wireframeEnabled;
  }

  onWindowResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(this.animate);

    // Simulation playback step
    if (this.isPlaying && this.totalAnimDistance > 0) {
      this.currentAnimDistance += this.simSpeed * 1.5;
      if (this.currentAnimDistance >= this.totalAnimDistance) {
        this.currentAnimDistance = this.totalAnimDistance;
        this.isPlaying = false;
        if (this.onPlaybackEnd) this.onPlaybackEnd();
      }
      this.updateToolPosition();
    } else if (this.isPlaying) {
      console.log('animate: isPlaying but totalAnimDistance is', this.totalAnimDistance);
    }

    this.controls.update();
    this.updateViewCube();
    this.renderer.render(this.scene, this.camera);
  }
}
