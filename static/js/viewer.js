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
    this.stockMesh = null;
    this.wcsMarker = null;
    this.toolpathGroup = new THREE.Group();
    this.featureHighlightGroup = new THREE.Group();
    this.toolGroup = new THREE.Group();
    this.toolVisible = true;
    this.fixtureGroup = new THREE.Group();
    this.clampZoneGroup = new THREE.Group();
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

    // Dynamic stock heightfield grid
    this.gridNX = 120;
    this.gridNY = 120;
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
    const gridSize = 300;
    const gridDivisions = 30;
    this.grid = new THREE.GridHelper(gridSize, gridDivisions, 0x00f0ff, 0x1e293b);
    this.grid.rotateX(Math.PI / 2); // Rotate to lie on XY plane (Z-up)
    this.grid.position.set(50, 30, 0);
    this.scene.add(this.grid);
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
    const rx = THREE.MathUtils.degToRad(this.currentSetupRotation[0] || 0);
    const ry = THREE.MathUtils.degToRad(this.currentSetupRotation[1] || 0);
    const rz = THREE.MathUtils.degToRad(this.currentSetupRotation[2] || 0);

    if (this.partGroup) this.partGroup.rotation.set(rx, ry, rz);
    if (this.toolpathGroup) this.toolpathGroup.rotation.set(rx, ry, rz);
    if (this.stockGroup) this.stockGroup.rotation.set(rx, ry, rz);
    if (this.fixturesGroup) this.fixturesGroup.rotation.set(rx, ry, rz);
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
    this.updateStock({ min: min, max: max }, stockConfig);
  }

  updateStock(stockBounds, stockConfig = null) {
    if (!stockBounds || !stockBounds.min || !stockBounds.max) return;
    this.rawStockBounds = stockBounds;
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

    const isCylinder = stockConfig && (
      stockConfig.stock_mode === 'cylinder' ||
      stockConfig.stock_mode === 'relative_cylinder' ||
      stockConfig.stock_mode === 'fixed_cylinder'
    );

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
      this.stockGeom = cylinderGeom;

      const stockMat = new THREE.MeshStandardMaterial({
        color: 0xf59e0b,
        roughness: 0.35,
        metalness: 0.25,
        transparent: true,
        opacity: 0.82,
        side: THREE.DoubleSide,
        depthWrite: true,
      });

      this.stockMesh = new THREE.Mesh(cylinderGeom, stockMat);
      this.stockMesh.position.set(min[0] + sx / 2, min[1] + sy / 2, min[2] + sz / 2);
      this.stockMesh.castShadow = true;
      this.stockMesh.receiveShadow = true;

      const edges = new THREE.EdgesGeometry(cylinderGeom, 30);
      const cylWire = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.6 })
      );
      this.stockMesh.add(cylWire);
      this.partGroup.add(this.stockMesh);
    } else {
      // 1. Dual-Surface Solid Stock Mesh for 3D In-Process Workpiece (IPW)
      const NX = this.gridNX;
      const NY = this.gridNY;
      const numVerts = NX * NY;
      const dx = sx / (NX - 1);
      const dy = sy / (NY - 1);

      this.initialTopHeights = new Float32Array(numVerts).fill(max[2]);
      this.initialBotHeights = new Float32Array(numVerts).fill(min[2]);
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
        opacity: 0.88,
        side: THREE.DoubleSide,
        depthWrite: true,
      });

      this.stockMesh = new THREE.Mesh(stockGeom, stockMat);
      this.stockMesh.castShadow = true;
      this.stockMesh.receiveShadow = true;

      // Bounding wireframe outline in part local coordinates
      const boxGeom = new THREE.BoxGeometry(sx, sy, sz);
      const edges = new THREE.EdgesGeometry(boxGeom);
      const boxWire = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.4 })
      );
      boxWire.position.set(min[0] + sx / 2, min[1] + sy / 2, min[2] + sz / 2);
      this.stockMesh.add(boxWire);

      // Add stockMesh directly to partGroup so it flips/rotates synchronously with the part!
      this.partGroup.add(this.stockMesh);
    }

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
  }

  updateFixtures(stockBounds, stockConfig = null) {
    if (!stockBounds || !stockBounds.min || !stockBounds.max) return;
    const min = stockBounds.min;
    const max = stockBounds.max;
    const sx = Math.max(max[0] - min[0], 1);
    const sy = Math.max(max[1] - min[1], 1);

    if (this.wcsMarker) {
      this.scene.remove(this.wcsMarker);
    }
    this.fixtureGroup.clear();
    this.clampZoneGroup.clear();

    // 2. WCS Origin Marker at Top Face Center of current setup
    const wcsGeom = new THREE.SphereGeometry(2, 16, 16);
    const wcsMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
    this.wcsMarker = new THREE.Mesh(wcsGeom, wcsMat);
    this.wcsMarker.position.set(min[0] + sx / 2, min[1] + sy / 2, max[2]);
    this.scene.add(this.wcsMarker);

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
  }

  carveSegmentIntoArrays(p0_setup, p1_setup, toolRadius, invRotMatrix, topHeights, botHeights) {
    if (!this.rawStockBounds || !topHeights || !botHeights) return;

    const min = this.rawStockBounds.min;
    const max = this.rawStockBounds.max;
    const sx = Math.max(max[0] - min[0], 1);
    const sy = Math.max(max[1] - min[1], 1);
    const NX = this.gridNX;
    const NY = this.gridNY;
    const dx = sx / (NX - 1);
    const dy = sy / (NY - 1);

    // Points are already in Part local frame (global coordinates)
    const p0 = p0_setup.clone();
    const p1 = p1_setup.clone();

    // Spindle vector in Part local frame: we still need to know which way the spindle is pointing!
    // Since the tool is coming from +Z in setup frame, we need to transform +Z to global frame
    const spindleSetup = new THREE.Vector3(0, 0, 1);
    const pZero = new THREE.Vector3(0, 0, 0).applyMatrix4(invRotMatrix);
    const spindlePart = spindleSetup.clone().applyMatrix4(invRotMatrix).sub(pZero).normalize();

    const isTopApproach = spindlePart.z > 0.4;
    const isBotApproach = spindlePart.z < -0.4;

    const minCutX = Math.min(p0.x, p1.x) - toolRadius;
    const maxCutX = Math.max(p0.x, p1.x) + toolRadius;
    const minCutY = Math.min(p0.y, p1.y) - toolRadius;
    const maxCutY = Math.max(p0.y, p1.y) + toolRadius;

    const iMin = Math.max(0, Math.floor((minCutX - min[0]) / dx));
    const iMax = Math.min(NX - 1, Math.ceil((maxCutX - min[0]) / dx));
    const jMin = Math.max(0, Math.floor((minCutY - min[1]) / dy));
    const jMax = Math.min(NY - 1, Math.ceil((maxCutY - min[1]) / dy));

    const segVecX = p1.x - p0.x;
    const segVecY = p1.y - p0.y;
    const segLenSq = segVecX * segVecX + segVecY * segVecY;
    const rSq = toolRadius * toolRadius;

    for (let j = jMin; j <= jMax; j++) {
      const gy = min[1] + j * dy;
      for (let i = iMin; i <= iMax; i++) {
        const gx = min[0] + i * dx;
        let t = 0;
        if (segLenSq > 1e-6) {
          t = Math.max(0, Math.min(1, ((gx - p0.x) * segVecX + (gy - p0.y) * segVecY) / segLenSq));
        }
        const nx = p0.x + t * segVecX;
        const ny = p0.y + t * segVecY;
        const dSq = (gx - nx) * (gx - nx) + (gy - ny) * (gy - ny);

        if (dSq <= rSq) {
          const zTool = p0.z + t * (p1.z - p0.z);
          const gridIdx = j * NX + i;

          if (isTopApproach) {
            // Cut into top surface
            if (zTool < topHeights[gridIdx]) {
              topHeights[gridIdx] = Math.max(min[2], zTool);
              if (topHeights[gridIdx] < botHeights[gridIdx]) {
                botHeights[gridIdx] = topHeights[gridIdx];
              }
            }
          } else if (isBotApproach) {
            // Cut into bottom surface (Flip 180°)
            if (zTool > botHeights[gridIdx]) {
              botHeights[gridIdx] = Math.min(max[2], zTool);
              if (botHeights[gridIdx] > topHeights[gridIdx]) {
                topHeights[gridIdx] = botHeights[gridIdx];
              }
            }
          } else {
            // Side approach
            if (zTool < topHeights[gridIdx] && zTool > botHeights[gridIdx]) {
              topHeights[gridIdx] = Math.max(min[2], zTool);
            }
          }
        }
      }
    }
  }

  computeAllSetupSnapshots(setups = [], allToolpaths = []) {
    if (!this.rawStockBounds || !this.initialTopHeights || !this.initialBotHeights) return;
    this.setupSnapshots = {};

    let currentTop = new Float32Array(this.initialTopHeights);
    let currentBot = new Float32Array(this.initialBotHeights);

    const cuttingTypes = new Set([
      'cut', 'plunge', 'ramp', 'helix',
      'arc_cw', 'arc_ccw', 'entry', 'exit'
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

      setupPaths.forEach((tp) => {
        let toolRadius = 5.0;
        if (tp.tool_id && this.toolCatalog[tp.tool_id]) {
          const dia = this.toolCatalog[tp.tool_id].diameter;
          if (dia && dia > 0) toolRadius = dia / 2.0;
        }

        tp.segments?.forEach((seg) => {
          if (cuttingTypes.has(seg.motion_type)) {
            const p0 = new THREE.Vector3(seg.start[0], seg.start[1], seg.start[2]);
            const p1 = new THREE.Vector3(seg.end[0], seg.end[1], seg.end[2]);
            let r = toolRadius;
            if (seg.tool_id && this.toolCatalog[seg.tool_id]) {
              const dia = this.toolCatalog[seg.tool_id].diameter;
              if (dia && dia > 0) r = dia / 2.0;
            }
            this.carveSegmentIntoArrays(p0, p1, r, invRotMatrix, currentTop, currentBot);
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
      'arc_cw', 'arc_ccw', 'entry', 'exit'
    ]);

    // If scrubbed backwards, reset to active setup base stock state
    if (targetDistance < this.lastCarvedDistance - 1e-3) {
      if (this.activeSetupId && this.activeSetupId !== 'all' && this.setupSnapshots[this.activeSetupId]) {
        this.currentTopHeights.set(this.setupSnapshots[this.activeSetupId].startTop);
        this.currentBotHeights.set(this.setupSnapshots[this.activeSetupId].startBot);
      } else {
        this.currentTopHeights.set(this.initialTopHeights);
        this.currentBotHeights.set(this.initialBotHeights);
      }
      this.lastCarvedDistance = 0;
    }

    const rx = THREE.MathUtils.degToRad(this.currentSetupRotation[0] || 0);
    const ry = THREE.MathUtils.degToRad(this.currentSetupRotation[1] || 0);
    const rz = THREE.MathUtils.degToRad(this.currentSetupRotation[2] || 0);
    const rotMatrix = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rx, ry, rz, 'XYZ'));
    const invRotMatrix = rotMatrix.clone().invert();

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

          this.carveSegmentIntoArrays(p0, p1, toolRadius, invRotMatrix, this.currentTopHeights, this.currentBotHeights);
        }
      }

      accumulatedDist += seg.length;
      if (accumulatedDist >= targetDistance) break;
    }

    this.lastCarvedDistance = targetDistance;
    this.updateStockMeshGeometry();
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

    const cutVerts = [];
    const rapidVerts = [];
    const plungeVerts = [];

    toolpathList.forEach((tp) => {
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
          rapidVerts.push(s[0], s[1], s[2], e[0], e[1], e[2]);
        } else if (seg.motion_type === 'plunge' || seg.motion_type === 'ramp') {
          plungeVerts.push(s[0], s[1], s[2], e[0], e[1], e[2]);
        } else {
          cutVerts.push(s[0], s[1], s[2], e[0], e[1], e[2]);
        }
      });
    });

    // Debug: log segment type distribution
    const typeCounts = {};
    this.animSegments.forEach(s => { typeCounts[s.type] = (typeCounts[s.type] || 0) + 1; });
    console.log('renderToolpaths: segment types:', typeCounts);
    console.log('renderToolpaths: total segments:', this.animSegments.length, 'total distance:', this.totalAnimDistance.toFixed(2));
    console.log('renderToolpaths: stock bounds:', this.stockBounds);
    console.log('renderToolpaths: tool catalog keys:', Object.keys(this.toolCatalog));

    // 1. Cutting Lines (Cyan)
    if (cutVerts.length > 0) {
      const cutGeom = new THREE.BufferGeometry();
      cutGeom.setAttribute('position', new THREE.Float32BufferAttribute(cutVerts, 3));
      const cutMat = new THREE.LineBasicMaterial({
        color: 0x00f0ff,
        linewidth: 2,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      });
      const lines = new THREE.LineSegments(cutGeom, cutMat);
      lines.renderOrder = 100;
      this.toolpathGroup.add(lines);
    }

    // 2. Rapid Lines (Amber / Orange)
    if (rapidVerts.length > 0) {
      const rapidGeom = new THREE.BufferGeometry();
      rapidGeom.setAttribute('position', new THREE.Float32BufferAttribute(rapidVerts, 3));
      const rapidMat = new THREE.LineBasicMaterial({
        color: 0xf59e0b,
        transparent: true,
        opacity: 0.85,
        depthTest: false,
      });
      const lines = new THREE.LineSegments(rapidGeom, rapidMat);
      lines.renderOrder = 100;
      this.toolpathGroup.add(lines);
    }

    // 3. Plunge / Ramp Lines (Yellow)
    if (plungeVerts.length > 0) {
      const plungeGeom = new THREE.BufferGeometry();
      plungeGeom.setAttribute('position', new THREE.Float32BufferAttribute(plungeVerts, 3));
      const plungeMat = new THREE.LineBasicMaterial({
        color: 0xeab308,
        linewidth: 2,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      });
      const lines = new THREE.LineSegments(plungeGeom, plungeMat);
      lines.renderOrder = 100;
      this.toolpathGroup.add(lines);
    }

    this.toolpathGroup.visible = true;

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
    this.featureHighlightGroup.clear();
    if (!feature) return;

    const min = feature.bounds_min;
    const max = feature.bounds_max;
    const sx = Math.max(max[0] - min[0], 2);
    const sy = Math.max(max[1] - min[1], 2);
    const sz = Math.max(max[2] - min[2], 2);

    const boxGeom = new THREE.BoxGeometry(sx, sy, sz);
    const boxMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      wireframe: true,
      transparent: true,
      opacity: 0.8,
    });

    const highlightBox = new THREE.Mesh(boxGeom, boxMat);
    highlightBox.position.set(min[0] + sx / 2, min[1] + sy / 2, min[2] + sz / 2);
    this.featureHighlightGroup.add(highlightBox);
  }

  setSimulationProgress(fraction) {
    if (this.totalAnimDistance <= 0 || this.animSegments.length === 0) return;
    this.currentAnimDistance = Math.max(0, Math.min(1, fraction)) * this.totalAnimDistance;
    this.updateToolPosition();
  }

  updateToolPosition() {
    if (!this.animSegments || this.animSegments.length === 0) return;
    console.log('updateToolPosition: currentAnimDistance=', this.currentAnimDistance.toFixed(2), 'total=', this.totalAnimDistance.toFixed(2));
    let accumulatedDist = 0;
    let accumulatedSec = 0;

    for (let i = 0; i < this.animSegments.length; i++) {
      const seg = this.animSegments[i];
      if (accumulatedDist + seg.length >= this.currentAnimDistance || i === this.animSegments.length - 1) {
        const segDist = this.currentAnimDistance - accumulatedDist;
        const alpha = seg.length > 0 ? Math.min(1, Math.max(0, segDist / seg.length)) : 0;
        
        const pos = new THREE.Vector3().lerpVectors(seg.start, seg.end, alpha);
        this.toolGroup.position.copy(pos);

        // Dynamically update tool model geometry if tool changed
        if (seg.tool_id && seg.tool_id !== this.currentSimToolId && this.toolCatalog[seg.tool_id]) {
          const tInfo = this.toolCatalog[seg.tool_id];
          this.buildSimTool(tInfo.diameter || 10, tInfo.flute_length || 35);
          this.currentSimToolId = seg.tool_id;
        }

        // Real-time stock carving update
        console.log('updateToolPosition: calling carveStockToDistance with', this.currentAnimDistance.toFixed(2), 'segment type:', seg.type, 'tool:', seg.tool_id);
        this.carveStockToDistance(this.currentAnimDistance);

        const currentElapsed = accumulatedSec + (alpha * (seg.durationSec || 0));
        const prog = this.totalAnimDistance > 0 ? this.currentAnimDistance / this.totalAnimDistance : 0;

        // Check for clamping / fixture breach
        let isClampBreach = false;
        if (this.stockBounds && this.clampingConfig && this.clampingConfig.clamp_type !== 'none') {
          const cfg = this.clampingConfig;
          const clampH = Math.max(parseFloat(cfg.clamp_height !== undefined ? cfg.clamp_height : (cfg.margin_z_bottom || 3.0)) || 3.0, 0.5);
          const zClampTop = this.stockBounds.min[2] + clampH;
          if (pos.z <= zClampTop + 1e-4) {
            isClampBreach = true;
          }
        }

        const info = {
          pos: { x: pos.x, y: pos.y, z: pos.z },
          spindle: seg.spindle || 0,
          feed: seg.feed || 0,
          tool: seg.tool_id || '--',
          progress: prog,
          elapsedSec: currentElapsed,
          totalSec: this.totalEstSeconds || (this.totalAnimDistance / 1500) * 60,
          isClampBreach: isClampBreach,
        };

        if (this.onProgressUpdate) {
          this.onProgressUpdate(info);
        }
        if (this.onTelemetryUpdate) {
          this.onTelemetryUpdate({
            x: pos.x,
            y: pos.y,
            z: pos.z,
            feed: info.feed,
            spindle: info.spindle,
            tool_id: info.tool,
            progress: prog,
            elapsedSec: currentElapsed,
            totalSec: info.totalSec,
            isClampBreach: isClampBreach,
          });
        }
        break;
      }
      accumulatedDist += seg.length;
      accumulatedSec += (seg.durationSec || 0);
    }
  }

  setCameraView(viewType) {
    const target = this.controls.target;
    const dist = 180;
    if (viewType === 'iso') {
      this.camera.position.set(target.x + 120, target.y - 120, target.z + 100);
    } else if (viewType === 'top') {
      this.camera.position.set(target.x, target.y, target.z + dist);
    } else if (viewType === 'front') {
      this.camera.position.set(target.x, target.y - dist, target.z);
    } else if (viewType === 'reset') {
      this.camera.position.set(target.x + 100, target.y - 150, target.z + 120);
    }
    this.camera.lookAt(target);
    this.controls.update();
  }

  togglePart(visible) {
    if (this.partMesh) this.partMesh.visible = visible;
    if (this.partEdges) this.partEdges.visible = visible;
  }

  toggleStock(visible) {
    if (this.stockMesh) this.stockMesh.visible = visible;
  }

  toggleFixture(visible) {
    if (this.fixtureGroup) this.fixtureGroup.visible = visible;
    if (this.clampZoneGroup) this.clampZoneGroup.visible = visible;
  }

  toggleToolpath(visible) {
    this.toolpathGroup.visible = visible;
  }

  toggleTool(visible) {
    this.toolVisible = visible !== undefined ? visible : !this.toolVisible;
    if (this.toolGroup) {
      this.toolGroup.visible = this.toolVisible;
    }
  }

  toggleWireframe(enabled) {
    if (this.partMesh) this.partMesh.material.wireframe = enabled;
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
        this.currentAnimDistance = 0; // loop or stop
      }
      console.log('animate: playing, currentAnimDistance=', this.currentAnimDistance.toFixed(2), 'simSpeed=', this.simSpeed);
      this.updateToolPosition();
    } else if (this.isPlaying) {
      console.log('animate: isPlaying but totalAnimDistance is', this.totalAnimDistance);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
