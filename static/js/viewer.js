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
    this.partMesh = null;
    this.partEdges = null;
    this.stockMesh = null;
    this.wcsMarker = null;
    this.toolpathGroup = new THREE.Group();
    this.featureHighlightGroup = new THREE.Group();
    this.toolGroup = new THREE.Group();
    this.fixtureGroup = new THREE.Group();
    this.clampZoneGroup = new THREE.Group();
    this.grid = null;
    
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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(100, -100, 200);
    dirLight1.castShadow = true;
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.45);
    dirLight2.position.set(-150, 150, 100);
    this.scene.add(dirLight2);

    const hemLight = new THREE.HemisphereLight(0x1e293b, 0x0f172a, 0.5);
    this.scene.add(hemLight);
  }

  setupEnvironment() {
    // Ground Grid in XY Plane
    this.grid = new THREE.GridHelper(300, 30, 0x00f0ff, 0x1e293b);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.position.set(50, 30, -0.1);
    this.scene.add(this.grid);

    // Coordinate Axes Triad (X-Red, Y-Green, Z-Blue)
    const axes = new THREE.AxesHelper(30);
    axes.renderOrder = 1;
    this.scene.add(axes);
  }

  buildSimTool(diameter = 10, length = 35) {
    this.toolGroup.clear();

    // Shank & Flute (Cylinder)
    const geomFlute = new THREE.CylinderGeometry(diameter / 2, diameter / 2, length, 24);
    geomFlute.rotateX(Math.PI / 2);
    geomFlute.translate(0, 0, length / 2);
    const matFlute = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      metalness: 0.9,
      roughness: 0.2,
    });
    const meshFlute = new THREE.Mesh(geomFlute, matFlute);
    this.toolGroup.add(meshFlute);

    // Collet Holder
    const geomCollet = new THREE.CylinderGeometry(15, diameter / 2 + 2, 20, 24);
    geomCollet.rotateX(Math.PI / 2);
    geomCollet.translate(0, 0, length + 10);
    const matCollet = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.8,
      roughness: 0.3,
    });
    const meshCollet = new THREE.Mesh(geomCollet, matCollet);
    this.toolGroup.add(meshCollet);

    // Spindle Body
    const geomSpindle = new THREE.CylinderGeometry(25, 20, 40, 24);
    geomSpindle.rotateX(Math.PI / 2);
    geomSpindle.translate(0, 0, length + 40);
    const matSpindle = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.5,
      roughness: 0.4,
    });
    const meshSpindle = new THREE.Mesh(geomSpindle, matSpindle);
    this.toolGroup.add(meshSpindle);

    // Initial position at safe Z
    this.toolGroup.position.set(0, 0, 35);
    this.toolGroup.visible = true;
  }

  loadModelMesh(meshData) {
    if (this.partMesh) {
      this.scene.remove(this.partMesh);
      if (this.partEdges) this.scene.remove(this.partEdges);
    }

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
    this.scene.add(this.partMesh);

    // Outline Edges for CAD wireframe appearance
    const edgesGeom = new THREE.EdgesGeometry(geometry, 24);
    this.partEdges = new THREE.LineSegments(
      edgesGeom,
      new THREE.LineBasicMaterial({ color: 0x94a3b8, linewidth: 1 })
    );
    this.scene.add(this.partEdges);

    // Recenter camera target to part center
    const bbox = meshData.bounding_box;
    const cx = (bbox.min[0] + bbox.max[0]) / 2;
    const cy = (bbox.min[1] + bbox.max[1]) / 2;
    const cz = (bbox.min[2] + bbox.max[2]) / 2;

    this.controls.target.set(cx, cy, cz);
    this.grid.position.set(cx, cy, bbox.min[2] - 0.1);
  }

  updateStockBounds(min, max, stockConfig = null) {
    this.updateStock({ min: min, max: max }, stockConfig);
  }

  updateStock(stockBounds, stockConfig = null) {
    if (!stockBounds || !stockBounds.min || !stockBounds.max) return;
    this.stockBounds = stockBounds;
    if (stockConfig) this.clampingConfig = stockConfig;

    if (this.stockMesh) {
      this.scene.remove(this.stockMesh);
      this.stockMesh = null;
    }
    if (this.wcsMarker) {
      this.scene.remove(this.wcsMarker);
    }
    this.fixtureGroup.clear();
    this.clampZoneGroup.clear();

    const min = stockBounds.min;
    const max = stockBounds.max;
    const sx = Math.max(max[0] - min[0], 1);
    const sy = Math.max(max[1] - min[1], 1);
    const sz = Math.max(max[2] - min[2], 1);

    // 1. Dynamic Solid Stock Heightfield
    const NX = this.gridNX;
    const NY = this.gridNY;
    const numTopVerts = NX * NY;
    const numVerts = numTopVerts + NX * 2 + NY * 2 + 4; // top + front/back bot + left/right bot + bottom corners
    const positions = new Float32Array(numVerts * 3);
    const indices = [];

    const dx = sx / (NX - 1);
    const dy = sy / (NY - 1);

    this.initialStockHeights = new Float32Array(numTopVerts).fill(max[2]);
    this.stockHeights = new Float32Array(this.initialStockHeights);
    this.lastCarvedDistance = 0;

    // Fill Top Face Vertices [0 .. numTopVerts-1]
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const idx = j * NX + i;
        positions[idx * 3] = min[0] + i * dx;
        positions[idx * 3 + 1] = min[1] + j * dy;
        positions[idx * 3 + 2] = max[2];
      }
    }

    // Top Face Triangles
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

    // Side Wall Bottom Vertices
    let offset = numTopVerts;
    const frontBotStart = offset;
    for (let i = 0; i < NX; i++) {
      positions[(offset + i) * 3] = min[0] + i * dx;
      positions[(offset + i) * 3 + 1] = min[1];
      positions[(offset + i) * 3 + 2] = min[2];
    }
    offset += NX;

    const backBotStart = offset;
    for (let i = 0; i < NX; i++) {
      positions[(offset + i) * 3] = min[0] + i * dx;
      positions[(offset + i) * 3 + 1] = max[1];
      positions[(offset + i) * 3 + 2] = min[2];
    }
    offset += NX;

    const leftBotStart = offset;
    for (let j = 0; j < NY; j++) {
      positions[(offset + j) * 3] = min[0];
      positions[(offset + j) * 3 + 1] = min[1] + j * dy;
      positions[(offset + j) * 3 + 2] = min[2];
    }
    offset += NY;

    const rightBotStart = offset;
    for (let j = 0; j < NY; j++) {
      positions[(offset + j) * 3] = max[0];
      positions[(offset + j) * 3 + 1] = min[1] + j * dy;
      positions[(offset + j) * 3 + 2] = min[2];
    }
    offset += NY;

    // Bottom Corners & Face
    const b0 = offset;
    positions[b0 * 3] = min[0]; positions[b0 * 3 + 1] = min[1]; positions[b0 * 3 + 2] = min[2];
    const b1 = offset + 1;
    positions[b1 * 3] = max[0]; positions[b1 * 3 + 1] = min[1]; positions[b1 * 3 + 2] = min[2];
    const b2 = offset + 2;
    positions[b2 * 3] = max[0]; positions[b2 * 3 + 1] = max[1]; positions[b2 * 3 + 2] = min[2];
    const b3 = offset + 3;
    positions[b3 * 3] = min[0]; positions[b3 * 3 + 1] = max[1]; positions[b3 * 3 + 2] = min[2];
    indices.push(b0, b2, b1);
    indices.push(b0, b3, b2);

    // Front Wall (j = 0)
    for (let i = 0; i < NX - 1; i++) {
      const t0 = i;
      const t1 = i + 1;
      const b_0 = frontBotStart + i;
      const b_1 = frontBotStart + i + 1;
      indices.push(t0, b_0, b_1);
      indices.push(t0, b_1, t1);
    }

    // Back Wall (j = NY - 1)
    for (let i = 0; i < NX - 1; i++) {
      const t0 = (NY - 1) * NX + i;
      const t1 = (NY - 1) * NX + i + 1;
      const b_0 = backBotStart + i;
      const b_1 = backBotStart + i + 1;
      indices.push(t0, b_1, b_0);
      indices.push(t0, t1, b_1);
    }

    // Left Wall (i = 0)
    for (let j = 0; j < NY - 1; j++) {
      const t0 = j * NX;
      const t1 = (j + 1) * NX;
      const b_0 = leftBotStart + j;
      const b_1 = leftBotStart + j + 1;
      indices.push(t0, b_1, b_0);
      indices.push(t0, t1, b_1);
    }

    // Right Wall (i = NX - 1)
    for (let j = 0; j < NY - 1; j++) {
      const t0 = j * NX + (NX - 1);
      const t1 = (j + 1) * NX + (NX - 1);
      const b_0 = rightBotStart + j;
      const b_1 = rightBotStart + j + 1;
      indices.push(t0, b_0, b_1);
      indices.push(t0, b_1, t1);
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

    // Outer bounding wireframe outline for initial stock envelope
    const boxGeom = new THREE.BoxGeometry(sx, sy, sz);
    const edges = new THREE.EdgesGeometry(boxGeom);
    const boxWire = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.4 })
    );
    boxWire.position.set(min[0] + sx / 2, min[1] + sy / 2, min[2] + sz / 2);
    this.stockMesh.add(boxWire);
    this.scene.add(this.stockMesh);

    // 2. WCS Origin Marker at Top Face Center
    const wcsGeom = new THREE.SphereGeometry(2, 16, 16);
    const wcsMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
    this.wcsMarker = new THREE.Mesh(wcsGeom, wcsMat);
    this.wcsMarker.position.set(min[0] + sx / 2, min[1] + sy / 2, max[2]);
    this.scene.add(this.wcsMarker);

    // 3. Clamping / Fixture Geometry & Safety Clearance Plane
    const cfg = this.clampingConfig || {};
    const clampType = cfg.clamp_type || 'vise_jaws';
    const clampH = Math.max(parseFloat(cfg.clamp_height !== undefined ? cfg.clamp_height : (cfg.margin_z_bottom || 3.0)) || 3.0, 0.5);
    const clampW = Math.max(parseFloat(cfg.clamp_width || 14.0) || 14.0, 5.0);

    if (clampType !== 'none') {
      const zBot = min[2];
      const zClampTop = zBot + clampH;
      const jawLength = sx + 20.0;
      const jawHeight = clampH + 15.0; // Vise body extends downward

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
        // 4 Corner Step/Toe Clamps
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

      // Clamping Safe Clearance Boundary Plane (Hatched Danger Boundary)
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

      // Border outline for clamp floor
      const planeBorderGeom = new THREE.EdgesGeometry(new THREE.BoxGeometry(sx, sy, 0.1));
      const planeBorder = new THREE.LineSegments(
        planeBorderGeom,
        new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 2 })
      );
      planeBorder.position.set(min[0] + sx / 2, min[1] + sy / 2, zClampTop);
      this.clampZoneGroup.add(planeBorder);
    }
  }

carveStockToDistance(targetDistance) {
    if (!this.stockHeights || !this.stockBounds || !this.animSegments || this.animSegments.length === 0) {
      // console.log('carveStockToDistance: early return - missing data', { hasHeights: !!this.stockHeights, hasBounds: !!this.stockBounds, hasSegments: !!this.animSegments, segCount: this.animSegments?.length });
      return;
    }

    const min = this.stockBounds.min;
    const max = this.stockBounds.max;
    const sx = Math.max(max[0] - min[0], 1);
    const sy = Math.max(max[1] - min[1], 1);
    const NX = this.gridNX;
    const NY = this.gridNY;
    const dx = sx / (NX - 1);
    const dy = sy / (NY - 1);

    // Cutting motion types that remove material
    const cuttingTypes = new Set([
      'cut', 'plunge', 'ramp', 'helix',
      'arc_cw', 'arc_ccw', 'entry', 'exit'
    ]);

    // If targetDistance is less than what was previously carved (scrubbed back or reset), reset heights
    if (targetDistance < this.lastCarvedDistance - 1e-3) {
      this.stockHeights.set(this.initialStockHeights);
      this.lastCarvedDistance = 0;
    }

    let dirty = false;
    let accumulatedDist = 0;
    let cuttingSegmentsProcessed = 0;
    let totalVoxelsCarved = 0;

    for (let sIdx = 0; sIdx < this.animSegments.length; sIdx++) {
      const seg = this.animSegments[sIdx];
      const segStartDist = accumulatedDist;
      const segEndDist = accumulatedDist + seg.length;

      // Check if this segment overlaps the slice [lastCarvedDistance, targetDistance]
      if (segEndDist > this.lastCarvedDistance && segStartDist < targetDistance) {
        const isCutting = cuttingTypes.has(seg.type);

        // Debug: log segment types
        // if (sIdx < 10 || isCutting) {
        //   console.log(`Segment ${sIdx}: type=${seg.type}, isCutting=${isCutting}, len=${seg.length.toFixed(2)}, tool_id=${seg.tool_id}, z=[${seg.start.z.toFixed(2)}, ${seg.end.z.toFixed(2)}]`);
        // }

        if (isCutting) {
          cuttingSegmentsProcessed++;
          // Determine tool radius
          let toolRadius = 5.0;
          if (seg.tool_id && this.toolCatalog[seg.tool_id]) {
            const dia = this.toolCatalog[seg.tool_id].diameter;
            if (dia && dia > 0) toolRadius = dia / 2.0;
          }
          // Clamp tool radius to reasonable range
          toolRadius = Math.max(0.1, Math.min(toolRadius, 100.0));

          // Fractional segment progress
          const sliceStartFrac = seg.length > 0 ? Math.max(0, Math.min(1, (this.lastCarvedDistance - segStartDist) / seg.length)) : 0;
          const sliceEndFrac = seg.length > 0 ? Math.max(0, Math.min(1, (targetDistance - segStartDist) / seg.length)) : 1;

          const p0 = new THREE.Vector3().lerpVectors(seg.start, seg.end, sliceStartFrac);
          const p1 = new THREE.Vector3().lerpVectors(seg.start, seg.end, sliceEndFrac);

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

          let segVoxelsCarved = 0;
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
                // Use small epsilon to ensure carving at stock surface
                if (zTool < this.stockHeights[gridIdx] - 1e-6) {
                  this.stockHeights[gridIdx] = Math.max(min[2], zTool);
                  dirty = true;
                  segVoxelsCarved++;
                }
              }
            }
          }
          totalVoxelsCarved += segVoxelsCarved;
          // if (segVoxelsCarved > 0) {
          //   console.log(`  Carved ${segVoxelsCarved} voxels in segment ${sIdx} (type=${seg.type}, toolRadius=${toolRadius.toFixed(2)}, zTool range=[${p0.z.toFixed(2)}, ${p1.z.toFixed(2)}])`);
          // }
        }
      }

      accumulatedDist += seg.length;
      if (accumulatedDist >= targetDistance) break;
    }

    this.lastCarvedDistance = targetDistance;

    // Debug summary
    if (cuttingSegmentsProcessed > 0 || targetDistance < 10) {
      console.log(`carveStockToDistance(${targetDistance.toFixed(2)}): processed ${cuttingSegmentsProcessed} cutting segments, carved ${totalVoxelsCarved} voxels, dirty=${dirty}`);
    }
    // }

    if (dirty && this.stockGeom) {
      const posAttr = this.stockGeom.attributes.position;
      const positions = posAttr.array;
      for (let k = 0; k < NX * NY; k++) {
        positions[k * 3 + 2] = this.stockHeights[k];
      }
      posAttr.needsUpdate = true;
      this.stockGeom.computeVertexNormals();
    }
  }

  renderToolpaths(toolpathList) {
    this.toolpathGroup.clear();
    this.toolpaths = toolpathList;
    this.animSegments = [];
    this.totalAnimDistance = 0;
    this.currentAnimDistance = 0;
    this.lastCarvedDistance = 0;
    this.totalEstSeconds = 0;

    // Reset stock mesh heights if present
    if (this.stockHeights && this.initialStockHeights) {
      this.stockHeights.set(this.initialStockHeights);
      if (this.stockGeom) {
        const posAttr = this.stockGeom.attributes.position;
        const positions = posAttr.array;
        for (let k = 0; k < this.gridNX * this.gridNY; k++) {
          positions[k * 3 + 2] = this.stockHeights[k];
        }
        posAttr.needsUpdate = true;
        this.stockGeom.computeVertexNormals();
      }
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
      const cutMat = new THREE.LineBasicMaterial({ color: 0x00f0ff, linewidth: 2 });
      this.toolpathGroup.add(new THREE.LineSegments(cutGeom, cutMat));
    }

    // 2. Rapid Lines (Amber / Orange)
    if (rapidVerts.length > 0) {
      const rapidGeom = new THREE.BufferGeometry();
      rapidGeom.setAttribute('position', new THREE.Float32BufferAttribute(rapidVerts, 3));
      const rapidMat = new THREE.LineBasicMaterial({
        color: 0xf59e0b,
        transparent: true,
        opacity: 0.7,
      });
      this.toolpathGroup.add(new THREE.LineSegments(rapidGeom, rapidMat));
    }

    // 3. Plunge / Ramp Lines (Yellow)
    if (plungeVerts.length > 0) {
      const plungeGeom = new THREE.BufferGeometry();
      plungeGeom.setAttribute('position', new THREE.Float32BufferAttribute(plungeVerts, 3));
      const plungeMat = new THREE.LineBasicMaterial({ color: 0xeab308, linewidth: 2 });
      this.toolpathGroup.add(new THREE.LineSegments(plungeGeom, plungeMat));
    }

    // Position tool at start of first segment
    if (this.animSegments.length > 0) {
      const p0 = this.animSegments[0].start;
      this.toolGroup.position.set(p0.x, p0.y, p0.z);
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
