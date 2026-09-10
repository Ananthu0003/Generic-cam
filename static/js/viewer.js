/**
 * 3D WebGL CAM Viewport powered by Three.js
 */

class Cam3DViewer {
  constructor(containerId) {
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
    this.grid = null;
    
    // Tool simulation state
    this.toolpaths = [];
    this.animSegments = [];
    this.totalAnimDistance = 0;
    this.currentAnimDistance = 0;
    this.isPlaying = false;
    this.simSpeed = 2.0; // mm per frame multiplier
    this.onTelemetryUpdate = null;

    this.init();
  }

  init() {
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
    dirLight1.position.set(100, -100, 200);
    dirLight1.castShadow = true;
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
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

  updateStock(stockBounds) {
    if (this.stockMesh) {
      this.scene.remove(this.stockMesh);
    }
    if (this.wcsMarker) {
      this.scene.remove(this.wcsMarker);
    }

    const min = stockBounds.min;
    const max = stockBounds.max;
    const sx = max[0] - min[0];
    const sy = max[1] - min[1];
    const sz = max[2] - min[2];

    const geom = new THREE.BoxGeometry(sx, sy, sz);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.15,
      roughness: 0.5,
      metalness: 0.1,
      wireframe: false,
    });

    this.stockMesh = new THREE.Mesh(geom, mat);
    this.stockMesh.position.set(min[0] + sx / 2, min[1] + sy / 2, min[2] + sz / 2);

    // Add stock wireframe box outline
    const edges = new THREE.EdgesGeometry(geom);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.6 })
    );
    this.stockMesh.add(line);
    this.scene.add(this.stockMesh);

    // WCS Origin Marker at Top Face Center
    const wcsGeom = new THREE.SphereGeometry(2, 16, 16);
    const wcsMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
    this.wcsMarker = new THREE.Mesh(wcsGeom, wcsMat);
    this.wcsMarker.position.set(min[0] + sx / 2, min[1] + sy / 2, max[2]);
    this.scene.add(this.wcsMarker);
  }

  renderToolpaths(toolpathList) {
    this.toolpathGroup.clear();
    this.toolpaths = toolpathList;
    this.animSegments = [];
    this.totalAnimDistance = 0;
    this.currentAnimDistance = 0;

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

        this.animSegments.push({
          start: new THREE.Vector3(s[0], s[1], s[2]),
          end: new THREE.Vector3(e[0], e[1], e[2]),
          length: len,
          type: seg.motion_type,
          feed: seg.feed || 1000,
          spindle: seg.spindle || 6000,
          tool_id: seg.tool_id || tp.tool_id,
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
    let accumulated = 0;
    for (let i = 0; i < this.animSegments.length; i++) {
      const seg = this.animSegments[i];
      if (accumulated + seg.length >= this.currentAnimDistance || i === this.animSegments.length - 1) {
        const segDist = this.currentAnimDistance - accumulated;
        const alpha = seg.length > 0 ? Math.min(1, Math.max(0, segDist / seg.length)) : 0;
        
        const pos = new THREE.Vector3().lerpVectors(seg.start, seg.end, alpha);
        this.toolGroup.position.copy(pos);

        if (this.onTelemetryUpdate) {
          this.onTelemetryUpdate({
            x: pos.x,
            y: pos.y,
            z: pos.z,
            feed: seg.feed,
            spindle: seg.spindle,
            tool_id: seg.tool_id,
            progress: this.totalAnimDistance > 0 ? this.currentAnimDistance / this.totalAnimDistance : 0,
          });
        }
        break;
      }
      accumulated += seg.length;
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
      this.updateToolPosition();
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
