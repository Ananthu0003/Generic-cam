# Generic-CAM Engineering Audit

## 1. Executive Summary

Generic-CAM is a geometry-driven CAM (Computer-Aided Manufacturing) system that imports STEP/STL CAD files, performs B-Rep topology-based feature recognition, plans machining operations with capability-driven tool selection, generates engagement-aware toolpaths, and post-processes to controller-specific G-code (GRBL, Fanuc, Haas, LinuxCNC).

**Current Maturity:** Early prototype (v0.1.0). The core pipeline works end-to-end for 3-axis milling of prismatic parts. Feature recognition, operation planning, and post-processing are functional. The system is well-designed architecturally with clear separation of concerns, explicit error handling, and a documented data-flow hierarchy.

**Major Strengths:**
- Clean, layered architecture with explicit data flow (no circular dependencies)
- Real B-Rep topology extraction from OpenCascade (not synthetic/mesh-only)
- Relational feature recognition (counterbores recognized as parent/child pairs)
- Explicit coordinate system chain with validation (MODEL -> SETUP -> WORK -> MACHINE)
- Comprehensive error taxonomy with provenance tracking
- Material-driven cutting parameters (not hardcoded constants)
- Multiple controller post-processors with modal state tracking
- Canned cycle support (G81-G86, G73, G76) with parse-back verification

**Major Weaknesses:**
- Global mutable session state in the API (race conditions, no multi-user support)
- API upload endpoint doesn't clean up temp files reliably on all error paths
- CORS allows all origins (security concern for production)
- No input size limits on file uploads
- No authentication or authorization
- Missing tests for geometry layer, machining calculations, coordinate transforms
- Demo model features are hardcoded fallbacks (not generated from real geometry)
- Some toolpath strategies use `object()` as placeholder for VoxelStock
- `pyproject.toml` missing `pydantic` dependency

**Overall Risk:** Medium. The core CAM logic is sound but the system lacks production hardening, multi-user support, comprehensive test coverage, and several safety checks needed for manufacturing use.

---

## 2. Repository Architecture

```
Generic-CAM/
├── app.py                          # FastAPI application entry point
├── cam_engine/
│   ├── __init__.py                 # Package docstring + version
│   ├── __main__.py                 # CLI entry point
│   ├── pipeline.py                 # End-to-end CAM pipeline orchestrator
│   ├── context.py                  # Planning context, Machine, Tool, Material, Stock, Setup
│   ├── coords.py                   # Coordinate system chain + transforms
│   ├── units.py                    # Unit conversion (mm <-> inch)
│   ├── errors.py                   # Error taxonomy + CamError + PlanningBlocker
│   ├── features.py                 # B-Rep feature recognizer (cylinders, pockets, slots, etc.)
│   ├── machinability.py            # Cutting parameters + tool selection/ranking
│   ├── operations.py               # Operation planner (facing, roughing, finishing, drilling, etc.)
│   ├── setup_sheet.py              # Shop floor documentation generator
│   ├── demo_models.py              # Demo CAD model + default tools/machine
│   ├── geometry/
│   │   ├── __init__.py             # Geometry layer exports
│   │   ├── topology.py             # B-Rep extraction (Shape, Face, Edge, SurfaceData)
│   │   ├── step_import.py          # STEP file import via OCP
│   │   ├── stl_import.py           # STL file import (ASCII + binary)
│   │   ├── mesh.py                 # TriangleMesh (ray casting, point-in-triangle, height queries)
│   │   └── voxel.py                # VoxelStock for material-removal simulation
│   ├── toolpaths/
│   │   ├── __init__.py             # Toolpath package
│   │   ├── semantic.py             # Controller-independent toolpath representation
│   │   ├── strategies.py           # Toolpath strategy implementations (~1784 lines)
│   │   └── validation.py           # Toolpath validation + gouge/collision detection
│   ├── post/
│   │   ├── __init__.py             # Post-processor exports
│   │   ├── post_processor.py       # G-code generation (GRBL, Fanuc, Haas, LinuxCNC)
│   │   └── gcode_parser.py         # G-code parse-back verification
│   └── api/
│       ├── __init__.py             # API router export
│       ├── models.py               # Pydantic request/response schemas
│       └── routes.py               # FastAPI REST endpoints
├── tests/
│   ├── test_cam_web_pipeline.py    # API + post-processor tests
│   ├── test_generic_cam_geometry.py # Geometry + feature recognition tests (requires OCP)
│   ├── test_hole_chamfer_strategies.py # Strategy tests (spot drill, ream, tap, chamfer)
│   └── test_canned_cycles.py       # Canned cycle + coolant tests
├── static/                         # Frontend HTML/CSS/JS
├── scratch/                        # Dev scratch files
├── sample_bracket.step             # Sample CAD model
├── sample_bracket_*.nc             # Generated G-code samples
└── pyproject.toml                  # Project metadata + dependencies
```

### Module Responsibilities

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `pipeline.py` | Orchestrates the full CAM flow | All modules |
| `context.py` | Central data model (Machine, Tool, Material, Stock, Setup) | coords, geometry, units |
| `coords.py` | Coordinate transforms between spaces | numpy, errors |
| `features.py` | B-Rep feature recognition | geometry, context |
| `machinability.py` | Cutting parameters + tool selection | context, features |
| `operations.py` | Operation planning from features | context, features, machinability |
| `toolpaths/strategies.py` | Toolpath generation strategies | context, operations, semantic |
| `toolpaths/semantic.py` | Controller-independent toolpath data | coords |
| `toolpaths/validation.py` | Toolpath safety checks | context, geometry, semantic |
| `post/post_processor.py` | G-code generation | semantic |
| `api/routes.py` | REST API endpoints | All modules |

---

## 3. Data Flow

```
Input: .step / .stp / .stl file
  │
  ▼
STEP Import (OCP) ──> ImportedModel (Shape + units + bounds)
  │
  ▼
Setup Transform (model_to_setup matrix)
  │
  ▼
PlanningContext assembly (Machine + Tools + Material + Stock + Setup)
  │
  ├──────────────────────────────┐
  ▼                              ▼
Feature Recognition         Demo Features (hardcoded fallback)
  │                              │
  ▼                              ▼
MachiningFeature[] ──────────────┘
  │
  ▼
Operation Planning (feature-by-feature, ordered: facing > roughing > rest > semi > finish > drill > bore > ream > tap > chamfer)
  │
  ▼
PlannedOperation[] (purpose + tool + CuttingParameters)
  │
  ▼
Toolpath Generation (engagement-aware strategies per operation type)
  │
  ▼
Toolpath Validation (axis limits, gouge, collision, reach)
  │
  ▼
Post-Processing (controller-specific G-code with modal state)
  │
  ▼
Output: .nc / .gcode file + PostResult statistics
```

---

## 4. Module-by-Module Analysis

### 4.1 `pipeline.py` — CAM Pipeline Orchestrator

**Purpose:** End-to-end orchestration from STEP file to G-code.

**Strengths:**
- Clean sequential flow with clear stage numbering
- Error collection (doesn't abort on first toolpath error)
- Validates stock, setup, and context

**Weaknesses:**
- Hardcoded finish allowances (`0.3mm` wall/floor at lines 137-138) — should be configurable
- No multi-setup support in the pipeline loop (only processes `ctx.setups` but no setup ordering logic)
- The pipeline doesn't call `ctx.validate()` — validation is implicit

**Risks:** MEDIUM — Missing validation call could allow invalid contexts through.

### 4.2 `context.py` — Planning Context

**Purpose:** Single source of truth for all machining state.

**Strengths:**
- Comprehensive validation methods on Machine, Tool, Stock
- Explicit material database with domain-appropriate cutting data
- Coordinate system chain validation

**Weaknesses:**
- `PlanningContext` is not truly immutable (list fields are mutable)
- `DEFAULT_MATERIALS` is a module-level mutable dict — could be accidentally modified
- `Tool.validate()` checks `tip_radius` for drills but drill tip geometry is actually point-angle based (the check is technically correct for the simplified model)

**Risks:** LOW

### 4.3 `coords.py` — Coordinate System

**Purpose:** Explicit coordinate transforms between MODEL, SETUP, WORK, MACHINE spaces.

**Strengths:**
- Proper rigid transform validation (orthonormal rotation, det=1, correct row)
- Inverse computed analytically (not numerically)
- Full chain composition with round-trip verification

**Weaknesses:**
- `TOOL` space is defined in the enum but never used in the chain
- No rotation-only or translation-only convenience constructors

**Risks:** LOW

### 4.4 `features.py` — Feature Recognition

**Purpose:** Extract machining features from B-Rep topology.

**Strengths:**
- Topology-driven (uses real face adjacency, surface types, oriented normals)
- Relational features (counterbores recognized as coaxial cylinder clusters)
- Boss vs hole distinction via radial normal analysis
- Evidence tracking on every recognized feature

**Weaknesses:**
- `_accessibility_of()` creates a new `TriangleMesh` from faces on every call (O(n) repeated work)
- Fillet recognition heuristic (`radius < 10.0 and area < 200.0`) is fragile
- No recognition of threads (tapping relies on `notes["tapped"]` which is never set by the recognizer)
- Through-hole detection uses `abs(bottom_z) < 0.5` which assumes Z=0 is the model bottom — fragile

**Risks:** HIGH — Through-hole detection logic is a likely bug for parts not positioned at Z=0.

### 4.5 `machinability.py` — Cutting Parameters

**Purpose:** Derive cutting parameters from material + tool + machine + operation.

**Strengths:**
- Material-driven (not hardcoded feeds/speeds)
- Proper power/torque estimation from MRR
- Tool ranking with explicit exclusion reasons
- Engagement angle calculation from stepover ratio

**Weaknesses:**
- Feed-per-tooth formula (`0.012 * d * (100/HB)^0.35`) is a reasonable approximation but should be validated against machinability handbooks
- `rank_tools()` score thresholds (40.0 cutoff, -60 for incompatibility) are arbitrary

**Risks:** MEDIUM — FORMULA NEEDS DOMAIN VALIDATION. The cutting parameter formulas are reasonable engineering approximations but should be validated against a machinability data source.

### 4.6 `operations.py` — Operation Planning

**Purpose:** Plan ordered machining operations from features.

**Strengths:**
- Operations only created when geometry demands them
- Dependency tracking between operations (rest_machining depends on roughing)
- Multiple operation types: facing, roughing, rest_machining, semi_finishing, finishing, drilling, boring, reaming, tapping, chamfering, spot_drilling

**Weaknesses:**
- `ToolType_mill()` is a module-level function that imports `ToolType` — should be a constant or method
- No operation for CONTOUR feature type in roughing (only in finishing)
- `_plan_semi_finishing` condition at line 181 (`wall_stock > 1.5 * min(t.diameter ...)`) is unclear
- `_plan_tapping` requires `notes["tapped"]` but the feature recognizer never sets this

**Risks:** MEDIUM — Tapping operations will never be planned because the recognizer never sets `notes["tapped"]`.

### 4.7 `toolpaths/strategies.py` — Toolpath Strategies

**Purpose:** Generate semantic toolpaths for each operation type.

**Strengths:**
- Engagement-aware roughing (contour-parallel with polygon offsets)
- Adaptive/trochoidal roughing option
- Multiple finishing strategies (parallel, waterline, scallop, pencil, radial)
- Helical entry for pocket roughing
- Arc lead-in/lead-out for finishing
- Drop-cutter Z for freeform surfaces

**Weaknesses:**
- `_offset_polygon()` (line 277) uses a simplified bisector approach — can fail for concave polygons or sharp corners
- `_feature_boundary_polygon()` always returns a bounding-box rectangle (never uses the actual boundary loop from the feature)
- `_finish_wall_floor` creates rectangular contours (not true boundary-following)
- Rest machining relies on voxel heightmap which is `object()` placeholder in some paths
- Nearest-neighbor sort for pencil trace is O(n²) — fine for small point sets but could be slow for dense curvature maps

**Risks:** HIGH — Polygon offset and boundary extraction are simplified; will produce incorrect toolpaths for non-rectangular pockets.

### 4.8 `toolpaths/semantic.py` — Toolpath Representation

**Purpose:** Controller-independent machining intent representation.

**Strengths:**
- Clean enum-based motion types
- Full provenance (operation, feature, tool on every segment)
- Arc length calculation handles full circles correctly

**Weaknesses:** None significant.

**Risks:** LOW

### 4.9 `toolpaths/validation.py` — Toolpath Validation

**Purpose:** Safety checks on generated toolpaths.

**Strengths:**
- Machine axis travel validation
- Gouge detection via mesh height queries
- Holder collision checking
- Flute reach validation

**Weaknesses:**
- `_gouge_depth()` samples 8 radial directions × 3 fractions = 24 points — coarse
- `_check_holder_collision()` only checks start/end points of segment, not intermediate
- No feedrate validation against material recommendations
- No spindle speed range validation

**Risks:** MEDIUM

### 4.10 `post/post_processor.py` — G-Code Generation

**Purpose:** Translate semantic toolpaths to controller-specific G-code.

**Strengths:**
- Modal state tracking (suppresses redundant G0/G1 mode words)
- Canned cycle support (G81-G86, G73, G76) with proper G80 cancellation
- Arc fitting to reduce G1 point count
- Controller-specific headers/footers (GRBL, Fanuc, Haas, LinuxCNC)
- Proper M29 rigid tapping support for Fanuc

**Weaknesses:**
- GRBL post uses `M0` (program pause) for tool change — correct for manual tool change but should be configurable
- No arc radius validation (could emit arcs with I/J that don't match endpoint)
- Time estimation uses constant rapid feed (3000 mm/min) — should use machine rapid_rate

**Risks:** MEDIUM

### 4.11 `api/routes.py` — REST API

**Purpose:** Web API for the CAM system.

**Strengths:**
- Clean RESTful endpoints for each pipeline stage
- Pydantic request/response validation
- Error mapping from CamError to HTTP 400

**Weaknesses:**
- Global mutable `session` object (line 92) — not thread-safe, no multi-user support
- CORS allows all origins (`allow_origins=["*"]`)
- No file size limits on upload
- Temp file cleanup in `upload_model_file` uses `try/finally` but if the `with` block fails before `tmp_path` is set, it could leak
- `_make_dummy_imported()` creates a `Shape.__new__(Shape)` — bypasses constructor, no `occ_shape` attribute
- `recognize_features()` for demo model returns hardcoded features, not computed from the demo mesh
- `generate_toolpaths()` catches `CamError` but doesn't catch general exceptions
- No rate limiting
- No authentication

**Risks:** HIGH — Security and concurrency issues.

### 4.12 `geometry/topology.py` — B-Rep Extraction

**Purpose:** Extract face/edge/surface data from OpenCascade shapes.

**Strengths:**
- Proper OCC surface type detection (Plane, Cylinder, Cone, Sphere, Torus, BSpline)
- Oriented normal evaluation with face orientation (REVERSED check)
- Wire loop extraction for boundary curves
- Face adjacency via shared edges (real topology, not proximity)

**Weaknesses:**
- Tessellation quality depends on `linear_deflection=0.2` and `angular_deflection=0.25` — hardcoded
- Edge discretization uses linear sampling (no adaptive refinement for small features)
- `is_internal` detection for cylinders is done twice (once in `_extract_face` via OCC, once in `_recognize_cylindrical_features` via normal analysis)

**Risks:** LOW

### 4.13 `geometry/mesh.py` — Triangle Mesh

**Purpose:** Mesh queries for ray casting, point classification, height fields.

**Strengths:**
- Grid-accelerated ray casting (Moller-Trumbore with spatial hash)
- Efficient `is_point_inside()` using vectorized parity test
- `heights_above()` for vertical line intersection

**Weaknesses:**
- Grid cell size is hardcoded at 5.0mm — may be too coarse for small features
- `ray_first_hit()` grid march can infinite-loop if `t_next <= t + 1e-12` — there's a safeguard (`t += step`) but `step` is `max(1e-9, cell)` which could be large
- No BVH or k-d tree acceleration

**Risks:** LOW — The grid approach works but isn't optimal for dense meshes.

### 4.14 `geometry/voxel.py` — Voxel Stock

**Purpose:** Voxel representation for material-removal simulation.

**Strengths:**
- Clean from_bounds construction
- Point-based mesh voxelization
- Heightmap generation for rest machining

**Weaknesses:**
- `remove_capsule()` is O(n_steps × n_voxels_in_sphere) — slow for long moves
- `from_mesh()` calls `is_point_inside()` sequentially if `is_point_inside_batch` not available — very slow for large meshes

**Risks:** LOW — Performance concern only.

### 4.15 `setup_sheet.py` — Shop Floor Documentation

**Purpose:** Generate human-readable setup sheets.

**Strengths:**
- Comprehensive tool list and operation summary
- Estimated cycle time from toolpath lengths

**Weaknesses:**
- Hardcoded rapid feed of 5000 mm/min in time estimation (line 187) — should use machine config
- No export to PDF or common formats

**Risks:** LOW

### 4.16 `demo_models.py` — Demo Model

**Purpose:** Procedural test geometry for UI demonstration.

**Strengths:**
- Creates a realistic bracket with pocket, hole, step, and facing features
- Comprehensive default tool catalog (12 tools)

**Weaknesses:**
- Hole cylinder face normals are inconsistent (some inward, some mixed)
- Pocket wall faces metadata at line 121 has `"normal": [0, 0, 0]` — zero normal

**Risks:** LOW

---

## 5. Critical Findings

| Severity | Issue | Location | Root Cause | Recommendation |
|----------|-------|----------|------------|----------------|
| HIGH | Through-hole detection uses `abs(bottom_z) < 0.5` as proxy for "reaches model bottom" | `features.py:273` | Fragile heuristic; fails for parts not at Z=0 or with stock margins | Compare against actual model bounding box bottom, not absolute Z |
| HIGH | `ToolType_mill()` function imports ToolType inside function body | `operations.py:350-353` | Circular import workaround; called from multiple methods | Move to module-level constant tuple |
| HIGH | Global mutable `session` state in API | `routes.py:92` | Single-user design assumption | Use dependency injection or per-request state |
| HIGH | `_feature_boundary_polygon()` always returns bounding box, never actual boundary | `strategies.py:317-327` | Boundary loop extraction not connected to strategy | Wire up `feature.boundary_loop` to strategy |
| MEDIUM | `pyproject.toml` missing `pydantic` dependency | `pyproject.toml:6-11` | Pydantic used in API models but not listed | Add `pydantic` to dependencies |
| MEDIUM | Tapping operations never planned (feature recognizer never sets `notes["tapped"]`) | `operations.py:323` | Missing integration between recognizer and planner | Add thread recognition or user annotation |
| MEDIUM | `ctx.validate()` never called in pipeline | `pipeline.py` | Validation is implicit via individual calls | Add explicit `ctx.validate()` call |
| MEDIUM | Temp file leak possible if `upload_model_file` fails before tmp_path assignment | `routes.py:129-132` | Error before `tmp_path` is set in `finally` block | Initialize `tmp_path = None` before try |
| MEDIUM | `_offset_polygon()` can fail for concave polygons | `strategies.py:277-315` | Simplified bisector approach | Use proven polygon offset library (e.g., `pyclipper`) |
| LOW | CORS allows all origins | `app.py:18-24` | Development convenience | Restrict to known origins for production |
| LOW | No file size limit on uploads | `routes.py:122` | Missing validation | Add `max_size` parameter |
| LOW | Finish allowances hardcoded at 0.3mm | `pipeline.py:137-138` | No configuration mechanism | Pass through PlanningContext or config |

---

## 6. CAM / CNC Domain Findings

### 6.1 Units

**CONFIRMED CORRECT:** Internal units are mm. Conversion happens only at import/export boundaries. `UnitConverter` handles mm<->inch explicitly.

**CONCERN:** The STEP import assumes OpenCascade handles unit conversion on import (line 69-71 of `step_import.py`). If the STEP file declares inches and OCP doesn't auto-convert, the explicit `file_units` parameter handles it. This is correct behavior.

### 6.2 Coordinate Systems

**CONFIRMED CORRECT:** The four-space chain (MODEL -> SETUP -> WORK -> MACHINE) is properly validated with rigid transform checks. The `_invert()` function correctly computes analytic inverse for rigid transforms.

**CONCERN:** `TOOL` space is defined but never used in the chain. If tool-center-point (TCP) control is ever needed, the chain needs extension.

### 6.3 Geometry Interpretation

**CONFIRMED CORRECT:** Diameter vs radius is handled consistently. Surface normals are properly oriented using OCC face orientation. Cylinder internal/external distinction uses radial normal analysis.

**LIKELY BUG:** Through-hole detection at `features.py:273`:
```python
is_through = abs(c["bottom_z"]) < 0.5 or (c["bottom_z"] <= min(f.bounds_min[2] for f in self.faces) + 0.5)
```
The first condition `abs(c["bottom_z"]) < 0.5` checks if bottom_z is near zero, which only works if the model bottom is at Z=0. For translated models, this fails. The second condition is correct.

**DOMAIN ASSUMPTION:** The system assumes 3-axis vertical milling only. No turning, 5-axis, or mill-turn operations.

### 6.4 Machining Logic

**SOFTWARE VERIFIED, DOMAIN VALIDATION REQUIRED:**
- Cutting parameter formulas in `machinability.py` are reasonable engineering approximations
- Feed-per-tooth: `fz = 0.012 * d * (100/HB)^0.35` — this is a simplified model
- Power estimation: `MRR * Kc / 60` — standard formula
- Torque: `P * 9550 / RPM` — standard formula
- Operation ordering (facing > roughing > rest > semi > finish) is correct for standard practice

**REQUIRES DOMAIN VALIDATION:** The specific numerical coefficients in the feed-per-tooth formula should be validated against machinability handbooks for each material class.

### 6.5 Tool Assumptions

**CONFIRMED:** Tool geometry is explicit (diameter, flute length, corner radius, tip radius, flutes). Ball endmill tip_radius is validated to equal diameter/2. Drill tip_radius is validated to be 0.

**CONCERN:** No tool wear model. No tool life tracking. No vibration/chatter analysis.

---

## 7. Geometry Findings

| Finding | Status | Location | Notes |
|---------|--------|----------|-------|
| B-Rep face extraction from OCC | CORRECT | `topology.py:129-150` | Proper face iteration with tessellation |
| Surface type detection | CORRECT | `topology.py:166-201` | Handles all standard OCC surface types |
| Oriented normal evaluation | CORRECT | `topology.py:204-241` | Uses face orientation + BRepLProp_SLProps |
| Face adjacency via shared edges | CORRECT | `topology.py:378-401` | Real topology, not proximity-based |
| TriangleMesh ray casting | CORRECT | `mesh.py:45-104` | Grid-accelerated Moller-Trumbore |
| Point-in-triangle (2D) | LIKELY BUG | `mesh.py:147-163` | Edge handling uses strict `<` comparisons; points on edges may be missed |
| Voxel from_bounds | CORRECT | `voxel.py:24-38` | Clean construction |
| STEP import unit handling | CORRECT | `step_import.py:69-76` | Explicit conversion with no silent scaling |

---

## 8. API Findings

### Endpoint: `GET /api/model/demo`

```text
Endpoint: GET /api/model/demo
Purpose: Load and return demo CAD model
Input: None
Validation: None needed
Processing: Creates demo mesh, resets session state
Output: ModelInfoResponse (mesh data + bounding box)
Status Codes: 200
Side Effects: Resets global session state
Potential Issues: Not thread-safe (global mutable session)
```

### Endpoint: `POST /api/model/upload`

```text
Endpoint: POST /api/model/upload
Purpose: Upload CAD file for processing
Input: File (.step, .stp, .stl)
Validation: File extension check only
Processing: Import STEP/STL, extract mesh, cache in session
Output: ModelInfoResponse
Status Codes: 200, 400 (bad format), 422 (processing error)
Side Effects: Creates temp file, modifies global session
Potential Issues: No file size limit, temp file leak on some error paths, no content-type validation
```

### Endpoint: `GET /api/tools-and-machines`

```text
Endpoint: GET /api/tools-and-machines
Purpose: Return tool catalog and machine config
Input: None
Processing: Read from session defaults
Output: Tool list + machine config
Status Codes: 200
Potential Issues: None
```

### Endpoint: `POST /api/recognize-features`

```text
Endpoint: POST /api/recognize-features
Purpose: Run feature recognition on active model
Input: None (uses session state)
Processing: FeatureRecognizer on imported model, or hardcoded demo features
Output: RecognizeFeaturesResponse
Status Codes: 200
Side Effects: Modifies session.features
Potential Issues: Demo model returns hardcoded features, not computed
```

### Endpoint: `POST /api/plan-operations`

```text
Endpoint: POST /api/plan-operations
Purpose: Plan machining operations
Input: PlanOpsRequest (stock config, optional feature selection)
Processing: OperationPlanner with material/tool/machine context
Output: PlanOpsResponse (ordered operations + stock bounds)
Status Codes: 200, 400 (CamError)
Side Effects: Modifies session.planned_ops
Potential Issues: Calls recognize_features() if none exist (side effect)
```

### Endpoint: `POST /api/generate-toolpaths`

```text
Endpoint: POST /api/generate-toolpaths
Purpose: Generate semantic toolpaths
Input: GenerateToolpathsRequest (optional operations + stock)
Processing: StrategyEngine per operation
Output: GenerateToolpathsResponse (toolpaths + statistics)
Status Codes: 200, 400 (CamError)
Side Effects: Modifies session.toolpaths
Potential Issues: Missing error handling for non-CamError exceptions
```

### Endpoint: `POST /api/generate-gcode`

```text
Endpoint: POST /api/generate-gcode
Purpose: Post-process toolpaths to G-code
Input: PostProcessRequest (controller, work_offset, program_name)
Processing: Controller-specific post processor
Output: PostProcessResponse (G-code + statistics)
Status Codes: 200, 400 (CamError)
Side Effects: None (reads session.toolpaths)
Potential Issues: None
```

### Endpoint: `GET /api/setup-sheet`

```text
Endpoint: GET /api/setup-sheet
Purpose: Generate shop floor setup sheet
Input: None
Processing: Full pipeline + setup sheet formatting
Output: Setup sheet text + structured data
Status Codes: 200, 400
Potential Issues: Triggers full pipeline if not already run (slow)
```

---

## 9. Security Findings

| Severity | Issue | Location | Recommendation |
|----------|-------|----------|----------------|
| HIGH | CORS allows all origins | `app.py:18-24` | Restrict to specific origins for production |
| HIGH | No file size limit on uploads | `routes.py:122` | Add max upload size (e.g., 50MB) |
| MEDIUM | No authentication/authorization | All API routes | Add auth middleware for production |
| MEDIUM | Temp file written to system temp dir | `routes.py:129-132` | Ensure cleanup; consider in-memory processing |
| LOW | Error messages expose internal details | `routes.py:244` | Sanitize error messages for production |
| LOW | No rate limiting | All API routes | Add rate limiting for DoS prevention |

**SECRETS:** No hardcoded secrets, API keys, or credentials found.

---

## 10. Testing Findings

### Current Coverage

| Module | Test Coverage | Notes |
|--------|--------------|-------|
| `post/post_processor.py` | GOOD | GRBL, Fanuc, canned cycles, coolant |
| `post/gcode_parser.py` | PARTIAL | Parse-back tested via pipeline |
| `api/routes.py` | GOOD | Demo model, tools, features, ops, toolpaths, G-code, upload |
| `toolpaths/strategies.py` | PARTIAL | Spot drill, ream, tap, chamfer tested; roughing/finishing not |
| `toolpaths/semantic.py` | MINIMAL | Only tested indirectly |
| `toolpaths/validation.py` | NONE | No direct tests |
| `features.py` | GOOD | Via test_generic_cam_geometry.py (requires OCP) |
| `machinability.py` | NONE | No direct tests |
| `operations.py` | MINIMAL | Tested via pipeline |
| `context.py` | NONE | No direct tests |
| `coords.py` | GOOD | Via test_generic_cam_geometry.py |
| `geometry/topology.py` | GOOD | Via test_generic_cam_geometry.py |
| `geometry/mesh.py` | NONE | No direct tests |
| `geometry/voxel.py` | NONE | No direct tests |
| `geometry/step_import.py` | PARTIAL | Via API upload test |
| `geometry/stl_import.py` | NONE | No tests |
| `pipeline.py` | GOOD | Via test_generic_cam_geometry.py and API tests |
| `errors.py` | NONE | No direct tests |
| `units.py` | NONE | No direct tests |

### Missing Critical Tests

1. **Geometry/mesh.py ray casting** — No tests for `ray_first_hit`, `is_point_inside`, `top_height`
2. **Toolpath validation** — No tests for gouge detection, collision detection, axis limit checks
3. **Machinability cutting parameters** — No tests for feed/speed calculation, power/torque estimation
4. **Coordinate transforms** — Only basic composition tested; no rotation, multi-setup, or edge cases
5. **STL import** — No tests for ASCII or binary STL parsing
6. **Voxel operations** — No tests for remove_sphere, remove_capsule, heightmap
7. **Error handling** — No tests for CamError propagation, missing data failures
8. **Edge cases** — No tests for empty geometry, degenerate inputs, single-face models
9. **Post-processor arc fitting** — No direct tests for `_fit_arc` correctness
10. **Concurrent API access** — No tests for session state races

---

## 11. Performance Findings

| Issue | Location | Impact | Notes |
|-------|----------|--------|-------|
| `_accessibility_of()` creates new TriangleMesh per feature | `features.py:614` | O(features × faces) | Should create mesh once per recognize() call |
| `from_mesh()` in voxel.py is sequential | `voxel.py:48-49` | O(n³) for large meshes | Should use batch point-in-mesh or GPU |
| `_nearest_neighbor_sort()` is O(n²) | `strategies.py:1213` | Fine for <1000 points | Could use scipy.spatial for larger sets |
| Grid cell size hardcoded at 5.0mm | `mesh.py:33` | May be too coarse/small | Should scale with mesh bounds |
| Demo model face creation uses Python loops | `demo_models.py:40-150` | Only runs once | Acceptable |

**Overall:** No critical performance issues for the current use case (single-part processing). The voxel simulation could be slow for large stocks with small tools.

---

## 12. Code Quality Findings

| Issue | Location | Recommendation |
|-------|----------|----------------|
| `ToolType_mill()` function should be a constant | `operations.py:350-353` | Define at module level |
| Mixed naming conventions (snake_case, camelCase in some OCC calls) | Various | Follow PEP 8 consistently |
| `object()` used as placeholder for VoxelStock | `strategies.py:32-33` | Use proper type hints |
| Some methods are very long (>100 lines) | `strategies.py` various | Break into smaller helpers |
| No `__all__` in most modules | Various | Add explicit exports |
| Docstrings are generally good | All modules | Maintain this standard |
| Error messages are informative | All modules | Maintain this standard |

---

## 13. Dependency Findings

```toml
dependencies = [
    "numpy",           # Used extensively - required
    "cadquery-ocp",    # OpenCascade bindings - required for STEP import
    "fastapi",         # Web framework - required for API
    "uvicorn",         # ASGI server - required for running
    "python-multipart", # File upload support - required for API
]

[project.optional-dependencies]
dev = ["pytest", "httpx"]  # httpx not used (TestClient from fastapi is used instead)
```

**Issues:**
1. `pydantic` is used in `api/models.py` but not listed in dependencies (comes with FastAPI transitively)
2. `httpx` is listed as dev dependency but never imported (tests use `fastapi.testclient.TestClient`)
3. No version pins — could break on major version bumps
4. `requests` is imported in `test_upload.py` but not in any dependency list (it's a script, not a test)

---

## 14. Requirements Gap Analysis

### CURRENTLY IMPLEMENTED (FACT)

- STEP file import via OpenCascade
- STL file import (ASCII + binary)
- B-Rep topology extraction (faces, edges, surface types, adjacency)
- Feature recognition: holes (through/blind), bores, counterbores, bosses, pockets, slots, steps, facing regions, chamfers, fillets, freeform surfaces
- Cutting parameter derivation from material properties
- Tool selection and ranking
- Operation planning with dependency tracking
- Toolpath generation: facing, roughing (contour-parallel), adaptive/trochoidal, rest machining, semi-finishing, finishing (wall/floor, freeform, waterline, scallop, pencil, radial), drilling (peck), helical interpolation, boring, spot drilling, reaming, tapping, chamfering
- Toolpath validation (axis limits, gouge detection, holder collision, reach)
- G-code post-processing for GRBL, Fanuc, Haas, LinuxCNC
- Canned cycle support (G81-G86, G73, G76)
- Arc fitting to reduce point count
- Setup sheet generation
- REST API for interactive use
- CLI interface
- Demo model for testing

### PARTIALLY IMPLEMENTED (INFERENCE)

- Voxel stock simulation — structure exists but `remove_capsule` is only used for visualization, not integrated into operation planning for stock tracking
- Multi-setup machining — the loop exists but no setup reorientation logic
- Coordinate system chain — full chain defined but only model->setup is used in practice
- Setup sheet PDF export — text format works, no PDF

### BROKEN (FACT)

- `test_generic_cam_geometry.py` collection fails when OCP is not installed (no skip marker)
- Through-hole detection heuristic (`abs(bottom_z) < 0.5`) fails for translated parts
- `ToolType_mill()` function imports inside function body (works but is non-standard)
- Tapping operations are unreachable (feature recognizer never sets `notes["tapped"]`)

### MISSING (RECOMMENDATION)

1. **Multi-axis support** — 4-axis and 5-axis toolpath strategies
2. **Turning operations** — The stock model supports CYLINDER kind but no turning strategies exist
3. **Thread recognition** — Automatic thread/tapping hole detection from B-Rep
4. **Surface finish requirements** — FinishRequirement is defined but never connected to strategies
5. **Work coordinate offsets** — Only G54 used; G55-G59 configured but not testable with current demo
6. **Collision simulation** — Voxel-based material removal not used for gouge/collision validation
7. **Tool library persistence** — No database, all tools are in-memory defaults
8. **Job queue** — Single-request processing, no async job management
9. **Undo/redo** — No session history
10. **G-code simulation** — Parse-back exists but no 3D visualization
11. **Feed/speed optimization** — No adaptive feeds based on engagement
12. **Rest machining from prior operations** — Structure exists but voxel state not updated between operations

---

## 15. Prioritized Roadmap

### P0 — Critical

| Issue | Priority | Files | Reason | Expected Impact | Complexity | Dependencies |
|-------|----------|-------|--------|-----------------|------------|--------------|
| `pyproject.toml` missing `pydantic` | P0 | `pyproject.toml` | Import fails if pydantic not installed transitively | Prevents installation | LOW | None |
| Through-hole detection Z=0 assumption | P0 | `features.py:273` | Incorrect feature classification for translated parts | Wrong machining operations | LOW | None |
| Temp file leak in upload | P0 | `routes.py:129-132` | Resource leak on error paths | Disk space exhaustion | LOW | None |

### P1 — High Priority

| Issue | Priority | Files | Reason | Expected Impact | Complexity | Dependencies |
|-------|----------|-------|--------|-----------------|------------|--------------|
| Global mutable session state | P1 | `routes.py:92` | Not thread-safe, no multi-user | Broken concurrent use | MEDIUM | None |
| `ToolType_mill()` circular import workaround | P1 | `operations.py:350-353` | Non-standard pattern | Maintainability | LOW | None |
| Demo model hardcoded features | P1 | `routes.py:302-356` | Demo doesn't test real recognition | False confidence in demo | MEDIUM | None |
| `_feature_boundary_polygon()` returns bounding box | P1 | `strategies.py:317-327` | Incorrect toolpath for non-rectangular pockets | Wrong machining | HIGH | None |
| Tapping unreachable | P1 | `operations.py:323`, `features.py` | Missing thread recognition integration | Missing functionality | MEDIUM | None |
| No `ctx.validate()` in pipeline | P1 | `pipeline.py` | Invalid context can propagate | Silent failures | LOW | None |

### P2 — Medium Priority

| Issue | Priority | Files | Reason | Expected Impact | Complexity | Dependencies |
|-------|----------|-------|--------|-----------------|------------|--------------|
| CORS allows all origins | P2 | `app.py:18-24` | Security risk | Unauthorized access | LOW | None |
| No file size limit on uploads | P2 | `routes.py:122` | DoS vector | Server crash | LOW | None |
| No authentication | P2 | All API routes | Security risk | Unauthorized use | HIGH | None |
| Hardcoded finish allowances | P2 | `pipeline.py:137-138` | Not configurable | Limited flexibility | LOW | None |
| `_offset_polygon()` simplified | P2 | `strategies.py:277-315` | Fails for concave polygons | Wrong toolpaths for complex pockets | MEDIUM | None |
| Voxel stock not updated between operations | P2 | `pipeline.py:125-127` | Rest machining uses fresh stock | Incorrect rest machining | MEDIUM | Voxel integration |
| No operation for CONTOUR in roughing | P2 | `operations.py:111-115` | Missing roughing for freeform | Limited functionality | LOW | None |
| Hardcoded rapid feed in setup_sheet | P2 | `setup_sheet.py:187` | Inaccurate time estimates | Misleading cycle times | LOW | None |
| Missing test coverage for geometry/mesh, validation, machinability | P2 | tests/ | Regression risk | Unknown bugs | HIGH | None |

### P3 — Performance

| Issue | Priority | Files | Reason | Expected Impact | Complexity | Dependencies |
|-------|----------|-------|--------|-----------------|------------|--------------|
| `_accessibility_of()` mesh recreation | P3 | `features.py:614` | O(features × faces) work | Slow recognition for complex parts | LOW | None |
| Grid cell size hardcoded | P3 | `mesh.py:33` | May be suboptimal | Slow ray casting | LOW | None |
| Sequential voxel from_mesh | P3 | `voxel.py:48-49` | O(n³) for large meshes | Slow stock creation | MEDIUM | None |

### P4 — Nice to Have

| Issue | Priority | Files | Reason | Expected Impact | Complexity | Dependencies |
|-------|----------|-------|--------|-----------------|------------|--------------|
| Multi-axis support | P4 | New modules | Major feature | Broader applicability | HIGH | Architecture changes |
| Turning operations | P4 | New modules | Major feature | Lathe support | HIGH | None |
- Thread recognition | P4 | `features.py` | Missing feature | Tapping support | MEDIUM | None |
| PDF setup sheets | P4 | `setup_sheet.py` | Export format | Better documentation | LOW | ReportLab dependency |
| Tool library persistence | P4 | New module | Data management | Session persistence | MEDIUM | Database |
| Undo/redo | P4 | API layer | UX improvement | Better workflow | HIGH | State management |

---

## 16. Recommended Architecture

The current architecture is sound and well-layered. The primary recommendation is **not** a rewrite but targeted improvements:

### Current Target (No Changes Needed):
```
API -> Pipeline -> [Features -> Machinability -> Operations -> Strategies -> Validation -> Post]
```

### Recommended Enhancements:

1. **Dependency Injection for Session State** — Replace global `session` with FastAPI dependency injection (per-request scoped)
2. **Configuration Object** — Add a `CAMConfig` dataclass for finish allowances, tessellation quality, grid sizes, etc.
3. **Voxel Integration** — Update voxel stock between operations for accurate rest machining
4. **Boundary Following** — Wire `feature.boundary_loop` into strategies for accurate pocket/slot contours
5. **Validation Gate Extension** — Add feedrate and spindle speed range validation

No architectural rewrite is justified. The current layered design is appropriate for the system's complexity.

---

## 17. Risk Register

| Risk | Probability | Impact | Severity | Mitigation |
|------|------------|--------|----------|------------|
| Incorrect feature classification on translated parts | HIGH | HIGH | CRITICAL | Fix Z=0 heuristic to use model bbox |
| Thread recognition never triggers | HIGH | MEDIUM | HIGH | Add user annotation or simple thread detection |
| Non-rectangular pocket toolpaths use bounding box | HIGH | HIGH | HIGH | Wire boundary_loop to strategies |
| Concurrent API access corrupts session | MEDIUM | HIGH | HIGH | Per-request state scoping |
| Undetected gouging from coarse validation sampling | LOW | HIGH | MEDIUM | Increase validation sample density |
| Cutting parameters outside safe range | LOW | HIGH | MEDIUM | Add min/max clamping from tool/material data |
| Temp file accumulation from uploads | LOW | MEDIUM | LOW | Ensure reliable cleanup |
| Missing tests allow regressions | MEDIUM | MEDIUM | MEDIUM | Add tests for critical paths |

---

## 18. Audit Limitations

1. **OCP Not Installed** — Tests requiring OpenCascade (`test_generic_cam_geometry.py`) could not be run. Feature recognition correctness is inferred from code review, not verified by execution.

2. **No Runtime Verification** — The pipeline was not executed against the sample STEP file during this audit. The analysis is based on code reading and the 20 passing tests.

3. **Domain Assumption Validation** — Cutting parameter formulas, machining operation ordering, and tool selection heuristics are reasonable engineering approximations but have not been validated against machinability handbooks or CNC expert review.

4. **Frontend Code** — The `static/` directory contains HTML/CSS/JS files that were not audited (outside scope of backend engineering audit).

5. **Large Geometry Testing** — No performance testing with complex CAD models (500+ faces, freeform surfaces, multi-body parts).

6. **G-code Verification** — Generated G-code was not verified against real CNC controllers. The parse-back verification tests confirm structural correctness but not controller compatibility.

7. **Security Testing** — No penetration testing was performed. Security findings are based on code review only.

---

*Audit performed: 2026-09-10*
*Auditor: opencode/mimo-v2.5-free*
*Codebase version: 0.1.0*
*Files reviewed: 36 Python files, 1 TOML config*
*Lines of code: ~8,500 (Python)*
