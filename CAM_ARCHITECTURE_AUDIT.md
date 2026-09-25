# CAM Architecture Audit

_Generated as part of the new-architecture.md refactor (Phase 1)._

---

## 1. Current Architecture (pre-refactor)

`
User (browser)
  POST /api/model/upload          -> ImportedModel stored in session (global coords) ✅
  POST /api/setups/auto-generate  -> BUG: always adds Bottom + Side from face scan ❌
  POST /api/recognize-features    -> BUG: per-setup on rotated shape, duplication ❌
  POST /api/plan-operations       -> filters via notes.get("setup_id") ❌
  POST /api/generate-toolpaths    -> BUG: fresh VoxelStock per setup, no IPW ❌
  POST /api/post-process          -> G-code output
`

## 2. Files

| File | Role |
|---|---|
| cam_engine/geometry/step_import.py | STEP import (OCP) |
| cam_engine/geometry/topology.py | B-Rep wrapper |
| cam_engine/geometry/voxel.py | VoxelStock IPW simulation |
| cam_engine/features.py | FeatureRecognizer, MachiningFeature |
| cam_engine/context.py | PlanningContext, Setup, Stock, Tool |
| cam_engine/operations.py | OperationPlanner, PlannedOperation |
| cam_engine/pipeline.py | Batch end-to-end pipeline |
| cam_engine/toolpaths/strategies.py | StrategyEngine toolpath algorithms |
| cam_engine/toolpaths/semantic.py | Toolpath, MotionSegment, MotionType |
| cam_engine/toolpaths/validation.py | ToolpathValidator |
| cam_engine/api/routes.py | All REST endpoints + SessionState |
| cam_engine/api/models.py | Pydantic schemas |
| static/js/viewer.js | Three.js renderer |

## 3. Bugs Found

| # | Severity | File:Line | Description |
|---|---|---|---|
| B1 | HIGH | routes.py:609 | Bottom setup always added regardless of features |
| B2 | HIGH | routes.py:598 | Side setups from face existence, not feature accessibility |
| B3 | HIGH | routes.py:794 | Feature recognition per-setup on rotated shape - duplicates |
| B4 | HIGH | routes.py:803 | setup_id only in notes dict, not a field |
| B5 | HIGH | routes.py:~1118 | Fresh VoxelStock per setup - IPW never carries over |
| B6 | MED | routes.py:1022 | brittle notes.get("setup_id") lookup |
| B7 | MED | viewer.js | No authoritative setup_to_world transform for toolpath rendering |
| B8 | MED | features.py:58 | MachiningFeature has no setup_id field |
| B9 | LOW | pipeline.py vs routes.py | Two IPW implementations that diverge |

## 4. Fixes Applied

### Phase 2 — features.py
Added setup_id: Optional[str] = None as a first-class field on MachiningFeature.

### Phase 3 — routes.py recognize_features
Feature recognition now runs ONCE in global model coordinates.
setup_id assigned by matching machining_direction to nearest setup tool_axis.
setup_feat_map synchronizes feature_ids on each SetupConfig.

### Phase 5 — routes.py auto_generate_setups
Replaced geometry-scan + always-add-bottom with feature-driven logic:
Run global recognition -> classify each feature direction -> create setup only if needed.

### Phase 6 — routes.py generate_toolpaths
Single VoxelStock created from initial stock.
IPW updated after every cutting operation across all setups.
Toolpath generation errors are warned, not silently swallowed.

### Phase 7 — routes.py + models.py
setup_transform (flat 4x4 row-major inverse rotation matrix) added to ToolpathItem.
Identity for top setup; correct inverse rotation for bottom/side setups.

### Phase 8 — viewer.js
renderToolpaths() now applies tp.setup_transform when present.
Toolpath points transformed to model/world space before rendering.
No double-rotation: the part stays in its position; only toolpath points are transformed.

### Phase 10 — tests/test_new_architecture.py
New test file covering all above fixes.

## 5. Files Modified

- cam_engine/features.py (Phase 2: setup_id field)
- cam_engine/api/routes.py (Phase 3, 5, 6, 7)
- cam_engine/api/models.py (Phase 7: ToolpathItem.setup_transform)
- static/js/viewer.js (Phase 8)
- tests/test_new_architecture.py (Phase 10)
- tests/test_contour_and_fillet_toolpaths.py (OCP 8.0 compat fix: TopoDS.Edge_s -> TopoDS.Edge)

## 6. Files Unchanged (API Compatible)

- cam_engine/context.py
- cam_engine/operations.py
- cam_engine/pipeline.py
- cam_engine/toolpaths/strategies.py
- cam_engine/toolpaths/semantic.py
- cam_engine/toolpaths/validation.py
- cam_engine/machinability.py
- cam_engine/coords.py
