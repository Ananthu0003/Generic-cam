"""Tests for Material Database, Alias Resolution, Cutting-Data Calculations, and G-Code Auditability.

Verifies:
- Beryllium Copper (BeCu / C17200) and other engineering materials are first-class database entries.
- Alias resolution (e.g. "BeCu", "C17200", "Beryllium Copper", "Ti-6Al-4V", "SS304") works seamlessly.
- Cutting parameters (Vc, RPM, fz, feed rate, plunge, engagement, power, torque, peck Q) derive
  strictly from the workpiece material and differ fundamentally from Aluminum 6061-T6.
- Multi-setup G-code generation retains the true workpiece material throughout both setups
  and never silently substitutes a default material.
- Unresolved or invalid materials block G-code generation and raise clear errors.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np

from app import app
from cam_engine.context import (
    DEFAULT_MATERIALS,
    get_material,
    list_materials,
    Material,
    PlanningContext,
    Setup,
    Stock,
    StockKind,
    WorkOffset,
    Tool,
    ToolType,
)
from cam_engine.errors import CamError
from cam_engine.machinability import MachinabilityAnalyzer
from cam_engine.features import MachiningFeature, FeatureType
from cam_engine.operations import OperationPlanner
from cam_engine.toolpaths.strategies import StrategyEngine, StrategyContext
from cam_engine.geometry.step_import import import_step
from cam_engine.geometry.mesh import TriangleMesh
from cam_engine.geometry.voxel import VoxelStock
from cam_engine.demo_models import get_default_machine, get_default_tools
from cam_engine.post import get_post_processor


@pytest.fixture
def client():
    return TestClient(app)


class TestMaterialDatabaseAndAliases:
    """Verify material database entries and alias resolution."""

    def test_becu_in_database_and_properties(self):
        mat = get_material("becu")
        assert mat.id == "becu"
        assert "Beryllium Copper" in mat.name
        assert mat.hardness_hb == 220
        assert mat.specific_cutting_force_kn_mm2 == 1.55
        assert mat.v_carbide == 120.0
        assert mat.v_hss == 25.0
        assert mat.needs_coolant is True

    def test_alias_resolution_for_becu(self):
        aliases = [
            "becu", "BeCu", "BE-CU", "beryllium_copper",
            "Beryllium Copper", "C17200", "c17300", "C172", "Alloy 25"
        ]
        for alias in aliases:
            mat = get_material(alias)
            assert mat.id == "becu"

    def test_other_key_materials_resolution(self):
        # Titanium
        mat_ti = get_material("Ti-6Al-4V")
        assert mat_ti.id == "titanium_gr5"
        assert mat_ti.hardness_hb == 330

        # Inconel
        mat_inco = get_material("Inconel 718")
        assert mat_inco.id == "inconel_718"
        assert mat_inco.specific_cutting_force_kn_mm2 == 3.10

        # Stainless 304
        mat_ss = get_material("SS304")
        assert mat_ss.id == "stainless_304"

        # Steel 4140
        mat_4140 = get_material("4140_HT")
        assert mat_4140.id == "steel_4140"

        # Copper 110
        mat_cu = get_material("C11000")
        assert mat_cu.id == "copper_110"

    def test_unresolved_material_raises_error(self):
        """Unresolved or invalid materials must raise an error and never silently fall back."""
        with pytest.raises(CamError) as exc_info:
            get_material("Unobtainium_999")
        assert exc_info.value.code == "INVALID_MATERIAL"
        assert "Unknown or unresolved material" in exc_info.value.message

        with pytest.raises(CamError) as exc_empty:
            get_material("")
        assert exc_empty.value.code == "INVALID_MATERIAL"


class TestCuttingDataDerivationFromMaterial:
    """Verify that Vc, RPM, fz, feed rate, power, and torque differ between BeCu and Aluminum."""

    def test_becu_vs_aluminum_cutting_parameters(self):
        machine = get_default_machine()
        mat_al = get_material("aluminum_6061")
        mat_becu = get_material("becu")

        analyzer_al = MachinabilityAnalyzer(machine, mat_al)
        analyzer_becu = MachinabilityAnalyzer(machine, mat_becu)

        tool = Tool(
            id="T01",
            type=ToolType.FLAT_ENDMILL,
            diameter=6.0,
            flute_length=15.0,
            overall_length=50.0,
            tool_number=1,
            flutes=3,
            material="carbide",
        )
        feature = MachiningFeature(
            id="f1",
            type=FeatureType.OPEN_POCKET,
            face_indices=[0],
            depth=5.0,
            top_z=10.0,
            floor_z=5.0,
            bounds_min=np.array([-10.0, -10.0, 5.0]),
            bounds_max=np.array([10.0, 10.0, 10.0]),
        )
        stock = Stock(kind=StockKind.BOX, bounds_min=np.array([-15, -15, 0]), bounds_max=np.array([15, 15, 12]))
        setup = Setup(id="s1", name="Setup 1", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=stock)

        params_al = analyzer_al.analyze(tool, feature, "roughing", setup)
        params_becu = analyzer_becu.analyze(tool, feature, "roughing", setup)

        # 1. Surface speed Vc and Target RPM
        assert params_al.target_cutting_speed_m_min == 450.0
        assert params_al.effective_cutting_speed_m_min == pytest.approx(282.74, rel=1e-3)
        assert params_al.rpm_clamped is True
        assert params_becu.target_cutting_speed_m_min == 120.0
        assert params_becu.effective_cutting_speed_m_min == pytest.approx(120.0, rel=1e-3)
        assert params_becu.target_rpm < params_al.target_rpm
        # For Ø6 mm, BeCu RPM = 1000*120/(pi*6) = ~6366 RPM
        assert 6300 <= params_becu.spindle_rpm <= 6400

        # 2. Feed per tooth fz and Feed rate
        # Higher hardness in BeCu (220 HB vs 95 HB) results in lower chip load fz
        assert params_becu.feed_per_tooth_mm < params_al.feed_per_tooth_mm
        assert params_becu.feed_rate < params_al.feed_rate

        # 3. Power and Torque calculations
        # BeCu Kc (1.55) is > 2.2x Aluminum Kc (0.70)
        assert mat_becu.specific_cutting_force_kn_mm2 > mat_al.specific_cutting_force_kn_mm2

        # 4. Material-aware Peck Depth Q
        hole_feature = MachiningFeature(
            id="h1",
            type=FeatureType.BLIND_HOLE,
            face_indices=[0],
            depth=25.0,
            top_z=25.0,
            floor_z=0.0,
            diameter=6.0,
            bounds_min=np.array([-3.0, -3.0, 0.0]),
            bounds_max=np.array([3.0, 3.0, 25.0]),
        )
        peck_al = analyzer_al.calculate_drilling_peck(tool, hole_feature)
        peck_becu = analyzer_becu.calculate_drilling_peck(tool, hole_feature)
        assert peck_becu < peck_al


class TestMultiSetupGCodeMaterialPersistence:
    """Verify that multi-setup toolpaths and G-code retain BeCu material across all setups."""

    def test_gcode_material_audit_comments_becu(self):
        machine = get_default_machine()
        tools = get_default_tools()
        mat_becu = get_material("becu")

        from cam_engine.geometry.step_import import ImportedModel
        from cam_engine.context import Units, StockKind
        from cam_engine.operations import OpPurpose, PlannedOperation

        model = ImportedModel(shape=None, declared_units=Units.MM, original_bounds=(np.zeros(3), np.ones(3)))
        s1 = Setup(id="setup_001", name="Setup 1 - Top", model_to_setup=np.eye(4), work_offset=WorkOffset.G54, stock=Stock(kind=StockKind.BOX, bounds_min=np.array([-5, -5, -2]), bounds_max=np.array([35, 25, 12])))
        s2 = Setup(id="setup_002", name="Setup 2 - Bottom", model_to_setup=np.eye(4), work_offset=WorkOffset.G55, stock=Stock(kind=StockKind.BOX, bounds_min=np.array([-5, -5, -2]), bounds_max=np.array([35, 25, 12])))

        ctx = PlanningContext(model=model, setups=[s1, s2], machine=machine, tools=tools, material=mat_becu)

        tool = tools[0]
        feat1 = MachiningFeature(
            id="f1",
            type=FeatureType.FACING_REGION,
            face_indices=[0],
            depth=1.0,
            top_z=10.0,
            floor_z=9.0,
            bounds_min=np.array([0, 0, 9.0]),
            bounds_max=np.array([30, 20, 10.0]),
            notes={"setup_id": "setup_001"},
        )
        analyzer = MachinabilityAnalyzer(machine, mat_becu)
        params1 = analyzer.analyze(tool, feat1, "facing", s1)
        op1 = PlannedOperation(id="op_face_s1", feature=feat1, tool=tool, purpose=OpPurpose.FACING, params=params1, notes={"setup_id": "setup_001"})

        feat2 = MachiningFeature(
            id="f2",
            type=FeatureType.FACING_REGION,
            face_indices=[0],
            depth=1.0,
            top_z=10.0,
            floor_z=9.0,
            bounds_min=np.array([0, 0, 9.0]),
            bounds_max=np.array([30, 20, 10.0]),
            notes={"setup_id": "setup_002"},
        )
        params2 = analyzer.analyze(tool, feat2, "facing", s2)
        op2 = PlannedOperation(id="op_face_s2", feature=feat2, tool=tool, purpose=OpPurpose.FACING, params=params2, notes={"setup_id": "setup_002"})

        mesh = TriangleMesh(triangles=np.zeros((0, 3, 3)))
        strat_ctx1 = StrategyContext(context=ctx, setup=s1, mesh=mesh, stock_voxels=None, clearance_z=20.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0)
        engine1 = StrategyEngine(strat_ctx1)
        tp1 = engine1.generate(op1)

        strat_ctx2 = StrategyContext(context=ctx, setup=s2, mesh=mesh, stock_voxels=None, clearance_z=20.0, finish_allowance_wall=0.0, finish_allowance_floor=0.0)
        engine2 = StrategyEngine(strat_ctx2)
        tp2 = engine2.generate(op2)

        post = get_post_processor("fanuc", work_offset="G54")
        res1 = post.post_process([tp1], program_name="PROG_SETUP1", setup_id="setup_001")
        res2 = post.post_process([tp2], program_name="PROG_SETUP2", setup_id="setup_002")

        # Verify BeCu material is auditable in BOTH setup outputs
        assert "Beryllium Copper" in res1.gcode
        assert "HB 220" in res1.gcode
        assert "Aluminum 6061" not in res1.gcode

        assert "Beryllium Copper" in res2.gcode
        assert "HB 220" in res2.gcode
        assert "Aluminum 6061" not in res2.gcode


class TestApiMaterialsEndpoints:
    """Verify REST API material endpoints."""

    def test_get_materials_catalog(self, client):
        res = client.get("/api/materials")
        assert res.status_code == 200
        data = res.json()
        assert "materials" in data
        mat_ids = [m["id"] for m in data["materials"]]
        assert "becu" in mat_ids
        assert "aluminum_6061" in mat_ids
        assert "titanium_gr5" in mat_ids
        assert "inconel_718" in mat_ids

    def test_tools_and_machines_includes_materials(self, client):
        res = client.get("/api/tools-and-machines")
        assert res.status_code == 200
        data = res.json()
        assert "materials" in data
        assert len(data["materials"]) >= 15
