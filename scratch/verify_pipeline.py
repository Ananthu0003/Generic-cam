from pathlib import Path
from cam_engine.pipeline import run_cam_pipeline

res = run_cam_pipeline(Path("sample_bracket.step"), controller="grbl", work_offset="G54")

print("=== RECOGNIZED FEATURES ===")
for f in res.features:
    print(f"ID: {f.id}, Type: {f.type.value}, Dia: {f.diameter}, Depth: {f.depth:.2f}, FloorZ: {f.floor_z:.2f}, TopZ: {f.top_z:.2f}, Faces: {f.face_indices}, Evidence: {f.recognition_evidence}")

print("\n=== PLANNED OPERATIONS ===")
for o in res.operations:
    print(f"Op ID: {o.id}, Purpose: {o.purpose.value}, Feature: {o.feature.id}, Tool: {o.tool.id} ({o.tool.type.value} D{o.tool.diameter}), Depth: {o.feature.depth:.2f}")

print("\n=== VALIDATED TOOLPATHS ===")
for tp in res.toolpaths:
    print(f"Toolpath Op: {tp.operation_id}, Tool: {tp.tool_id}, Segments: {len(tp.segments)}, Rapid: {tp.rapid_length():.2f}mm, Cutting: {tp.cutting_length():.2f}mm")
    for i, m in enumerate(tp.segments[:5]):
        s = [round(x, 2) for x in m.start] if m.start is not None else None
        e = [round(x, 2) for x in m.end] if m.end is not None else None
        print(f"  Segment {i}: Type={m.motion_type.value}, Start={s}, End={e}, Feed={m.feed}")
    if len(tp.segments) > 5:
        print(f"  ... ({len(tp.segments) - 5} more segments)")

print("\n=== GENERATED G-CODE ===")
lines = res.gcode.splitlines()
for line in lines[:45]:
    print(line)
print(f"... Total lines: {len(lines)}")
