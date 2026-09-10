"""Command-line interface for Generic CAM Engine."""

import argparse
import sys
from pathlib import Path

from .pipeline import run_cam_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generic CAM CLI: Generate CNC toolpaths & G-code directly from a STEP file.",
    )
    parser.add_argument("step_file", type=str, help="Path to input .step / .stp CAD file")
    parser.add_argument(
        "--controller",
        choices=["grbl", "fanuc", "haas", "linuxcnc"],
        default="grbl",
        help="Target CNC controller (default: grbl)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output .nc / .gcode file path (default: stdout / <filename>.nc)",
    )
    parser.add_argument(
        "--material",
        type=str,
        default="aluminum_6061",
        help="Workpiece material (default: aluminum_6061)",
    )

    args = parser.parse_args()
    step_path = Path(args.step_file)

    if not step_path.exists():
        print(f"Error: STEP file not found: {step_path}", file=sys.stderr)
        sys.exit(1)

    out_file = args.output or f"{step_path.stem}_{args.controller}.nc"
    print(f"[*] Processing STEP file: {step_path}")
    print(f"[*] Target Controller: {args.controller.upper()}")
    print(f"[*] Material: {args.material}")

    try:
        result = run_cam_pipeline(
            step_path=step_path,
            controller=args.controller,
            material_name=args.material,
            output_gcode_path=out_file,
        )
        print(f"[+] Successfully recognized {len(result.features)} features:")
        for f in result.features:
            print(f"    - {f.id} ({f.type.value}): depth={f.depth:.1f}mm")
        print(f"[+] Planned {len(result.operations)} machining operations")
        print(f"[+] Generated {len(result.toolpaths)} toolpaths ({result.post_result.total_cut_dist_mm:.1f} mm cutting motion)")
        print(f"[+] Saved G-Code to: {out_file} ({result.post_result.line_count} lines, est. cycle time: {result.post_result.total_time_seconds:.1f}s)")
    except Exception as e:
        print(f"Error during CAM processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
