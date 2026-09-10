"""cam_engine: generic, geometry-driven CAM planning and G-code generation.

Architecture (data flows strictly downward; no stage reaches around another):

    STEP (B-Rep)  ->  geometry layer
                  ->  planning context (single source of truth)
                  ->  feature recognition (topology-driven)
                  ->  machinability + tool selection (capability-driven)
                  ->  operation planning (dependency-driven sequence)
                  ->  toolpath generation (engagement-aware strategies)
                  ->  semantic toolpath (controller-independent)
                  ->  validation gates (geometry / setup / planning / toolpath)
                  ->  voxel stock-removal simulation
                  ->  optimization (intent-preserving)
                  ->  post processor (controller profiles, modal state)
                  ->  G-code validation + parse-back verification
                  ->  controller-specific G-code
"""

__version__ = "0.1.0"
