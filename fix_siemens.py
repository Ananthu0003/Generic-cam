import re

with open('cam_engine/post/post_processor.py', 'r') as f:
    content = f.read()

# Add Siemens renderers inside the class, before 'SinumerikPostProcessor = SiemensPostProcessor'
pattern = r'(        return lines\n\n\nSinumerikPostProcessor = SiemensPostProcessor)'

replacement = r'''        return lines

    # --- Siemens Renderers ---
    def _render_tool_change(self, cmd: ToolChange) -> list[str]:
        lines = []
        actual_tool_num = self._resolve_tool_number(cmd.tool_id, cmd.tool_number)
        desc = cmd.description
        h = self._resolve_length_offset(cmd.tool_id, actual_tool_num, cmd.length_offset_h, None)
        if self.current_tool_id != cmd.tool_id:
            if self.canned_cycle_active:
                lines.append("MCALL")
                self.canned_cycle_active = False
                self._cycle_op_id = None
            if self.spindle_active:
                lines.append("M5")
                self.spindle_active = False
            safe_z = self._get_safe_retract_z(fallback=50.0)
            if self.is_pos_established:
                lines.append(f"G0 Z{self._fmt(safe_z)}")
            lines.append(f"T{actual_tool_num} M6 (Tool: T{actual_tool_num:02d} - {desc})")
            lines.append("D1")
            rpm = int(round(self.current_spindle or 3000.0))
            lines.append(f"S{rpm} M3")
            self.spindle_active = True
            self.current_tool_id = cmd.tool_id
            self.is_pos_established = False
            if cmd.tool_id not in self.used_tools:
                self.used_tools.append(cmd.tool_id)
        return lines

    def _render_cycle(self, cmd: Cycle) -> list[str]:
        lines = []
        cycle_type = cmd.cycle_type
        x = cmd.x
        y = cmd.y
        z_r = cmd.r_plane
        z_depth = cmd.depth_z
        peck = cmd.peck
        dwell = cmd.dwell
        feed = cmd.feed
        safe_z = self._get_safe_retract_z(fallback=50.0)
        if feed != self.current_feed:
            self.current_feed = feed
        rtp = safe_z
        rfp = z_r
        sdis = max(1.0, abs(z_r - 0.0)) if abs(z_r) < 1e-4 else 2.0
        dp = z_depth
        if cycle_type in ("G83", "G73", "CYCLE83") and peck > 0:
            fdep = z_depth + (abs(z_depth - z_r) * 0.5) if abs(z_depth - z_r) > 2.0 else z_depth
            lines.append(
                f"CYCLE83({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {fdep:.3f}, , {peck:.3f}, {dwell:.2f}, 0.0, 1.0, 0)"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        elif cycle_type in ("G82", "CYCLE82") and dwell > 0:
            lines.append(
                f"CYCLE82({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f})"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        elif cycle_type in ("G85", "CYCLE85"):
            lines.append(
                f"CYCLE85({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f}, {feed:.1f}, {feed:.1f})"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        elif cycle_type in ("G86", "CYCLE86"):
            lines.append(
                f"CYCLE86({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f}, , {dwell:.2f}, 3)"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        else:
            lines.append(
                f"CYCLE81({rtp:.3f}, {rfp:.3f}, {sdis:.3f}, {dp:.3f})"
            )
            lines.append(f"G0 X{self._fmt(x)} Y{self._fmt(y)}")
        self.current_pos = np.array([x, y, z_r])
        self.canned_cycle_active = True
        self._cycle_op_id = ""
        self._cycle_r_plane = z_r
        self.is_pos_established = True
        return lines

    def _render_cycle_cancel(self, cmd: CycleCancel) -> list[str]:
        self.canned_cycle_active = False
        self._cycle_op_id = None
        return ["MCALL"]

    def _render_safe_return(self, cmd: SafeReturn) -> list[str]:
        if cmd.axis == "z":
            return ["SUPA G0 Z0.0 D0"]
        elif cmd.axis == "xy":
            return ["SUPA G0 X0.0 Y0.0"]
        return []

    def _render_program_end(self, cmd: ProgramEnd) -> list[str]:
        return ["M30"]

    def _render_machine_command(self, cmd: MachineCommand) -> list[str]:
        if cmd.name == "hsm_mode":
            mode = dict(cmd.params).get("mode", "")
            if self.hsm_enabled and mode:
                tol = 0.05 if mode == "roughing" else (0.02 if mode == "medium" else 0.005)
                mode_code = 1 if mode == "roughing" else (2 if mode == "medium" else 3)
                return [f"CYCLE832({tol:.4f}, {mode_code}, 1) ; High-Speed Settings {mode.upper()}"]
        elif cmd.name == "hsm_reset":
            if self.hsm_enabled and self.current_hsm_mode is not None:
                return ["CYCLE832() ; Reset High-Speed Settings"]
        return []


SinumerikPostProcessor = SiemensPostProcessor'''

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('cam_engine/post/post_processor.py', 'w') as f:
    f.write(new_content)

print('Added Siemens renderers inside class')