from ..utils.logging import debug_print
import bpy

from .state import ToolState
from ..utils.cursor import set_active_tool_cursor


class ToolManager:

    def _redraw_view3d(self, context):
        try:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        except Exception:
            pass

    WORKSPACE_TOOL_IDS = {
        'OBJECT': {
            'line': "sketchtools.line_tool",
            'rectangle': "sketchtools.rectangle_tool",
            'circle': "sketchtools.circle_tool",
            'freehand': "sketchtools.freehand_tool",
            'eraser': "sketchtools.eraser_tool",
        },
        'EDIT_MESH': {
            'line': "sketchtools.line_tool_edit_mesh",
            'rectangle': "sketchtools.rectangle_tool_edit_mesh",
            'circle': "sketchtools.circle_tool_edit_mesh",
            'freehand': "sketchtools.freehand_tool_edit_mesh",
            'eraser': "sketchtools.eraser_tool_edit_mesh",
        },
    }

    ALL_WORKSPACE_TOOL_IDS = {
        tool_id
        for mode_map in WORKSPACE_TOOL_IDS.values()
        for tool_id in mode_map.values()
    }

    def __init__(self):
        self.state = ToolState()
        self.previous_blender_tool_id = None
        self.claimed_workspace_tool_id = None

    def _tool_key(self, tool):
        name = getattr(tool, "tool_name", type(tool).__name__).lower()
        for key in ("rectangle", "freehand", "eraser", "circle", "line"):
            if key in name:
                return key
        return "line"

    def workspace_tool_id_for_context(self, context, tool=None):
        mode_map = self.WORKSPACE_TOOL_IDS.get(
            context.mode,
            self.WORKSPACE_TOOL_IDS['OBJECT'],
        )
        key = self._tool_key(tool) if tool is not None else "line"
        return mode_map.get(key, mode_map["line"])

    # -------------------------------------
    # Blender workspace-tool helpers
    # -------------------------------------

    def get_blender_tool(self, context):
        try:
            tool = context.workspace.tools.from_space_view3d_mode(
                context.mode,
                create=False,
            )
            return tool.idname if tool is not None else None
        except Exception as exc:
            debug_print("SketchTools: unable to read Blender tool:", exc)
            return None

    def set_blender_tool(self, context, tool_id):
        if not tool_id:
            return False

        try:
            bpy.ops.wm.tool_set_by_id(
                name=tool_id,
                space_type='VIEW_3D',
            )
            return self.get_blender_tool(context) == tool_id
        except Exception as exc:
            debug_print(
                "SketchTools: unable to set Blender tool",
                tool_id,
                ":",
                exc,
            )
            return False

    def claim_blender_tool(self, context, tool=None):
        current = self.get_blender_tool(context)
        target = self.workspace_tool_id_for_context(context, tool)

        if current not in self.ALL_WORKSPACE_TOOL_IDS:
            self.previous_blender_tool_id = current

        self.claimed_workspace_tool_id = target

        if current == target:
            return True

        return self.set_blender_tool(context, target)

    def blender_tool_changed(self, context):
        if self.state.active_tool is None:
            return False

        current = self.get_blender_tool(context)

        if current is None:
            return False

        expected = self.claimed_workspace_tool_id
        if expected is None:
            expected = self.workspace_tool_id_for_context(
                context,
                self.state.active_tool,
            )
            self.claimed_workspace_tool_id = expected

        return current != expected

    def restore_previous_blender_tool(self, context):
        previous = self.previous_blender_tool_id
        self.previous_blender_tool_id = None
        self.claimed_workspace_tool_id = None

        if not previous or previous in self.ALL_WORKSPACE_TOOL_IDS:
            return

        self.set_blender_tool(context, previous)

    # -------------------------------------
    # Activate SketchTool
    # -------------------------------------

    def activate(self, tool, context):
        if self.state.active_tool is not None:
            self.state.active_tool.cancel(context)

        if not self.claim_blender_tool(context, tool):
            debug_print(
                "SketchTools: warning - workspace tool ownership "
                "could not be confirmed"
            )

        self.state.active_tool = tool
        set_active_tool_cursor(tool)
        tool.start(context)
        self._redraw_view3d(context)

    def deactivate(self, context, restore_blender_tool=True):
        if self.state.active_tool is not None:
            self.state.active_tool.finish(context)
            self.state.active_tool = None

        self._redraw_view3d(context)

        if restore_blender_tool:
            self.restore_previous_blender_tool(context)
        else:
            self.previous_blender_tool_id = None
            self.claimed_workspace_tool_id = None

    def cancel(self, context, restore_blender_tool=True):
        if self.state.active_tool is not None:
            self.state.active_tool.cancel(context)
            self.state.active_tool = None

        self._redraw_view3d(context)

        if restore_blender_tool:
            self.restore_previous_blender_tool(context)
        else:
            self.previous_blender_tool_id = None
            self.claimed_workspace_tool_id = None

    @property
    def active_tool(self):
        return self.state.active_tool
