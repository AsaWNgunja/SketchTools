import bpy
from pathlib import Path


_ICON_DIR = Path(__file__).resolve().parent.parent / "icons"

def _icon_path(name):
    # Blender WorkSpaceTool expects the custom geometry-icon path without ".dat".
    return (_ICON_DIR / name).as_posix()


def _draw_circle_settings(context, layout, tool):
    layout.prop(
        context.scene,
        "sketchtools_circle_segments",
        text="Segments",
    )


def _draw_freehand_settings(context, layout, tool):
    layout.prop(
        context.scene,
        "sketchtools_freehand_spacing",
        text="Spacing",
    )


class _SketchToolsToolbarBase:
    bl_space_type = 'VIEW_3D'
    bl_widget = None
    bl_cursor = 'DEFAULT'


_NATIVE_TOOL_KEYMAP = (
    ("sketchtools.native_event", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    ("sketchtools.native_event", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'RIGHTMOUSE', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'X', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'Y', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'Z', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'ESC', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'RET', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_ENTER', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'BACK_SPACE', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'DEL', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'ZERO', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'ONE', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'TWO', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'THREE', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'FOUR', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'FIVE', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'SIX', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'SEVEN', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'EIGHT', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NINE', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_0', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_1', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_2', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_3', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_4', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_5', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_6', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_7', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_8', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_9', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'PERIOD', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_PERIOD', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'MINUS', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'NUMPAD_MINUS', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'M', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'C', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'F', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'T', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'I', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'N', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'K', "value": 'PRESS'}, None),
    ("sketchtools.native_event", {"type": 'S', "value": 'PRESS'}, None),
)


# v167: Freehand is click-move-click/move, not press-drag-release.
# A first click starts the stroke; mouse movement samples continuously without
# holding LMB. Further clicks only finish when they acquire a valid vertex/edge
# (or the stroke start for Close Loop). Therefore Freehand uses the same
# press-only native event stream as the frozen tools.
_FREEHAND_NATIVE_TOOL_KEYMAP = _NATIVE_TOOL_KEYMAP


# ---------------------------------------------------------
# Object Mode
# ---------------------------------------------------------

class SketchToolsLineWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.line_tool'
    bl_label = 'SketchTools Line'
    bl_description = 'Draw connected SketchTools line segments'
    bl_icon = _icon_path("line")
    # v118 prototype: Line uses one-shot WorkSpaceTool keymap events instead
    # of starting the shared persistent modal controller. Blender's own gizmo
    # handlers therefore retain first chance to own navigation gestures.
    bl_keymap = (
        ("sketchtools.line_native_event", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
        ("sketchtools.line_native_event", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'RIGHTMOUSE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'X', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'Y', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'Z', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'ESC', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'RET', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'NUMPAD_ENTER', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'BACK_SPACE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'DEL', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'ZERO', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'ONE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'TWO', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'THREE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'FOUR', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'FIVE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'SIX', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'SEVEN', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'EIGHT', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'NINE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'PERIOD', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'M', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'C', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'F', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'T', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'I', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'N', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'K', "value": 'PRESS'}, None),
    )


class SketchToolsRectangleWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.rectangle_tool'
    bl_label = 'SketchTools Rectangle'
    bl_description = 'Draw SketchTools rectangles'
    bl_icon = _icon_path("rectangle")
    bl_keymap = _NATIVE_TOOL_KEYMAP


class SketchToolsCircleWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.circle_tool'
    bl_label = 'SketchTools Circle'
    bl_description = 'Draw SketchTools circles'
    bl_icon = _icon_path("circle")
    bl_keymap = _NATIVE_TOOL_KEYMAP

    @staticmethod
    def draw_settings(context, layout, tool):
        _draw_circle_settings(context, layout, tool)


class SketchToolsFreehandWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.freehand_tool'
    bl_label = 'SketchTools Freehand'
    bl_description = 'Draw smooth freehand geometry'
    bl_icon = _icon_path("freehand")
    bl_keymap = _FREEHAND_NATIVE_TOOL_KEYMAP

    @staticmethod
    def draw_settings(context, layout, tool):
        _draw_freehand_settings(context, layout, tool)


class SketchToolsEraserWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.eraser_tool'
    bl_label = 'SketchTools Eraser'
    bl_description = 'Erase SketchTools mesh vertices, edges and faces'
    bl_icon = _icon_path("eraser")
    bl_keymap = _NATIVE_TOOL_KEYMAP


# ---------------------------------------------------------
# Mesh Edit Mode
# Blender stores a separate active toolbar tool per mode, so each Free tool
# needs an EDIT_MESH counterpart.
# ---------------------------------------------------------

class SketchToolsLineWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.line_tool_edit_mesh'
    bl_label = 'SketchTools Line'
    bl_description = 'Draw connected SketchTools line segments'
    bl_icon = _icon_path("line")
    # v118 prototype: Line uses one-shot WorkSpaceTool keymap events instead
    # of starting the shared persistent modal controller. Blender's own gizmo
    # handlers therefore retain first chance to own navigation gestures.
    bl_keymap = (
        ("sketchtools.line_native_event", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
        ("sketchtools.line_native_event", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'RIGHTMOUSE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'X', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'Y', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'Z', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'ESC', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'RET', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'NUMPAD_ENTER', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'BACK_SPACE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'DEL', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'ZERO', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'ONE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'TWO', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'THREE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'FOUR', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'FIVE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'SIX', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'SEVEN', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'EIGHT', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'NINE', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'PERIOD', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'M', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'C', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'F', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'T', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'I', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'N', "value": 'PRESS'}, None),
        ("sketchtools.line_native_event", {"type": 'K', "value": 'PRESS'}, None),
    )


class SketchToolsRectangleWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.rectangle_tool_edit_mesh'
    bl_label = 'SketchTools Rectangle'
    bl_description = 'Draw SketchTools rectangles and cut mesh faces'
    bl_icon = _icon_path("rectangle")
    bl_keymap = _NATIVE_TOOL_KEYMAP


class SketchToolsCircleWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.circle_tool_edit_mesh'
    bl_label = 'SketchTools Circle'
    bl_description = 'Draw SketchTools circles and cut mesh faces'
    bl_icon = _icon_path("circle")
    bl_keymap = _NATIVE_TOOL_KEYMAP

    @staticmethod
    def draw_settings(context, layout, tool):
        _draw_circle_settings(context, layout, tool)


class SketchToolsFreehandWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.freehand_tool_edit_mesh'
    bl_label = 'SketchTools Freehand'
    bl_description = 'Draw smooth freehand cuts on mesh faces'
    bl_icon = _icon_path("freehand")
    bl_keymap = _FREEHAND_NATIVE_TOOL_KEYMAP

    @staticmethod
    def draw_settings(context, layout, tool):
        _draw_freehand_settings(context, layout, tool)


class SketchToolsEraserWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.eraser_tool_edit_mesh'
    bl_label = 'SketchTools Eraser'
    bl_description = 'Erase mesh vertices, edges and faces'
    bl_icon = _icon_path("eraser")
    bl_keymap = _NATIVE_TOOL_KEYMAP


OBJECT_WORKSPACE_TOOLS = (
    SketchToolsLineWorkspaceTool,
    SketchToolsRectangleWorkspaceTool,
    SketchToolsCircleWorkspaceTool,
    SketchToolsFreehandWorkspaceTool,
    SketchToolsEraserWorkspaceTool,
)

EDIT_WORKSPACE_TOOLS = (
    SketchToolsLineWorkspaceToolEditMesh,
    SketchToolsRectangleWorkspaceToolEditMesh,
    SketchToolsCircleWorkspaceToolEditMesh,
    SketchToolsFreehandWorkspaceToolEditMesh,
    SketchToolsEraserWorkspaceToolEditMesh,
)

ALL_WORKSPACE_TOOLS = OBJECT_WORKSPACE_TOOLS + EDIT_WORKSPACE_TOOLS
