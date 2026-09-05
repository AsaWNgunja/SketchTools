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


# ---------------------------------------------------------
# Object Mode
# ---------------------------------------------------------

class SketchToolsLineWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.line_tool'
    bl_label = 'SketchTools Line'
    bl_description = 'Draw connected SketchTools line segments'
    bl_icon = _icon_path("line")
    bl_keymap = (
        ("sketchtools.line", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )


class SketchToolsRectangleWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.rectangle_tool'
    bl_label = 'SketchTools Rectangle'
    bl_description = 'Draw SketchTools rectangles'
    bl_icon = _icon_path("rectangle")
    bl_keymap = (
        ("sketchtools.rectangle", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )


class SketchToolsCircleWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.circle_tool'
    bl_label = 'SketchTools Circle'
    bl_description = 'Draw SketchTools circles'
    bl_icon = _icon_path("circle")
    bl_keymap = (
        ("sketchtools.circle", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )

    @staticmethod
    def draw_settings(context, layout, tool):
        _draw_circle_settings(context, layout, tool)


class SketchToolsFreehandWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.freehand_tool'
    bl_label = 'SketchTools Freehand'
    bl_description = 'Draw smooth freehand geometry'
    bl_icon = _icon_path("freehand")
    bl_keymap = (
        ("sketchtools.freehand", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )

    @staticmethod
    def draw_settings(context, layout, tool):
        _draw_freehand_settings(context, layout, tool)


class SketchToolsEraserWorkspaceTool(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'OBJECT'
    bl_idname = 'sketchtools.eraser_tool'
    bl_label = 'SketchTools Eraser'
    bl_description = 'Erase SketchTools mesh vertices, edges and faces'
    bl_icon = _icon_path("eraser")
    bl_keymap = (
        ("sketchtools.eraser", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )


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
    bl_keymap = (
        ("sketchtools.line", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )


class SketchToolsRectangleWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.rectangle_tool_edit_mesh'
    bl_label = 'SketchTools Rectangle'
    bl_description = 'Draw SketchTools rectangles and cut mesh faces'
    bl_icon = _icon_path("rectangle")
    bl_keymap = (
        ("sketchtools.rectangle", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )


class SketchToolsCircleWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.circle_tool_edit_mesh'
    bl_label = 'SketchTools Circle'
    bl_description = 'Draw SketchTools circles and cut mesh faces'
    bl_icon = _icon_path("circle")
    bl_keymap = (
        ("sketchtools.circle", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )

    @staticmethod
    def draw_settings(context, layout, tool):
        _draw_circle_settings(context, layout, tool)


class SketchToolsFreehandWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.freehand_tool_edit_mesh'
    bl_label = 'SketchTools Freehand'
    bl_description = 'Draw smooth freehand cuts on mesh faces'
    bl_icon = _icon_path("freehand")
    bl_keymap = (
        ("sketchtools.freehand", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )

    @staticmethod
    def draw_settings(context, layout, tool):
        _draw_freehand_settings(context, layout, tool)


class SketchToolsEraserWorkspaceToolEditMesh(_SketchToolsToolbarBase, bpy.types.WorkSpaceTool):
    bl_context_mode = 'EDIT_MESH'
    bl_idname = 'sketchtools.eraser_tool_edit_mesh'
    bl_label = 'SketchTools Eraser'
    bl_description = 'Erase mesh vertices, edges and faces'
    bl_icon = _icon_path("eraser")
    bl_keymap = (
        ("sketchtools.eraser", {"type": 'MOUSEMOVE', "value": 'ANY'}, None),
    )


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
