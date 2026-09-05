import bpy

from ..core.session import tool_manager
from ..tools.eraser import EraserTool


class SketchToolsEraserOperator(bpy.types.Operator):
    bl_idname = "sketchtools.eraser"
    bl_label = "Eraser"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "SketchTools Eraser must be started from a 3D View")
            return {'CANCELLED'}

        if context.mode not in {'OBJECT', 'EDIT_MESH'}:
            self.report({'INFO'}, "Eraser supports Object Mode and Mesh Edit Mode")
            return {'CANCELLED'}

        tool_manager.activate(EraserTool(), context)

        window_region = next(
            (region for region in context.area.regions if region.type == 'WINDOW'),
            None,
        )
        if window_region is None:
            tool_manager.cancel(context)
            return {'CANCELLED'}

        with context.temp_override(area=context.area, region=window_region):
            bpy.ops.sketchtools.modal_controller('INVOKE_DEFAULT')

        return {'FINISHED'}
