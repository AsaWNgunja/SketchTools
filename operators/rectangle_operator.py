import bpy
from ..core.session import tool_manager
from ..tools.rectangle import RectangleTool


class SketchToolsRectangleOperator(bpy.types.Operator):
    bl_idname = "sketchtools.rectangle"
    bl_label = "Rectangle"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            return {'CANCELLED'}

        tool_manager.activate(RectangleTool(), context)

        window_region = next(
            (r for r in context.area.regions if r.type == 'WINDOW'),
            None,
        )
        if window_region is None:
            tool_manager.cancel(context)
            return {'CANCELLED'}

        with context.temp_override(area=context.area, region=window_region):
            bpy.ops.sketchtools.modal_controller('INVOKE_DEFAULT')

        return {'FINISHED'}
