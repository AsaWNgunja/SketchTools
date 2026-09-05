import bpy

from ..core.session import tool_manager
from ..tools.freehand import FreehandTool


class SketchToolsFreehandOperator(bpy.types.Operator):
    bl_idname = "sketchtools.freehand"
    bl_label = "Freehand"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report(
                {'WARNING'},
                "SketchTools Freehand must be started from a 3D View"
            )
            return {'CANCELLED'}

        if context.mode not in {'OBJECT', 'EDIT_MESH'}:
            self.report(
                {'INFO'},
                "Freehand supports Object Mode and Mesh Edit Mode"
            )
            return {'CANCELLED'}

        spacing = getattr(
            context.scene,
            "sketchtools_freehand_spacing",
            5,
        )

        tool_manager.activate(
            FreehandTool(spacing_px=spacing),
            context,
        )

        window_region = next(
            (
                region
                for region in context.area.regions
                if region.type == 'WINDOW'
            ),
            None,
        )

        if window_region is None:
            tool_manager.cancel(context)
            return {'CANCELLED'}

        with context.temp_override(
            area=context.area,
            region=window_region,
        ):
            bpy.ops.sketchtools.modal_controller('INVOKE_DEFAULT')

        return {'FINISHED'}
