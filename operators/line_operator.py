import bpy

from ..core.session import tool_manager
from ..tools.line import LineTool


class SketchToolsLineOperator(
    bpy.types.Operator
):

    bl_idname = "sketchtools.line"
    bl_label = "Line"

    bl_options = {
        'REGISTER',
        'UNDO'
    }

    def invoke(
        self,
        context,
        event
    ):

        if context.area is None or context.area.type != 'VIEW_3D':
            self.report(
                {'WARNING'},
                "SketchTools Line must be started from a 3D View"
            )
            return {
                'CANCELLED'
            }

        # Activate the Line tool.  ToolManager first makes the dedicated
        # SketchTools workspace tool active so a later click on Select Box,
        # Move, Rotate, Scale, Cursor, etc. can be detected explicitly.
        tool = LineTool()
        tool_manager.activate(
            tool,
            context
        )

        # A panel button is invoked from the View3D's UI region.  The modal
        # controller, however, must own the View3D WINDOW region so ray-casting
        # and event.mouse_region_x/y use the actual viewport coordinate system.
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
            self.report(
                {'WARNING'},
                "SketchTools could not find the 3D View window region"
            )
            return {
                'CANCELLED'
            }

        with context.temp_override(
            area=context.area,
            region=window_region,
        ):
            bpy.ops.sketchtools.modal_controller(
                'INVOKE_DEFAULT'
            )

        return {
            'FINISHED'
        }
