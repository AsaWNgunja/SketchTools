import bpy

from ..core.session import tool_manager
from ..tools.circle import CircleTool, rebuild_parametric_circle, circle_is_pristine


class SketchToolsCircleOperator(bpy.types.Operator):
    bl_idname = "sketchtools.circle"
    bl_label = "Circle"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "SketchTools Circle must be started from a 3D View")
            return {'CANCELLED'}

        if context.mode not in {'OBJECT', 'EDIT_MESH'}:
            self.report({'INFO'}, "Circle supports Object Mode and Mesh Edit Mode")
            return {'CANCELLED'}

        segments = getattr(context.scene, "sketchtools_circle_segments", 32)
        tool_manager.activate(CircleTool(segments=segments), context)

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



class SketchToolsCircleSegmentsOperator(bpy.types.Operator):
    bl_idname = "sketchtools.circle_set_segments"
    bl_label = "Update Circle Segments"
    bl_options = {'REGISTER', 'UNDO'}

    segments: bpy.props.IntProperty(name="Segments", default=32, min=3, max=256)

    def execute(self, context):
        obj = context.active_object
        if obj is None or not obj.get("sketchtools_circle"):
            self.report({'INFO'}, "Select a SketchTools circle")
            return {'CANCELLED'}

        if not circle_is_pristine(obj):
            obj["sketchtools_circle_pristine"] = False
            self.report({'WARNING'}, "Circle geometry was manually edited; automatic rebuild disabled")
            return {'CANCELLED'}

        if rebuild_parametric_circle(obj, self.segments):
            context.scene.sketchtools_circle_segments = self.segments
            self.report({'INFO'}, f"Circle changed to {self.segments} segments")
            return {'FINISHED'}

        return {'CANCELLED'}
