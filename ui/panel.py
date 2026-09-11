import bpy


class SketchToolsPanel(bpy.types.Panel):
    bl_label = "SketchTools Free"
    bl_idname = "VIEW3D_PT_sketchtools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SketchTools"

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Tool Settings")

        box.prop(
            context.scene,
            "sketchtools_circle_segments",
            text="Circle Segments",
        )

        box.prop(
            context.scene,
            "sketchtools_circle_radius",
            text="Circle Radius",
        )

        box.prop(
            context.scene,
            "sketchtools_freehand_spacing",
            text="Freehand Spacing",
        )

        box.separator()
        box.label(text="Tools are in the left toolbar")
