__build__ = 93
__edition__ = "Free"

bl_info = {
    "name": "SketchTools Free",
    "author": "Asa W Ngunja",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": (
        "View3D > Toolbar > SketchTools"
    ),
    "description": (
        "SketchUp-style drawing tools for Blender"
    ),
    "category": "3D View",
}


import bpy
from .core.session import tool_manager


from .ui.panel import SketchToolsPanel
from .core.workspace_tool import (
    OBJECT_WORKSPACE_TOOLS,
    EDIT_WORKSPACE_TOOLS,
    ALL_WORKSPACE_TOOLS,
)

from .operators.modal_controller import (
    SketchToolsModalController
)

from .operators.line_operator import (
    SketchToolsLineOperator
)

from .operators.rectangle_operator import (
    SketchToolsRectangleOperator
)

from .operators.circle_operator import (
    SketchToolsCircleOperator,
    SketchToolsCircleSegmentsOperator,
)


from .operators.freehand_operator import (
    SketchToolsFreehandOperator
)

from .operators.eraser_operator import (
    SketchToolsEraserOperator
)


# -------------------------------------
# Normal Blender classes
# -------------------------------------

classes = (

    SketchToolsPanel,

    SketchToolsModalController,

    SketchToolsLineOperator,

    SketchToolsRectangleOperator,

    SketchToolsCircleOperator,

    SketchToolsCircleSegmentsOperator,

    SketchToolsFreehandOperator,

    SketchToolsEraserOperator,

)



def _circle_segments_update(scene, context):
    """v93: changing Circle Segments also updates the selected pristine circle."""
    try:
        obj = context.active_object if context is not None else None
        if obj is None or obj.type != 'MESH' or not obj.get("sketchtools_circle"):
            return
        from .tools.circle import rebuild_parametric_circle, circle_is_pristine
        if circle_is_pristine(obj):
            rebuild_parametric_circle(obj, int(scene.sketchtools_circle_segments))
    except Exception:
        # Tool setting must remain usable even if the selected object cannot be rebuilt.
        pass


# -------------------------------------
# Register
# -------------------------------------

def register():

    bpy.types.Scene.sketchtools_freehand_spacing = bpy.props.IntProperty(
        name="Freehand Spacing",
        description="Minimum screen-pixel movement before another freehand point is sampled",
        default=5,
        min=1,
        max=50,
    )

    bpy.types.Scene.sketchtools_circle_segments = bpy.props.IntProperty(
        name="Circle Segments",
        description="Number of edges used by newly created SketchTools circles",
        default=32,
        min=3,
        max=256,
        update=_circle_segments_update,
    )

    for cls in classes:

        bpy.utils.register_class(
            cls
        )


    # Native Blender left-toolbar integration.
    # Register Line as a separated entry, then keep the remaining Free tools
    # together directly below it.
    previous_id = None
    for index, tool_cls in enumerate(OBJECT_WORKSPACE_TOOLS):
        kwargs = {}
        if index == 0:
            kwargs["separator"] = True
        elif previous_id:
            kwargs["after"] = {previous_id}
        bpy.utils.register_tool(tool_cls, **kwargs)
        previous_id = tool_cls.bl_idname

    previous_id = None
    for index, tool_cls in enumerate(EDIT_WORKSPACE_TOOLS):
        kwargs = {}
        if index == 0:
            kwargs["separator"] = True
        elif previous_id:
            kwargs["after"] = {previous_id}
        bpy.utils.register_tool(tool_cls, **kwargs)
        previous_id = tool_cls.bl_idname


# -------------------------------------
# Unregister
# -------------------------------------

def unregister():

    # Stop modal GPU/UI callbacks before unregistering workspace tools/RNA classes.
    try: SketchToolsModalController.shutdown_all()
    except Exception: pass
    try:
        active = tool_manager.active_tool
        if active is not None:
            try: active.cancel(bpy.context)
            except Exception: pass
        tool_manager.state.active_tool = None
        tool_manager.previous_blender_tool_id = None
        tool_manager.claimed_workspace_tool_id = None
    except Exception: pass

    if hasattr(bpy.types.Scene, "sketchtools_freehand_spacing"):
        del bpy.types.Scene.sketchtools_freehand_spacing

    if hasattr(bpy.types.Scene, "sketchtools_circle_segments"):
        del bpy.types.Scene.sketchtools_circle_segments

    for tool_cls in reversed(ALL_WORKSPACE_TOOLS):
        try:
            bpy.utils.unregister_tool(tool_cls)
        except Exception:
            pass



    for cls in reversed(
        classes
    ):

        bpy.utils.unregister_class(
            cls
        )
