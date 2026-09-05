import bpy

from ..core.session import tool_manager


class SketchToolsEventHandler:

    """
    Central event dispatcher for SketchTools.

    Sends Blender events to the active SketchTool.
    """


    def handle_event(self, context, event):

        tool = tool_manager.active_tool

        if tool is None:
            return


        # ----------------------------
        # Mouse movement
        # ----------------------------

        if event.type == 'MOUSEMOVE':

            tool.on_mouse_move(
                context,
                event
            )


        # ----------------------------
        # Left click
        # ----------------------------

        elif event.type == 'LEFTMOUSE':

            if event.value == 'PRESS':

                tool.on_left_click(
                    context,
                    event
                )

            elif event.value == 'RELEASE' and hasattr(
                tool,
                "on_left_release"
            ):

                tool.on_left_release(
                    context,
                    event
                )


        # ----------------------------
        # Right click
        # ----------------------------

        elif event.type == 'RIGHTMOUSE':

            if event.value == 'PRESS':

                tool.on_right_click(
                    context,
                    event
                )


        # ----------------------------
        # Keyboard
        # ----------------------------

        elif event.value == 'PRESS':

            tool.on_key_press(
                context,
                event
            )


        # ----------------------------
        # ESC cancellation
        # ----------------------------

        if event.type == 'ESC':

            tool_manager.cancel(
                context
            )