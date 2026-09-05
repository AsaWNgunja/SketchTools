class SketchToolBase:
    """
    Base class for all SketchTools tools.

    Every tool inherits:
    - start()
    - finish()
    - cancel()
    - mouse events
    - keyboard events
    - preview drawing
    """

    tool_name = "Tool"

    def __init__(self):

        self.running = False

        self.mouse_x = 0
        self.mouse_y = 0

        self.start_point = None
        self.current_point = None

        self.preview_data = {}

    # -------------------------------------------------
    # Lifecycle
    # -------------------------------------------------

    def start(self, context):

        self.running = True

        self.set_status(
            context,
            f"{self.tool_name} active"
        )

    def finish(self, context):

        self.running = False

        self.clear_status(context)

    def cancel(self, context):

        self.running = False

        self.clear_status(context)

    # -------------------------------------------------
    # Mouse Events
    # -------------------------------------------------

    def on_mouse_move(self, context, event):

        self.mouse_x = event.mouse_region_x
        self.mouse_y = event.mouse_region_y

    def on_left_click(self, context, event):

        pass

    def on_left_release(self, context, event):

        pass

    def on_right_click(self, context, event):

        pass

    # -------------------------------------------------
    # Keyboard
    # -------------------------------------------------

    def on_key_press(self, context, event):

        pass

    # -------------------------------------------------
    # Drawing
    # -------------------------------------------------

    def draw_preview(self, context):

        pass

    # -------------------------------------------------
    # Status Bar
    # -------------------------------------------------

    def set_status(self, context, text):

        workspace = context.workspace

        if hasattr(workspace, "status_text_set"):
            workspace.status_text_set(text)

        # v94: force Blender's Status Bar to repaint immediately.  Without
        # this, modal keyboard input can update Workspace.status_text_set()
        # internally while the visible Status Bar keeps showing the previous
        # workspace-tool hint until another UI event causes a redraw.
        screen = getattr(context, "screen", None)
        if screen is not None:
            for area in screen.areas:
                if area.type in {'STATUSBAR', 'VIEW_3D'}:
                    try:
                        area.tag_redraw()
                    except Exception:
                        pass

    def clear_status(self, context):

        workspace = context.workspace

        if hasattr(workspace, "status_text_set"):
            workspace.status_text_set(None)