class ToolState:

    def __init__(self):
        self.active_tool = None

        self.snap_point = None

        self.hover_vertex = None
        self.hover_edge = None
        self.hover_face = None

        self.input_text = ""