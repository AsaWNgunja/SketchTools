import blf


class SnapFeedbackMixin:
    def _init_snap_feedback(self):
        self.snap_label = None
        self.snap_mouse = None
        self.measurement_label = None
        self.measurement_mouse = None

    def set_snap_feedback(self, label, mouse_x=None, mouse_y=None):
        self.snap_label = label
        if mouse_x is not None and mouse_y is not None:
            self.snap_mouse = (float(mouse_x), float(mouse_y))

    def clear_snap_feedback(self):
        self.snap_label = None
        self.snap_mouse = None

    def set_measurement_feedback(self, label, mouse_x=None, mouse_y=None):
        self.measurement_label = label
        if mouse_x is not None and mouse_y is not None:
            self.measurement_mouse = (float(mouse_x), float(mouse_y))

    def clear_measurement_feedback(self):
        self.measurement_label = None
        self.measurement_mouse = None

    def draw_label(self):
        font_id = 0
        blf.size(font_id, 14.0)

        if self.snap_label and self.snap_mouse is not None:
            x, y = self.snap_mouse
            blf.position(font_id, x + 18.0, y + 14.0, 0.0)
            blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
            blf.draw(font_id, self.snap_label)

        if self.measurement_label and self.measurement_mouse is not None:
            x, y = self.measurement_mouse
            # Keep measurements below the snap hint so both remain readable.
            blf.position(font_id, x + 18.0, y - 8.0, 0.0)
            blf.color(font_id, 1.0, 1.0, 0.65, 1.0)
            blf.draw(font_id, self.measurement_label)
