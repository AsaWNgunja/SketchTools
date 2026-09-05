import gpu

from gpu_extras.batch import batch_for_shader
from .snap_feedback import SnapFeedbackMixin


class PreviewLine(SnapFeedbackMixin):

    def __init__(self):

        self.start = None
        self.end = None

        self.snap_point = None

        # v94 equal-length inference guide.  These are world-space points so
        # the guide remains anchored correctly while the view changes.
        self.equal_reference_start = None
        self.equal_reference_end = None
        self.equal_current_mid = None
        self.equal_reference_mid = None

        self._init_snap_feedback()


    def set_points(
        self,
        start,
        end
    ):

        self.start = start
        self.end = end


    def set_snap_point(
        self,
        point
    ):

        self.snap_point = point


    def clear_snap_point(self):

        self.snap_point = None
        self.clear_snap_feedback()


    def set_equal_length_guide(self, reference_start, reference_end, current_start, current_end):
        self.equal_reference_start = reference_start.copy()
        self.equal_reference_end = reference_end.copy()
        self.equal_reference_mid = (reference_start + reference_end) * 0.5
        self.equal_current_mid = (current_start + current_end) * 0.5

    def clear_equal_length_guide(self):
        self.equal_reference_start = None
        self.equal_reference_end = None
        self.equal_current_mid = None
        self.equal_reference_mid = None

    def draw(self):

        shader = gpu.shader.from_builtin(
            "UNIFORM_COLOR"
        )


        # ---------------------------------
        # Draw preview line
        # ---------------------------------

        if (
            self.start is not None
            and self.end is not None
        ):

            line_batch = batch_for_shader(
                shader,
                "LINES",
                {
                    "pos": [
                        self.start,
                        self.end
                    ]
                }
            )

            gpu.state.line_width_set(
                3.0
            )

            shader.bind()

            shader.uniform_float(
                "color",
                (
                    1.0,
                    1.0,
                    0.0,
                    1.0
                )
            )

            line_batch.draw(
                shader
            )

            gpu.state.line_width_set(
                1.0
            )


        # ---------------------------------
        # v94 equal-length inference guide
        # ---------------------------------

        if (
            self.equal_reference_start is not None
            and self.equal_reference_end is not None
            and self.equal_current_mid is not None
            and self.equal_reference_mid is not None
        ):
            guide_batch = batch_for_shader(
                shader,
                "LINES",
                {
                    "pos": [
                        self.equal_reference_start, self.equal_reference_end,
                        self.equal_reference_mid, self.equal_current_mid,
                    ]
                }
            )

            gpu.state.line_width_set(2.0)
            shader.bind()
            # Magenta-style inference guide, matching the visual language used
            # by SketchUp for relationship/inference feedback.
            shader.uniform_float("color", (1.0, 0.15, 1.0, 1.0))
            guide_batch.draw(shader)
            gpu.state.line_width_set(1.0)

        # ---------------------------------
        # Draw snap marker
        # ---------------------------------

        if self.snap_point is not None:

            point_batch = batch_for_shader(
                shader,
                "POINTS",
                {
                    "pos": [
                        self.snap_point
                    ]
                }
            )

            gpu.state.point_size_set(
                10.0
            )

            shader.bind()

            shader.uniform_float(
                "color",
                (
                    1.0,
                    1.0,
                    0.0,
                    1.0
                )
            )

            point_batch.draw(
                shader
            )

            gpu.state.point_size_set(
                1.0
            )


    def clear(self):

        self.start = None
        self.end = None
        self.snap_point = None
        self.clear_snap_feedback()
        self.clear_measurement_feedback()