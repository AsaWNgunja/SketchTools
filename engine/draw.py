import bpy
import gpu

from gpu_extras.batch import batch_for_shader


class SketchToolsDraw:

    """
    Handles temporary viewport drawing.

    Used for:
    - preview lines
    - preview arcs
    - snap markers
    - guides
    - measurements
    """


    def __init__(self):

        self.handler = None

        self.draw_items = []


    # ---------------------------------
    # Register drawing callback
    # ---------------------------------

    def start(self):

        if self.handler:
            return


        self.handler = bpy.types.SpaceView3D.draw_handler_add(
            self.draw,
            (),
            'WINDOW',
            'POST_VIEW'
        )


    # ---------------------------------
    # Remove callback
    # ---------------------------------

    def stop(self):

        if self.handler:

            bpy.types.SpaceView3D.draw_handler_remove(
                self.handler,
                'WINDOW'
            )

            self.handler = None


    # ---------------------------------
    # Add preview item
    # ---------------------------------

    def add_line(self, start, end):

        self.draw_items.append(
            {
                "type": "LINE",
                "start": start,
                "end": end
            }
        )


    # ---------------------------------
    # Clear previews
    # ---------------------------------

    def clear(self):

        self.draw_items.clear()


    # ---------------------------------
    # Draw callback
    # ---------------------------------

    def draw(self):

        if not self.draw_items:
            return


        shader = gpu.shader.from_builtin(
            '3D_UNIFORM_COLOR'
        )


        for item in self.draw_items:


            if item["type"] == "LINE":

                coords = [
                    item["start"],
                    item["end"]
                ]


                batch = batch_for_shader(
                    shader,
                    'LINES',
                    {
                        "pos": coords
                    }
                )


                shader.bind()

                shader.uniform_float(
                    "color",
                    (1, 1, 1, 1)
                )


                batch.draw(shader)