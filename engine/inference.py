from mathutils import Vector


class InferenceResult:

    """
    Stores the result of an inference test.
    """

    def __init__(
        self,
        location=None,
        inference_type=None,
        element=None
    ):

        self.location = location
        self.type = inference_type
        self.element = element



class InferenceEngine:

    """
    SketchUp-style inference system.

    Finds the best snapping target
    near the cursor.
    """


    def __init__(self):

        self.current = None


    # ---------------------------------
    # Find best snap
    # ---------------------------------

    def find_snap(
        self,
        context,
        location,
        objects
    ):

        """
        location:
            3D cursor position candidate

        objects:
            objects being tested
        """


        # Placeholder for now

        self.current = InferenceResult(
            location=location,
            inference_type="NONE"
        )


        return self.current


    # ---------------------------------
    # Clear
    # ---------------------------------

    def clear(self):

        self.current = None