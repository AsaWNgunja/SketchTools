from dataclasses import dataclass
from mathutils import Vector


@dataclass
class SelectionResult:
    """
    Stores everything the engine knows
    about the current mouse position.
    """

    hit: bool = False

    point: Vector | None = None

    normal: Vector | None = None

    obj = None

    face = None

    edge = None

    vertex = None

    snap_type: str = "NONE"


class SelectionManager:
    """
    Central selection system.

    Every SketchTool asks THIS class
    where the mouse is.
    """

    def __init__(self):

        self.result = SelectionResult()


    def clear(self):

        self.result = SelectionResult()


    def update(
        self,
        context,
        event
    ):
        """
        This will later perform
        raycasting and snapping.

        For now it simply clears
        the previous result.
        """

        self.clear()

        return self.result