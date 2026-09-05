class SketchToolsInput:

    """
    Handles typed numeric input.

    Examples:
    2500
    45
    12

    Tools decide what the value means.
    """


    def __init__(self):

        self.buffer = ""


    # ---------------------------------
    # Add keyboard character
    # ---------------------------------

    def add_character(self, char):

        self.buffer += char


    # ---------------------------------
    # Remove last character
    # ---------------------------------

    def backspace(self):

        self.buffer = self.buffer[:-1]


    # ---------------------------------
    # Clear input
    # ---------------------------------

    def clear(self):

        self.buffer = ""


    # ---------------------------------
    # Get numeric value
    # ---------------------------------

    def get_value(self):

        if self.buffer == "":
            return None


        try:

            return float(self.buffer)

        except ValueError:

            return None