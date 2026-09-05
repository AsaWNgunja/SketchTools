"""SketchTools release logging.

Debug output is disabled in public builds by default.  Set DEBUG = True
while diagnosing a local development build.
"""

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
