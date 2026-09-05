from .logging import debug_print
import ctypes
import os


_cursor_handles = {}
_active_cursor_name = "line"


def _user32():
    try:
        return ctypes.windll.user32
    except Exception:
        return None


def _normalize_tool_name(tool):
    if tool is None:
        return "line"

    if isinstance(tool, str):
        name = tool
    else:
        name = tool.__class__.__name__

    name = name.lower()
    if "rectangle" in name:
        return "rectangle"
    if "circle" in name:
        return "circle"
    if "freehand" in name:
        return "freehand"
    if "eraser" in name:
        return "eraser"
    return "line"


def set_active_tool_cursor(tool):
    """Select the cursor family used by the currently active SketchTool."""
    global _active_cursor_name
    _active_cursor_name = _normalize_tool_name(tool)
    return show_pencil_cursor()


def load_tool_cursor(name=None):
    """Load one SketchTools cursor once and cache its Windows handle."""
    name = _normalize_tool_name(name or _active_cursor_name)

    if _cursor_handles.get(name):
        return _cursor_handles[name]

    cursor_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "cursors",
        name + ".cur",
    )

    # Safe fallback for old/custom packages.
    if not os.path.exists(cursor_path):
        cursor_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "cursors",
            "pencil.cur",
        )

    if not os.path.exists(cursor_path):
        debug_print("SketchTools: cursor not found:", cursor_path)
        return None

    user32 = _user32()
    if user32 is None:
        return None

    try:
        user32.LoadCursorFromFileW.argtypes = [ctypes.c_wchar_p]
        user32.LoadCursorFromFileW.restype = ctypes.c_void_p
        user32.SetCursor.argtypes = [ctypes.c_void_p]
        user32.SetCursor.restype = ctypes.c_void_p

        handle = user32.LoadCursorFromFileW(cursor_path)
        if not handle:
            debug_print("SketchTools: failed to load cursor:", cursor_path)
            return None

        _cursor_handles[name] = handle
        return handle
    except Exception as exc:
        debug_print("SketchTools cursor error:", exc)
        return None


def load_pencil_cursor():
    # Backwards-compatible API.
    return bool(load_tool_cursor())


def show_pencil_cursor():
    """Show the cursor belonging to the active SketchTool."""
    handle = load_tool_cursor()
    if not handle:
        return False

    user32 = _user32()
    if user32 is None:
        return False

    try:
        user32.SetCursor(handle)
        return True
    except Exception:
        return False


def set_pencil_cursor():
    return show_pencil_cursor()


def refresh_pencil_cursor():
    return show_pencil_cursor()


def show_blender_cursor(context):
    try:
        context.window.cursor_set("DEFAULT")
    except Exception:
        pass


def restore_cursor(context):
    """Restore Blender's cursor without discarding cached custom cursors."""
    show_blender_cursor(context)
