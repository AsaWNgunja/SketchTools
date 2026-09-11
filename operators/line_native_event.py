"""v118 experimental native WorkSpaceTool event path for Line only.

The key architectural difference is that no persistent SketchTools modal operator
is installed for the Line tool. Blender's WorkSpaceTool keymap invokes this
operator for individual drawing events, then the operator immediately returns.
That leaves Blender's own gizmo system free to own navigation interactions before
the tool keymap sees them.
"""

import bpy
import gpu

from ..core.session import tool_manager
from ..tools.line import LineTool
from ..engine.selection import SelectionManager
from ..utils.cursor import show_pencil_cursor
from ..utils.logging import debug_print




def _line_workspace_is_active():
    try:
        context = bpy.context
        if context is None or context.workspace is None:
            return False
        wt = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
        wid = wt.idname if wt else None
        return wid in {'sketchtools.line_tool', 'sketchtools.line_tool_edit_mesh'}
    except Exception:
        return False

class _LineNativeDrawService:
    """GPU preview handlers for native-event Line without a modal controller."""

    draw_handler = None
    label_handler = None

    @classmethod
    def ensure(cls):
        if cls.draw_handler is None:
            cls.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                cls.draw_callback, (), 'WINDOW', 'POST_VIEW'
            )
        if cls.label_handler is None:
            cls.label_handler = bpy.types.SpaceView3D.draw_handler_add(
                cls.draw_label_callback, (), 'WINDOW', 'POST_PIXEL'
            )

    @classmethod
    def shutdown(cls):
        for attr in ("draw_handler", "label_handler"):
            handle = getattr(cls, attr, None)
            if handle is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
                except Exception:
                    pass
                setattr(cls, attr, None)

    @staticmethod
    def draw_callback():
        tool = tool_manager.active_tool
        if not isinstance(tool, LineTool) or not hasattr(tool, "preview"):
            return
        # v122: a cancelled/switched Line must never leave its GPU preview
        # behind while another Blender/SketchTools workspace tool is active.
        if not _line_workspace_is_active():
            try:
                tool.preview.clear()
            except Exception:
                pass
            return
        pushed = False
        try:
            gpu.matrix.push()
            pushed = True
            tool.preview.draw()
        except (ReferenceError, RuntimeError):
            return
        except Exception as exc:
            debug_print("SketchTools v118 Line draw skipped:", exc)
        finally:
            if pushed:
                try:
                    gpu.matrix.pop()
                except Exception:
                    pass

    @staticmethod
    def draw_label_callback():
        tool = tool_manager.active_tool
        if not isinstance(tool, LineTool) or not hasattr(tool, "preview"):
            return
        # v122: a cancelled/switched Line must never leave its GPU preview
        # behind while another Blender/SketchTools workspace tool is active.
        if not _line_workspace_is_active():
            try:
                tool.preview.clear()
            except Exception:
                pass
            return
        try:
            preview = tool.preview
            if hasattr(preview, "draw_label"):
                preview.draw_label()
        except (ReferenceError, RuntimeError):
            return
        except Exception as exc:
            debug_print("SketchTools v118 Line label skipped:", exc)


_LINE_KEY_TYPES = {
    'X', 'Y', 'Z',
    'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR',
    'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE',
    'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3', 'NUMPAD_4',
    'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7', 'NUMPAD_8', 'NUMPAD_9',
    'PERIOD', 'NUMPAD_PERIOD', 'MINUS', 'NUMPAD_MINUS',
    'M', 'C', 'F', 'T', 'I', 'N', 'K',
    'RET', 'NUMPAD_ENTER', 'BACK_SPACE', 'DEL', 'ESC',
}


class SketchToolsLineNativeEventOperator(bpy.types.Operator):
    """Process one Line-tool event and finish immediately (never modal)."""

    bl_idname = "sketchtools.line_native_event"
    bl_label = "SketchTools Line Native Event"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        # The operator is attached to the Line WorkSpaceTool keymap. If Blender
        # changed tools between keymap resolution and invocation, do nothing.
        try:
            current = context.workspace.tools.from_space_view3d_mode(
                context.mode, create=False
            )
            current_id = current.idname if current is not None else None
        except Exception:
            current_id = None

        valid_ids = {
            'sketchtools.line_tool',
            'sketchtools.line_tool_edit_mesh',
        }
        if current_id not in valid_ids:
            return {'PASS_THROUGH'}

        # Native keymap events should already carry the View3D WINDOW region.
        # Never manufacture a modal override here; that is the behavior v118
        # is explicitly testing against Blender's own gizmo event priority.
        if context.region is None or context.region.type != 'WINDOW':
            return {'PASS_THROUGH'}

        tool = tool_manager.active_tool
        if not isinstance(tool, LineTool):
            tool = LineTool()
            tool_manager.activate(tool, context)

        _LineNativeDrawService.ensure()
        show_pencil_cursor()

        # Each invocation owns only this single event. No modal handler is
        # registered, so after this return Blender is completely free to route
        # the next event to navigation gizmos/keymaps first.
        consumed = False

        if event.type == 'MOUSEMOVE':
            tool.on_mouse_move(context, event)
            # Preserve native gizmo hover/highlight handling.
            self._tag_redraw(context)
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            tool.on_left_click(context, event)
            consumed = True

        elif event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            tool.on_right_click(context, event)
            consumed = True

        elif event.value == 'PRESS' and event.type in _LINE_KEY_TYPES:
            consumed = bool(tool.on_key_press(context, event))
            if event.type == 'ESC' and not consumed:
                tool_manager.cancel(context)
                consumed = True

        self._tag_redraw(context)

        # If the LineTool ended itself, leave the draw service installed but
        # dormant; it owns no modal/input state and is removed on unregister.
        return {'FINISHED'} if consumed else {'PASS_THROUGH'}

    @staticmethod
    def _tag_redraw(context):
        try:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        except Exception:
            pass


def shutdown_line_native_draw_service():
    _LineNativeDrawService.shutdown()
