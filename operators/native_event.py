import bpy, gpu
from ..core.session import tool_manager
from ..tools.rectangle import RectangleTool
from ..tools.circle import CircleTool
from ..tools.freehand import FreehandTool
from ..tools.eraser import EraserTool
from ..utils.logging import debug_print
from ..utils.cursor import set_active_tool_cursor, show_pencil_cursor



def _current_workspace_tool_id():
    try:
        context = bpy.context
        if context is None or context.workspace is None:
            return None
        wt = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
        return wt.idname if wt else None
    except Exception:
        return None


def _tool_matches_workspace(tool):
    wid = _current_workspace_tool_id()
    if isinstance(tool, RectangleTool):
        return wid in {'sketchtools.rectangle_tool', 'sketchtools.rectangle_tool_edit_mesh'}
    if isinstance(tool, CircleTool):
        return wid in {'sketchtools.circle_tool', 'sketchtools.circle_tool_edit_mesh'}
    if isinstance(tool, FreehandTool):
        return wid in {'sketchtools.freehand_tool', 'sketchtools.freehand_tool_edit_mesh'}
    if isinstance(tool, EraserTool):
        return wid in {'sketchtools.eraser_tool', 'sketchtools.eraser_tool_edit_mesh'}
    return False

class _NativeDrawService:
    draw_handler = None; label_handler = None
    @classmethod
    def ensure(cls):
        if cls.draw_handler is None:
            cls.draw_handler = bpy.types.SpaceView3D.draw_handler_add(cls.draw_callback, (), 'WINDOW', 'POST_VIEW')
        if cls.label_handler is None:
            cls.label_handler = bpy.types.SpaceView3D.draw_handler_add(cls.draw_label_callback, (), 'WINDOW', 'POST_PIXEL')
    @classmethod
    def shutdown(cls):
        for attr in ('draw_handler','label_handler'):
            h=getattr(cls,attr,None)
            if h is not None:
                try: bpy.types.SpaceView3D.draw_handler_remove(h,'WINDOW')
                except Exception: pass
                setattr(cls,attr,None)
    @staticmethod
    def draw_callback():
        tool=tool_manager.active_tool
        if not isinstance(tool,(RectangleTool,CircleTool,FreehandTool,EraserTool)) or not hasattr(tool,'preview'): return
        # v122: never draw stale preview/inference feedback after Blender has
        # switched away from the owning SketchTool.  This removes lingering
        # yellow snap dots and labels such as "On Grid" without restoring a
        # persistent modal controller.
        if not _tool_matches_workspace(tool):
            try: tool.preview.clear()
            except Exception: pass
            return
        pushed=False
        try:
            gpu.matrix.push(); pushed=True; tool.preview.draw()
        except (ReferenceError,RuntimeError): return
        except Exception as exc: debug_print('SketchTools v119 native draw skipped:',exc)
        finally:
            if pushed:
                try: gpu.matrix.pop()
                except Exception: pass
    @staticmethod
    def draw_label_callback():
        tool=tool_manager.active_tool
        if not isinstance(tool,(RectangleTool,CircleTool,FreehandTool,EraserTool)) or not hasattr(tool,'preview'): return
        # v122: never draw stale preview/inference feedback after Blender has
        # switched away from the owning SketchTool.  This removes lingering
        # yellow snap dots and labels such as "On Grid" without restoring a
        # persistent modal controller.
        if not _tool_matches_workspace(tool):
            try: tool.preview.clear()
            except Exception: pass
            return
        try:
            if hasattr(tool.preview,'draw_label'): tool.preview.draw_label()
        except (ReferenceError,RuntimeError): return
        except Exception as exc: debug_print('SketchTools v119 native label skipped:',exc)

_TOOL_IDS={
'sketchtools.rectangle_tool':('rectangle',RectangleTool),'sketchtools.rectangle_tool_edit_mesh':('rectangle',RectangleTool),
'sketchtools.circle_tool':('circle',CircleTool),'sketchtools.circle_tool_edit_mesh':('circle',CircleTool),
'sketchtools.freehand_tool':('freehand',FreehandTool),'sketchtools.freehand_tool_edit_mesh':('freehand',FreehandTool),
'sketchtools.eraser_tool':('eraser',EraserTool),'sketchtools.eraser_tool_edit_mesh':('eraser',EraserTool),}

class SketchToolsNativeEventOperator(bpy.types.Operator):
    bl_idname='sketchtools.native_event'; bl_label='SketchTools Native Event'; bl_options={'INTERNAL'}
    @classmethod
    def poll(cls,context): return context.area is not None and context.area.type=='VIEW_3D'
    def invoke(self,context,event):
        if context.region is None or context.region.type!='WINDOW': return {'PASS_THROUGH'}
        try:
            wt=context.workspace.tools.from_space_view3d_mode(context.mode,create=False); wid=wt.idname if wt else None
        except Exception: wid=None
        spec=_TOOL_IDS.get(wid)
        if spec is None: return {'PASS_THROUGH'}
        key,tool_cls=spec; tool=tool_manager.active_tool
        if not isinstance(tool,tool_cls):
            if key=='circle': tool=CircleTool(segments=getattr(context.scene,'sketchtools_circle_segments',32))
            elif key=='freehand': tool=FreehandTool(spacing_px=getattr(context.scene,'sketchtools_freehand_spacing',5))
            else: tool=tool_cls()
            tool_manager.activate(tool,context)
        # v120: the native-event architecture returns after every event, so
        # Blender may restore its normal pointer between invocations. Reassert
        # the selected tool cursor on every native event, just as the working
        # v118/v119 Line path does, without reintroducing a modal controller.
        set_active_tool_cursor(tool)
        show_pencil_cursor()
        _NativeDrawService.ensure(); consumed=False
        if event.type=='MOUSEMOVE':
            tool.on_mouse_move(context,event); self._redraw(context); return {'PASS_THROUGH'}
        if event.type=='LEFTMOUSE':
            if event.value=='PRESS': tool.on_left_click(context,event); consumed=True
            elif event.value=='RELEASE' and hasattr(tool,'on_left_release'): tool.on_left_release(context,event); consumed=True
        elif event.type=='RIGHTMOUSE' and event.value=='PRESS': tool.on_right_click(context,event); consumed=True
        elif event.value=='PRESS' and hasattr(tool,'on_key_press'):
            result=tool.on_key_press(context,event); consumed=bool(result)
            if event.type=='ESC' and not consumed: tool_manager.cancel(context); consumed=True
        self._redraw(context); return {'FINISHED'} if consumed else {'PASS_THROUGH'}
    @staticmethod
    def _redraw(context):
        try:
            for area in context.screen.areas:
                if area.type=='VIEW_3D': area.tag_redraw()
        except Exception: pass

def shutdown_native_draw_service(): _NativeDrawService.shutdown()
