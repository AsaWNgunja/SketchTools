from ..utils.logging import debug_print
import bpy
import gpu
import weakref

from ..engine.selection import SelectionManager
from ..core.session import tool_manager
from ..engine.events import SketchToolsEventHandler
from ..utils.cursor import show_blender_cursor, show_pencil_cursor


class SketchToolsModalController(
    bpy.types.Operator
):

    bl_idname = "sketchtools.modal_controller"
    bl_label = "SketchTools Controller"

    bl_options = {
        'REGISTER'
    }

    # Only one shared controller should dispatch SketchTools events.
    _running = False
    _running_generation = None
    _instances = weakref.WeakSet()


    # -------------------------------------
    # Invoke
    # -------------------------------------

    def invoke(
        self,
        context,
        event
    ):

        if type(self)._running:
            return {
                'CANCELLED'
            }

        type(self)._running = True
        type(self)._instances.add(self)

        # Persistent generation token.  Unlike the class-level _running flag,
        # this survives Python module reloads.  Any controller from an older
        # generation exits before it can dispatch the same mouse event.
        wm = context.window_manager
        generation = int(wm.get("sketchtools_modal_generation", 0)) + 1
        wm["sketchtools_modal_generation"] = generation
        self._generation = generation
        type(self)._running_generation = generation

        self.event_handler = (
            SketchToolsEventHandler()
        )

        self.selection_manager = (
            SelectionManager()
        )


        # True while Blender navigation gizmo
        # owns the current LEFTMOUSE action.
        self.navigation_gizmo_active = False



        self.draw_handler = (
            bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback,
                (),
                'WINDOW',
                'POST_VIEW'
            )
        )

        self.snap_label_handler = (
            bpy.types.SpaceView3D.draw_handler_add(
                self.draw_label_callback,
                (),
                'WINDOW',
                'POST_PIXEL'
            )
        )


        context.window_manager.modal_handler_add(
            self
        )


        return {
            'RUNNING_MODAL'
        }


    # -------------------------------------
    # Find area / region under mouse
    # -------------------------------------

    def get_mouse_area_region(
        self,
        context,
        event
    ):

        screen = context.screen

        if screen is None:

            return None, None


        mouse_x = event.mouse_x
        mouse_y = event.mouse_y


        for area in screen.areas:

            inside_area = (
                mouse_x >= area.x
                and
                mouse_x < (
                    area.x
                    + area.width
                )
                and
                mouse_y >= area.y
                and
                mouse_y < (
                    area.y
                    + area.height
                )
            )


            if not inside_area:

                continue


            # ---------------------------------
            # Check UI regions first
            # ---------------------------------

            for region in area.regions:

                if region.type == 'WINDOW':

                    continue


                inside_region = (
                    mouse_x >= region.x
                    and
                    mouse_x < (
                        region.x
                        + region.width
                    )
                    and
                    mouse_y >= region.y
                    and
                    mouse_y < (
                        region.y
                        + region.height
                    )
                )


                if inside_region:

                    return (
                        area,
                        region
                    )


            # ---------------------------------
            # Main viewport region
            # ---------------------------------

            for region in area.regions:

                if region.type != 'WINDOW':

                    continue


                inside_region = (
                    mouse_x >= region.x
                    and
                    mouse_x < (
                        region.x
                        + region.width
                    )
                    and
                    mouse_y >= region.y
                    and
                    mouse_y < (
                        region.y
                        + region.height
                    )
                )


                if inside_region:

                    return (
                        area,
                        region
                    )


            return (
                area,
                None
            )


        return (
            None,
            None
        )


    # -------------------------------------
    # View3D WINDOW region under mouse
    # -------------------------------------

    def get_view3d_window_region_at_mouse(
        self,
        event,
        area
    ):

        if area is None or area.type != 'VIEW_3D':
            return None

        for region in area.regions:
            if region.type != 'WINDOW':
                continue

            if (
                event.mouse_x >= region.x
                and event.mouse_x < region.x + region.width
                and event.mouse_y >= region.y
                and event.mouse_y < region.y + region.height
            ):
                return region

        return None


    # -------------------------------------
    # Visible SketchTools sidebar hit test
    # -------------------------------------

    def mouse_over_visible_sketchtools_panel(
        self,
        context,
        event,
        area,
        region
    ):
        """Return True only over the visible SketchTools panel content.

        Blender's VIEW_3D UI region (the N-sidebar) can extend all the way to
        the bottom of the editor even when the visible SketchTools panel ends
        much higher.  Treating the whole UI region as occupied makes the Line
        cursor turn back to Blender's normal cursor over visually empty 3D
        space below the panel.

        SketchTools currently has a compact fixed-height panel.  Estimate the
        panel's visible content height using Blender's UI scale and leave the
        transparent area below it under Line-tool ownership.
        """

        if area is None or region is None:
            return False

        if area.type != 'VIEW_3D' or region.type != 'UI':
            return False

        # The panel begins immediately below the View3D header/tool-header.
        # The compact Free panel extends through Drawing Tools and Modify/Eraser.
        # Keep the hit area large enough to include Eraser so direct one-click
        # switching between all Free tools continues to work.
        # Do NOT add a safety/bottom margin here: that creates an invisible
        # horizontal barrier below the visible N-panel.  The first pixel below
        # the visible panel must immediately return to viewport ownership.
        try:
            ui_scale = context.preferences.system.ui_scale
        except Exception:
            ui_scale = 1.0

        visible_panel_height = 145.0 * max(0.75, float(ui_scale))

        panel_top = region.y + region.height
        panel_bottom = panel_top - visible_panel_height

        return event.mouse_y >= panel_bottom


    # -------------------------------------
    # Navigation gizmo detection
    # -------------------------------------

    def mouse_over_navigation_gizmo(
        self,
        context,
        event,
        area
    ):

        if area is None:

            return False


        if area.type != 'VIEW_3D':

            return False


        window_region = None


        for region in area.regions:

            if region.type == 'WINDOW':

                window_region = region

                break


        if window_region is None:

            return False


        local_x = (
            event.mouse_x
            - window_region.x
        )

        local_y = (
            event.mouse_y
            - window_region.y
        )


        try:

            gizmo_size = (
                context.preferences
                .view
                .gizmo_size
            )

        except Exception:

            gizmo_size = 75


        # Keep this zone tight.  Older builds added 100 px of padding,
        # which created a visible dead strip where Line changed back to the
        # normal cursor even though the mouse was still in usable viewport.
        # The gizmo itself is the only thing that needs special click routing.
        gizmo_zone = max(48.0, float(gizmo_size) + 8.0)

        return (
            local_x >= (window_region.width - gizmo_zone)
            and
            local_y >= (window_region.height - gizmo_zone)
        )


    # -------------------------------------
    # Draw callback
    # -------------------------------------

    def draw_callback(
        self
    ):

        tool = (
            tool_manager.active_tool
        )


        if tool is None:

            return


        if hasattr(tool, "preview"):
            pushed = False
            try:
                gpu.matrix.push(); pushed = True
                tool.preview.draw()
            except ReferenceError:
                return
            except Exception as exc:
                debug_print("SketchTools draw callback skipped:", exc)
            finally:
                if pushed:
                    try: gpu.matrix.pop()
                    except Exception: pass


    def draw_label_callback(self):
        try:
            tool = tool_manager.active_tool
            if tool is None or not hasattr(tool, 'preview'):
                return
            preview = tool.preview
            if hasattr(preview, 'draw_label'):
                preview.draw_label()
        except ReferenceError:
            return
        except Exception as exc:
            debug_print("SketchTools label callback skipped:", exc)


    # -------------------------------------
    # Modal
    # -------------------------------------

    def modal(
        self,
        context,
        event
    ):

        # Kill stale controllers created by previous add-on reloads.
        # Without this, one physical click can be dispatched many times.
        if int(context.window_manager.get("sketchtools_modal_generation", -1)) != getattr(self, "_generation", -2):
            self.cleanup()
            return {'FINISHED'}

        # -------------------------------------
        # IMPORTANT:
        #
        # Has the user selected another
        # Blender workspace tool?
        #
        # Examples:
        #
        # Select Box
        # Move
        # Rotate
        # Scale
        # Cursor
        #
        # If yes, Blender owns the tool now.
        # Cancel SketchTools immediately.
        # -------------------------------------

        if tool_manager.blender_tool_changed(
            context
        ):

            current_tool = (
                tool_manager.get_blender_tool(
                    context
                )
            )


            debug_print(
                "SketchTools cancelled because "
                "Blender tool changed to:",
                current_tool
            )


            # IMPORTANT:
            #
            # restore_blender_tool=False
            #
            # The user deliberately chose the
            # new Blender tool. Do not replace
            # it with the previous one.
            tool_manager.cancel(
                context,
                restore_blender_tool=False
            )


            self.cleanup()


            return {
                'FINISHED'
            }


        # -------------------------------------
        # Tool disappeared for another reason
        # -------------------------------------

        if tool_manager.active_tool is None:

            self.cleanup()


            return {
                'FINISHED'
            }


        # -------------------------------------
        # Find area / region under mouse
        # -------------------------------------

        mouse_area, raw_mouse_region = (
            self.get_mouse_area_region(
                context,
                event
            )
        )

        # Keep two independent notions of what is under the mouse:
        #
        # 1) raw_mouse_region = Blender UI ownership for actual clicks.
        #    This MUST be preserved so toolbar/header/sidebar controls can
        #    still receive LEFTMOUSE even where their region overlaps WINDOW.
        #
        # 2) mouse_region = SketchTools drawing/hover ownership.  Here the
        #    real View3D WINDOW wins wherever possible so Line stays active
        #    through visually empty overlay areas and right up to UI edges.
        mouse_region = raw_mouse_region


        # -------------------------------------
        # Prefer the real View3D WINDOW for drawing/hover only
        # -------------------------------------

        if mouse_area is not None and mouse_area.type == 'VIEW_3D':
            window_under_mouse = self.get_view3d_window_region_at_mouse(
                event,
                mouse_area
            )

            if window_under_mouse is not None:
                mouse_region = window_under_mouse

            elif (
                raw_mouse_region is not None
                and raw_mouse_region.type == 'UI'
                and not self.mouse_over_visible_sketchtools_panel(
                    context,
                    event,
                    mouse_area,
                    raw_mouse_region
                )
            ):
                # Empty area below the visible N-panel behaves as viewport.
                mouse_region = next(
                    (r for r in mouse_area.regions if r.type == 'WINDOW'),
                    raw_mouse_region
                )

        # -------------------------------------
        # Navigation gizmo
        # -------------------------------------

        inside_gizmo_zone = (
            self.mouse_over_navigation_gizmo(
                context,
                event,
                mouse_area
            )
        )


        # -------------------------------------
        # Start gizmo interaction
        # -------------------------------------

        if (
            inside_gizmo_zone
            and
            event.type == 'LEFTMOUSE'
            and
            event.value == 'PRESS'
        ):

            self.navigation_gizmo_active = True
            show_blender_cursor(context)

            return {
                'RUNNING_MODAL',
                'PASS_THROUGH'
            }


        # -------------------------------------
        # Gizmo owns interaction until release
        # -------------------------------------

        if self.navigation_gizmo_active:

            if (
                event.type == 'LEFTMOUSE'
                and
                event.value == 'RELEASE'
            ):

                self.navigation_gizmo_active = (
                    False
                )


            return {
                'RUNNING_MODAL',
                'PASS_THROUGH'
            }


        # -------------------------------------
        # Hovering navigation gizmo
        # -------------------------------------

        if inside_gizmo_zone:
            # Hover alone must NEVER steal Line's mouse-move stream.
            # Previous builds passed MOUSEMOVE through here; because the
            # detector is rectangular while Blender's navigation controls are
            # not, that produced an invisible horizontal barrier immediately
            # below/around the controls.  Only an actual LEFTMOUSE press above
            # the controls is routed to Blender (handled above).
            show_pencil_cursor()


        # -------------------------------------
        # Outside any editor
        # -------------------------------------

        if mouse_area is None:

            show_blender_cursor(context)

            # Do not consume events outside Blender editors.  PASS_THROUGH
            # alone keeps this modal controller alive while allowing the
            # window/UI keymaps to receive the event normally.
            return {'PASS_THROUGH'}


        # -------------------------------------
        # Other Blender editors
        # -------------------------------------

        if mouse_area.type != 'VIEW_3D':

            show_blender_cursor(context)

            # Outliner, Timeline, Properties, etc. own these events.
            # Keep SketchTools modal, but do not consume the event.
            return {'PASS_THROUGH'}


        # -------------------------------------
        # View3D UI owns LEFTMOUSE completely
        #
        # Sidebar / Toolbar / Header / Tool Header
        #
        # A modal operator gets the event before Blender's
        # toolbar has actually changed the active workspace
        # tool.  Waiting for blender_tool_changed() here can
        # therefore be one event too late and, depending on
        # the UI interaction, can make a toolbar button appear
        # to hover correctly but not click correctly.
        #
        # Rule: when the user presses LEFTMOUSE anywhere in a
        # View3D UI region, SketchTools gives up ownership first
        # and passes that exact click through to Blender.
        # Blender is then free to activate Select Box, Move,
        # Rotate, another panel operator, etc.
        # -------------------------------------

        # CLICK ownership is deliberately based on Blender's original region,
        # not the drawing/hover region above.  This is what allows Line to stay
        # active over every drawable viewport pixel while still letting a real
        # Blender control win when it is actually clicked.
        click_region = raw_mouse_region

        # v106: Never reinterpret an actual View3D UI-region click as a
        # viewport click.  The N-panel tab strip and its controls live in the
        # same UI region, and the old visible-panel-height heuristic could
        # classify lower tabs (including Tool / SketchTools) as WINDOW.  That
        # made those tabs unclickable while a SketchTool was active.
        #
        # Hover/drawing ownership may still use the heuristic above so the
        # pencil can remain responsive near visually empty overlay space, but
        # CLICK ownership must follow Blender's raw region exactly.

        if (
            click_region is None
            or
            click_region.type != 'WINDOW'
        ):

            # Hovering Blender UI must not create a normal-cursor moat around
            # the viewport.  Keep the Line cursor until the user actually
            # clicks Blender UI.  The click below cancels SketchTools and is
            # passed through unchanged, so Select Box / Move / panels still
            # activate normally.
            show_pencil_cursor()

            # Blender owns all events in View3D UI regions (toolbar,
            # header, sidebar, tool header).  Crucially, do NOT cancel the
            # SketchTool here: passing the event through lets Blender UI work
            # while the modal tool remains ready to resume in the viewport.
            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                show_blender_cursor(context)
                debug_print("SketchTools: passing View3D UI click to Blender")

            return {'PASS_THROUGH'}


        # -------------------------------------
        # Blender viewport orbit
        # -------------------------------------

        if event.type == 'MIDDLEMOUSE':
            # IMPORTANT:
            # Do not keep a persistent "orbit active" flag here.  Blender's
            # own View3D rotate operator becomes the top modal handler after
            # the MMB press, which means this SketchTools controller may never
            # see the corresponding MMB RELEASE.  A sticky flag would then
            # leave every later event permanently passed through, making the
            # yellow Line preview look frozen and preventing the same chain
            # from continuing.
            #
            # Instead, simply pass MMB to Blender.  While Blender is orbiting
            # it owns the drag stream.  When its rotate modal finishes, this
            # controller naturally starts receiving events again with the
            # existing LineTool object untouched: start_point, points, drawing
            # plane and preview all remain live.
            show_blender_cursor(context)
            return {'PASS_THROUGH'}

        # -------------------------------------
        # Standard viewport navigation
        # -------------------------------------

        if event.type in {

            'MIDDLEMOUSE',

            'WHEELUPMOUSE',
            'WHEELDOWNMOUSE',
            'WHEELINMOUSE',
            'WHEELOUTMOUSE',

            'TRACKPADPAN',
            'TRACKPADZOOM',
            'MOUSEROTATE',
            'MOUSESMARTZOOM',

            'NDOF_MOTION',
            'NDOF_BUTTON_FIT',

        }:

            show_blender_cursor(context)

            # Let Blender's normal View3D keymap process wheel/trackpad/NDOF
            # navigation directly.  SketchTools does not consume these events.
            return {'PASS_THROUGH'}


        # Keep the pencil visible only in the actual 3D viewport.
        show_pencil_cursor()


        # -------------------------------------
        # SketchTools handles viewport event
        # -------------------------------------

        self.selection_manager.update(
            context,
            event
        )

        # Give active SketchTools a chance to consume text/numeric keyboard
        # input before the generic event handler.  Circle uses this for
        # SketchUp-style entries such as "48s" + Enter.
        active_tool = tool_manager.active_tool
        if (
            active_tool is not None
            and event.value == 'PRESS'
            and hasattr(active_tool, "on_key_press")
        ):
            if active_tool.on_key_press(context, event):
                return {'RUNNING_MODAL'}

        self.event_handler.handle_event(
            context,
            event
        )


        # -------------------------------------
        # Tool may have ended while handling
        # the event.
        # -------------------------------------

        if tool_manager.active_tool is None:

            self.cleanup()


            return {
                'FINISHED'
            }


        # -------------------------------------
        # Mouse movement reaches BOTH systems
        #
        # SketchTools already processed it.
        #
        # Blender receives it afterwards so:
        #
        # - navigation gizmo highlights
        # - UI hover remains responsive
        # -------------------------------------

        if event.type == 'MOUSEMOVE':

            return {
                'RUNNING_MODAL',
                'PASS_THROUGH'
            }


        # -------------------------------------
        # Other drawing events remain owned
        # by SketchTools.
        # -------------------------------------

        return {
            'RUNNING_MODAL'
        }


    # -------------------------------------
    # Cleanup
    # -------------------------------------

    def cleanup(self):
        """Idempotently release every Blender-facing resource owned here."""
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True
        if type(self)._running_generation == getattr(self, "_generation", None):
            type(self)._running = False
            type(self)._running_generation = None
        for attr in ("draw_handler", "snap_label_handler"):
            handle = getattr(self, attr, None)
            if handle:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
                except (ReferenceError, RuntimeError, ValueError):
                    pass
                except Exception as exc:
                    debug_print("SketchTools handler cleanup:", exc)
                setattr(self, attr, None)
        self.event_handler = None
        self.selection_manager = None
        self.navigation_gizmo_active = False
        try: type(self)._instances.discard(self)
        except Exception: pass

    @classmethod
    def shutdown_all(cls):
        """Invalidate all old modal callbacks before add-on unregister/reload."""
        cls._running = False
        cls._running_generation = None
        for instance in list(cls._instances):
            try: instance.cleanup()
            except Exception: pass
        cls._instances.clear()


    # -------------------------------------
    # Cancel
    # -------------------------------------

    def cancel(
        self,
        context
    ):

        if tool_manager.active_tool is not None:

            tool_manager.cancel(
                context,
                restore_blender_tool=True
            )


        self.cleanup()