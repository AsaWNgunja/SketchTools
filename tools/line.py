from ..utils.logging import debug_print
import bpy
from itertools import combinations
import bmesh
from mathutils import Vector
from bpy_extras import view3d_utils


from ..geometry.raycast import (
    get_view_ray,
    mouse_to_plane
)

from ..geometry.snapping import resolve_snap, snap_label, SnapResult, _result_is_occluded
from ..geometry.projection import object_or_floor_plane

from ..geometry.topology import (
    create_edge,
    create_face_from_points
)
from ..core.session import tool_manager
from ..drawing.preview import PreviewLine

from ..utils.cursor import (
    set_pencil_cursor,
    refresh_pencil_cursor,
    restore_cursor
)

from ..utils.geometry_object import (
    get_or_create_geometry_object
)
from ..utils.measurements import format_length, parse_length

from ..core.tool_base import SketchToolBase



class LineTool(SketchToolBase):


    def __init__(self):

        self.start_point = None
        self.end_point = None
        
        self.drawing_plane_point = None
        self.drawing_plane_normal = None
        self._last_snap_type = None

        self.state = "READY"
        
        self.points = []
        
        self.geometry_object = None
        
        self.preview = PreviewLine()

        # v93 precision memory for SketchUp-style post-draw length entry.
        self.last_segment_start = None
        self.last_segment_end = None
        self.last_segment_obj_name = None
        self.length_input = ""



    # -------------------------------------
    # Lifecycle
    # -------------------------------------

    def start(
        self,
        context
    ):

        super().start(context)

        self.reset()

        set_pencil_cursor()
        
        debug_print(
            "Line Tool Active"
        )



    def _start_point_snap(self, context, event, threshold_px=16.0):
        """High-priority snap back to the first point of the current chain."""
        if self.state != "DRAWING" or len(self.points) < 3:
            return None

        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            return None

        screen = view3d_utils.location_3d_to_region_2d(
            region,
            rv3d,
            self.points[0],
        )
        if screen is None:
            return None

        dx = float(event.mouse_region_x) - float(screen.x)
        dy = float(event.mouse_region_y) - float(screen.y)
        if (dx * dx + dy * dy) ** 0.5 > threshold_px:
            return None

        point = Vector(self.points[0]).copy()
        self.preview.set_snap_point(point)
        self.preview.set_snap_feedback(
            "Close Loop",
            event.mouse_region_x,
            event.mouse_region_y,
        )
        return point

    def _preferred_axis(self):
        """Return the X/Y/Z inference family retained by the last line snap."""
        return {
            "X_AXIS": "X_AXIS",
            "Y_AXIS": "Y_AXIS",
            "Z_AXIS": "Z_AXIS",
            "X_GRID": "X_AXIS",
            "Y_GRID": "Y_AXIS",
            "Z_GRID": "Z_AXIS",
        }.get(self._last_snap_type)

    def _snap(self, context, event):
        start_snap = self._start_point_snap(context, event)
        if start_snap is not None:
            return start_snap

        snap = resolve_snap(
            context,
            event.mouse_region_x,
            event.mouse_region_y,
            plane_point=self.drawing_plane_point,
            plane_normal=self.drawing_plane_normal,
            inference_origin=(self.start_point if self.state == 'DRAWING' else None),
            include_face=True,
            include_grid=(context.mode != 'EDIT_MESH'),
            world_axis_inference=(
                context.mode != 'EDIT_MESH'
                and context.region_data is not None
                and context.region_data.view_perspective == 'PERSP'
            ),
            preferred_axis=self._preferred_axis(),
            # v95: while Line is being drawn, prefer discrete Blender grid
            # crossings that lie on the same inferred X/Y/Z ray. Other tools
            # retain v94 snapping unchanged.
            include_axis_grid=(self.state == 'DRAWING'),
        )
        if snap is not None and snap.valid:
            self._last_snap_type = snap.snap_type
            self.preview.set_snap_point(snap.location)
            self.preview.set_snap_feedback(
                snap_label(snap), event.mouse_region_x, event.mouse_region_y
            )
            return Vector(snap.location)
        # Keep the previous axis inference as short-lived hysteresis memory.
        # It will be replaced as soon as another valid snap is acquired.
        self.preview.clear_snap_point()
        self.preview.clear_snap_feedback()
        return None

    def _cut_closed_loop_edit_mode(self, context):
        """
        Dedicated Line polygon cutter.
        Creates one centre face plus a small number of outer ring faces,
        without Freehand's companion/support strip.
        """
        obj = context.edit_object
        if obj is None or obj.type != 'MESH' or len(self.points) < 3:
            return False

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        inv = obj.matrix_world.inverted()
        inner_pos = [inv @ Vector(p) for p in self.points]

        # Remove repeated close point if ever supplied.
        if len(inner_pos) > 1 and (inner_pos[0] - inner_pos[-1]).length < 1e-7:
            inner_pos.pop()
        if len(inner_pos) < 3:
            return False

        centre = sum(inner_pos, Vector((0.0, 0.0, 0.0))) / len(inner_pos)

        # Nested-cut fix: after the first polygon cut there are several
        # coplanar faces. Plane distance alone can therefore select a ring
        # sector instead of the face actually underneath the new polygon.
        # Choose the SMALLEST coplanar face that contains every new point.
        def _candidate_face_info(face):
            if len(face.verts) < 3:
                return None
            fn = face.normal.normalized()
            fo = face.verts[0].co
            if any(abs(fn.dot(p - fo)) > 2e-4 for p in inner_pos):
                return None
            fu = (face.verts[1].co - fo)
            if fu.length < 1e-9:
                return None
            fu.normalize()
            fv = fn.cross(fu)
            if fv.length < 1e-9:
                return None
            fv.normalize()

            def xy(p):
                d = p - fo
                return Vector((d.dot(fu), d.dot(fv)))

            poly = [xy(vtx.co) for vtx in face.verts]
            pts = [xy(p) for p in inner_pos]

            def on_segment(p, a, b, tol=2e-5):
                ab = b - a
                if ab.length_squared < 1e-14:
                    return (p - a).length <= tol
                t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
                return (p - (a + ab * t)).length <= tol

            def contains(p):
                for i, a in enumerate(poly):
                    if on_segment(p, a, poly[(i + 1) % len(poly)]):
                        return True
                hit = False
                j = len(poly) - 1
                for i in range(len(poly)):
                    a, b = poly[i], poly[j]
                    if ((a.y > p.y) != (b.y > p.y)):
                        x = (b.x-a.x) * (p.y-a.y) / (b.y-a.y) + a.x
                        if p.x < x:
                            hit = not hit
                    j = i
                return hit

            if not all(contains(p) for p in pts):
                return None
            face_area = abs(0.5 * sum(
                poly[i].x * poly[(i + 1) % len(poly)].y
                - poly[(i + 1) % len(poly)].x * poly[i].y
                for i in range(len(poly))
            ))
            return face_area

        candidates = []
        for face in bm.faces:
            info = _candidate_face_info(face)
            if info is not None:
                candidates.append((info, face))
        target = min(candidates, key=lambda item: item[0])[1] if candidates else None
        if target is None:
            debug_print("SketchTools Line: no containing coplanar face for closed loop")
            return False

        normal = target.normal.normalized()
        origin = target.verts[0].co.copy()
        if abs(normal.dot(centre - origin)) > 1e-4:
            return False
        if any(abs(normal.dot(p - origin)) > 2e-4 for p in inner_pos):
            return False

        outer = list(target.verts)
        if len(outer) < 3:
            return False

        # Stable 2D basis.
        u = (outer[1].co - origin).normalized()
        v = normal.cross(u).normalized()

        def to2(p):
            d = p - origin
            return Vector((d.dot(u), d.dot(v)))

        def area(poly):
            return 0.5 * sum(
                poly[i].x * poly[(i + 1) % len(poly)].y
                - poly[(i + 1) % len(poly)].x * poly[i].y
                for i in range(len(poly))
            )

        def inside(pt, poly):
            hit = False
            j = len(poly) - 1
            for i in range(len(poly)):
                a, b = poly[i], poly[j]
                if ((a.y > pt.y) != (b.y > pt.y)):
                    x = (b.x-a.x) * (pt.y-a.y) / (b.y-a.y) + a.x
                    if pt.x < x:
                        hit = not hit
                j = i
            return hit

        outer2 = [to2(vtx.co) for vtx in outer]
        inner2 = [to2(p) for p in inner_pos]

        if not all(inside(q, outer2) for q in inner2):
            return False

        if area(outer2) < 0:
            outer.reverse()
            outer2.reverse()
        if area(inner2) < 0:
            inner_pos.reverse()
            inner2.reverse()

        n_outer = len(outer)
        n_inner = len(inner_pos)

        # Map the outer boundary onto the new inner polygon in cyclic order.
        # The original cutter required n_inner >= n_outer, so a second smaller
        # polygon inside the first centre face could never cut. Keep the old
        # distinct-anchor solution when possible; otherwise use an ordered
        # perimeter mapping that permits repeated inner anchors.
        best = None
        best_cost = None

        if n_inner >= n_outer:
            for combo in combinations(range(n_inner), n_outer):
                combo = tuple(combo)
                for shift in range(n_outer):
                    anchors = combo[shift:] + combo[:shift]
                    cost = 0.0
                    for i in range(n_outer):
                        d = outer2[i] - inner2[anchors[i]]
                        cost += d.length_squared
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best = anchors
        else:
            # Ordered cyclic mapping. Repeated anchors create valid triangular
            # sectors where necessary, while the full inner loop is traversed
            # exactly once around the annulus.
            for shift in range(n_inner):
                anchors = tuple(
                    (shift + int(round(i * n_inner / n_outer))) % n_inner
                    for i in range(n_outer)
                )
                cost = sum(
                    (outer2[i] - inner2[anchors[i]]).length_squared
                    for i in range(n_outer)
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = anchors

        if best is None:
            return False

        inner_verts = [bm.verts.new(p) for p in inner_pos]
        bm.verts.index_update()

        def inner_chain(start_idx, end_idx):
            chain = [inner_verts[start_idx]]
            if start_idx == end_idx:
                return chain
            idx = start_idx
            safety = 0
            while idx != end_idx:
                idx = (idx + 1) % n_inner
                chain.append(inner_verts[idx])
                safety += 1
                if safety > n_inner:
                    raise RuntimeError("invalid Line inner chain")
            return chain

        original_outer = tuple(target.verts)
        created = []

        try:
            # Prepare all ring loops before destructive mutation.
            ring_loops = []
            for i in range(n_outer):
                j = (i + 1) % n_outer
                chain = inner_chain(best[i], best[j])

                # Broad sector bounded by one outer edge and the corresponding
                # inner polygon arc. No radial fan from every Line vertex.
                loop = [outer[i], outer[j]] + list(reversed(chain))
                clean = []
                for vert in loop:
                    if not clean or clean[-1] is not vert:
                        clean.append(vert)
                if len(clean) > 1 and clean[0] is clean[-1]:
                    clean.pop()
                if len(set(clean)) < 3:
                    raise RuntimeError("degenerate Line ring sector")
                ring_loops.append(tuple(clean))

            # Only now remove the original face.
            bmesh.ops.delete(bm, geom=[target], context='FACES_ONLY')

            centre_face = bm.faces.new(tuple(inner_verts))
            created.append(centre_face)

            for loop in ring_loops:
                try:
                    face = bm.faces.new(loop)
                except ValueError:
                    face = bm.faces.new(tuple(reversed(loop)))
                created.append(face)

            if len(created) != n_outer + 1:
                raise RuntimeError("Line closed-loop reconstruction incomplete")

            for face in bm.faces:
                face.select = False
            centre_face.select = True

            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data,
                loop_triangles=True,
                destructive=True,
            )
            debug_print(
                "SketchTools Line: dedicated closed-loop cut",
                "inner_points=", n_inner,
                "outer_sectors=", n_outer,
            )
            return True

        except Exception as exc:
            valid_faces = [f for f in created if f.is_valid]
            if valid_faces:
                bmesh.ops.delete(bm, geom=valid_faces, context='FACES_ONLY')

            valid_inner = [vtx for vtx in inner_verts if vtx.is_valid]
            if valid_inner:
                bmesh.ops.delete(bm, geom=valid_inner, context='VERTS')

            try:
                if all(vtx.is_valid for vtx in original_outer):
                    bm.faces.new(original_outer)
            except Exception as rollback_exc:
                debug_print(
                    "SketchTools Line rollback warning:",
                    repr(rollback_exc),
                )

            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data,
                loop_triangles=False,
                destructive=True,
            )
            debug_print(
                "SketchTools Line dedicated cut rolled back:",
                repr(exc),
            )
            return False


    # -------------------------------------
    # v94 Equal-length inference
    # -------------------------------------

    def _equal_length_inference(self, context, event, current_point, pixel_radius=12.0):
        """Snap the live endpoint when it reaches the length of a visible parallel edge.

        The comparison is screen-space driven: for every visible mesh edge that
        is parallel to the line currently being drawn, calculate where the
        current endpoint would be if its length exactly matched that edge.  If
        that equal-length endpoint falls close to the mouse, use it and show a
        relationship guide. Hidden/occluded reference edges are rejected using
        the same front-surface rule as normal SketchTools snapping.
        """
        if self.state != "DRAWING" or self.start_point is None or current_point is None:
            self.preview.clear_equal_length_guide()
            return current_point

        # Never replace a stronger explicit geometry snap. Equal Length is a
        # secondary inference, just like SketchUp: endpoints/midpoints/edges
        # and origins remain authoritative when the cursor is actually on one.
        if self._last_snap_type in {"ENDPOINT", "MIDPOINT", "EDGE", "ORIGIN"}:
            self.preview.clear_equal_length_guide()
            return current_point

        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None:
            self.preview.clear_equal_length_guide()
            return current_point

        start = Vector(self.start_point)
        live = Vector(current_point)
        live_vec = live - start
        if live_vec.length <= 1e-8:
            self.preview.clear_equal_length_guide()
            return current_point
        live_dir = live_vec.normalized()
        mouse = Vector((float(event.mouse_region_x), float(event.mouse_region_y)))

        best = None
        best_px = float(pixel_radius)

        for obj in context.visible_objects:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            mw = obj.matrix_world
            for edge in mesh.edges:
                va = mesh.vertices[edge.vertices[0]]
                vb = mesh.vertices[edge.vertices[1]]
                wa = mw @ va.co
                wb = mw @ vb.co
                ref_vec = wb - wa
                ref_len = ref_vec.length
                if ref_len <= 1e-7:
                    continue

                # Parallel or anti-parallel within about 4 degrees.
                if abs(ref_vec.normalized().dot(live_dir)) < 0.9975:
                    continue

                # The midpoint of the reference edge must itself be visible;
                # this prevents equal-length inference from reading geometry
                # through a foreground face.
                ref_mid = (wa + wb) * 0.5
                ref_result = SnapResult(location=ref_mid, snap_type="MIDPOINT", obj=obj, element=edge)
                if _result_is_occluded(context, ref_result):
                    continue

                target = start + live_dir * ref_len
                screen_target = view3d_utils.location_3d_to_region_2d(region, rv3d, target)
                if screen_target is None:
                    continue
                px = (Vector(screen_target) - mouse).length
                if px > best_px:
                    continue

                # Avoid treating the segment currently being interactively
                # extended as its own reference when Blender data happens to
                # contain a coincident edge.
                if (wa - start).length < 1e-6 and (wb - target).length < 1e-6:
                    continue
                if (wb - start).length < 1e-6 and (wa - target).length < 1e-6:
                    continue

                best_px = px
                best = (target, wa, wb)

        if best is None:
            self.preview.clear_equal_length_guide()
            return current_point

        target, ref_a, ref_b = best
        self.preview.set_equal_length_guide(ref_a, ref_b, start, target)
        self.preview.set_snap_point(target)
        self.preview.set_snap_feedback("Equal Length", event.mouse_region_x, event.mouse_region_y)
        return target


    # -------------------------------------
    # Mouse Move
    # -------------------------------------

    def on_mouse_move(self, context, event):
        refresh_pencil_cursor()

        if self.state not in {"READY", "DRAWING"}:
            return

        current_point = self._snap(context, event)

        if current_point is None and self.state == "DRAWING":
            current_point = mouse_to_plane(
                context,
                event,
                self.drawing_plane_point,
                self.drawing_plane_normal,
            )

        if current_point is not None and self.state == "DRAWING":
            current_point = self._equal_length_inference(context, event, current_point)
            self.preview.set_points(self.start_point, current_point)
            self.preview.set_measurement_feedback(
                "Length: " + format_length(context, (Vector(current_point) - Vector(self.start_point)).length),
                event.mouse_region_x,
                event.mouse_region_y,
            )

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    # -------------------------------------
    # Left Click
    # -------------------------------------

    def on_left_click(self, context, event):
        if self.state == "READY":
            plane_point, plane_normal, hit_obj, face_index = object_or_floor_plane(
                context, event
            )
            self.drawing_plane_point = Vector(plane_point)
            self.drawing_plane_normal = Vector(plane_normal).normalized()

        point = self._snap(context, event)
        if point is None:
            point = mouse_to_plane(
                context,
                event,
                self.drawing_plane_point,
                self.drawing_plane_normal,
            )

        if point is None:
            debug_print("Invalid point")
            return

        if self.state == "DRAWING":
            point = self._equal_length_inference(context, event, point)

        # Planar snaps stay on the established drawing plane. In Perspective
        # Object Mode, a true X/Y/Z inference is intentionally allowed to
        # leave that plane (e.g. drawing vertically on world Z).
        if self._last_snap_type not in {
            "X_AXIS", "Y_AXIS", "Z_AXIS",
            "X_GRID", "Y_GRID", "Z_GRID",
        }:
            delta = Vector(point) - self.drawing_plane_point
            point = Vector(point) - self.drawing_plane_normal * delta.dot(self.drawing_plane_normal)

        if self.state == "READY":
            self.start_point = point
            self.points.append(point)
            self.state = "DRAWING"
            debug_print("Line started:", point)
            debug_print("Drawing plane established")
            return

        if self.state == "DRAWING":
            if len(self.points) >= 3 and (point - self.points[0]).length < 1e-4:
                debug_print("Polygon closed")
                self.end_point = self.points[0]
                self.create_line(context)

                obj = self.geometry_object if hasattr(self, "geometry_object") else None
                if obj is not None:
                    # In Edit Mode every Line segment has already been inserted
                    # with create_edge() -> _autocut_edge_bmesh().  Closing the
                    # final segment therefore splits the existing surface and
                    # creates the enclosed region as topology.  Creating a new
                    # face here would lay a duplicate coplanar polygon over that
                    # region and cause z-fighting/striping.
                    if context.mode != 'EDIT_MESH':
                        create_face_from_points(obj, self.points)
                    else:
                        # Use the same closed-loop surface reconstruction that
                        # Freehand uses successfully. The Line chain is already
                        # exact/low-density, so this creates an independently
                        # selectable inner region while preserving the outer
                        # surface and avoiding a duplicate coplanar face.
                        if self._cut_closed_loop_edit_mode(context):
                            debug_print(
                                "SketchTools Line: closed Edit Mode loop "
                                "converted to dedicated surface topology"
                            )
                        else:
                            debug_print(
                                "SketchTools Line: dedicated closed-loop cut "
                                "was not applicable; keeping autocut edges"
                            )

                self.preview.clear()
                self.start_point = None
                self.end_point = None
                self.drawing_plane_point = None
                self.drawing_plane_normal = None
                self.points.clear()
                self.state = "READY"
                debug_print("Polygon finished - Line Tool still active")
                return

            self.end_point = point
            debug_print("Adding Point:", point)
            segment_start = Vector(self.start_point).copy()
            segment_end = Vector(point).copy()
            self.create_line(context)
            self.last_segment_start = segment_start
            self.last_segment_end = segment_end
            self.last_segment_obj_name = self.geometry_object.name if self.geometry_object is not None else None
            self.length_input = ""
            self.points.append(point)
            self.start_point = point
            self._last_snap_type = None
            self.preview.clear_equal_length_guide()
            if context.mode != 'EDIT_MESH':
                # Continue free drawing on a plane parallel to the original
                # one but passing through the newest anchor. This prevents a
                # vertical axis segment from snapping the next free point back
                # to the old Z level.
                self.drawing_plane_point = Vector(point)
            self.end_point = None

    # -------------------------------------
    # Right Click / Cancel
    # -------------------------------------

    def on_right_click(
        self,
        context,
        event
    ):

        debug_print(
            "Line chain cancelled"
        )
        
        self.preview.clear()

        self.start_point = None
        self.end_point = None

        self.drawing_plane_point = None
        self.drawing_plane_normal = None

        self.points.clear()

        self.state = "READY"

        debug_print(
            "Line Tool still active"
        )


    def finish(
        self,
        context
    ):

        restore_cursor(
            context
        )

        super().finish(
            context
        )

        self.reset()


    def cancel(
        self,
        context
    ):

        restore_cursor(
            context
        )
                
        super().cancel(
            context
        )


        self.reset()



    # -------------------------------------
    # Keyboard
    # -------------------------------------

    def _adjust_last_segment_length(self, context, new_length):
        if (
            self.last_segment_start is None
            or self.last_segment_end is None
            or not self.last_segment_obj_name
        ):
            return False

        obj = bpy.data.objects.get(self.last_segment_obj_name)
        if obj is None or obj.type != 'MESH':
            return False

        direction = Vector(self.last_segment_end) - Vector(self.last_segment_start)
        if direction.length <= 1e-9:
            return False
        direction.normalize()
        new_end = Vector(self.last_segment_start) + direction * float(new_length)

        inv = obj.matrix_world.inverted()
        old_local = inv @ Vector(self.last_segment_end)
        new_local = inv @ new_end

        moved = False
        tolerance = 2e-5
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            candidates = [v for v in bm.verts if (v.co - old_local).length <= tolerance]
            if candidates:
                min(candidates, key=lambda v: (v.co - old_local).length_squared).co = new_local
                bm.normal_update()
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                moved = True
        else:
            candidates = [v for v in obj.data.vertices if (v.co - old_local).length <= tolerance]
            if candidates:
                min(candidates, key=lambda v: (v.co - old_local).length_squared).co = new_local
                obj.data.update()
                obj.update_tag()
                moved = True

        if not moved:
            return False

        self.last_segment_end = new_end.copy()
        # Continue the active Line chain from the adjusted endpoint.
        if self.points and (Vector(self.points[-1]) - Vector(self.start_point)).length <= tolerance:
            self.points[-1] = new_end.copy()
        self.start_point = new_end.copy()
        if context.mode != 'EDIT_MESH':
            self.drawing_plane_point = new_end.copy()
        context.view_layer.update()
        return True

    def on_key_press(self, context, event):
        # After a segment is placed, typing a value and Enter adjusts that most
        # recent segment along its established direction (SketchUp behaviour).
        if self.last_segment_start is None or self.last_segment_end is None:
            return False

        if event.type in {'BACK_SPACE', 'DEL'}:
            self.length_input = self.length_input[:-1]
            typed = self.length_input or "_"
            self.set_status(context, "Measurements  |  Length: " + typed + "  |  Enter to apply  |  Esc to cancel")
            self.preview.set_measurement_feedback("Length Input: " + typed)
            return True

        if event.type == 'ESC':
            if self.length_input:
                self.length_input = ""
                self.preview.clear_measurement_feedback()
                self.set_status(context, "Line: click next point")
                return True
            return False

        if event.type in {'RET', 'NUMPAD_ENTER'}:
            value = parse_length(self.length_input)
            if value is not None and self._adjust_last_segment_length(context, value):
                self.length_input = ""
                self.set_status(context, "Line length: " + format_length(context, value) + " | click next point")
                self.preview.set_measurement_feedback(
                    "Length: " + format_length(context, value),
                    getattr(event, 'mouse_region_x', 0),
                    getattr(event, 'mouse_region_y', 0),
                )
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
                return True
            if self.length_input:
                self.length_input = ""
                self.set_status(context, "Line: invalid length")
                return True
            return False

        ch = getattr(event, 'unicode', '')
        if ch and (ch.isdigit() or ch.lower() in '.mcftink"'):
            self.length_input += ch.lower()
            self.set_status(
                context,
                "Measurements  |  Length: " + self.length_input + "  |  Enter to apply  |  Esc to cancel"
            )
            # Keep the typed value visible near the cursor as a secondary
            # safeguard while the Blender Status Bar shows the same input.
            self.preview.set_measurement_feedback("Length Input: " + self.length_input)
            return True

        return False


    # -------------------------------------
    # Create Geometry
    # -------------------------------------

    def create_line(
        self,
        context
    ):

        obj = self.geometry_object
        
        if obj is None:
            
            obj = context.active_object

        if obj is None or obj.type != "MESH":

            obj = get_or_create_geometry_object()

            debug_print(
                "Using SketchTools Geometry"
            )
        
        self.geometry_object = obj
        
        
        if self.start_point is None:

            return



        if self.end_point is None:

            return



        if (
            self.start_point
            -
            self.end_point
        ).length < 1e-6:


            debug_print(
                "Line too short"
            )

            return



        create_edge(
            obj,
            self.start_point,
            self.end_point
        )
        
        context.view_layer.update()


        for area in context.screen.areas:

            if area.type == 'VIEW_3D':

                area.tag_redraw()

        debug_print(
            "Line Created"
        )



    # -------------------------------------
    # Reset
    # -------------------------------------

    def reset(
        self
    ):


        self.start_point = None

        self.end_point = None
        
        self.drawing_plane_point = None
        self.drawing_plane_normal = None    

        self.points.clear()
        
        self.geometry_object = None
        self.last_segment_start = None
        self.last_segment_end = None
        self.last_segment_obj_name = None
        self.length_input = ""
        
        self.state = "READY"
        
        self.preview.clear()