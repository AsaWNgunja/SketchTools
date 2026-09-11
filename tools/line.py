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
from ..geometry.projection import object_or_floor_plane, view_fallback_plane
from ..geometry.edit_free_geometry import intersect_face_at_points

from ..geometry.topology import (
    create_edge,
    create_face_from_points,
    _autocut_edge_bmesh,
)
from ..core.session import tool_manager
from ..drawing.preview import PreviewLine

from ..utils.cursor import (
    set_pencil_cursor,
    refresh_pencil_cursor,
    restore_cursor
)

from ..utils.measurements import format_length, parse_length
from ..utils.axis_lock import (
    axis_vector,
    closest_point_on_axis_to_view_ray,
    AXIS_SNAP_TYPES,
)

from ..core.tool_base import SketchToolBase

def _create_object_mode_line_object(context):
    """Create a fresh Object-Mode Line mesh for the current drawing chain.

    v130 rule: Object Mode never appends a newly drawn polygon to an existing
    SketchTools or arbitrary active mesh.  A completed closed polygon therefore
    remains an independent Blender object.
    """
    mesh = bpy.data.meshes.new("SketchTools Line Mesh")
    obj = bpy.data.objects.new("SketchTools Line", mesh)
    collection = getattr(context, "collection", None) or context.scene.collection
    collection.objects.link(obj)
    return obj


class LineTool(SketchToolBase):


    def __init__(self):

        self.start_point = None
        self.end_point = None
        
        self.drawing_plane_point = None
        self.drawing_plane_normal = None
        self._last_snap_type = None
        self._last_snap_result = None
        self._last_axis_direction = None

        self.state = "READY"
        
        self.points = []
        
        self.geometry_object = None
        
        self.preview = PreviewLine()

        # v93 precision memory for SketchUp-style post-draw length entry.
        self.last_segment_start = None
        self.last_segment_end = None
        self.last_segment_obj_name = None
        self.length_input = ""

        # Explicit world-axis lock.  X/Y/Z toggles the current segment onto
        # that axis through start_point; pressing the same key again unlocks.
        self.axis_lock = None
        self.edit_free_space = False
        # v163: when a chain starts on an Edit-Mode face, keep the host mesh
        # completely unchanged until the loop closes.
        self.defer_host_face_topology = False


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
        self.set_status(context, "Line: click first point | X/Y/Z after first point: lock axis")
        
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
        """Return the directed automatic inference retained by the last snap.

        v109 remembers +X/-X/+Y/-Y/+Z/-Z separately. This prevents the
        opposite ray of the same axis from inheriting hysteresis.
        """
        if self._last_axis_direction is not None:
            return self._last_axis_direction
        return {
            "X_AXIS": "X_AXIS",
            "Y_AXIS": "Y_AXIS",
            "Z_AXIS": "Z_AXIS",
            "X_GRID": "X_AXIS",
            "Y_GRID": "Y_AXIS",
            "Z_GRID": "Z_AXIS",
        }.get(self._last_snap_type)

    def _update_axis_guide(self, current_point):
        """Draw the active/inferred world-axis guide through the segment anchor."""
        if self.state != "DRAWING" or self.start_point is None or current_point is None:
            self.preview.clear_axis_guide()
            return

        axis_name = self.axis_lock
        if axis_name is None:
            axis_name = {
                "X_AXIS": "X", "X_GRID": "X",
                "Y_AXIS": "Y", "Y_GRID": "Y",
                "Z_AXIS": "Z", "Z_GRID": "Z",
            }.get(self._last_snap_type)

        if axis_name is None:
            self.preview.clear_axis_guide()
            return

        start = Vector(self.start_point)
        end = Vector(current_point)
        a = axis_vector(axis_name)
        amount = (end - start).dot(a)
        sign = 1.0 if amount >= 0.0 else -1.0
        axis = a * sign
        span = max(abs(amount), 0.5)
        guide_start = start - axis * (span * 0.20)
        guide_end = end + axis * (span * 0.35)
        self.preview.set_axis_guide(guide_start, guide_end, AXIS_SNAP_TYPES[axis_name])

    def _locked_axis_point(self, context, event):
        if self.axis_lock is None or self.start_point is None:
            return None
        ray_origin, ray_direction = get_view_ray(context, event)
        point = closest_point_on_axis_to_view_ray(
            ray_origin, ray_direction, self.start_point, self.axis_lock
        )
        self._last_snap_type = AXIS_SNAP_TYPES[self.axis_lock]
        self.preview.set_snap_point(point)
        self.preview.set_snap_feedback(
            f"{self.axis_lock} Axis Locked",
            event.mouse_region_x, event.mouse_region_y,
        )
        return point

    def _snap(self, context, event):
        start_snap = self._start_point_snap(context, event)
        if start_snap is not None:
            return start_snap

        locked = self._locked_axis_point(context, event)
        if locked is not None:
            return locked

        snap = resolve_snap(
            context,
            event.mouse_region_x,
            event.mouse_region_y,
            plane_point=self.drawing_plane_point,
            plane_normal=self.drawing_plane_normal,
            inference_origin=(self.start_point if self.state == 'DRAWING' else None),
            include_face=True,
            include_grid=True,
            # v108: automatic XYZ inference uses screen-projected world axes
            # in Perspective for both Object and Edit Mode. Manual axis locks
            # remain authoritative in _locked_axis_point().
            world_axis_inference=(
                context.region_data is not None
                and context.region_data.view_perspective == 'PERSP'
            ),
            preferred_axis=self._preferred_axis(),
            # v95: while Line is being drawn, prefer discrete Blender grid
            # crossings that lie on the same inferred X/Y/Z ray. Other tools
            # retain v94 snapping unchanged.
            include_axis_grid=(self.state == 'DRAWING'),
            # Object Mode Line is a true 3D construction tool.  Once an
            # Endpoint/Midpoint/Edge/Center is acquired, preserve that exact
            # world coordinate even when it is not on the fallback plane.
            allow_off_plane_geometry=(context.mode != 'EDIT_MESH'),
            endpoint_pixel_radius=26 if context.mode == 'EDIT_MESH' else 20,
            # v170: Line boundary acquisition is intentionally stronger than
            # ordinary hover snapping. A chain that is crossing an existing
            # face must reliably catch the boundary so the click can terminate
            # and cut the host cell SketchUp-style. Endpoint still outranks Edge
            # in resolve_snap(), so corners remain exact rather than sticky-edge.
            edge_pixel_radius=24 if context.mode == 'EDIT_MESH' else 20,
            midpoint_pixel_radius=22,
            center_pixel_radius=22,
            previous_snap_type=self._last_snap_type,
            snap_hysteresis_pixels=7,
        )
        if snap is not None and snap.valid:
            self._last_snap_result = snap
            self._last_snap_type = snap.snap_type
            if snap.snap_type in {"X_AXIS", "Y_AXIS", "Z_AXIS", "X_GRID", "Y_GRID", "Z_GRID"}:
                self._last_axis_direction = getattr(snap, "axis_direction", None)
            else:
                self._last_axis_direction = None
            self.preview.set_snap_point(snap.location)
            self.preview.set_snap_feedback(
                snap_label(snap), event.mouse_region_x, event.mouse_region_y
            )
            return Vector(snap.location)
        # Keep the previous axis inference as short-lived hysteresis memory.
        # It will be replaced as soon as another valid snap is acquired.
        self._last_snap_result = None
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

        # v177 sparse Line-host sectors: a contained Line polygon must not
        # create one bridge for every polygon vertex.  Use at most FOUR
        # principal host connections (fewer when either loop has <4 vertices).
        # The inner boundary between those anchors is absorbed into the broad
        # surrounding sectors.  This gives Z-shaped polygons the requested
        # minimal 1/2/3/4 boundary connections instead of a radial fan.
        if n_outer < 3 or n_inner < 3:
            return False

        bridge_count = min(4, n_outer, n_inner)

        def _forward_outer_chain(start_idx, end_idx):
            chain = [outer[start_idx]]
            idx = start_idx
            safety = 0
            while idx != end_idx:
                idx = (idx + 1) % n_outer
                chain.append(outer[idx])
                safety += 1
                if safety > n_outer:
                    raise RuntimeError("invalid Line outer chain")
            return chain

        # Pick bridge_count inner anchors around the polygon perimeter.  Search
        # cyclic phases so the four anchors naturally land on representative
        # extremities/corners rather than being tied to the first-click phase.
        best = None
        best_cost = None
        best_inner_ids = None
        for inner_shift in range(n_inner):
            inner_ids = tuple(
                (inner_shift + int(round(k * n_inner / bridge_count))) % n_inner
                for k in range(bridge_count)
            )
            if len(set(inner_ids)) != bridge_count:
                continue
            for outer_shift in range(n_outer):
                anchors = tuple(
                    (outer_shift + int(round(k * n_outer / bridge_count))) % n_outer
                    for k in range(bridge_count)
                )
                if len(set(anchors)) != bridge_count:
                    continue
                cost = sum(
                    (outer2[anchors[k]] - inner2[inner_ids[k]]).length_squared
                    for k in range(bridge_count)
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = anchors
                    best_inner_ids = inner_ids

        if best is None or best_inner_ids is None:
            return False

        # v161: reuse the Line vertices already inserted by the open chain.
        # Creating a second coincident vertex ring here would leave stacked
        # topology on the host face and make later edits unreliable.
        inner_verts = []
        for p in inner_pos:
            existing = None
            for vert in bm.verts:
                if vert.is_valid and (vert.co - p).length < 1e-5:
                    existing = vert
                    break
            inner_verts.append(existing if existing is not None else bm.verts.new(p))
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
            # Exactly bridge_count annular sectors.  Each sector follows the
            # host boundary forward, then follows the corresponding portion of
            # the inner polygon backward.  Intermediate inner vertices remain
            # genuine Z-boundary vertices without receiving helper bridges.
            ring_loops = []
            for k in range(bridge_count):
                nk = (k + 1) % bridge_count
                ii = best_inner_ids[k]
                jj = best_inner_ids[nk]
                ai = best[k]
                aj = best[nk]
                outer_chain = _forward_outer_chain(ai, aj)
                inner_part = inner_chain(ii, jj)

                loop = list(outer_chain) + list(reversed(inner_part))
                clean = []
                for vert in loop:
                    if not clean or clean[-1] is not vert:
                        clean.append(vert)
                if len(clean) > 1 and clean[0] is clean[-1]:
                    clean.pop()
                if len(set(clean)) < 3:
                    raise RuntimeError("degenerate sparse Line ring sector")
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

            if len(created) != bridge_count + 1:
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
                "sparse_sectors=", bridge_count,
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


    def _point_on_face_boundary_local(self, face, point, tol=3e-5):
        """Return True when local-space *point* lies on any boundary edge of face."""
        p = Vector(point)
        for edge in face.edges:
            a = edge.verts[0].co
            b = edge.verts[1].co
            ab = b - a
            if ab.length_squared <= 1e-16:
                continue
            t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
            if (p - (a + ab * t)).length <= tol:
                return True
        return False

    def _face_contains_chain_local(self, face, points_local, tol=4e-5):
        """Planar point-in-polygon test including boundary points."""
        if face is None or len(face.verts) < 3 or not points_local:
            return False
        normal = face.normal.normalized()
        origin = face.verts[0].co.copy()
        if any(abs(normal.dot(Vector(p) - origin)) > 2e-4 for p in points_local):
            return False
        u = face.verts[1].co - origin
        if u.length <= 1e-9:
            return False
        u.normalize()
        v = normal.cross(u)
        if v.length <= 1e-9:
            return False
        v.normalize()
        def xy(p):
            d = Vector(p) - origin
            return Vector((d.dot(u), d.dot(v)))
        poly = [xy(vtx.co) for vtx in face.verts]
        def on_seg(p, a, b):
            ab = b - a
            if ab.length_squared <= 1e-16:
                return (p-a).length <= tol
            t = max(0.0, min(1.0, (p-a).dot(ab)/ab.length_squared))
            return (p-(a+ab*t)).length <= tol
        def inside(p):
            for i,a in enumerate(poly):
                if on_seg(p, a, poly[(i+1)%len(poly)]):
                    return True
            hit=False
            j=len(poly)-1
            for i in range(len(poly)):
                a,b=poly[i],poly[j]
                if ((a.y>p.y)!=(b.y>p.y)):
                    x=(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x
                    if p.x < x:
                        hit=not hit
                j=i
            return hit
        return all(inside(xy(p)) for p in points_local)

    def _topological_closure_face(self, context, end_point):
        """Find a host cell closed by new chain + an existing boundary path.

        v164 deliberately does not mutate BMesh here.  Start/end may lie in the
        middle of existing edges; those edges are split later by create_edge()
        when the completed transaction is committed.
        """
        if context.mode != 'EDIT_MESH' or self.edit_free_space:
            return None
        # v170: a topological cut does NOT need a closed/new polygon.
        # Even a single new chord from one host boundary point to another
        # (Edge -> Edge, Vertex -> Edge, etc.) divides the existing face.
        # self.points already contains the start and any intermediate anchors;
        # end_point is the boundary point currently being clicked.
        if len(self.points) < 1:
            return None
        obj = context.edit_object
        if obj is None or obj.type != 'MESH':
            return None
        inv = obj.matrix_world.inverted()
        chain = [inv @ Vector(p) for p in self.points] + [inv @ Vector(end_point)]
        if (chain[0] - chain[-1]).length <= 1e-5:
            return None
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        candidates=[]
        for face in bm.faces:
            if not face.is_valid:
                continue
            # Both ends must lie on the SAME existing cell boundary.  That
            # existing boundary supplies the missing return path (7 -> 1).
            if not self._point_on_face_boundary_local(face, chain[0]):
                continue
            if not self._point_on_face_boundary_local(face, chain[-1]):
                continue

            # v179: SketchUp-style existing-boundary closure must not require
            # the newly drawn chain to remain inside the host face.  A common
            # case is drawing a new polygon OUTSIDE a concave face, starting
            # at one boundary vertex/edge and ending at another; the existing
            # face boundary between those endpoints is the missing return path.
            #
            # Keep the operation planar with the candidate face, but allow the
            # intermediate Line points to lie on either side of its 2D polygon.
            try:
                normal = face.normal.normalized()
                origin = face.verts[0].co
                if any(abs(normal.dot(Vector(p) - origin)) > 2e-4 for p in chain):
                    continue
            except Exception:
                continue

            # Prefer the smallest qualifying cell so an already-intersected
            # Circle×Rectangle patch is chosen instead of an exterior region.
            try:
                area = float(face.calc_area())
            except Exception:
                area = 1e30
            candidates.append((area, face))
        return min(candidates, key=lambda item:item[0])[1] if candidates else None

    def _finish_current_chain(self, context, dedicated_face_cut=False, topological=False):
        """Reset Line state after a completed explicit/topological loop."""
        obj = self.geometry_object if hasattr(self, "geometry_object") else None
        if obj is not None and context.mode != 'EDIT_MESH':
            try:
                if any(poly for poly in obj.data.polygons):
                    obj.name = "SketchTools Polygon"
                    obj.data.name = "SketchTools Polygon Mesh"
            except ReferenceError:
                pass
        if context.mode == 'EDIT_MESH':
            debug_print(
                "SketchTools Line: Edit Mode loop finalized by "
                + ("topological closure through existing boundary" if topological else
                   "dedicated host-face cut" if dedicated_face_cut else
                   "clean planar graph retopology")
            )
        self.preview.clear()
        self.start_point = None
        self.end_point = None
        self.drawing_plane_point = None
        self.drawing_plane_normal = None
        self.points.clear()
        if context.mode != 'EDIT_MESH':
            self.geometry_object = None
        self.axis_lock = None
        self.defer_host_face_topology = False
        self._last_snap_type = None
        self.state = "READY"
        self.set_status(context, "Line: click first point | X/Y/Z after first point: lock axis")
        debug_print("Polygon finished - Line Tool still active")


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
            self.preview.clear_axis_guide()
            current_point = mouse_to_plane(
                context,
                event,
                self.drawing_plane_point,
                self.drawing_plane_normal,
            )

        if current_point is not None and self.state == "DRAWING":
            if self.axis_lock is None:
                current_point = self._equal_length_inference(context, event, current_point)
            self._update_axis_guide(current_point)
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
        precision_types = {"ENDPOINT", "MIDPOINT", "CENTER", "EDGE", "ORIGIN"}

        # Resolve the snap before establishing the fallback plane.  This keeps
        # a visible 3D vertex/edge authoritative instead of first choosing a
        # nearby screen-facing plane and later projecting the snap onto it.
        point = self._snap(context, event)
        acquired_type = self._last_snap_type if point is not None else None

        if self.state == "READY":
            plane_point, plane_normal, hit_obj, face_index = object_or_floor_plane(
                context, event
            )
            if context.mode == 'EDIT_MESH' and (hit_obj is None or hit_obj != context.edit_object or face_index is None):
                # v174: ALL Edit-Mode Line chains are transactional while open.
                # A chain may begin in free space and only cross existing faces later;
                # committing each intermediate click used to let create_edge() rebuild
                # the connected planar network and make unrelated faces disappear,
                # then reappear on a later click.  Keep BMesh completely untouched
                # until the chain actually terminates/closes.
                #
                # If the first click is a precise boundary snap and a linked host face
                # is available, use that face's plane.  Otherwise retain the normal
                # free-space fallback plane, but STILL defer topology.
                snap_result = getattr(self, "_last_snap_result", None)

                # v175: Never dereference the BMVert/BMEdge stored inside a
                # SnapResult here.  A prior Edit-Mode topology update can invalidate
                # that BMesh element even though the snapped world coordinate is still
                # perfectly valid.  Reacquire the host face from the CURRENT BMesh
                # using only stable data: object identity + exact snapped coordinate.
                host_face = None
                boundary_host = False
                if (
                    acquired_type in {"ENDPOINT", "MIDPOINT", "EDGE"}
                    and snap_result is not None
                    and getattr(snap_result, "obj", None) is context.edit_object
                    and point is not None
                ):
                    try:
                        bm_now = bmesh.from_edit_mesh(context.edit_object.data)
                        bm_now.verts.ensure_lookup_table()
                        bm_now.edges.ensure_lookup_table()
                        bm_now.faces.ensure_lookup_table()
                        p_local = context.edit_object.matrix_world.inverted() @ Vector(point)

                        candidates = []
                        for face in bm_now.faces:
                            try:
                                if self._point_on_face_boundary_local(face, p_local, tol=8e-5):
                                    candidates.append(face)
                            except ReferenceError:
                                continue

                        if candidates:
                            # Prefer the smallest local host cell when several coplanar
                            # faces share the same snapped vertex/edge.
                            host_face = min(candidates, key=lambda f: max(float(f.calc_area()), 0.0))
                            boundary_host = True
                    except (ReferenceError, RuntimeError, ValueError):
                        host_face = None
                        boundary_host = False

                if boundary_host and host_face is not None:
                    plane_point = context.edit_object.matrix_world @ host_face.verts[0].co
                    plane_normal = (context.edit_object.matrix_world.to_3x3() @ host_face.normal).normalized()
                    self.edit_free_space = False
                else:
                    plane_point, plane_normal = view_fallback_plane(context)
                    self.edit_free_space = True
                self.defer_host_face_topology = True
            else:
                self.edit_free_space = False
                # v174: even a chain that starts directly on a visible host face
                # remains preview-only until a real completion event.
                self.defer_host_face_topology = (context.mode == 'EDIT_MESH')
            self.drawing_plane_point = Vector(plane_point)
            self.drawing_plane_normal = Vector(plane_normal).normalized()

            # In Object Mode start the working plane through the exact acquired
            # geometry coordinate.  Its normal is still the sensible face/view
            # normal selected above, so ordinary free drawing remains familiar.
            if (
                point is not None
                and context.mode != 'EDIT_MESH'
                and acquired_type in precision_types
            ):
                self.drawing_plane_point = Vector(point).copy()

        if point is None:
            point = mouse_to_plane(
                context,
                event,
                self.drawing_plane_point,
                self.drawing_plane_normal,
            )
            acquired_type = None

        if point is None:
            debug_print("Invalid point")
            return

        if self.state == "DRAWING" and self.axis_lock is None:
            point = self._equal_length_inference(context, event, point)

        # v100: geometry snaps are authoritative in Object Mode.  Do not show
        # "Endpoint" and then numerically move the click onto the drawing plane.
        # Edit Mode stays planar because its purpose is cutting the hit surface.
        preserve_exact = (
            context.mode != 'EDIT_MESH'
            and acquired_type in precision_types
        )
        if (
            not preserve_exact
            and self._last_snap_type not in {
                "X_AXIS", "Y_AXIS", "Z_AXIS",
                "X_GRID", "Y_GRID", "Z_GRID",
            }
        ):
            delta = Vector(point) - self.drawing_plane_point
            point = Vector(point) - self.drawing_plane_normal * delta.dot(self.drawing_plane_normal)

        if self.state == "READY":
            self.start_point = point
            self.points.append(point)
            if self.defer_host_face_topology:
                self.preview.set_chain_points(self.points)
            self.state = "DRAWING"
            self.set_status(context, "Line: click next point | X/Y/Z: lock axis")
            debug_print("Line started:", point)
            debug_print("Drawing plane established")
            return

        if self.state == "DRAWING":
            if len(self.points) >= 3 and (point - self.points[0]).length < 1e-4:
                debug_print("Polygon closed")
                self.end_point = self.points[0]

                # v161 — closed Line polygon drawn wholly on an existing Edit
                # Mode face gets a dedicated local face subdivision.  The
                # general v137 Line graph remains the authority for crossings
                # and free-space polygons; this narrow path is used only when
                # one containing coplanar host face can be identified.
                dedicated_face_cut = False
                if context.mode == 'EDIT_MESH' and not self.edit_free_space:
                    try:
                        dedicated_face_cut = self._cut_closed_loop_edit_mode(context)
                    except Exception as exc:
                        debug_print("SketchTools Line v161 host-face cut failed:", repr(exc))
                        dedicated_face_cut = False

                if not dedicated_face_cut:
                    if context.mode == 'EDIT_MESH' and self.defer_host_face_topology:
                        self._commit_deferred_chain_edit_mode(context, close_loop=True)
                    else:
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
                        # v130: create_edge() already performs live minimal-cell
                        # filling.  Do not lay a second polygon over that topology.
                        # Finalize this chain as its own Blender object, then the
                        # next Line chain will allocate a fresh object.
                        try:
                            if any(poly for poly in obj.data.polygons):
                                obj.name = "SketchTools Polygon"
                                obj.data.name = "SketchTools Polygon Mesh"
                        except ReferenceError:
                            pass
                    else:
                        # v136: every Edit-Mode segment, including the closing
                        # segment, is already handled by create_edge()'s clean
                        # planar-graph retopology.  Do NOT run the legacy second
                        # face/cutter pipeline here: that competing closure path
                        # was able to recreate the external/spanning faces seen
                        # in v130-v135.
                        debug_print(
                            "SketchTools Line: Edit Mode loop finalized by "
                            + ("dedicated host-face cut" if dedicated_face_cut else "clean planar graph retopology")
                        )

                self.preview.clear()
                self.start_point = None
                self.end_point = None
                self.drawing_plane_point = None
                self.drawing_plane_normal = None
                self.points.clear()
                if context.mode != 'EDIT_MESH':
                    # The completed polygon remains independent.  The next
                    # Object-Mode chain must not reuse it.
                    self.geometry_object = None
                self.axis_lock = None
                self.defer_host_face_topology = False
                self._last_snap_type = None
                self.state = "READY"
                self.set_status(context, "Line: click first point | X/Y/Z after first point: lock axis")
                debug_print("Polygon finished - Line Tool still active")
                return

            # v170 SketchUp-style boundary termination: if the current click
            # lands anywhere on the boundary of the same existing host cell as
            # the chain start, the existing mesh boundary supplies the return
            # path. This includes direct Edge->Edge chords and multi-segment
            # Vertex/Edge/Midpoint combinations; no redraw back to the start.
            topo_face = self._topological_closure_face(context, point)
            if topo_face is not None and self.defer_host_face_topology:
                debug_print("Topological Line cut terminated on existing boundary")
                self.end_point = Vector(point).copy()
                self.points.append(Vector(point).copy())
                # Commit ONLY the newly drawn chain. _autocut_edge_bmesh splits
                # start/end host edges at their exact On Edge positions; v137
                # local planar retopology then recognizes the new bounded cell.
                self._commit_deferred_chain_edit_mode(context, close_loop=False)
                self._finish_current_chain(context, topological=True)
                return

            self.end_point = point
            debug_print("Adding Point:", point)
            segment_start = Vector(self.start_point).copy()
            segment_end = Vector(point).copy()
            if context.mode == 'EDIT_MESH' and self.defer_host_face_topology:
                # v163: display/store the segment only. Do not call create_edge()
                # because its live planar rebuild can temporarily delete a host
                # cell created by Circle×Rectangle or other prior intersections.
                pass
            else:
                self.create_line(context)
            self.last_segment_start = segment_start
            self.last_segment_end = segment_end
            self.last_segment_obj_name = self.geometry_object.name if self.geometry_object is not None else None
            self.length_input = ""
            self.points.append(point)
            if self.defer_host_face_topology:
                self.preview.set_chain_points(self.points)
            self.start_point = point
            self._last_snap_type = None
            self.preview.clear_equal_length_guide()
            self.preview.clear_axis_guide()
            if context.mode != 'EDIT_MESH':
                # Continue free drawing on a plane parallel to the original
                # one but passing through the newest anchor. This prevents a
                # vertical axis segment from snapping the next free point back
                # to the old Z level.
                self.drawing_plane_point = Vector(point)
            self.end_point = None
            self._set_axis_status(context)

    # -------------------------------------
    # Right Click / Cancel
    # -------------------------------------

    def on_right_click(
        self,
        context,
        event
    ):

        # v174: prior to the all-transactional Edit-Mode change, each clicked
        # segment had already been committed before RMB ended the chain.  Preserve
        # that user-visible behaviour: RMB ends an open transactional chain and
        # commits all accepted points once, rather than silently discarding them.
        if context.mode == 'EDIT_MESH' and self.defer_host_face_topology and len(self.points) >= 2:
            try:
                # v176: RMB finishing an OPEN interior chain is a wire-only
                # commit.  Running v137 whole-planar-cell reconstruction here
                # created the unwanted long corner diagonals/fans seen when a
                # simple Z was drawn on a face.
                self._commit_deferred_chain_wire_edit_mode(context)
                debug_print("Line open transactional chain committed on RMB (wire-only)")
            except Exception as exc:
                debug_print("SketchTools Line v174 RMB commit failed:", repr(exc))
        else:
            debug_print("Line chain cancelled")
        
        self.preview.clear()

        self.start_point = None
        self.end_point = None

        self.drawing_plane_point = None
        self.drawing_plane_normal = None

        self.points.clear()

        self.state = "READY"
        self.axis_lock = None
        self.defer_host_face_topology = False
        self.set_status(context, "Line: click first point | X/Y/Z after first point: lock axis")

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

    def _set_axis_status(self, context):
        if self.state != "DRAWING":
            self.set_status(context, "Line: click first point | X/Y/Z after first point: lock axis")
            return
        if self.axis_lock:
            axis = self.axis_lock
            self.set_status(
                context,
                f"Line | {axis} AXIS LOCKED | LMB: place point | {axis}: unlock | X/Y/Z: switch axis | RMB: cancel chain",
            )
        else:
            self.set_status(context, "Line: click next point | X/Y/Z: lock axis | RMB: cancel chain")

    def _toggle_axis_lock(self, context, axis):
        if self.state != "DRAWING" or self.start_point is None:
            return False
        axis = axis.upper()
        self.axis_lock = None if self.axis_lock == axis else axis
        self._last_snap_type = AXIS_SNAP_TYPES.get(self.axis_lock) if self.axis_lock else None
        if self.axis_lock is None:
            self.preview.clear_axis_guide()
            self.preview.clear_snap_feedback()
        self._set_axis_status(context)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return True

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
        if event.type in {'X', 'Y', 'Z'} and not self.length_input:
            if self._toggle_axis_lock(context, event.type):
                return True

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

    def _commit_deferred_chain_wire_edit_mode(self, context):
        """Commit an OPEN transactional Line chain without planar face rebuild.

        v176: an unfinished/open polyline drawn across the interior of a face
        (for example a Z) must not force Blender to invent bridge diagonals to
        distant host-boundary vertices.  Insert/split only the geometry that the
        user actually drew and leave existing faces intact.  If the chain later
        reaches a valid host boundary, the normal topological-closure path is
        used instead and performs the real face split.
        """
        obj = context.edit_object
        if obj is None or obj.type != 'MESH' or len(self.points) < 2:
            return False

        inv = obj.matrix_world.inverted()
        pts = [inv @ Vector(p) for p in self.points]
        bm = bmesh.from_edit_mesh(obj.data)
        changed = False
        for a, b in zip(pts, pts[1:]):
            if (b - a).length <= 1e-7:
                continue
            changed = _autocut_edge_bmesh(bm, a, b) or changed

        bm.normal_update()
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
        self.geometry_object = obj
        context.view_layer.update()
        debug_print("Line v176 open wire committed without host-face retopology")
        return changed


    def _commit_deferred_chain_edit_mode(self, context, close_loop=False):
        """Commit a v163 transactional Edit-Mode chain only when needed.

        This fallback is used when a chain began on a face but did not qualify
        for the dedicated contained host-face cut (for example it crossed out
        of that face). All coordinates are stored as plain vectors, so no stale
        BMesh element references survive between topology operations.
        """
        obj = context.edit_object
        if obj is None or obj.type != 'MESH' or len(self.points) < 2:
            return False
        pts = [Vector(p).copy() for p in self.points]
        if close_loop and len(pts) >= 3:
            pts.append(pts[0].copy())
        for a, b in zip(pts, pts[1:]):
            if (b - a).length <= 1e-7:
                continue
            create_edge(obj, a, b)
        self.geometry_object = obj
        context.view_layer.update()
        return True


    def create_line(
        self,
        context
    ):

        # v127: Blender RNA references can outlive the object they point to.
        # Never dereference a cached geometry object until it has been proven
        # alive; an empty scene after deleting SketchTools Geometry used to
        # raise: ReferenceError: StructRNA of type Object has been removed.
        obj = self.geometry_object
        try:
            cached_valid = obj is not None and obj.name in bpy.data.objects and obj.type == "MESH"
        except ReferenceError:
            cached_valid = False
            obj = None

        if not cached_valid:
            if context.mode == 'EDIT_MESH':
                obj = context.edit_object
                try:
                    active_valid = obj is not None and obj.name in bpy.data.objects and obj.type == "MESH"
                except ReferenceError:
                    active_valid = False
                    obj = None
                if not active_valid:
                    debug_print("Line: no valid Edit Mode mesh object")
                    return
            else:
                # v130: every Object-Mode drawing chain owns a fresh mesh.
                # Never append Line geometry to an arbitrary active object.
                obj = _create_object_mode_line_object(context)
                debug_print("Created fresh SketchTools Line object")

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
        self.axis_lock = None
        self.edit_free_space = False
        self.defer_host_face_topology = False

        # v126: automatic axis inference is transient UI state.  It must not
        # survive cancel/tool-switch and reappear when Line is selected again.
        self._last_snap_type = None
        self._last_axis_direction = None
        
        self.state = "READY"
        
        self.preview.clear()