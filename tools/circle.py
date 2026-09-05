from ..utils.logging import debug_print
import bpy
import math
import time
import bmesh
from itertools import combinations
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import gpu
from gpu_extras.batch import batch_for_shader
from ..drawing.snap_feedback import SnapFeedbackMixin

from ..core.tool_base import SketchToolBase
from ..core.session import tool_manager
from ..geometry.raycast import get_view_ray, mouse_to_plane
from ..geometry.projection import view_fallback_plane
from ..geometry.snapping import (
    find_nearest_endpoint,
    find_nearest_midpoint,
    find_nearest_edge,
    find_origin_snap,
    find_grid_snap,
    resolve_snap,
    snap_label,
)
from ..utils.cursor import restore_cursor
from ..utils.measurements import format_length
from ..utils.axis_lock import PLANE_LABELS, set_world_axis_plane


class CirclePreview(SnapFeedbackMixin):
    def __init__(self):
        self.points = []
        self.snap_point = None
        self._init_snap_feedback()

    def set_points(self, points):
        self.points = list(points) if points else []

    def set_snap_point(self, point):
        self.snap_point = point

    def clear_snap_point(self):
        self.snap_point = None
        self.clear_snap_feedback()

    def clear(self):
        self.points = []
        self.snap_point = None
        self.clear_snap_feedback()
        self.clear_measurement_feedback()

    def draw(self):
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")

        if len(self.points) >= 3:
            coords = self.points + [self.points[0]]
            batch = batch_for_shader(shader, "LINE_STRIP", {"pos": coords})
            gpu.state.line_width_set(3.0)
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))
            batch.draw(shader)
            gpu.state.line_width_set(1.0)

        if self.snap_point is not None:
            batch = batch_for_shader(shader, "POINTS", {"pos": [self.snap_point]})
            gpu.state.point_size_set(10.0)
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))
            batch.draw(shader)
            gpu.state.point_size_set(1.0)


def rebuild_parametric_circle(obj, segments):
    """Rebuild a pristine SketchTools circle without changing its transform."""
    if obj is None or obj.type != 'MESH' or not obj.get("sketchtools_circle"):
        return False

    segments = max(3, min(256, int(segments)))
    radius = float(obj.get("sketchtools_circle_radius", 0.0))
    if radius <= 0.0:
        return False

    u = Vector(obj.get("sketchtools_circle_axis_u", (1.0, 0.0, 0.0))).normalized()
    v = Vector(obj.get("sketchtools_circle_axis_v", (0.0, 1.0, 0.0))).normalized()

    verts = [
        tuple(
            u * (math.cos((2.0 * math.pi * i) / segments) * radius)
            + v * (math.sin((2.0 * math.pi * i) / segments) * radius)
        )
        for i in range(segments)
    ]

    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata(verts, [], [tuple(range(segments))])
    mesh.update()

    obj["sketchtools_circle_segments"] = segments
    return True


def circle_is_pristine(obj):
    if obj is None or obj.type != 'MESH' or not obj.get("sketchtools_circle"):
        return False
    expected = int(obj.get("sketchtools_circle_segments", 0))
    mesh = obj.data
    return (
        expected >= 3
        and len(mesh.vertices) == expected
        and len(mesh.edges) == expected
        and len(mesh.polygons) == 1
        and len(mesh.polygons[0].vertices) == expected
    )


class CircleTool(SketchToolBase):
    tool_name = "Circle"

    def __init__(self, segments=32):
        super().__init__()
        self.segments = max(3, min(256, int(segments)))
        self.state = "READY"
        self.center = None
        self.drawing_plane_point = None
        self.drawing_plane_normal = None
        self.axis_u = None
        self.axis_v = None
        self.preview = CirclePreview()
        self.last_circle_name = None
        self.segment_input = ""
        self.target_edit_face = None
        self.target_edit_face_index = None
        self.target_edit_face_vert_indices = None
        self.target_edit_face_signature = None
        self._last_edit_error = None
        self.known_centers = []
        self._last_snap_type = None
        self.axis_lock = None
        self._base_plane_point = None
        self._base_plane_normal = None
        self._base_axis_u = None
        self._base_axis_v = None

    def start(self, context):
        super().start(context)
        self.reset_operation()
        self.set_status(context, "Circle: click center | after center X/Y/Z: lock circle plane")

    def _redraw(self, context):
        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

    def _snap(self, context, event):
        snap = resolve_snap(
            context,
            event.mouse_region_x,
            event.mouse_region_y,
            plane_point=self.drawing_plane_point,
            plane_normal=self.drawing_plane_normal,
            extra_centers=getattr(self, "known_centers", None),
            inference_origin=(self.center if self.center is not None else None),
            include_face=True,
            include_grid=(context.mode != 'EDIT_MESH'),
            preferred_axis=(
                self._last_snap_type
                if self._last_snap_type in {"X_AXIS", "Y_AXIS", "Z_AXIS"}
                else None
            ),
        )
        if snap is not None and snap.valid:
            self._last_snap_type = snap.snap_type
            self.preview.set_snap_point(snap.location)
            self.preview.set_snap_feedback(
                snap_label(snap), event.mouse_region_x, event.mouse_region_y
            )
            return Vector(snap.location)
        self.preview.clear_snap_point()
        self.preview.clear_snap_feedback()
        return None

    def _finish_plane_basis(self, center, candidates):
        axis_u = None
        normal = self.drawing_plane_normal

        for seed in candidates:
            projected = seed - normal * seed.dot(normal)
            if projected.length > 1e-8:
                axis_u = projected.normalized()
                break

        if axis_u is None:
            return False

        self.axis_u = axis_u
        self.axis_v = normal.cross(axis_u).normalized()

        delta = Vector(center) - self.drawing_plane_point
        self.center = Vector(center) - normal * delta.dot(normal)
        return True

    def _setup_plane(self, context, event, center):
        ray_origin, ray_direction = get_view_ray(context, event)

        self.target_edit_face = None
        self.target_edit_face_index = None
        self.target_edit_face_vert_indices = None
        self.target_edit_face_signature = None

        if context.mode == 'EDIT_MESH':
            obj = context.edit_object
            if obj is None or obj.type != 'MESH':
                return False

            bm = bmesh.from_edit_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            bm.verts.index_update()
            bm.faces.index_update()

            inv = obj.matrix_world.inverted()
            local_origin = inv @ ray_origin
            local_direction = (inv.to_3x3() @ ray_direction).normalized()
            tree = BVHTree.FromBMesh(bm)

            hit_location, hit_normal, face_index, distance = tree.ray_cast(
                local_origin,
                local_direction,
            )

            if hit_location is None or face_index is None:
                return False

            if not (0 <= face_index < len(bm.faces)):
                return False

            face = bm.faces[face_index]
            self.target_edit_face = face
            self.target_edit_face_index = face.index
            self.target_edit_face_vert_indices = tuple(
                sorted(v.index for v in face.verts)
            )
            self.target_edit_face_signature = tuple(
                sorted(
                    (round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6))
                    for v in face.verts
                )
            )

            self.drawing_plane_point = obj.matrix_world @ hit_location
            self.drawing_plane_normal = (
                obj.matrix_world.to_3x3() @ face.normal
            ).normalized()

            candidates = (
                (obj.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized(),
                (obj.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized(),
                (obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized(),
            )
            return self._finish_plane_basis(center, candidates)

        depsgraph = context.evaluated_depsgraph_get()
        hit, location, normal, face_index, obj, matrix = context.scene.ray_cast(
            depsgraph,
            ray_origin,
            ray_direction,
        )

        if hit and obj is not None and obj.type == 'MESH':
            self.drawing_plane_point = Vector(location)
            self.drawing_plane_normal = Vector(normal).normalized()
            candidates = (
                (obj.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized(),
                (obj.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized(),
                (obj.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized(),
            )
        else:
            self.drawing_plane_point, self.drawing_plane_normal = view_fallback_plane(context)
            candidates = (
                Vector((1.0, 0.0, 0.0)),
                Vector((0.0, 1.0, 0.0)),
                Vector((0.0, 0.0, 1.0)),
            )

        return self._finish_plane_basis(center, candidates)

    def _event_point(self, context, event):
        snap = self._snap(context, event)
        if snap is not None:
            delta = snap - self.drawing_plane_point
            return snap - self.drawing_plane_normal * delta.dot(self.drawing_plane_normal)

        return mouse_to_plane(
            context,
            event,
            self.drawing_plane_point,
            self.drawing_plane_normal,
        )

    def _circle_points(self, radius_point):
        delta = Vector(radius_point) - self.center
        u = delta.dot(self.axis_u)
        v = delta.dot(self.axis_v)
        radius = math.sqrt(u * u + v * v)

        if radius < 1e-8:
            return []

        # Rotate the basis so preview vertex 0 follows the radius cursor.
        radial_u = (self.axis_u * u + self.axis_v * v).normalized()
        radial_v = self.drawing_plane_normal.cross(radial_u).normalized()

        return [
            self.center
            + radial_u * (math.cos((2.0 * math.pi * i) / self.segments) * radius)
            + radial_v * (math.sin((2.0 * math.pi * i) / self.segments) * radius)
            for i in range(self.segments)
        ]

    def on_mouse_move(self, context, event):
        if self.state == "READY":
            self._snap(context, event)
            self._redraw(context)
            return

        point = self._event_point(context, event)
        if point is None:
            return

        points = self._circle_points(point)
        self.preview.set_points(points)
        if points:
            radius = (Vector(points[0]) - Vector(self.center)).length
            self.preview.set_measurement_feedback(
                f"Radius: {format_length(context, radius)}   Segments: {self.segments}",
                event.mouse_region_x,
                event.mouse_region_y,
            )
        self._redraw(context)

    def _create_concentric_outer_cut(self, context, obj, inner_face, points):
        """Expand a circle concentrically through the four surrounding ring faces.

        This is the Edit Mode counterpart of SketchUp's concentric-circle
        workflow: drawing a larger circle from the center of an existing
        SketchTools circle splits the surrounding planar ring instead of
        creating coplanar/overlapping faces.
        """
        bm = bmesh.from_edit_mesh(obj.data)
        inv = obj.matrix_world.inverted()

        def to_2d(local_point):
            world = obj.matrix_world @ local_point
            delta = world - self.drawing_plane_point
            return Vector((delta.dot(self.axis_u), delta.dot(self.axis_v)))

        def signed_area(poly):
            area = 0.0
            for i, a in enumerate(poly):
                b = poly[(i + 1) % len(poly)]
                area += a.x * b.y - b.x * a.y
            return area * 0.5

        def point_in_polygon(point, poly):
            inside = False
            px, py = point.x, point.y
            for i, a in enumerate(poly):
                b = poly[(i + 1) % len(poly)]
                ab = b - a
                ap = point - a
                cross = ab.x * ap.y - ab.y * ap.x
                if abs(cross) <= 1e-7:
                    dot = ap.dot(ab)
                    if -1e-7 <= dot <= ab.length_squared + 1e-7:
                        return True
                if (a.y > py) != (b.y > py):
                    x_hit = (b.x - a.x) * (py - a.y) / (b.y - a.y) + a.x
                    if px < x_hit:
                        inside = not inside
            return inside

        center_local = inv @ self.center
        center_2d = to_2d(center_local)
        old_verts = list(inner_face.verts)
        old_2d = [to_2d(v.co) for v in old_verts]
        if len(old_verts) < 4:
            return False

        old_radii = [(p - center_2d).length for p in old_2d]
        old_radius = sum(old_radii) / len(old_radii)
        if old_radius <= 1e-8:
            return False
        if max(abs(r - old_radius) for r in old_radii) > max(1e-4, old_radius * 0.03):
            return False

        new_positions = [inv @ Vector(point) for point in points]
        new_2d = [to_2d(p) for p in new_positions]
        new_radius = sum((p - center_2d).length for p in new_2d) / len(new_2d)
        if new_radius <= old_radius * 1.001:
            return False

        ring_faces = set()
        for edge in inner_face.edges:
            for face in edge.link_faces:
                if face is not inner_face and face.is_valid:
                    # Only consume the directly surrounding coplanar ring.
                    if abs(face.normal.dot(inner_face.normal)) > 0.999:
                        ring_faces.add(face)

        if len(ring_faces) < 2:
            return False

        patch_faces = set(ring_faces)
        patch_faces.add(inner_face)

        # Outer boundary of center + surrounding ring.  Internal circle and
        # radial edges have two patch faces; only the original outer boundary
        # occurs once.
        boundary_edges = []
        seen_edges = set()
        for face in patch_faces:
            for edge in face.edges:
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                count = sum(1 for f in edge.link_faces if f in patch_faces)
                if count == 1:
                    boundary_edges.append(edge)

        if len(boundary_edges) < 3:
            return False

        adjacency = {}
        for edge in boundary_edges:
            a, b = edge.verts
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return False

        outer_verts = []
        start = boundary_edges[0].verts[0]
        previous = None
        current = start
        for _ in range(len(boundary_edges) + 1):
            outer_verts.append(current)
            choices = adjacency[current]
            nxt = choices[0] if choices[0] is not previous else choices[1]
            previous, current = current, nxt
            if current is start:
                break
        if current is not start or len(outer_verts) < 3:
            return False

        outer_2d = [to_2d(v.co) for v in outer_verts]
        if signed_area(outer_2d) < 0.0:
            outer_verts.reverse()
            outer_2d.reverse()

        if not all(point_in_polygon(p, outer_2d) for p in new_2d):
            return False

        def normalize_loop(verts, pts):
            verts = list(verts)
            pts = list(pts)
            if signed_area(pts) < 0.0:
                verts.reverse()
                pts.reverse()
            angles = [math.atan2((p-center_2d).y, (p-center_2d).x) % (2.0*math.pi) for p in pts]
            start_idx = min(range(len(angles)), key=lambda i: angles[i])
            return (
                verts[start_idx:] + verts[:start_idx],
                pts[start_idx:] + pts[:start_idx],
            )

        old_verts, old_2d = normalize_loop(old_verts, old_2d)

        # New loop vertices do not exist yet, but orient/rotate positions now.
        dummy = list(range(len(new_positions)))
        dummy, new_2d = normalize_loop(dummy, new_2d)
        new_positions = [new_positions[i] for i in dummy]

        def quarter_indices(poly):
            angles = [math.atan2((p-center_2d).y, (p-center_2d).x) % (2.0*math.pi) for p in poly]
            targets = (0.0, 0.5*math.pi, math.pi, 1.5*math.pi)
            result = []
            for target in targets:
                def angular_distance(i):
                    d = abs(angles[i] - target)
                    return min(d, 2.0*math.pi-d)
                result.append(min(range(len(poly)), key=angular_distance))
            if len(set(result)) < 4:
                return None
            return result

        old_q = quarter_indices(old_2d)
        new_q = quarter_indices(new_2d)
        if old_q is None or new_q is None:
            return False

        n_outer = len(outer_verts)
        best_anchors = None
        best_cost = None
        new_quarter_pts = [new_2d[i] for i in new_q]
        for combo in combinations(range(n_outer), 4):
            combo = tuple(combo)
            for shift in range(4):
                anchors = combo[shift:] + combo[:shift]
                cost = sum((outer_2d[anchors[i]] - new_quarter_pts[i]).length_squared for i in range(4))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_anchors = anchors
        if best_anchors is None:
            return False

        def arc(loop, start_idx, end_idx):
            result = [loop[start_idx]]
            idx = start_idx
            for _ in range(len(loop) + 1):
                if idx == end_idx:
                    return result
                idx = (idx + 1) % len(loop)
                result.append(loop[idx])
            raise RuntimeError("invalid concentric arc")

        def outer_chain(start_idx, end_idx):
            result = [outer_verts[start_idx]]
            idx = start_idx
            for _ in range(n_outer + 1):
                if idx == end_idx:
                    return result
                idx = (idx + 1) % n_outer
                result.append(outer_verts[idx])
            raise RuntimeError("invalid outer chain")

        original_ring_faces = [tuple(face.verts) for face in ring_faces if face.is_valid]
        created_faces = []
        new_verts = []

        try:
            bmesh.ops.delete(bm, geom=list(ring_faces), context='FACES_ONLY')
            new_verts = [bm.verts.new(position) for position in new_positions]

            # Four annulus sectors between the existing circle and new circle.
            for sector in range(4):
                nxt = (sector + 1) % 4
                outer_arc = arc(new_verts, new_q[sector], new_q[nxt])
                inner_arc = arc(old_verts, old_q[sector], old_q[nxt])
                verts = outer_arc + list(reversed(inner_arc))
                clean = []
                for v in verts:
                    if not clean or clean[-1] is not v:
                        clean.append(v)
                if len(clean) > 1 and clean[0] is clean[-1]:
                    clean.pop()
                created_faces.append(bm.faces.new(tuple(clean)))

            # Four sectors between the new circle and the original patch boundary.
            for sector in range(4):
                nxt = (sector + 1) % 4
                chain = outer_chain(best_anchors[sector], best_anchors[nxt])
                circle_arc = arc(new_verts, new_q[sector], new_q[nxt])
                verts = chain + list(reversed(circle_arc))
                clean = []
                for v in verts:
                    if not clean or clean[-1] is not v:
                        clean.append(v)
                if len(clean) > 1 and clean[0] is clean[-1]:
                    clean.pop()
                created_faces.append(bm.faces.new(tuple(clean)))

            bm.normal_update()
            bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
            self._last_edit_error = None
            debug_print(
                "SketchTools Circle: concentric outer cut committed",
                "old_radius=", round(old_radius, 6),
                "new_radius=", round(new_radius, 6),
                "faces=", len(created_faces),
            )
            return True

        except Exception as exc:
            valid_faces = [f for f in created_faces if f.is_valid]
            if valid_faces:
                bmesh.ops.delete(bm, geom=valid_faces, context='FACES_ONLY')
            valid_verts = [v for v in new_verts if v.is_valid]
            if valid_verts:
                bmesh.ops.delete(bm, geom=valid_verts, context='VERTS')
            for verts in original_ring_faces:
                try:
                    if all(v.is_valid for v in verts):
                        bm.faces.new(verts)
                except Exception:
                    pass
            bm.normal_update()
            bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
            self._last_edit_error = "concentric outer cut failed: " + str(exc)
            debug_print("SketchTools Circle:", self._last_edit_error)
            return False

    def _create_circle_in_edit_face(self, context, obj, points):
        """Cut a circle into the clicked face.

        Topology rule: one center circle face plus exactly four surrounding
        ring faces.  Each ring face absorbs an arc of the circle and any extra
        vertices on an n-gon boundary, so 5+ vertex target faces stay clean
        without triangulating the surrounding ring.
        """
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()
        bm.faces.index_update()

        face = None

        stored_vert_ids = self.target_edit_face_vert_indices
        if stored_vert_ids is not None:
            wanted = tuple(stored_vert_ids)
            for candidate in bm.faces:
                if len(candidate.verts) != len(wanted):
                    continue
                ids = tuple(sorted(v.index for v in candidate.verts))
                if ids == wanted:
                    face = candidate
                    break

        if face is None and self.target_edit_face_signature is not None:
            wanted_sig = self.target_edit_face_signature
            for candidate in bm.faces:
                if len(candidate.verts) != len(wanted_sig):
                    continue
                sig = tuple(
                    sorted(
                        (round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6))
                        for v in candidate.verts
                    )
                )
                if sig == wanted_sig:
                    face = candidate
                    break

        if face is None and self.target_edit_face_index is not None:
            idx = self.target_edit_face_index
            if 0 <= idx < len(bm.faces):
                candidate = bm.faces[idx]
                if candidate.is_valid:
                    face = candidate

        if face is None or not face.is_valid:
            self._last_edit_error = "could not reacquire target face"
            debug_print("SketchTools Circle Edit commit blocked:", self._last_edit_error)
            return False

        if len(face.verts) < 4:
            self._last_edit_error = (
                "four-face circle ring currently requires a target face "
                "with at least 4 boundary vertices"
            )
            debug_print("SketchTools Circle Edit commit blocked:", self._last_edit_error)
            return False

        inv = obj.matrix_world.inverted()
        inner_positions = [inv @ Vector(point) for point in points]
        outer_verts = list(face.verts)

        def to_2d(local_point):
            world = obj.matrix_world @ local_point
            delta = world - self.drawing_plane_point
            return Vector((delta.dot(self.axis_u), delta.dot(self.axis_v)))

        def signed_area(poly):
            area = 0.0
            for i, a in enumerate(poly):
                b = poly[(i + 1) % len(poly)]
                area += a.x * b.y - b.x * a.y
            return area * 0.5

        def point_in_polygon(point, poly):
            inside = False
            px, py = point.x, point.y
            count = len(poly)
            for i in range(count):
                a = poly[i]
                b = poly[(i + 1) % count]
                ab = b - a
                ap = point - a
                cross = ab.x * ap.y - ab.y * ap.x
                if abs(cross) <= 1e-7:
                    dot = ap.dot(ab)
                    if -1e-7 <= dot <= ab.length_squared + 1e-7:
                        return True
                if ((a.y > py) != (b.y > py)):
                    x_hit = (b.x - a.x) * (py - a.y) / (b.y - a.y) + a.x
                    if px < x_hit:
                        inside = not inside
            return inside

        outer_2d = [to_2d(v.co) for v in outer_verts]
        inner_2d = [to_2d(p) for p in inner_positions]

        if abs(signed_area(outer_2d)) < 1e-10:
            self._last_edit_error = "target face has invalid area"
            return False

        if not all(point_in_polygon(p, outer_2d) for p in inner_2d):
            if self._create_concentric_outer_cut(context, obj, face, points):
                return True
            self._last_edit_error = "circle must stay inside the available planar region"
            debug_print("SketchTools Circle Edit commit blocked:", self._last_edit_error)
            return False

        if signed_area(outer_2d) < 0.0:
            outer_verts.reverse()
            outer_2d.reverse()

        if signed_area(inner_2d) < 0.0:
            inner_positions.reverse()
            inner_2d.reverse()

        # Split the circle into four quarter arcs.  Segment counts that are not
        # divisible by four are distributed naturally by rounded quarter indices.
        n_inner = len(inner_positions)
        quarter = [
            0,
            int(round(n_inner * 0.25)) % n_inner,
            int(round(n_inner * 0.50)) % n_inner,
            int(round(n_inner * 0.75)) % n_inner,
        ]

        # Guarantee four distinct ordered indices for very low segment counts.
        if len(set(quarter)) < 4:
            self._last_edit_error = "Edit Mode circle cut requires at least 4 segments"
            debug_print("SketchTools Circle Edit commit blocked:", self._last_edit_error)
            return False

        quarter_2d = [inner_2d[i] for i in quarter]

        n_outer = len(outer_verts)
        best_anchors = None
        best_cost = None
        for combo in combinations(range(n_outer), 4):
            combo = tuple(combo)
            for shift in range(4):
                anchors = combo[shift:] + combo[:shift]
                cost = 0.0
                for i in range(4):
                    d = outer_2d[anchors[i]] - quarter_2d[i]
                    cost += d.length_squared
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_anchors = anchors

        if best_anchors is None:
            self._last_edit_error = "could not divide target boundary into four ring sectors"
            debug_print("SketchTools Circle Edit commit blocked:", self._last_edit_error)
            return False

        def outer_chain(start_idx, end_idx):
            chain = [outer_verts[start_idx]]
            idx = start_idx
            safety = 0
            while idx != end_idx:
                idx = (idx + 1) % n_outer
                chain.append(outer_verts[idx])
                safety += 1
                if safety > n_outer:
                    raise RuntimeError("invalid outer boundary chain")
            return chain

        def inner_arc(start_idx, end_idx, inner_verts):
            arc = [inner_verts[start_idx]]
            idx = start_idx
            safety = 0
            while idx != end_idx:
                idx = (idx + 1) % n_inner
                arc.append(inner_verts[idx])
                safety += 1
                if safety > n_inner:
                    raise RuntimeError("invalid circle arc")
            return arc

        created_faces = []
        inner_verts = []
        original_outer = tuple(outer_verts)

        try:
            bmesh.ops.delete(bm, geom=[face], context='FACES_ONLY')

            inner_verts = [bm.verts.new(position) for position in inner_positions]
            bm.verts.index_update()

            inner_face = bm.faces.new(tuple(inner_verts))
            created_faces.append(inner_face)

            for sector in range(4):
                next_sector = (sector + 1) % 4
                q0 = quarter[sector]
                q1 = quarter[next_sector]

                arc = inner_arc(q0, q1, inner_verts)
                chain = outer_chain(
                    best_anchors[sector],
                    best_anchors[next_sector],
                )

                # Ring polygon walks forward on the outer boundary, then back
                # along the circle arc so the region between them becomes one face.
                ring_verts = chain + list(reversed(arc))

                clean = []
                for v in ring_verts:
                    if not clean or clean[-1] is not v:
                        clean.append(v)
                if len(clean) > 1 and clean[0] is clean[-1]:
                    clean.pop()

                if len(clean) < 3 or len(set(clean)) < 3:
                    raise RuntimeError(f"circle ring sector {sector + 1} is degenerate")

                try:
                    ring_face = bm.faces.new(tuple(clean))
                except ValueError:
                    ring_face = bm.faces.new(tuple(reversed(clean)))
                created_faces.append(ring_face)

            if len(created_faces) != 5:
                raise RuntimeError(
                    f"circle cut must create 1 center + 4 ring faces, got {len(created_faces)}"
                )

            for f in bm.faces:
                f.select = False
            if inner_face.is_valid:
                inner_face.select = True

            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data,
                loop_triangles=True,
                destructive=True,
            )

            self._last_edit_error = None
            debug_print(
                "SketchTools Circle: Edit Mode cut committed successfully",
                "segments=", n_inner,
                "target_verts=", len(original_outer),
                "ring_faces=4",
            )
            return True

        except Exception as exc:
            valid_faces = [f for f in created_faces if f.is_valid]
            if valid_faces:
                bmesh.ops.delete(bm, geom=valid_faces, context='FACES_ONLY')

            valid_verts = [v for v in inner_verts if v.is_valid]
            if valid_verts:
                bmesh.ops.delete(bm, geom=valid_verts, context='VERTS')

            try:
                if all(v.is_valid for v in original_outer):
                    bm.faces.new(original_outer)
            except Exception as rollback_exc:
                debug_print("SketchTools Circle rollback warning:", repr(rollback_exc))

            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data,
                loop_triangles=False,
                destructive=True,
            )
            self._last_edit_error = str(exc)
            debug_print("SketchTools Circle Edit commit failed:", repr(exc))
            return False


    def on_left_click(self, context, event):
        # Shared debounce so stale modal instances cannot commit one click
        # multiple times after add-on reloads.
        now = time.monotonic()
        wm = context.window_manager
        mouse_x = int(getattr(event, "mouse_x", event.mouse_region_x))
        mouse_y = int(getattr(event, "mouse_y", event.mouse_region_y))
        click_state = self.state
        last_press = float(wm.get("sketchtools_circle_last_press", -1000.0))
        last_x = int(wm.get("sketchtools_circle_last_x", -100000))
        last_y = int(wm.get("sketchtools_circle_last_y", -100000))
        last_state = str(wm.get("sketchtools_circle_last_state", ""))

        if (
            (now - last_press) < 0.75
            and abs(mouse_x - last_x) <= 2
            and abs(mouse_y - last_y) <= 2
            and click_state == last_state
        ):
            return

        wm["sketchtools_circle_last_press"] = now
        wm["sketchtools_circle_last_x"] = mouse_x
        wm["sketchtools_circle_last_y"] = mouse_y
        wm["sketchtools_circle_last_state"] = click_state

        if self.state == "READY":
            snap = self._snap(context, event)
            ray_origin, ray_direction = get_view_ray(context, event)
            point = None

            if context.mode == 'EDIT_MESH':
                obj = context.edit_object
                if obj is not None and obj.type == 'MESH':
                    bm = bmesh.from_edit_mesh(obj.data)
                    bm.faces.ensure_lookup_table()

                    inv = obj.matrix_world.inverted()
                    local_origin = inv @ ray_origin
                    local_direction = (inv.to_3x3() @ ray_direction).normalized()
                    tree = BVHTree.FromBMesh(bm)
                    hit_location, hit_normal, face_index, distance = tree.ray_cast(
                        local_origin,
                        local_direction,
                    )
                    if hit_location is not None and face_index is not None:
                        point = obj.matrix_world @ hit_location
                        if snap is not None:
                            hit_world = Vector(point)
                            hit_world_normal = (obj.matrix_world.to_3x3() @ bm.faces[face_index].normal).normalized()
                            candidate = Vector(snap)
                            if abs((candidate - hit_world).dot(hit_world_normal)) <= 1e-4:
                                point = candidate

                if point is None:
                    self.set_status(context, "Circle: click on a mesh face")
                    return

            else:
                depsgraph = context.evaluated_depsgraph_get()
                hit, location, normal, face_index, obj, matrix = context.scene.ray_cast(
                    depsgraph,
                    ray_origin,
                    ray_direction,
                )

                if hit and obj is not None and obj.type == 'MESH':
                    point = Vector(location)
                elif snap is not None:
                    point = Vector(snap)
                else:
                    fallback_point, fallback_normal = view_fallback_plane(context)
                    point = mouse_to_plane(
                        context,
                        event,
                        fallback_point,
                        fallback_normal,
                    )

            if point is None:
                return

            if not self._setup_plane(context, event, point):
                self.set_status(context, "Circle: could not establish drawing plane")
                return

            self._base_plane_point = self.drawing_plane_point.copy()
            self._base_plane_normal = self.drawing_plane_normal.copy()
            self._base_axis_u = self.axis_u.copy()
            self._base_axis_v = self.axis_v.copy()
            self.axis_lock = None
            self.state = "DRAWING"
            self.preview.clear_snap_point()
            self._set_axis_status(context)
            return

        # Commit the visible preview exactly as drawn.
        if len(self.preview.points) >= 4:
            points = [Vector(p).copy() for p in self.preview.points]
        else:
            point = self._event_point(context, event)
            if point is None:
                return
            points = self._circle_points(point)

        if len(points) < 4:
            return

        if context.mode == 'EDIT_MESH':
            obj = context.edit_object
            if obj is None or obj.type != 'MESH':
                return

            debug_print("SketchTools Circle: committing second click EDIT_MESH")
            if self._create_circle_in_edit_face(context, obj, points):
                self.known_centers.append(self.center.copy())
                self.reset_operation()
                self.set_status(context, "Circle: click center | after center X/Y/Z: lock circle plane")
                self._redraw(context)
            else:
                self.set_status(
                    context,
                    "Circle cut failed: " + (self._last_edit_error or "unknown error")
                )
            return

        debug_print("SketchTools Circle: committing second click OBJECT")

        center = self.center.copy()
        local_points = [point - center for point in points]

        mesh = bpy.data.meshes.new("Circle")
        mesh.from_pydata(
            [tuple(point) for point in local_points],
            [],
            [tuple(range(len(local_points)))],
        )
        mesh.update()

        obj = bpy.data.objects.new("Circle", mesh)
        obj.location = center

        radial_u = (points[0] - center).normalized()
        radial_v = self.drawing_plane_normal.cross(radial_u).normalized()
        obj["sketchtools_circle"] = True
        obj["sketchtools_circle_segments"] = int(self.segments)
        obj["sketchtools_circle_radius"] = float((points[0] - center).length)
        obj["sketchtools_circle_axis_u"] = tuple(radial_u)
        obj["sketchtools_circle_axis_v"] = tuple(radial_v)
        obj["sketchtools_circle_pristine"] = True

        context.collection.objects.link(obj)

        for selected in list(context.selected_objects):
            selected.select_set(False)

        obj.select_set(True)
        context.view_layer.objects.active = obj
        context.view_layer.update()

        debug_print(
            "SketchTools Circle: created",
            "segments=", self.segments,
            "radius=", round((points[0] - center).length, 6),
        )

        self.last_circle_name = obj.name
        self.known_centers.append(center.copy())
        self.segment_input = ""
        self.reset_operation()
        self.set_status(
            context,
            "Circle: click center | type e.g. 48s + Enter to change last circle"
        )
        self._redraw(context)

    def _set_axis_status(self, context):
        if self.state != "DRAWING":
            self.set_status(context, "Circle: click center | after center X/Y/Z: lock circle plane")
            return
        if self.axis_lock:
            axis = self.axis_lock
            plane = PLANE_LABELS[axis]
            self.set_status(
                context,
                f"Circle | {axis} PLANE LOCKED ({plane}) | LMB: set radius | {axis}: unlock | X/Y/Z: switch plane | RMB: cancel",
            )
        else:
            self.set_status(context, "Circle: click radius | X/Y/Z: lock plane | RMB: cancel")

    def _toggle_axis_lock(self, context, axis):
        if self.state != "DRAWING" or self.center is None:
            return False
        if context.mode == 'EDIT_MESH':
            self.set_status(context, "Circle: face plane is fixed in Edit Mode | X/Y/Z plane lock available in Object Mode")
            return True

        axis = axis.upper()
        if self.axis_lock == axis:
            self.axis_lock = None
            if self._base_plane_point is not None:
                self.drawing_plane_point = self._base_plane_point.copy()
                self.drawing_plane_normal = self._base_plane_normal.copy()
                self.axis_u = self._base_axis_u.copy()
                self.axis_v = self._base_axis_v.copy()
        else:
            self.axis_lock = axis
            set_world_axis_plane(self, axis, self.center)

        self.preview.clear()
        self._last_snap_type = None
        self._set_axis_status(context)
        self._redraw(context)
        return True

    def on_key_press(self, context, event):
        """Axis-plane locking while drawing; segment entry while ready."""
        if event.type in {'X', 'Y', 'Z'} and self.state == "DRAWING":
            return self._toggle_axis_lock(context, event.type)

        # SketchUp-style post-create segment entry, e.g. 48s + Enter.
        if self.state != "READY" or not self.last_circle_name:
            return False

        obj = bpy.data.objects.get(self.last_circle_name)
        if obj is None or not circle_is_pristine(obj):
            self.last_circle_name = None
            self.segment_input = ""
            return False

        if event.type in {'BACK_SPACE', 'DEL'}:
            self.segment_input = self.segment_input[:-1]
            self.set_status(context, "Circle segments: " + (self.segment_input or "_"))
            return True

        if event.type in {'ESC'}:
            self.segment_input = ""
            self.set_status(context, "Circle: click center | after center X/Y/Z: lock circle plane")
            return True

        if event.type in {'RET', 'NUMPAD_ENTER'}:
            raw = self.segment_input.lower().strip()
            if raw.endswith("s"):
                raw = raw[:-1]

            if raw.isdigit():
                segments = max(3, min(256, int(raw)))
                if rebuild_parametric_circle(obj, segments):
                    self.segments = segments
                    context.scene.sketchtools_circle_segments = segments
                    debug_print("SketchTools Circle: segments adjusted to", segments)
                    self.segment_input = ""
                    self.set_status(context, f"Circle: {segments} segments | click center")
                    self._redraw(context)
                    return True

            self.segment_input = ""
            self.set_status(context, "Circle: invalid segment count")
            return True

        ch = getattr(event, "unicode", "")
        if ch and (ch.isdigit() or ch.lower() == "s"):
            self.segment_input += ch.lower()
            self.set_status(context, "Circle segments: " + self.segment_input)
            return True

        return False


    def on_right_click(self, context, event):
        if self.state == "DRAWING":
            self.reset_operation()
            self.set_status(context, "Circle: click center | after center X/Y/Z: lock circle plane")
            self._redraw(context)
        else:
            tool_manager.cancel(context)

    def finish(self, context):
        restore_cursor(context)
        self.reset_operation()
        super().finish(context)

    def cancel(self, context):
        restore_cursor(context)
        self.reset_operation()
        super().cancel(context)

    def reset_operation(self):
        self.state = "READY"
        self.center = None
        self.drawing_plane_point = None
        self.drawing_plane_normal = None
        self.axis_u = None
        self.axis_v = None
        self.target_edit_face = None
        self.target_edit_face_index = None
        self.target_edit_face_vert_indices = None
        self.target_edit_face_signature = None
        self._last_edit_error = None
        self._last_snap_type = None
        self.axis_lock = None
        self._base_plane_point = None
        self._base_plane_normal = None
        self._base_axis_u = None
        self._base_axis_v = None
        self.preview.clear()
