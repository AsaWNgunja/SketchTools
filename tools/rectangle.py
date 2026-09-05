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
from ..geometry.topology import create_edge, create_face_from_points
from ..utils.geometry_object import get_or_create_geometry_object
from ..utils.cursor import restore_cursor
from ..utils.measurements import format_length
from ..utils.axis_lock import PLANE_LABELS, set_world_axis_plane


class RectanglePreview(SnapFeedbackMixin):
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

        if len(self.points) == 4:
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


class RectangleTool(SketchToolBase):
    tool_name = "Rectangle"

    def __init__(self):
        super().__init__()
        self.state = "READY"
        self.start_point = None
        self.drawing_plane_point = None
        self.drawing_plane_normal = None
        self.axis_u = None
        self.axis_v = None
        self.target_edit_face = None
        self.target_edit_face_index = None
        self.target_edit_face_vert_indices = None
        self.target_edit_face_signature = None
        self._last_left_press_time = 0.0
        self._last_edit_error = None
        self.preview = RectanglePreview()
        self._last_snap_type = None
        self.axis_lock = None
        self._base_plane_point = None
        self._base_plane_normal = None
        self._base_axis_u = None
        self._base_axis_v = None

    def start(self, context):
        super().start(context)
        self.reset_operation()
        self.set_status(context, "Rectangle: click first corner | after first corner X/Y/Z: lock plane")

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
            extra_centers=None,
            inference_origin=(self.start_point if self.start_point is not None else None),
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

    def _setup_plane(self, context, event, point):
        """Choose a stable rectangle drawing plane.

        Object Mode:
            empty space -> world XY
            existing face -> that face plane

        Edit Mode:
            clicked active-mesh face -> that exact BMesh face plane
        """
        ray_origin, ray_direction = get_view_ray(context, event)

        self.target_edit_face = None
        self.target_edit_face_index = None
        self.target_edit_face_vert_indices = None
        self.target_edit_face_signature = None

        if context.mode == 'EDIT_MESH':
            obj = context.edit_object

            if obj is not None and obj.type == 'MESH':
                bm = bmesh.from_edit_mesh(obj.data)
                inv = obj.matrix_world.inverted()

                local_origin = inv @ ray_origin
                local_direction = (
                    inv.to_3x3() @ ray_direction
                ).normalized()

                tree = BVHTree.FromBMesh(bm)

                hit_location, hit_normal, face_index, distance = tree.ray_cast(
                    local_origin,
                    local_direction,
                )

                if face_index is not None:
                    bm.faces.ensure_lookup_table()

                    if 0 <= face_index < len(bm.faces):
                        face = bm.faces[face_index]
                        self.target_edit_face = face

                        # Never rely on the BMFace Python wrapper surviving
                        # until the second click.  Some Edit Mode operations
                        # performed by Blender while the modal tool is alive
                        # can invalidate that wrapper without changing the
                        # visible topology.  Store several stable descriptors
                        # so commit can reacquire the live face.
                        bm.verts.index_update()
                        bm.faces.index_update()
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

                        self.drawing_plane_point = (
                            obj.matrix_world @ hit_location
                        )

                        self.drawing_plane_normal = (
                            obj.matrix_world.to_3x3()
                            @ face.normal
                        ).normalized()

                        candidates = (
                            (
                                obj.matrix_world.to_3x3()
                                @ Vector((1.0, 0.0, 0.0))
                            ).normalized(),
                            (
                                obj.matrix_world.to_3x3()
                                @ Vector((0.0, 1.0, 0.0))
                            ).normalized(),
                            (
                                obj.matrix_world.to_3x3()
                                @ Vector((0.0, 0.0, 1.0))
                            ).normalized(),
                        )

                        return self._finish_plane_basis(
                            point,
                            candidates,
                        )

        depsgraph = context.evaluated_depsgraph_get()

        hit, location, normal, face_index, obj, matrix = (
            context.scene.ray_cast(
                depsgraph,
                ray_origin,
                ray_direction,
            )
        )

        if (
            hit
            and obj is not None
            and obj.type == 'MESH'
        ):
            self.drawing_plane_point = Vector(location)
            self.drawing_plane_normal = Vector(normal).normalized()

            candidates = (
                (
                    obj.matrix_world.to_3x3()
                    @ Vector((1.0, 0.0, 0.0))
                ).normalized(),
                (
                    obj.matrix_world.to_3x3()
                    @ Vector((0.0, 1.0, 0.0))
                ).normalized(),
                (
                    obj.matrix_world.to_3x3()
                    @ Vector((0.0, 0.0, 1.0))
                ).normalized(),
            )
        else:
            self.drawing_plane_point, self.drawing_plane_normal = view_fallback_plane(context)

            world_axes = (
                Vector((1.0, 0.0, 0.0)),
                Vector((0.0, 1.0, 0.0)),
                Vector((0.0, 0.0, 1.0)),
            )
            candidates = world_axes

        return self._finish_plane_basis(
            point,
            candidates,
        )

    def _finish_plane_basis(
        self,
        point,
        candidates,
    ):
        axis_u = None

        for seed in candidates:
            projected = (
                seed
                - self.drawing_plane_normal
                * seed.dot(self.drawing_plane_normal)
            )

            if projected.length > 1e-8:
                axis_u = projected.normalized()
                break

        if axis_u is None:
            return False

        self.axis_u = axis_u

        self.axis_v = (
            self.drawing_plane_normal
            .cross(self.axis_u)
            .normalized()
        )

        delta = Vector(point) - self.drawing_plane_point

        self.start_point = (
            Vector(point)
            - self.drawing_plane_normal
            * delta.dot(self.drawing_plane_normal)
        )

        return True

    def _event_point(self, context, event):
        snap = self._snap(context, event)
        if snap is not None:
            delta = snap - self.drawing_plane_point
            return snap - self.drawing_plane_normal * delta.dot(self.drawing_plane_normal)
        return mouse_to_plane(
            context, event,
            self.drawing_plane_point,
            self.drawing_plane_normal,
        )

    def _corners(self, opposite):
        delta = Vector(opposite) - self.start_point
        u = delta.dot(self.axis_u)
        v = delta.dot(self.axis_v)
        p0 = self.start_point.copy()
        p1 = p0 + self.axis_u * u
        p2 = p1 + self.axis_v * v
        p3 = p0 + self.axis_v * v
        return [p0, p1, p2, p3]

    def on_mouse_move(self, context, event):
        if self.state == "READY":
            self._snap(context, event)
            self._redraw(context)
            return

        point = self._event_point(context, event)
        if point is None:
            return
        corners = self._corners(point)
        self.preview.set_points(corners)
        if len(corners) == 4:
            width = (Vector(corners[1]) - Vector(corners[0])).length
            height = (Vector(corners[3]) - Vector(corners[0])).length
            self.preview.set_measurement_feedback(
                f"{format_length(context, width)} × {format_length(context, height)}",
                event.mouse_region_x,
                event.mouse_region_y,
            )
        self._redraw(context)

    def _create_rectangle_in_edit_face(
        self,
        context,
        obj,
        points,
    ):
        """Insert a rectangle into the clicked Edit Mode face.

        Topology rule: every successful cut creates exactly five faces total:
        one centre rectangle plus exactly four surrounding ring faces.  Extra
        boundary vertices from a 5-vertex (or larger) n-gon are absorbed into
        one or more of those four ring faces instead of triangulating the ring.
        """
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()
        bm.faces.index_update()

        # Reacquire the live face instead of trusting the BMFace wrapper saved
        # at first click. Edit-mode BMesh wrappers may be invalidated while the
        # mouse moves even when the visible topology has not changed.
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
            debug_print("SketchTools Rectangle Edit commit blocked:", self._last_edit_error)
            return False

        self.target_edit_face = face

        if len(face.verts) < 4:
            self._last_edit_error = (
                "four-face rectangle ring currently requires a face with "
                "at least 4 boundary vertices"
            )
            debug_print("SketchTools Rectangle Edit commit blocked:", self._last_edit_error)
            return False

        inv = obj.matrix_world.inverted()
        inner_positions = [inv @ Vector(point) for point in points]
        outer_verts = list(face.verts)

        def to_2d(local_point):
            world = obj.matrix_world @ local_point
            delta = world - self.drawing_plane_point
            return Vector((
                delta.dot(self.axis_u),
                delta.dot(self.axis_v),
            ))

        def signed_area_2d(poly):
            area = 0.0
            for i, a in enumerate(poly):
                b = poly[(i + 1) % len(poly)]
                area += a.x * b.y - b.x * a.y
            return area * 0.5

        def point_in_polygon(point, poly):
            # Boundary-inclusive ray crossing test.
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

        min_x = min(p.x for p in inner_2d)
        max_x = max(p.x for p in inner_2d)
        min_y = min(p.y for p in inner_2d)
        max_y = max(p.y for p in inner_2d)
        if (max_x - min_x) < 1e-6 or (max_y - min_y) < 1e-6:
            self._last_edit_error = "rectangle is too small"
            return False

        outer_area = signed_area_2d(outer_2d)
        inner_area = signed_area_2d(inner_2d)
        if abs(outer_area) < 1e-10:
            self._last_edit_error = "target face has invalid area"
            return False

        if not all(point_in_polygon(p, outer_2d) for p in inner_2d):
            self._last_edit_error = "rectangle must stay inside the target face"
            debug_print("SketchTools Rectangle Edit commit blocked:", self._last_edit_error)
            return False

        # Work in CCW order in the drawing plane. This lets each ring face use
        # one inner edge and the matching forward chain of the outer boundary.
        if outer_area < 0.0:
            outer_verts.reverse()
            outer_2d.reverse()

        if inner_area < 0.0:
            inner_positions.reverse()
            inner_2d.reverse()

        # Choose four distinct outer boundary vertices as the four sector
        # anchors. For a 5-vertex n-gon, one of the four outer chains therefore
        # contains two boundary edges; that extra vertex is absorbed by that
        # ring face instead of spawning another face.
        n_outer = len(outer_verts)
        best_anchors = None
        best_cost = None

        for combo in combinations(range(n_outer), 4):
            combo = tuple(combo)
            # The same four anchors can start at any rectangle corner. Try all
            # cyclic alignments and keep the geometrically closest mapping.
            for shift in range(4):
                anchors = combo[shift:] + combo[:shift]
                cost = 0.0
                for i in range(4):
                    d = outer_2d[anchors[i]] - inner_2d[i]
                    cost += d.length_squared
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_anchors = anchors

        if best_anchors is None:
            self._last_edit_error = "could not divide target boundary into four ring sectors"
            debug_print("SketchTools Rectangle Edit commit blocked:", self._last_edit_error)
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

        created_faces = []
        inner_verts = []
        original_outer = tuple(outer_verts)

        try:
            bmesh.ops.delete(
                bm,
                geom=[face],
                context='FACES_ONLY',
            )

            inner_verts = [bm.verts.new(position) for position in inner_positions]
            bm.verts.index_update()

            # Centre rectangle: face #1.
            inner_face = bm.faces.new(tuple(inner_verts))
            created_faces.append(inner_face)

            # Ring faces: exactly four, one per rectangle side. Each face may
            # itself be an n-gon when the original boundary has extra vertices.
            for i in range(4):
                j = (i + 1) % 4
                chain = outer_chain(best_anchors[i], best_anchors[j])
                ring_verts = [inner_verts[i]] + chain + [inner_verts[j]]

                # Remove accidental consecutive duplicates while preserving the
                # polygon order.
                clean = []
                for v in ring_verts:
                    if not clean or clean[-1] is not v:
                        clean.append(v)
                if len(clean) > 1 and clean[0] is clean[-1]:
                    clean.pop()

                if len(clean) < 3 or len(set(clean)) < 3:
                    raise RuntimeError(f"ring sector {i + 1} is degenerate")

                try:
                    ring_face = bm.faces.new(tuple(clean))
                except ValueError:
                    ring_face = bm.faces.new(tuple(reversed(clean)))
                created_faces.append(ring_face)

            if len(created_faces) != 5:
                raise RuntimeError(
                    f"rectangle cut must create exactly 5 faces, got {len(created_faces)}"
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
                "SketchTools Rectangle: Edit Mode cut committed successfully",
                "target_verts=", len(original_outer),
                "ring_faces=4",
            )
            return True

        except Exception as exc:
            valid_faces = [f for f in created_faces if f.is_valid]
            if valid_faces:
                bmesh.ops.delete(
                    bm,
                    geom=valid_faces,
                    context='FACES_ONLY',
                )

            valid_verts = [v for v in inner_verts if v.is_valid]
            if valid_verts:
                bmesh.ops.delete(
                    bm,
                    geom=valid_verts,
                    context='VERTS',
                )

            try:
                if all(v.is_valid for v in original_outer):
                    bm.faces.new(original_outer)
            except Exception as rollback_exc:
                debug_print("SketchTools Rectangle rollback warning:", repr(rollback_exc))

            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data,
                loop_triangles=False,
                destructive=True,
            )
            self._last_edit_error = str(exc)
            debug_print("SketchTools Rectangle Edit commit failed:", repr(exc))
            return False



    def on_left_click(self, context, event):
        # Debounce through WindowManager, not only this Python object.  That
        # makes the guard survive duplicate/stale modal instances created by
        # add-on reloads, since all of them see the same WindowManager value.
        now = time.monotonic()
        wm = context.window_manager

        # Guard against the exact same physical click being dispatched more
        # than once by stale/reloaded modal handlers.  First and second corners
        # naturally have different screen coordinates, so they are never
        # blocked by this guard even when drawn quickly.
        mouse_x = int(getattr(event, "mouse_x", event.mouse_region_x))
        mouse_y = int(getattr(event, "mouse_y", event.mouse_region_y))
        click_state = self.state
        last_press = float(wm.get("sketchtools_rectangle_last_press", -1000.0))
        last_x = int(wm.get("sketchtools_rectangle_last_x", -100000))
        last_y = int(wm.get("sketchtools_rectangle_last_y", -100000))
        last_state = str(wm.get("sketchtools_rectangle_last_state", ""))

        same_physical_click = (
            (now - last_press) < 0.75
            and abs(mouse_x - last_x) <= 2
            and abs(mouse_y - last_y) <= 2
            and click_state == last_state
        )
        if same_physical_click:
            return

        wm["sketchtools_rectangle_last_press"] = now
        wm["sketchtools_rectangle_last_x"] = mouse_x
        wm["sketchtools_rectangle_last_y"] = mouse_y
        wm["sketchtools_rectangle_last_state"] = click_state
        self._last_left_press_time = now

        if self.state == "READY":
            snap = self._snap(context, event)
            ray_origin, ray_direction = get_view_ray(context, event)

            # In Edit Mode, Scene.ray_cast() is not a reliable source for the
            # live edit BMesh.  Ray-cast the actual edit BMesh instead so the
            # very first click can acquire both the point and target face.
            point = None

            if context.mode == 'EDIT_MESH':
                edit_obj = context.edit_object

                if edit_obj is not None and edit_obj.type == 'MESH':
                    bm = bmesh.from_edit_mesh(edit_obj.data)
                    bm.faces.ensure_lookup_table()

                    inv = edit_obj.matrix_world.inverted()
                    local_origin = inv @ ray_origin
                    local_direction = (
                        inv.to_3x3() @ ray_direction
                    ).normalized()

                    tree = BVHTree.FromBMesh(bm)
                    hit_location, hit_normal, face_index, distance = tree.ray_cast(
                        local_origin,
                        local_direction,
                    )

                    if hit_location is not None and face_index is not None:
                        point = edit_obj.matrix_world @ hit_location
                        if snap is not None:
                            hit_world = Vector(point)
                            hit_world_normal = (edit_obj.matrix_world.to_3x3() @ bm.faces[face_index].normal).normalized()
                            candidate = Vector(snap)
                            if abs((candidate - hit_world).dot(hit_world_normal)) <= 1e-4:
                                point = candidate

                # Edit Mode rectangles are face-cutting operations.  Do not
                # silently fall back to the world XY plane when no edit face
                # is under the cursor.
                if point is None:
                    self.set_status(
                        context,
                        "Rectangle: click on a mesh face",
                    )
                    return

            else:
                # Object Mode can use Blender's evaluated scene ray cast.
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

            self.start_point = Vector(point)

            if not self._setup_plane(context, event, point):
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

        # SECOND CLICK: commit the rectangle that is already visible.
        # Do not ray/project the click again if a valid yellow preview exists;
        # Blender can occasionally give a click event that fails projection even
        # though the preceding MOUSEMOVE produced a perfectly valid rectangle.
        if len(self.preview.points) == 4:
            points = [Vector(p).copy() for p in self.preview.points]
        else:
            point = self._event_point(context, event)
            if point is None:
                self._last_edit_error = "second click has no valid drawing point"
                self.set_status(context, "Rectangle: second click has no valid drawing point")
                debug_print("SketchTools Rectangle: second click projection failed")
                return
            points = self._corners(point)

        debug_print("SketchTools Rectangle: committing second click", context.mode)
        if (
            (points[1] - points[0]).length < 1e-6
            or (points[3] - points[0]).length < 1e-6
        ):
            return

        if context.mode == 'EDIT_MESH':
            obj = context.edit_object or context.active_object

            if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
                self._last_edit_error = "no active mesh edit object at commit"
                self.set_status(context, "Rectangle: " + self._last_edit_error)
                debug_print("SketchTools Rectangle Edit commit blocked:", self._last_edit_error)
                return

            if not self._create_rectangle_in_edit_face(
                context,
                obj,
                points,
            ):
                self.set_status(
                    context,
                    "Rectangle: " + (self._last_edit_error or "could not cut face"),
                )
                return

        else:
            # Object Mode: every rectangle is its own mesh object.
            # Put mesh coordinates around the geometric center, then locate
            # the object at that center. This makes the object's origin truly
            # sit at the rectangle center without using the 3D Cursor.
            center = (
                points[0]
                + points[1]
                + points[2]
                + points[3]
            ) * 0.25

            local_points = [
                Vector(point) - center
                for point in points
            ]

            mesh = bpy.data.meshes.new(
                "Rectangle"
            )

            mesh.from_pydata(
                [
                    tuple(point)
                    for point in local_points
                ],
                [],
                [(0, 1, 2, 3)],
            )

            mesh.update()

            obj = bpy.data.objects.new(
                "Rectangle",
                mesh,
            )

            obj.location = center

            context.collection.objects.link(
                obj
            )

            for selected in context.selected_objects:
                selected.select_set(False)

            obj.select_set(True)
            context.view_layer.objects.active = obj

        context.view_layer.update()

        self.reset_operation()
        self.set_status(context, "Rectangle: click first corner | after first corner X/Y/Z: lock plane")
        self._redraw(context)

    def _set_axis_status(self, context):
        if self.state != "DRAWING":
            self.set_status(context, "Rectangle: click first corner | after first corner X/Y/Z: lock plane")
            return
        if self.axis_lock:
            axis = self.axis_lock
            plane = PLANE_LABELS[axis]
            self.set_status(
                context,
                f"Rectangle | {axis} PLANE LOCKED ({plane}) | LMB: opposite corner | {axis}: unlock | X/Y/Z: switch plane | RMB: cancel",
            )
        else:
            self.set_status(context, "Rectangle: click opposite corner | X/Y/Z: lock plane | RMB: cancel")

    def _toggle_axis_lock(self, context, axis):
        if self.state != "DRAWING" or self.start_point is None:
            return False
        if context.mode == 'EDIT_MESH':
            self.set_status(context, "Rectangle: face plane is fixed in Edit Mode | X/Y/Z plane lock available in Object Mode")
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
            set_world_axis_plane(self, axis, self.start_point)

        self.preview.clear()
        self._last_snap_type = None
        self._set_axis_status(context)
        self._redraw(context)
        return True

    def on_key_press(self, context, event):
        if event.type in {'X', 'Y', 'Z'}:
            return self._toggle_axis_lock(context, event.type)
        return False

    def on_right_click(self, context, event):
        if self.state == "DRAWING":
            self.reset_operation()
            self.set_status(context, "Rectangle: click first corner | after first corner X/Y/Z: lock plane")
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
        self.start_point = None
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
