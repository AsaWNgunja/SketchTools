from ..utils.logging import debug_print
import bpy
import math
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils
import gpu
from mathutils.geometry import tessellate_polygon
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
from ..geometry.topology import _autocut_edge_bmesh, _retopologize_edit_planar_region, _edges_on_segment
from ..geometry.edit_free_geometry import create_disconnected_chain, create_disconnected_face


class FreehandPreview(SnapFeedbackMixin):
    def __init__(self):
        self.points = []
        self.snap_point = None
        self._init_snap_feedback()

    def set_points(self, points):
        self.points = [Vector(p).copy() for p in points]

    def set_snap_point(self, point):
        self.snap_point = Vector(point).copy() if point is not None else None

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

        if len(self.points) >= 2:
            batch = batch_for_shader(
                shader,
                "LINE_STRIP",
                {"pos": self.points},
            )
            gpu.state.line_width_set(3.0)
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))
            batch.draw(shader)
            gpu.state.line_width_set(1.0)

        if self.snap_point is not None:
            batch = batch_for_shader(
                shader,
                "POINTS",
                {"pos": [self.snap_point]},
            )
            gpu.state.point_size_set(10.0)
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))
            batch.draw(shader)
            gpu.state.point_size_set(1.0)


class FreehandTool(SketchToolBase):
    tool_name = "Freehand"

    def __init__(self, spacing_px=5):
        super().__init__()
        self.spacing_px = max(1, min(50, int(spacing_px)))
        self.state = "READY"
        self.points = []
        self.preview = FreehandPreview()

        self.drawing_plane_point = None
        self.drawing_plane_normal = None

        self.last_mouse_x = None
        self.last_mouse_y = None
        self.target_edit_object_name = None
        self.closed_loop = False
        self._last_snap_type = None
        self.edit_free_space = False

    def start(self, context):
        super().start(context)
        self.reset_stroke()
        self.set_status(context, "Freehand: click to start; move freely")
        debug_print("Freehand Tool Active")

    def _redraw(self, context):
        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

    def _start_point_snap(self, context, event, threshold_px=16.0):
        """High-priority Close Loop snap to the first point of this stroke."""
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

    def _release_close_point(self, context, event, candidate=None):
        """Return the exact first point when a stroke is released near it.

        v166: closure must not depend on the last sampled mouse-move landing
        exactly on the first point.  Freehand is sampled at ``spacing_px``
        intervals, so a fast final motion can otherwise release inside the
        visual Close Loop marker without ever appending that snapped sample.
        Use screen-space proximity as the authority and always reuse the exact
        first vertex.  A small geometry fallback covers projection edge cases.
        """
        if self.state != "DRAWING" or len(self.points) < 3:
            return None

        first = Vector(self.points[0]).copy()
        threshold_px = max(18.0, min(28.0, float(self.spacing_px) * 3.0))

        region = context.region
        rv3d = context.region_data
        if region is not None and rv3d is not None:
            try:
                screen = view3d_utils.location_3d_to_region_2d(region, rv3d, first)
            except Exception:
                screen = None
            if screen is not None:
                dx = float(event.mouse_region_x) - float(screen.x)
                dy = float(event.mouse_region_y) - float(screen.y)
                if (dx * dx + dy * dy) ** 0.5 <= threshold_px:
                    self.preview.set_snap_point(first)
                    self.preview.set_snap_feedback(
                        "Close Loop", event.mouse_region_x, event.mouse_region_y
                    )
                    return first

        # Defensive world-space fallback.  Keep it scale-aware and deliberately
        # tight so an open stroke cannot accidentally close across the shape.
        if candidate is not None:
            try:
                cand = Vector(candidate)
                pts = [Vector(p) for p in self.points]
                lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
                hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
                diag = (hi - lo).length
                tol = max(1e-5, min(diag * 0.025, 0.05))
                if (cand - first).length <= tol:
                    return first
            except Exception:
                pass

        return None

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
            extra_centers=None,
            inference_origin=(self.points[-1] if self.state == 'DRAWING' and self.points else None),
            include_face=True,
            include_grid=True,
            preferred_axis=(
                self._last_snap_type
                if self._last_snap_type in {"X_AXIS", "Y_AXIS", "Z_AXIS"}
                else None
            ),
            # v169: Freehand needs a generous acquisition zone because the
            # cursor is normally moving continuously rather than being placed
            # carefully like Line.  Endpoint remains higher priority than Edge
            # inside resolve_snap().
            endpoint_pixel_radius=30,
            edge_pixel_radius=24,
            midpoint_pixel_radius=22,
            center_pixel_radius=22,
            previous_snap_type=self._last_snap_type,
            snap_hysteresis_pixels=7,
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

    def _raycast_edit_surface(self, context, event):
        obj = context.edit_object
        if obj is None or obj.type != 'MESH':
            return None, None, None

        ray_origin, ray_direction = get_view_ray(context, event)
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

        if hit_location is None or face_index is None:
            return None, None, None

        if not (0 <= face_index < len(bm.faces)):
            return None, None, None

        face = bm.faces[face_index]
        world_point = obj.matrix_world @ hit_location
        world_normal = (
            obj.matrix_world.to_3x3() @ face.normal
        ).normalized()

        return world_point, world_normal, obj

    def _first_point_and_plane(self, context, event):
        if context.mode == 'EDIT_MESH':
            surface_point, surface_normal, obj = self._raycast_edit_surface(
                context,
                event,
            )

            if surface_point is None:
                # v121: free-space Edit Mode stroke.  Use the same fallback
                # construction plane as Object Mode but commit directly into
                # the active edit BMesh as a disconnected island.
                self.drawing_plane_point, self.drawing_plane_normal = view_fallback_plane(context)
                self.target_edit_object_name = context.edit_object.name if context.edit_object else None
                self.edit_free_space = True
                return mouse_to_plane(context, event, self.drawing_plane_point, self.drawing_plane_normal)

            self.edit_free_space = False
            # Establish the hovered surface FIRST.  Only then resolve snaps,
            # so geometry/grid behind the face can never steal the stroke.
            self.drawing_plane_point = Vector(surface_point)
            self.drawing_plane_normal = Vector(surface_normal)
            self.target_edit_object_name = obj.name
            snap = self._snap(context, event)
            return Vector(snap) if snap is not None else Vector(surface_point)

        snap = self._snap(context, event)

        ray_origin, ray_direction = get_view_ray(context, event)
        depsgraph = context.evaluated_depsgraph_get()
        hit, location, normal, face_index, obj, matrix = context.scene.ray_cast(
            depsgraph,
            ray_origin,
            ray_direction,
        )

        if hit and obj is not None and obj.type == 'MESH':
            self.drawing_plane_point = Vector(location)
            self.drawing_plane_normal = Vector(normal).normalized()
            snap = self._snap(context, event)
            return Vector(snap) if snap is not None else Vector(location)

        self.drawing_plane_point, self.drawing_plane_normal = view_fallback_plane(context)

        if snap is not None:
            return Vector(snap)

        return mouse_to_plane(
            context,
            event,
            self.drawing_plane_point,
            self.drawing_plane_normal,
        )

    def _event_point(self, context, event):
        """
        Resolve the point used by the stroke.

        v71: once an Edit Mode Freehand stroke has started on a surface, the
        live edited surface has priority over global snapping.  This prevents
        endpoints/edges/axis/grid geometry behind the face from pulling
        successive samples sideways and creating a staircase path.

        Close Loop remains the highest-priority intentional snap.
        """
        if context.mode == 'EDIT_MESH' and self.state == "DRAWING":
            if self.edit_free_space:
                close_point = self._start_point_snap(context, event)
                if close_point is not None:
                    return Vector(close_point)
                return mouse_to_plane(context, event, self.drawing_plane_point, self.drawing_plane_normal)

            # v169: continuously acquire real topology before the broad On Face
            # raycast.  This makes Endpoint/Edge visible and exact while the
            # freehand preview approaches its intended termination target.
            terminal_point, terminal_type = self._terminal_snap(context, event)
            if terminal_point is not None:
                return Vector(terminal_point)

            surface_point, surface_normal, obj = self._raycast_edit_surface(
                context,
                event,
            )
            if (
                surface_point is not None
                and obj is not None
                and (
                    not self.target_edit_object_name
                    or obj.name == self.target_edit_object_name
                )
            ):
                self.preview.set_snap_point(surface_point)
                self.preview.set_snap_feedback(
                    "On Face",
                    event.mouse_region_x,
                    event.mouse_region_y,
                )
                return Vector(surface_point)

            # If the ray momentarily misses the edited face, stay on the
            # established drawing plane rather than snapping to background
            # geometry.
            if (
                self.drawing_plane_point is not None
                and self.drawing_plane_normal is not None
            ):
                self.preview.clear_snap_point()
                self.preview.clear_snap_feedback()
                return mouse_to_plane(
                    context,
                    event,
                    self.drawing_plane_point,
                    self.drawing_plane_normal,
                )
            return None

        # v93 Object Mode stroke behaviour: once drawing has begun, follow the
        # first visible surface under the cursor instead of repeatedly snapping
        # every sample to grids/edges. This removes staircase artifacts and also
        # prevents background geometry from pulling the stroke through a face.
        if self.state == "DRAWING":
            # v169: same strong topology acquisition in Object Mode.
            terminal_point, terminal_type = self._terminal_snap(context, event)
            if terminal_point is not None:
                return Vector(terminal_point)

            ray_origin, ray_direction = get_view_ray(context, event)
            try:
                depsgraph = context.evaluated_depsgraph_get()
                hit, location, normal, face_index, obj, matrix = context.scene.ray_cast(
                    depsgraph, ray_origin, ray_direction
                )
                if hit and obj is not None and obj.type == 'MESH':
                    point = Vector(location)
                    self.preview.set_snap_point(point)
                    self.preview.set_snap_feedback(
                        "On Face",
                        event.mouse_region_x,
                        event.mouse_region_y,
                    )
                    return point
            except Exception:
                pass

            if self.drawing_plane_point is not None and self.drawing_plane_normal is not None:
                self.preview.clear_snap_point()
                self.preview.clear_snap_feedback()
                return mouse_to_plane(
                    context, event, self.drawing_plane_point, self.drawing_plane_normal
                )
            return None

        # Before a stroke starts, keep the shared snap engine for intentional
        # endpoint/midpoint/face placement of the first point.
        snap = self._snap(context, event)
        if snap is not None:
            return Vector(snap)

        if self.drawing_plane_point is None or self.drawing_plane_normal is None:
            return None
        return mouse_to_plane(
            context, event, self.drawing_plane_point, self.drawing_plane_normal
        )

    def _screen_distance_from_last(self, event):
        if self.last_mouse_x is None or self.last_mouse_y is None:
            return 999999.0
        dx = event.mouse_region_x - self.last_mouse_x
        dy = event.mouse_region_y - self.last_mouse_y
        return (dx * dx + dy * dy) ** 0.5

    def _smooth_path(self, points):
        """
        Produce a curve-like Freehand polyline without adding extra vertices.

        v71 first redistributes samples uniformly along arc length, then applies
        three corner-aware relaxation passes.  Intentional sharp turns remain;
        shallow mouse jitter is removed.  Closed loops stay exactly closed.
        """
        pts = [Vector(p).copy() for p in points]
        if len(pts) < 4:
            return pts

        closed = len(pts) >= 4 and (pts[0] - pts[-1]).length < 1e-6
        work = pts[:-1] if closed else pts
        if len(work) < 3:
            return pts

        # Uniform arc-length resampling using the same point count. This removes
        # bunching caused by uneven mouse-event timing while preserving detail.
        def uniform_resample(src, is_closed):
            count = len(src)
            if count < 3:
                return [p.copy() for p in src]

            seg_count = count if is_closed else count - 1
            lengths = []
            total = 0.0
            for i in range(seg_count):
                a = src[i]
                b = src[(i + 1) % count]
                d = (b - a).length
                lengths.append(d)
                total += d

            if total <= 1e-9:
                return [p.copy() for p in src]

            if is_closed:
                targets = [total * i / count for i in range(count)]
            else:
                targets = [total * i / (count - 1) for i in range(count)]

            result = []
            seg_i = 0
            acc = 0.0
            for target in targets:
                while (
                    seg_i < seg_count - 1
                    and acc + lengths[seg_i] < target
                ):
                    acc += lengths[seg_i]
                    seg_i += 1

                length = lengths[seg_i]
                if length <= 1e-12:
                    result.append(src[seg_i % count].copy())
                    continue

                t = max(0.0, min(1.0, (target - acc) / length))
                a = src[seg_i % count]
                b = src[(seg_i + 1) % count]
                result.append(a.lerp(b, t))

            # Open strokes retain their exact user endpoints.
            if not is_closed:
                result[0] = src[0].copy()
                result[-1] = src[-1].copy()
            return result

        work = uniform_resample(work, closed)

        # Three moderate relaxation passes. Curves become visibly flowing while
        # deliberate corners above ~55 degrees remain untouched.
        corner_limit = math.radians(55.0)
        for _ in range(3):
            src = [p.copy() for p in work]
            dst = [p.copy() for p in src]
            count = len(src)
            indices = range(count) if closed else range(1, count - 1)

            for i in indices:
                prev = src[(i - 1) % count]
                cur = src[i]
                nxt = src[(i + 1) % count]
                a = cur - prev
                b = nxt - cur
                if a.length < 1e-8 or b.length < 1e-8:
                    continue

                turn = a.angle(b)
                if turn >= corner_limit:
                    continue

                strength = 0.42 * (1.0 - turn / corner_limit)
                dst[i] = cur.lerp((prev + nxt) * 0.5, strength)

            work = dst

        if closed:
            work.append(work[0].copy())
        return work

    def _display_points(self):
        """Smoothed preview of the current raw stroke."""
        return self._smooth_path(self.points)

    def _update_preview_points(self):
        self.preview.set_points(self._display_points())

    def _terminal_snap(self, context, event):
        """Return a topology endpoint that is allowed to finish Freehand.

        v167 deliberately excludes Face/Grid/Midpoint/Center/Origin: clicking
        empty space or ordinary face space must keep the same freehand stroke
        alive. Only the stroke start, an existing mesh endpoint, or an existing
        mesh edge can terminate it.
        """
        close = self._start_point_snap(context, event, threshold_px=28.0)
        if close is not None:
            return Vector(close), "CLOSE"

        snap = resolve_snap(
            context,
            event.mouse_region_x,
            event.mouse_region_y,
            plane_point=self.drawing_plane_point,
            plane_normal=self.drawing_plane_normal,
            extra_centers=None,
            inference_origin=None,
            include_face=False,
            include_grid=False,
            include_axis_grid=False,
            allow_off_plane_geometry=False,
            endpoint_pixel_radius=30,
            edge_pixel_radius=20,
        )
        if snap is not None and snap.valid and snap.snap_type in {"ENDPOINT", "EDGE"}:
            self.preview.set_snap_point(snap.location)
            self.preview.set_snap_feedback(
                snap_label(snap), event.mouse_region_x, event.mouse_region_y
            )
            return Vector(snap.location), snap.snap_type
        return None, None

    def _commit_active_stroke(self, context):
        if len(self.points) < 2:
            return False
        if context.mode == 'EDIT_MESH':
            self._commit_edit_mode(context)
        else:
            self._commit_object_mode(context)
        self.reset_stroke()
        self.set_status(context, "Freehand: click to start; move freely")
        self._redraw(context)
        return True

    def on_left_click(self, context, event):
        if context.mode not in {'OBJECT', 'EDIT_MESH'}:
            self.set_status(context, "Freehand supports Object Mode and Mesh Edit Mode")
            return

        # First click starts a persistent freehand stroke. LMB does not need to
        # remain held after this point.
        if self.state == "READY":
            point = self._first_point_and_plane(context, event)
            if point is None:
                if context.mode == 'EDIT_MESH':
                    self.set_status(context, "Freehand: start on a mesh face")
                return
            self.points = [Vector(point).copy()]
            self._update_preview_points()
            self.last_mouse_x = event.mouse_region_x
            self.last_mouse_y = event.mouse_region_y
            self.state = "DRAWING"
            self.set_status(
                context,
                "Freehand: move to draw; reach a vertex/edge to finish",
            )
            self._redraw(context)
            return

        if self.state != "DRAWING":
            return

        # Subsequent clicks are anchors unless they land on valid topology.
        # This permits any number of click/move sections in one continuous
        # freehand stroke.
        terminal, terminal_type = self._terminal_snap(context, event)
        if terminal is not None:
            terminal = Vector(terminal)
            if terminal_type == "CLOSE":
                self.closed_loop = True
                terminal = Vector(self.points[0]).copy()
            if not self.points or (terminal - self.points[-1]).length > 1e-6:
                self.points.append(terminal)
            elif terminal_type == "CLOSE" and (self.points[-1] - self.points[0]).length > 1e-6:
                self.points.append(Vector(self.points[0]).copy())
            self._update_preview_points()
            self._commit_active_stroke(context)
            return

        # Not on a vertex/edge: record the current free-space/surface point as
        # another anchor and continue drawing; do NOT finish.
        point = self._event_point(context, event)
        if point is not None:
            point = Vector(point)
            if not self.points or (point - self.points[-1]).length > 1e-6:
                self.points.append(point)
                self._update_preview_points()
            self.last_mouse_x = event.mouse_region_x
            self.last_mouse_y = event.mouse_region_y
        self.set_status(context, "Freehand: continuing; reach a vertex/edge to finish")
        self._redraw(context)

    def on_mouse_move(self, context, event):
        # Show normal SketchTools snapping before the stroke begins.
        if self.state == "READY":
            self._snap(context, event)
            self._redraw(context)
            return

        if self.state != "DRAWING":
            return

        # Keep Close Loop feedback responsive even before another geometry
        # sample is due.
        # Only intentional Close Loop snapping remains active during a stroke.
        # Normal edge/grid snapping is intentionally disabled sample-by-sample.
        self._start_point_snap(context, event)

        if self._screen_distance_from_last(event) < self.spacing_px:
            self._redraw(context)
            return

        point = self._event_point(context, event)
        if point is None:
            return

        point = Vector(point)
        if self.points and (point - self.points[-1]).length < 1e-6:
            return

        self.points.append(point)
        self._update_preview_points()

        self.last_mouse_x = event.mouse_region_x
        self.last_mouse_y = event.mouse_region_y
        self._redraw(context)

    def on_left_release(self, context, event):
        # v167: releasing LMB never finishes Freehand. Drawing continues from
        # mouse movement until an explicit topology termination click.
        return

    def _object_simplify_epsilon(self, points):
        """
        Scale-aware tolerance for Object Mode Freehand.

        The tolerance is deliberately conservative: about 0.45% of the
        stroke bounding-box diagonal.  This removes mouse-event stair-steps
        without flattening intentional corners or small shape changes.
        """
        pts = [Vector(p) for p in points]
        if len(pts) < 3:
            return 1e-5

        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        zs = [p.z for p in pts]
        diagonal = Vector((
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs),
        )).length

        if diagonal <= 1e-8:
            return 1e-5
        return max(1e-5, min(diagonal * 0.0045, 0.05))

    def _prepare_object_points(self):
        """
        Raw samples -> RDP simplification -> one corner-aware smoothing pass.

        RDP retains points that define the actual shape, so deliberate sharp
        turns survive while redundant near-collinear mouse samples disappear.
        """
        raw = [Vector(p).copy() for p in self.points]
        if len(raw) < 3:
            return raw

        closed = len(raw) >= 4 and (raw[0] - raw[-1]).length < 1e-6
        epsilon = self._object_simplify_epsilon(raw)

        # Only simplify genuinely dense strokes. Small deliberate strokes are
        # left mathematically intact apart from the existing gentle smoothing.
        if len(raw) > 12:
            if closed:
                simplified = self._simplify_closed_stroke(raw, epsilon)
            else:
                simplified = self._rdp_open(raw, epsilon)

            # Safety fallback: never collapse a usable stroke.
            minimum = 4 if closed else 2
            if len(simplified) >= minimum:
                before = len(raw) - (1 if closed else 0)
                after = len(simplified) - (1 if closed else 0)
                if after < before:
                    debug_print(
                        "SketchTools Freehand: object simplification",
                        "points=", before,
                        "->", after,
                        "epsilon=", round(epsilon, 6),
                    )
                raw = simplified

        return self._smooth_path(raw)

    def _commit_object_mode(self, context):
        points = self._prepare_object_points()

        # Closed strokes carry a repeated first point only for preview/commit
        # convenience.  Use one real vertex and add an explicit closing edge.
        closed = (
            len(points) >= 4
            and (points[0] - points[-1]).length < 1e-6
        )
        geometry_points = points[:-1] if closed else points

        center = Vector((0.0, 0.0, 0.0))
        for point in geometry_points:
            center += point
        center /= len(geometry_points)

        local_points = [p - center for p in geometry_points]
        edges = [(i, i + 1) for i in range(len(local_points) - 1)]
        if closed and len(local_points) >= 3:
            edges.append((len(local_points) - 1, 0))

        mesh = bpy.data.meshes.new("Freehand")
        faces = []
        if closed and len(local_points) >= 3:
            # v166: a closed Freehand stroke is real closed geometry, not only
            # a cyclic wire.  Object Mode mirrors Line/Rectangle/Circle by
            # creating an independent filled object immediately.
            faces = [tuple(range(len(local_points)))]
        mesh.from_pydata(
            [tuple(p) for p in local_points],
            edges,
            faces,
        )
        mesh.update()

        obj = bpy.data.objects.new("Freehand", mesh)
        obj.location = center
        obj["sketchtools_freehand"] = True
        obj["sketchtools_freehand_points"] = len(points)
        context.collection.objects.link(obj)

        for selected in list(context.selected_objects):
            selected.select_set(False)

        obj.select_set(True)
        context.view_layer.objects.active = obj
        context.view_layer.update()

        debug_print(
            "SketchTools Freehand: stroke created in Object Mode",
            "points=", len(points),
            "closed=", closed,
            "face=", bool(faces),
        )

    def _point_segment_distance(self, point, a, b):
        """3D distance from point to finite segment a-b."""
        point = Vector(point)
        a = Vector(a)
        b = Vector(b)
        ab = b - a
        denom = ab.length_squared
        if denom <= 1e-16:
            return (point - a).length
        t = max(0.0, min(1.0, (point - a).dot(ab) / denom))
        closest = a + ab * t
        return (point - closest).length

    def _rdp_open(self, points, epsilon):
        """Ramer-Douglas-Peucker for an open 3D polyline."""
        if len(points) <= 2:
            return [Vector(p).copy() for p in points]

        a = Vector(points[0])
        b = Vector(points[-1])
        max_dist = -1.0
        index = -1

        for i in range(1, len(points) - 1):
            dist = self._point_segment_distance(points[i], a, b)
            if dist > max_dist:
                max_dist = dist
                index = i

        if max_dist > epsilon and index > 0:
            left = self._rdp_open(points[:index + 1], epsilon)
            right = self._rdp_open(points[index:], epsilon)
            return left[:-1] + right

        return [a.copy(), b.copy()]

    def _simplify_closed_stroke(self, points, epsilon):
        """Simplify a closed stroke without changing its exact closing point."""
        pts = [Vector(p).copy() for p in points]
        if len(pts) < 6:
            return pts

        # Remove repeated closing vertex during simplification.
        if (pts[0] - pts[-1]).length < 1e-7:
            pts = pts[:-1]

        if len(pts) < 5:
            return pts + [pts[0].copy()]

        # Split the ring at the point farthest from index 0. This avoids the
        # degenerate RDP case where first == last.
        start = pts[0]
        split = max(
            range(1, len(pts)),
            key=lambda i: (pts[i] - start).length_squared,
        )

        chain_a = pts[:split + 1]
        chain_b = pts[split:] + [pts[0]]

        simp_a = self._rdp_open(chain_a, epsilon)
        simp_b = self._rdp_open(chain_b, epsilon)

        simplified = simp_a[:-1] + simp_b[:-1]

        # Preserve at least a usable polygon.
        if len(simplified) < 3:
            simplified = pts

        simplified.append(simplified[0].copy())
        return simplified

    def _topology_simplify_epsilon(self, context):
        """
        Conservative world-space tolerance for Edit Mode topology.
        About 0.15% of the active object's bounding-box diagonal, clamped.
        """
        obj = context.edit_object
        if obj is None:
            return 1e-4

        try:
            corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            diagonal = max((a - b).length for a in corners for b in corners)
        except Exception:
            diagonal = 1.0

        return max(1e-5, min(diagonal * 0.0015, 0.02))

    def _closed_loop_face_cut(self, context):
        """Cut a closed Freehand loop with a clean annular strip."""
        if not self.closed_loop or len(self.points) < 4:
            return False
        obj=context.edit_object
        if obj is None or obj.type!='MESH':
            return False

        bm=bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        inv=obj.matrix_world.inverted()

        source_points = [Vector(p).copy() for p in self.points]
        # Only simplify genuinely dense Freehand loops. Line reuses this cutter
        # with 3-10 points and must remain mathematically exact.
        if len(source_points) > 12:
            epsilon = self._topology_simplify_epsilon(context)
            simplified = self._simplify_closed_stroke(
                source_points,
                epsilon,
            )
            if len(simplified) >= 4:
                debug_print(
                    "SketchTools Freehand: topology simplification",
                    "points=", len(source_points) - 1,
                    "->", len(simplified) - 1,
                    "epsilon=", round(epsilon, 6),
                )
                source_points = simplified

        inner_pos=[inv @ Vector(p) for p in source_points[:-1]]
        if len(inner_pos)<3:
            return False

        centre=sum(inner_pos,Vector((0,0,0)))/len(inner_pos)

        # Nested-cut fix: select the smallest coplanar face containing the
        # whole new stroke, not simply the first/nearest coplanar face.
        def _candidate_face_info(face):
            if len(face.verts) < 3:
                return None
            fn=face.normal.normalized()
            fo=face.verts[0].co
            if any(abs(fn.dot(p-fo))>2e-4 for p in inner_pos):
                return None
            fu=face.verts[1].co-fo
            if fu.length<1e-9:
                return None
            fu.normalize()
            fv=fn.cross(fu)
            if fv.length<1e-9:
                return None
            fv.normalize()
            def xy(p):
                d=p-fo
                return Vector((d.dot(fu),d.dot(fv)))
            poly=[xy(vtx.co) for vtx in face.verts]
            pts=[xy(p) for p in inner_pos]
            def on_segment(p,a,b,tol=2e-5):
                ab=b-a
                if ab.length_squared<1e-14:
                    return (p-a).length<=tol
                t=max(0.0,min(1.0,(p-a).dot(ab)/ab.length_squared))
                return (p-(a+ab*t)).length<=tol
            def contains(p):
                for i,a in enumerate(poly):
                    if on_segment(p,a,poly[(i+1)%len(poly)]):
                        return True
                c=False;j=len(poly)-1
                for i in range(len(poly)):
                    a=poly[i];b=poly[j]
                    if ((a.y>p.y)!=(b.y>p.y)):
                        x=(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x
                        if p.x<x:c=not c
                    j=i
                return c
            if not all(contains(p) for p in pts):
                return None
            ar=abs(.5*sum(
                poly[i].x*poly[(i+1)%len(poly)].y
                - poly[(i+1)%len(poly)].x*poly[i].y
                for i in range(len(poly))
            ))
            return ar

        candidates=[]
        for face in bm.faces:
            info=_candidate_face_info(face)
            if info is not None:
                candidates.append((info,face))
        target=min(candidates,key=lambda item:item[0])[1] if candidates else None
        if target is None:
            debug_print("SketchTools Freehand: no containing coplanar face for closed loop")
            return False

        normal=target.normal.normalized()
        origin=target.verts[0].co.copy()
        if any(abs(normal.dot(p-origin))>2e-4 for p in inner_pos):
            return False

        outer=list(target.verts)
        if len(outer)<3:
            return False

        # Stable 2D coordinates.
        u=(outer[1].co-origin).normalized()
        v=normal.cross(u).normalized()
        def p2(p):
            d=p-origin
            return Vector((d.dot(u),d.dot(v)))
        def signed_area(poly):
            return .5*sum(
                poly[i].x*poly[(i+1)%len(poly)].y -
                poly[(i+1)%len(poly)].x*poly[i].y
                for i in range(len(poly))
            )
        def inside(pt,poly):
            c=False;j=len(poly)-1
            for i in range(len(poly)):
                a=poly[i];b=poly[j]
                if ((a.y>pt.y)!=(b.y>pt.y)):
                    x=(b.x-a.x)*(pt.y-a.y)/(b.y-a.y)+a.x
                    if pt.x<x:c=not c
                j=i
            return c

        outer2=[p2(q.co) for q in outer]
        inner2=[p2(q) for q in inner_pos]
        if not all(inside(q,outer2) for q in inner2):
            return False
        if signed_area(outer2)<0:
            outer.reverse();outer2.reverse()
        if signed_area(inner2)<0:
            inner_pos.reverse();inner2.reverse()

        # v74: single visible Freehand boundary.
        #
        # Earlier versions created a tiny offset "companion" loop around the
        # actual Freehand loop, which made the cut robust but left two nearly
        # parallel visible edge loops when zoomed in.  We no longer create that
        # offset strip.  The real Freehand loop itself is now the boundary used
        # by the surrounding sector faces.
        n=len(inner2)
        inner_verts=[bm.verts.new(p) for p in inner_pos]
        bm.verts.index_update()

        # The sector builders below historically referenced comp_verts /
        # companion2. Point them directly at the real Freehand boundary so all
        # topology meets one clean loop.
        comp_verts=inner_verts
        companion2=inner2
        gap=0.0

        # Four-ish anchors are enough for the broad outer region. Assign each
        # original boundary corner to its nearest Freehand boundary vertex.
        anchors=[]
        for ov in outer:
            idx=min(
                range(n),
                key=lambda i:(inner_verts[i].co-ov.co).length_squared,
            )
            anchors.append(idx)

        original_outer=tuple(target.verts)
        # v72: every face created by this cut must inherit the orientation of
        # the source surface. Store it before deleting the target face.
        target_normal=target.normal.copy().normalized()
        created=[]
        support_verts=[]
        try:
            # Validate before destructive mutation.
            if len(set(anchors))<3:
                raise RuntimeError("freehand loop too close to one side of target face")

            bmesh.ops.delete(bm,geom=[target],context='FACES_ONLY')

            centre_face=bm.faces.new(tuple(inner_verts));created.append(centre_face)

            # v75: minimal-bridge topology.
            #
            # Blender cannot represent an isolated inner boundary inside one
            # BMFace, so a real cut needs a seam. Instead of spreading four
            # sector connections around the loop, create one narrow seam using
            # exactly TWO bridge edges between one outer edge and one inner
            # edge. The rest of the surrounding region is a single large
            # planar n-gon. This keeps the Freehand cut real while making the
            # unavoidable connecting topology predictable and unobtrusive.
            m=len(outer)
            outer_region_faces=0
            sectors_repaired=0

            def cyclic_indices(count, start_idx, end_idx, step=1):
                out=[start_idx]
                idx=start_idx
                safety=0
                while idx != end_idx:
                    idx=(idx+step) % count
                    out.append(idx)
                    safety += 1
                    if safety > count:
                        raise RuntimeError("invalid minimal-bridge chain")
                return out

            # Find the closest pairing between an OUTER EDGE and an INNER EDGE.
            # Testing both inner-edge directions prevents a crossed/twisted seam.
            best=None
            for oi in range(m):
                oj=(oi+1)%m
                for ii in range(n):
                    ij=(ii+1)%n

                    # Pair outer oi->oj with inner ii->ij.
                    cost_a=(
                        (outer[oi].co-inner_verts[ii].co).length_squared
                        +(outer[oj].co-inner_verts[ij].co).length_squared
                    )
                    if best is None or cost_a < best[0]:
                        best=(cost_a,oi,oj,ii,ij)

                    # Pair outer oi->oj with inner ij->ii.
                    cost_b=(
                        (outer[oi].co-inner_verts[ij].co).length_squared
                        +(outer[oj].co-inner_verts[ii].co).length_squared
                    )
                    if best is None or cost_b < best[0]:
                        best=(cost_b,oi,oj,ij,ii)

            if best is None:
                raise RuntimeError("could not find minimal bridge")

            _,oa,ob,ia,ib=best

            # Main surrounding face:
            # take the LONG way around the outer boundary (ob -> oa), bridge to
            # inner ia, then take the LONG way around the inner boundary
            # (ia -> ib opposite the direct seam edge), and bridge back to ob.
            outer_chain_idx=cyclic_indices(m,ob,oa,step=1)

            # ia and ib are adjacent. Choose the direction that does NOT use
            # their direct edge, so the chain travels around the rest of the
            # Freehand boundary.
            if (ia+1)%n == ib:
                inner_step=-1
            elif (ia-1)%n == ib:
                inner_step=1
            else:
                # Defensive fallback; the selected pair should always be adjacent.
                inner_step=-1

            inner_chain_idx=cyclic_indices(n,ia,ib,step=inner_step)

            main_loop=[outer[i] for i in outer_chain_idx]
            main_loop += [inner_verts[i] for i in inner_chain_idx]

            clean=[]
            for q in main_loop:
                if not clean or clean[-1] is not q:
                    clean.append(q)

            if len(set(clean)) < 3:
                raise RuntimeError("minimal-bridge main face degenerate")

            try:
                f=bm.faces.new(tuple(clean))
            except ValueError:
                f=bm.faces.new(tuple(reversed(clean)))
            created.append(f)
            outer_region_faces += 1

            # Fill the narrow seam between the chosen adjacent outer and inner
            # edges. Its side edges are the ONLY two bridge connections.
            seam_loop=(outer[oa],outer[ob],inner_verts[ib],inner_verts[ia])
            if len(set(seam_loop)) >= 3:
                try:
                    sf=bm.faces.new(seam_loop)
                except ValueError:
                    sf=bm.faces.new(tuple(reversed(seam_loop)))
                created.append(sf)
                outer_region_faces += 1

            sectors_repaired=1

            if outer_region_faces < 1 or len(created) < 2:
                raise RuntimeError("outer reconstruction incomplete")

            # v72 normal/winding fix:
            # Some broad-sector loops can be assembled in the opposite winding
            # depending on stroke direction and cyclic seam placement. Update
            # normals, then flip only faces that disagree with the original
            # target face. This keeps centre, local strip and outer sectors
            # consistently front-facing on either horizontal or vertical faces.
            bm.normal_update()
            flipped_faces=0
            for f in created:
                if not f.is_valid:
                    continue
                if f.normal.dot(target_normal) < 0.0:
                    try:
                        f.normal_flip()
                        flipped_faces += 1
                    except Exception:
                        # Fallback supported by Blender's BMesh operator API.
                        try:
                            bmesh.ops.reverse_faces(bm, faces=[f])
                            flipped_faces += 1
                        except Exception:
                            pass

            bm.normal_update()

            for f in bm.faces:
                f.select=False
            centre_face.select=True
            bmesh.update_edit_mesh(obj.data,loop_triangles=True,destructive=True)
            debug_print(
                "SketchTools Freehand: clean local topology cut",
                "loop_points=",n,
                "local_strip_faces=",0,
                "outer_faces=", outer_region_faces,
                "outer_mode=", "minimal_two_bridge_seam",
                "flipped_faces=", flipped_faces,
                "sector_pairing_ok=", bool(sectors_repaired),
                "support_gap=", 0.0,
            )
            return True
        except Exception as exc:
            valid=[f for f in created if f.is_valid]
            if valid:bmesh.ops.delete(bm,geom=valid,context='FACES_ONLY')
            verts=[]
            seen=set()
            for q in inner_verts+support_verts:
                if q.is_valid and id(q) not in seen:
                    seen.add(id(q))
                    verts.append(q)
            if verts:
                bmesh.ops.delete(bm,geom=verts,context='VERTS')
            try:
                if all(q.is_valid for q in original_outer):
                    bm.faces.new(original_outer)
            except Exception as rb:
                debug_print("SketchTools Freehand rollback warning:",repr(rb))
            bm.normal_update()
            bmesh.update_edit_mesh(obj.data,loop_triangles=False,destructive=True)
            debug_print("SketchTools Freehand clean cut rolled back:",repr(exc))
            return False

    def _knife_project_edit_mode(self, context):
        """Use Blender's native Knife Project for a true surface topology cut."""
        obj = context.edit_object
        if obj is None or obj.type != 'MESH' or len(self.points) < 2:
            return False

        points = [Vector(p).copy() for p in self.points]
        closed = (
            len(points) >= 4
            and (points[0] - points[-1]).length < 1e-6
        )
        geometry_points = points[:-1] if closed else points
        if len(geometry_points) < 2:
            return False

        mesh = bpy.data.meshes.new("_SketchTools_Freehand_Cutter")
        edges = [(i, i + 1) for i in range(len(geometry_points) - 1)]
        if closed and len(geometry_points) >= 3:
            edges.append((len(geometry_points) - 1, 0))
        mesh.from_pydata(
            [tuple(p) for p in geometry_points],
            edges,
            [],
        )
        mesh.update()

        cutter = bpy.data.objects.new("_SketchTools_Freehand_Cutter", mesh)
        context.collection.objects.link(cutter)

        success = False
        try:
            # Knife Project reads the non-edit selected object(s) as cutters.
            bpy.ops.object.mode_set(mode='OBJECT')
            for selected in list(context.selected_objects):
                selected.select_set(False)

            obj.select_set(True)
            cutter.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')

            bpy.ops.mesh.knife_project(cut_through=False)
            success = True
        except Exception as exc:
            debug_print("SketchTools Freehand: Knife Project fallback:", exc)
        finally:
            try:
                if obj.mode == 'EDIT':
                    bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass

            try:
                if cutter and cutter.name in bpy.data.objects:
                    bpy.data.objects.remove(cutter, do_unlink=True)
            except Exception:
                pass

            try:
                context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.mode_set(mode='EDIT')
            except Exception:
                pass

        return success

    def _prepare_edit_points(self, context):
        """
        Prepare one authoritative final stroke for Edit Mode.

        All topology paths, including fallbacks, must consume the same
        simplified/smoothed points so a 201 -> 74 simplification actually
        commits ~74 segments instead of reverting to all raw mouse samples.
        """
        raw = [Vector(p).copy() for p in self.points]
        if len(raw) < 3:
            return raw

        closed = len(raw) >= 4 and (raw[0] - raw[-1]).length < 1e-6
        epsilon = self._topology_simplify_epsilon(context)

        prepared = raw
        if len(raw) > 12:
            try:
                if closed:
                    simplified = self._simplify_closed_stroke(raw, epsilon)
                else:
                    simplified = self._rdp_open(raw, epsilon)

                minimum = 4 if closed else 2
                if len(simplified) >= minimum:
                    prepared = simplified
            except Exception as exc:
                debug_print("SketchTools Freehand: edit simplification skipped:", exc)

        return self._smooth_path(prepared)

    def _commit_edit_mode(self, context):
        final_points = self._prepare_edit_points(context)
        if len(final_points) < 2:
            return
        raw_points_backup = self.points
        self.points = final_points
        try:
            obj = context.edit_object

            if (
                obj is None
                or obj.type != 'MESH'
                or (
                    self.target_edit_object_name
                    and obj.name != self.target_edit_object_name
                )
            ):
                debug_print("SketchTools Freehand: Edit Mode target object changed")
                return False

            # v121 free-space strokes never enter Knife/cut/autocut code.
            if self.edit_free_space:
                is_closed = len(final_points) >= 4 and (Vector(final_points[0]) - Vector(final_points[-1])).length < 1e-6
                if is_closed:
                    return create_disconnected_face(context, final_points)
                return create_disconnected_chain(context, final_points, closed=False)

            # Closed loops wholly inside a planar face need an explicit face-with-
            # hole rebuild; a wire chain alone cannot split that face in BMesh.
            if self._closed_loop_face_cut(context):
                return True

            # v173: use the same topology-aware segment insertion as Line as the
            # PRIMARY path for open / cross-boundary Freehand strokes. Knife
            # Project could succeed visually while leaving crossed mesh edges
            # unsplit or creating unnecessary local cells. _autocut_edge_bmesh
            # gives every intermediate Freehand segment a true intersection
            # vertex, then one planar rebuild at transaction completion produces
            # only the real bounded cells.
            bm = bmesh.from_edit_mesh(obj.data)
            inv = obj.matrix_world.inverted()

            local_points = [inv @ Vector(p) for p in final_points]

            # Insert every Freehand segment through the same topology-aware
            # autocut engine used by Line.  We keep one live BMesh for the entire
            # stroke and update Blender only once at the end, avoiding the older
            # per-segment Edit Mesh rebuild problem.
            created_segments = 0
            for p1, p2 in zip(local_points, local_points[1:]):
                if (p1 - p2).length < 1e-7:
                    continue
                if _autocut_edge_bmesh(bm, p1, p2):
                    created_segments += 1

            # v168: an open curved chain from one host boundary point to another
            # can leave intermediate Freehand vertices as wire geometry because
            # connect_vert_pair only splits a face when both vertices already
            # belong to that face boundary.  Once the complete chain is present,
            # rebuild the affected coplanar graph into its true bounded cells.
            # This is the same proven planar-cell engine used by Line, applied
            # only at Freehand transaction completion.
            stroke_edges = []
            for p1, p2 in zip(local_points, local_points[1:]):
                stroke_edges.extend(_edges_on_segment(bm, p1, p2))
            # Preserve order while removing duplicate BMesh edge references.
            stroke_edges = list(dict.fromkeys(e for e in stroke_edges if e is not None and e.is_valid))
            rebuilt_faces = _retopologize_edit_planar_region(bm, stroke_edges) if stroke_edges else 0

            # If the local graph engine could not insert anything, keep the old
            # Knife Project route only as a last-resort compatibility fallback.
            if created_segments == 0 and rebuilt_faces == 0:
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                if self._knife_project_edit_mode(context):
                    debug_print("SketchTools Freehand: Knife Project last-resort fallback")
                    return True

            bm.verts.index_update()
            bm.edges.index_update()
            bm.faces.index_update()
            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data,
                loop_triangles=True,
                destructive=True,
            )

            debug_print(
                "SketchTools Freehand: topology cut in Edit Mode",
                "segments=", created_segments,
                "sampled_points=", len(local_points),
                "rebuilt_faces=", rebuilt_faces,
                "closed=", bool(
                    len(local_points) >= 4
                    and (local_points[0] - local_points[-1]).length < 1e-6
                ),
            )
            return created_segments > 0
        finally:
            self.points = raw_points_backup

    def on_right_click(self, context, event):
        if self.state == "DRAWING":
            self.reset_stroke()
            self.set_status(context, "Freehand: click to start; move freely")
            self._redraw(context)
            debug_print("Freehand stroke cancelled")
        else:
            tool_manager.cancel(context)

    def on_key_press(self, context, event):
        pass

    def finish(self, context):
        restore_cursor(context)
        self.reset_stroke()
        super().finish(context)

    def cancel(self, context):
        restore_cursor(context)
        self.reset_stroke()
        super().cancel(context)

    def reset_stroke(self):
        self.state = "READY"
        self.points = []
        self.preview.clear()
        self.drawing_plane_point = None
        self.drawing_plane_normal = None
        self.last_mouse_x = None
        self.last_mouse_y = None
        self.target_edit_object_name = None
        self.closed_loop = False
        self._last_snap_type = None
        self.edit_free_space = False
