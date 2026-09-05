from ..utils.logging import debug_print
import bpy
import bmesh
import gpu

from mathutils import Vector
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

from ..core.tool_base import SketchToolBase
from ..utils.cursor import restore_cursor


class EraserPreview:
    """Small SketchUp-style hover highlight for the element about to erase."""

    def __init__(self):
        self.kind = None
        self.points = []

    def clear(self):
        self.kind = None
        self.points = []

    def set_vertex(self, point):
        self.kind = "VERT"
        self.points = [Vector(point).copy()]

    def set_edge(self, a, b):
        self.kind = "EDGE"
        self.points = [Vector(a).copy(), Vector(b).copy()]

    def set_face(self, points):
        self.kind = "FACE"
        self.points = [Vector(p).copy() for p in points]

    def draw(self):
        if not self.points:
            return

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        shader.bind()
        # Warm red/orange hover cue, similar to an erase warning.
        shader.uniform_float("color", (1.0, 0.22, 0.08, 1.0))

        if self.kind == "VERT":
            gpu.state.point_size_set(10.0)
            batch = batch_for_shader(shader, "POINTS", {"pos": self.points})
            batch.draw(shader)
            gpu.state.point_size_set(1.0)

        elif self.kind == "EDGE" and len(self.points) == 2:
            gpu.state.line_width_set(4.0)
            batch = batch_for_shader(shader, "LINES", {"pos": self.points})
            batch.draw(shader)
            gpu.state.line_width_set(1.0)

        elif self.kind == "FACE" and len(self.points) >= 3:
            # Outline the face that will be removed.  Vertex/edge hover still
            # has priority, so this appears only when the pointer is inside a
            # face rather than close to its boundary.
            line_points = []
            count = len(self.points)
            for i in range(count):
                line_points.extend((
                    self.points[i],
                    self.points[(i + 1) % count],
                ))
            gpu.state.line_width_set(3.0)
            batch = batch_for_shader(shader, "LINES", {"pos": line_points})
            batch.draw(shader)
            gpu.state.line_width_set(1.0)


class EraserTool(SketchToolBase):
    """
    SketchUp-style Eraser.

    Hold LEFTMOUSE and move across visible vertices/edges. Geometry is erased
    continuously.  The tool remains active after mouse release.

    v77 uses true DELETE semantics: erasing an edge removes its linked faces,
    opening closed geometry. Faces can also be erased directly while retaining
    their boundary edges, so deleting one cube face leaves an open cube.
    """

    tool_name = "Eraser"

    VERTEX_RADIUS_PX = 11.0
    EDGE_RADIUS_PX = 10.0

    def __init__(self):
        super().__init__()
        self.dragging = False
        self.hover = None
        self.preview = EraserPreview()

    def start(self, context):
        super().start(context)
        self.dragging = False
        self.hover = None
        self.preview.clear()
        self.set_status(
            context,
            "Eraser: hold Left Mouse and drag over vertices or edges",
        )
        debug_print("Eraser Tool Active")

    def cancel(self, context):
        # Match Line / Rectangle / Circle / Freehand lifecycle: releasing
        # Eraser must restore Blender's normal window cursor before the modal
        # controller hands the UI click to another tool.
        restore_cursor(context)
        self.dragging = False
        self.hover = None
        self.preview.clear()
        super().cancel(context)

    def finish(self, context):
        # Same lifecycle contract as every established SketchTools tool.
        restore_cursor(context)
        self.dragging = False
        self.hover = None
        self.preview.clear()
        super().finish(context)

    def _redraw(self, context):
        try:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        except Exception:
            pass

    @staticmethod
    def _distance_to_segment_2d(px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-12:
            dx = px - ax
            dy = py - ay
            return (dx * dx + dy * dy) ** 0.5

        t = ((px - ax) * abx + (py - ay) * aby) / denom
        t = max(0.0, min(1.0, t))
        qx = ax + abx * t
        qy = ay + aby * t
        dx = px - qx
        dy = py - qy
        return (dx * dx + dy * dy) ** 0.5

    def _project(self, context, world_point):
        if context.region is None or context.region_data is None:
            return None
        return view3d_utils.location_3d_to_region_2d(
            context.region,
            context.region_data,
            world_point,
        )

    def _object_under_cursor(self, context, event):
        """Return visible Object-Mode mesh and face index under the cursor."""
        if context.region is None or context.region_data is None:
            return None, None
        try:
            coord = (event.mouse_region_x, event.mouse_region_y)
            ray_origin = view3d_utils.region_2d_to_origin_3d(
                context.region, context.region_data, coord
            )
            ray_direction = view3d_utils.region_2d_to_vector_3d(
                context.region, context.region_data, coord
            )
            depsgraph = context.evaluated_depsgraph_get()
            hit, location, normal, face_index, obj, matrix = context.scene.ray_cast(
                depsgraph, ray_origin, ray_direction
            )
            if hit and obj is not None and obj.type == 'MESH':
                original = obj.original if hasattr(obj, "original") else obj
                return original, int(face_index)
        except Exception:
            pass
        return None, None

    def _edit_face_under_cursor(self, context, event):
        """Return the visible active Edit-Mode face under the cursor."""
        obj = context.edit_object
        if (
            obj is None
            or obj.type != 'MESH'
            or context.region is None
            or context.region_data is None
        ):
            return None

        try:
            from mathutils.bvhtree import BVHTree

            coord = (event.mouse_region_x, event.mouse_region_y)
            ray_origin = view3d_utils.region_2d_to_origin_3d(
                context.region, context.region_data, coord
            )
            ray_direction = view3d_utils.region_2d_to_vector_3d(
                context.region, context.region_data, coord
            )

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
                return None
            if not (0 <= face_index < len(bm.faces)):
                return None

            face = bm.faces[face_index]
            return (
                "FACE",
                obj,
                int(face.index),
                [obj.matrix_world @ v.co for v in face.verts],
            )
        except Exception:
            return None


    def _candidate_objects(self, context, event):
        if context.mode == 'EDIT_MESH':
            obj = context.edit_object
            return [obj] if obj is not None and obj.type == 'MESH' else []

        hit_obj, hit_face = self._object_under_cursor(context, event)
        if hit_obj is not None:
            return [hit_obj]

        # Wire-only meshes cannot be found by scene.ray_cast, so fall back to
        # visible meshes and choose by screen distance.
        result = []
        try:
            for obj in context.visible_objects:
                if (
                    obj.type == 'MESH'
                    and not obj.hide_get()
                    and obj.visible_get()
                ):
                    result.append(obj)
        except Exception:
            pass
        return result

    def _find_hover(self, context, event):
        px = float(event.mouse_region_x)
        py = float(event.mouse_region_y)

        best_vert = None
        best_vert_dist = self.VERTEX_RADIUS_PX + 1.0
        best_edge = None
        best_edge_dist = self.EDGE_RADIUS_PX + 1.0

        # Face is a fallback only. Vertices/edges remain SketchUp-style
        # high-priority targets when the pointer is near their boundary.
        face_candidate = None
        if context.mode == 'EDIT_MESH':
            face_candidate = self._edit_face_under_cursor(context, event)
        else:
            face_obj, face_index = self._object_under_cursor(context, event)
            if (
                face_obj is not None
                and face_index is not None
                and 0 <= face_index < len(face_obj.data.polygons)
            ):
                poly = face_obj.data.polygons[face_index]
                face_candidate = (
                    "FACE",
                    face_obj,
                    int(face_index),
                    [face_obj.matrix_world @ face_obj.data.vertices[i].co
                     for i in poly.vertices],
                )

        for obj in self._candidate_objects(context, event):
            if obj is None:
                continue

            matrix = obj.matrix_world

            if context.mode == 'EDIT_MESH' and obj == context.edit_object:
                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()

                projected = {}
                for v in bm.verts:
                    if v.hide:
                        continue
                    wp = matrix @ v.co
                    sp = self._project(context, wp)
                    if sp is None:
                        continue
                    projected[v.index] = (sp, wp)
                    dx = px - float(sp.x)
                    dy = py - float(sp.y)
                    d = (dx * dx + dy * dy) ** 0.5
                    if d <= self.VERTEX_RADIUS_PX and d < best_vert_dist:
                        best_vert_dist = d
                        best_vert = ("VERT", obj, v.index, wp.copy())

                for e in bm.edges:
                    if e.hide:
                        continue
                    a = projected.get(e.verts[0].index)
                    b = projected.get(e.verts[1].index)
                    if a is None or b is None:
                        continue
                    d = self._distance_to_segment_2d(
                        px, py,
                        float(a[0].x), float(a[0].y),
                        float(b[0].x), float(b[0].y),
                    )
                    if d <= self.EDGE_RADIUS_PX and d < best_edge_dist:
                        best_edge_dist = d
                        best_edge = (
                            "EDGE", obj, e.index,
                            a[1].copy(), b[1].copy(),
                        )

            else:
                mesh = obj.data
                projected = {}
                for v in mesh.vertices:
                    wp = matrix @ v.co
                    sp = self._project(context, wp)
                    if sp is None:
                        continue
                    projected[v.index] = (sp, wp)
                    dx = px - float(sp.x)
                    dy = py - float(sp.y)
                    d = (dx * dx + dy * dy) ** 0.5
                    if d <= self.VERTEX_RADIUS_PX and d < best_vert_dist:
                        best_vert_dist = d
                        best_vert = ("VERT", obj, v.index, wp.copy())

                for e in mesh.edges:
                    a = projected.get(e.vertices[0])
                    b = projected.get(e.vertices[1])
                    if a is None or b is None:
                        continue
                    d = self._distance_to_segment_2d(
                        px, py,
                        float(a[0].x), float(a[0].y),
                        float(b[0].x), float(b[0].y),
                    )
                    if d <= self.EDGE_RADIUS_PX and d < best_edge_dist:
                        best_edge_dist = d
                        best_edge = (
                            "EDGE", obj, e.index,
                            a[1].copy(), b[1].copy(),
                        )

        # SketchUp-style priority: vertex -> edge -> face.
        if best_vert is not None:
            return best_vert
        if best_edge is not None:
            return best_edge
        return face_candidate

    def _update_preview(self):
        if self.hover is None:
            self.preview.clear()
            return
        if self.hover[0] == "VERT":
            self.preview.set_vertex(self.hover[3])
        elif self.hover[0] == "EDGE":
            self.preview.set_edge(self.hover[3], self.hover[4])
        else:
            self.preview.set_face(self.hover[3])

    def _remove_empty_object(self, context, obj):
        """
        Remove an object from the scene/Outliner when no mesh geometry remains.

        In Edit Mode Blender's Mesh datablock can remain temporarily stale
        after a destructive BMesh edit, so test the live edit BMesh first.
        """
        if obj is None or obj.type != 'MESH':
            return False

        empty = False

        # Edit Mode must be checked from the live BMesh.  len(obj.data.vertices)
        # may still report the pre-edit geometry until Blender exits Edit Mode.
        if context.mode == 'EDIT_MESH' and context.edit_object == obj:
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                empty = (
                    len(bm.verts) == 0
                    and len(bm.edges) == 0
                    and len(bm.faces) == 0
                )
            except Exception:
                empty = False

            if not empty:
                return False

            # Leaving Edit Mode flushes the empty BMesh back to the Mesh
            # datablock before the object is unlinked.
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception as exc:
                debug_print(
                    "SketchTools Eraser: could not leave Edit Mode for empty cleanup:",
                    exc,
                )
                return False

        else:
            mesh = obj.data
            empty = (
                len(mesh.vertices) == 0
                and len(mesh.edges) == 0
                and len(mesh.polygons) == 0
            )
            if not empty:
                return False

        name = obj.name

        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            debug_print("SketchTools Eraser: removed empty object:", name)
            return True
        except Exception as exc:
            debug_print("SketchTools Eraser: could not remove empty object:", exc)
            return False

    def _erase_edit_mode(self, context, candidate):
        obj = context.edit_object
        if obj is None or candidate[1] != obj:
            return False

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        kind = candidate[0]
        index = candidate[2]

        try:
            if kind == "VERT":
                if not (0 <= index < len(bm.verts)):
                    return False
                vert = bm.verts[index]
                if not vert.is_valid:
                    return False

                # v77: erasing means DELETE, not dissolve. Connected edges and
                # faces disappear, so closed geometry visibly opens.
                bmesh.ops.delete(bm, geom=[vert], context='VERTS')

            elif kind == "EDGE":
                if not (0 <= index < len(bm.edges)):
                    return False
                edge = bm.edges[index]
                if not edge.is_valid:
                    return False

                # Delete the edge AND its linked faces. This is the requested
                # "open the object" behaviour, unlike v76's dissolve/merge.
                bmesh.ops.delete(
                    bm,
                    geom=[edge],
                    context='EDGES_FACES',
                )

            else:  # FACE
                if not (0 <= index < len(bm.faces)):
                    return False
                face = bm.faces[index]
                if not face.is_valid:
                    return False

                # Remove the face only, preserving its boundary edges/vertices.
                # Example: erase one cube face -> open cube.
                bmesh.ops.delete(
                    bm,
                    geom=[face],
                    context='FACES_ONLY',
                )

            bm.verts.index_update()
            bm.edges.index_update()
            bm.faces.index_update()
            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data,
                loop_triangles=True,
                destructive=True,
            )
            return True

        except (ReferenceError, RuntimeError, ValueError) as exc:
            debug_print("SketchTools Eraser Edit Mode:", exc)
            return False

    def _erase_object_mode(self, context, candidate):
        obj = candidate[1]
        if obj is None or obj.type != 'MESH':
            return False

        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            kind = candidate[0]
            index = candidate[2]

            if kind == "VERT":
                if not (0 <= index < len(bm.verts)):
                    return False
                vert = bm.verts[index]
                bmesh.ops.delete(bm, geom=[vert], context='VERTS')

            elif kind == "EDGE":
                if not (0 <= index < len(bm.edges)):
                    return False
                edge = bm.edges[index]
                bmesh.ops.delete(
                    bm,
                    geom=[edge],
                    context='EDGES_FACES',
                )

            else:  # FACE
                if not (0 <= index < len(bm.faces)):
                    return False
                face = bm.faces[index]
                bmesh.ops.delete(
                    bm,
                    geom=[face],
                    context='FACES_ONLY',
                )

            bm.normal_update()
            bm.to_mesh(obj.data)
            obj.data.update()
            return True

        except (ReferenceError, RuntimeError, ValueError) as exc:
            debug_print("SketchTools Eraser Object Mode:", exc)
            return False
        finally:
            bm.free()

    def _erase_hover(self, context):
        candidate = self.hover
        if candidate is None:
            return False

        obj = candidate[1]

        if context.mode == 'EDIT_MESH':
            changed = self._erase_edit_mode(context, candidate)
        else:
            changed = self._erase_object_mode(context, candidate)

        if not changed:
            return False

        # Re-read the mesh after destructive update. If nothing remains, remove
        # the Blender object completely so it disappears from the Outliner.
        self._remove_empty_object(context, obj)

        self.hover = None
        self.preview.clear()
        self._redraw(context)
        return True

    def on_mouse_move(self, context, event):
        self.mouse_x = event.mouse_region_x
        self.mouse_y = event.mouse_region_y
        self.hover = self._find_hover(context, event)
        self._update_preview()

        if self.dragging and self.hover is not None:
            self._erase_hover(context)

        self._redraw(context)

    def on_left_click(self, context, event):
        self.dragging = True
        self.hover = self._find_hover(context, event)
        self._update_preview()
        if self.hover is not None:
            self._erase_hover(context)

    def on_left_release(self, context, event):
        self.dragging = False
        self.hover = self._find_hover(context, event)
        self._update_preview()
        self._redraw(context)

    def on_right_click(self, context, event):
        # Keep normal SketchTools convention: right-click ends the active tool.
        from ..core.session import tool_manager
        tool_manager.cancel(context)

    def on_key_press(self, context, event):
        if event.type == 'ESC':
            from ..core.session import tool_manager
            tool_manager.cancel(context)
