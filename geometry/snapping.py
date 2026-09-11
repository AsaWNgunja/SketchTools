import bmesh
from mathutils import Vector
from mathutils.geometry import intersect_line_line
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils


class SnapResult:
    """Stores the result of a SketchTools snapping operation."""

    def __init__(
        self,
        location=None,
        snap_type=None,
        obj=None,
        element=None,
        screen_distance=None,
    ):
        self.location = (
            Vector(location)
            if location is not None
            else None
        )

        self.snap_type = snap_type
        self.obj = obj
        self.element = element
        self.screen_distance = screen_distance
        # Optional directed inference key used by Line (X+/X-/Y+/Y-/Z+/Z-).
        self.axis_direction = None

    @property
    def valid(self):
        return (
            self.location is not None
            and self.snap_type is not None
        )


def find_nearest_endpoint(
    context,
    mouse_x,
    mouse_y,
    pixel_radius=12
):
    """
    Find the nearest visible mesh vertex to the mouse cursor.

    Returns:
        SnapResult if an endpoint is found.
        None otherwise.
    """

    region = context.region
    rv3d = context.region_data

    best_result = None
    best_distance = pixel_radius

    # v138: Edit Mode must read the live BMesh. obj.data vertices/edges can
    # lag behind the edit mesh, which made Endpoint/Midpoint/Edge snapping
    # inconsistent for Rectangle/Circle/Freehand after topology changes.
    edit_obj = context.edit_object if context.mode == 'EDIT_MESH' else None
    if edit_obj is not None and edit_obj.type == 'MESH':
        try:
            bm = bmesh.from_edit_mesh(edit_obj.data)
            matrix_world = edit_obj.matrix_world
            for vertex in bm.verts:
                world_pos = matrix_world @ vertex.co
                screen_pos = view3d_utils.location_3d_to_region_2d(region, rv3d, world_pos)
                if screen_pos is None:
                    continue
                distance = (screen_pos - Vector((mouse_x, mouse_y))).length
                if distance < best_distance:
                    best_distance = distance
                    best_result = SnapResult(
                        location=world_pos, snap_type="ENDPOINT", obj=edit_obj,
                        element=vertex, screen_distance=distance,
                    )
        except Exception:
            pass

    for obj in context.visible_objects:
        if obj is edit_obj:
            continue

        if obj.type != 'MESH':
            continue

        mesh = obj.data
        matrix_world = obj.matrix_world

        for vertex in mesh.vertices:

            world_pos = (
                matrix_world
                @ vertex.co
            )

            screen_pos = (
                view3d_utils.location_3d_to_region_2d(
                    region,
                    rv3d,
                    world_pos
                )
            )

            if screen_pos is None:
                continue

            dx = screen_pos.x - mouse_x
            dy = screen_pos.y - mouse_y

            distance = (
                dx * dx
                + dy * dy
            ) ** 0.5

            if distance < best_distance:

                best_distance = distance

                best_result = SnapResult(
                    location=world_pos,
                    snap_type="ENDPOINT",
                    obj=obj,
                    element=vertex,
                    screen_distance=distance,
                )

    return best_result
    
    
def find_nearest_midpoint(
    context,
    mouse_x,
    mouse_y,
    pixel_radius=12
):
    """
    Find the nearest visible mesh edge midpoint
    to the mouse cursor.

    Returns:
        SnapResult if a midpoint is found.
        None otherwise.
    """

    region = context.region
    rv3d = context.region_data

    best_result = None
    best_distance = pixel_radius

    edit_obj = context.edit_object if context.mode == 'EDIT_MESH' else None
    if edit_obj is not None and edit_obj.type == 'MESH':
        try:
            bm = bmesh.from_edit_mesh(edit_obj.data)
            matrix_world = edit_obj.matrix_world
            for edge in bm.edges:
                world_a = matrix_world @ edge.verts[0].co
                world_b = matrix_world @ edge.verts[1].co
                midpoint = (world_a + world_b) * 0.5
                screen_pos = view3d_utils.location_3d_to_region_2d(region, rv3d, midpoint)
                if screen_pos is None:
                    continue
                distance = (screen_pos - Vector((mouse_x, mouse_y))).length
                if distance < best_distance:
                    best_distance = distance
                    best_result = SnapResult(
                        location=midpoint, snap_type="MIDPOINT", obj=edit_obj,
                        element=edge, screen_distance=distance,
                    )
        except Exception:
            pass

    for obj in context.visible_objects:
        if obj is edit_obj:
            continue

        if obj.type != 'MESH':
            continue

        mesh = obj.data
        matrix_world = obj.matrix_world

        for edge in mesh.edges:

            vertex_a = mesh.vertices[
                edge.vertices[0]
            ]

            vertex_b = mesh.vertices[
                edge.vertices[1]
            ]

            world_a = (
                matrix_world
                @ vertex_a.co
            )

            world_b = (
                matrix_world
                @ vertex_b.co
            )

            midpoint = (
                world_a + world_b
            ) * 0.5

            screen_pos = (
                view3d_utils.location_3d_to_region_2d(
                    region,
                    rv3d,
                    midpoint
                )
            )

            if screen_pos is None:
                continue

            dx = screen_pos.x - mouse_x
            dy = screen_pos.y - mouse_y

            distance = (
                dx * dx
                + dy * dy
            ) ** 0.5

            if distance < best_distance:

                best_distance = distance

                best_result = SnapResult(
                    location=midpoint,
                    snap_type="MIDPOINT",
                    obj=obj,
                    element=edge,
                    screen_distance=distance,
                )

    return best_result 



def find_nearest_edge(
    context,
    mouse_x,
    mouse_y,
    pixel_radius=12
):
    """
    Find the nearest point along a visible mesh edge to the mouse cursor.

    The proximity test is performed in screen space so the snap tolerance
    remains consistent while zooming.  The returned 3D point is interpolated
    along the matching world-space edge.

    Returns:
        SnapResult if an edge is found.
        None otherwise.
    """

    region = context.region
    rv3d = context.region_data

    if region is None or rv3d is None:
        return None

    mouse = Vector((mouse_x, mouse_y))

    best_result = None
    best_distance = float(pixel_radius)

    edit_obj = context.edit_object if context.mode == 'EDIT_MESH' else None
    if edit_obj is not None and edit_obj.type == 'MESH':
        try:
            bm = bmesh.from_edit_mesh(edit_obj.data)
            matrix_world = edit_obj.matrix_world
            for edge in bm.edges:
                world_a = matrix_world @ edge.verts[0].co
                world_b = matrix_world @ edge.verts[1].co
                screen_a = view3d_utils.location_3d_to_region_2d(region, rv3d, world_a)
                screen_b = view3d_utils.location_3d_to_region_2d(region, rv3d, world_b)
                if screen_a is None or screen_b is None:
                    continue
                screen_edge = screen_b - screen_a
                length_squared = screen_edge.length_squared
                if length_squared <= 1e-12:
                    continue
                screen_t = (mouse - screen_a).dot(screen_edge) / length_squared
                screen_t = max(0.0, min(1.0, screen_t))
                closest_screen = screen_a + screen_edge * screen_t
                distance = (mouse - closest_screen).length
                if distance <= best_distance:
                    # v172: screen_t is NOT the same as the 3D edge parameter
                    # in perspective view. Convert the projected interpolation
                    # to a perspective-correct world-space parameter so the
                    # yellow snap marker stays under the tool tip while sliding.
                    clip_a = rv3d.perspective_matrix @ world_a.to_4d()
                    clip_b = rv3d.perspective_matrix @ world_b.to_4d()
                    wa = abs(float(clip_a.w))
                    wb = abs(float(clip_b.w))
                    denom = wb * (1.0 - screen_t) + wa * screen_t
                    world_t = (wa * screen_t / denom) if denom > 1e-12 else screen_t
                    world_t = max(0.0, min(1.0, world_t))
                    best_distance = distance
                    best_result = SnapResult(
                        location=world_a.lerp(world_b, world_t), snap_type="EDGE",
                        obj=edit_obj, element=edge, screen_distance=distance,
                    )
        except Exception:
            pass

    for obj in context.visible_objects:
        if obj is edit_obj:
            continue

        if obj.type != 'MESH':
            continue

        mesh = obj.data
        matrix_world = obj.matrix_world

        for edge in mesh.edges:

            vertex_a = mesh.vertices[edge.vertices[0]]
            vertex_b = mesh.vertices[edge.vertices[1]]

            world_a = matrix_world @ vertex_a.co
            world_b = matrix_world @ vertex_b.co

            screen_a = view3d_utils.location_3d_to_region_2d(
                region, rv3d, world_a
            )
            screen_b = view3d_utils.location_3d_to_region_2d(
                region, rv3d, world_b
            )

            if screen_a is None or screen_b is None:
                continue

            screen_edge = screen_b - screen_a
            length_squared = screen_edge.length_squared

            if length_squared <= 1e-12:
                continue

            # Closest point on the finite projected segment.
            screen_t = (mouse - screen_a).dot(screen_edge) / length_squared
            screen_t = max(0.0, min(1.0, screen_t))

            closest_screen = screen_a + screen_edge * screen_t
            distance = (mouse - closest_screen).length

            if distance <= best_distance:
                best_distance = distance

                # v172 perspective-correct edge tracking. A linear parameter
                # on the projected 2D segment is not a linear parameter on the
                # 3D edge under perspective projection. Convert it using clip
                # W so the 3D snap point reprojects to the point nearest mouse.
                clip_a = rv3d.perspective_matrix @ world_a.to_4d()
                clip_b = rv3d.perspective_matrix @ world_b.to_4d()
                wa = abs(float(clip_a.w))
                wb = abs(float(clip_b.w))
                denom = wb * (1.0 - screen_t) + wa * screen_t
                world_t = (wa * screen_t / denom) if denom > 1e-12 else screen_t
                world_t = max(0.0, min(1.0, world_t))
                world_point = world_a.lerp(world_b, world_t)

                best_result = SnapResult(
                    location=world_point,
                    snap_type="EDGE",
                    obj=obj,
                    element=edge,
                    screen_distance=distance,
                )

    return best_result

def find_origin_snap(
    context,
    mouse_x,
    mouse_y,
    pixel_radius=12
):
    """
    Snap to the world origin (0, 0, 0)
    when the mouse cursor is close enough.
    """

    region = context.region
    rv3d = context.region_data

    origin = Vector(
        (0.0, 0.0, 0.0)
    )

    screen_pos = (
        view3d_utils.location_3d_to_region_2d(
            region,
            rv3d,
            origin
        )
    )

    if screen_pos is None:
        return None

    dx = screen_pos.x - mouse_x
    dy = screen_pos.y - mouse_y

    distance = (
        dx * dx
        + dy * dy
    ) ** 0.5

    if distance <= pixel_radius:

        return SnapResult(
            location=origin,
            snap_type="ORIGIN",
            screen_distance=distance,
        )

    return None


def find_grid_snap(
    context,
    mouse_x,
    mouse_y,
    pixel_radius=12
):
    """
    Snap to the nearest grid intersection.

    Supports:
        Top / Bottom  -> XY grid
        Front / Back  -> XZ grid
        Left / Right  -> YZ grid
        Perspective   -> XY grid
    """

    region = context.region
    rv3d = context.region_data

    if region is None or rv3d is None:
        return None

    mouse_coord = Vector(
        (mouse_x, mouse_y)
    )

    ray_origin = (
        view3d_utils.region_2d_to_origin_3d(
            region,
            rv3d,
            mouse_coord
        )
    )

    ray_direction = (
        view3d_utils.region_2d_to_vector_3d(
            region,
            rv3d,
            mouse_coord
        )
    )

    # -------------------------------------
    # Grid size
    # -------------------------------------

    space = context.space_data

    if (
        space is not None
        and space.type == 'VIEW_3D'
    ):
        grid_size = space.overlay.grid_scale
    else:
        grid_size = 1.0

    if grid_size <= 0:
        grid_size = 1.0

    # -------------------------------------
    # Determine current view direction
    # -------------------------------------

    view_direction = (
        rv3d.view_rotation
        @ Vector((0.0, 0.0, -1.0))
    )

    abs_x = abs(view_direction.x)
    abs_y = abs(view_direction.y)
    abs_z = abs(view_direction.z)
    
    is_perspective = (
        rv3d.view_perspective == 'PERSP'
    )

    # -------------------------------------
    # PERSPECTIVE / TOP / BOTTOM
    # XY plane
    # -------------------------------------

    if (
        is_perspective
        or (
            abs_z >= abs_x
            and abs_z >= abs_y
        )
    ):

        if abs(ray_direction.z) < 0.000001:
            return None

        t = (
            -ray_origin.z
            / ray_direction.z
        )

        point = (
            ray_origin
            + ray_direction * t
        )

        grid_point = Vector(
            (
                round(point.x / grid_size)
                * grid_size,

                round(point.y / grid_size)
                * grid_size,

                0.0
            )
        )

    # -------------------------------------
    # FRONT / BACK
    # XZ plane
    # -------------------------------------

    elif abs_y >= abs_x and abs_y >= abs_z:

        if abs(ray_direction.y) < 0.000001:
            return None

        t = (
            -ray_origin.y
            / ray_direction.y
        )

        point = (
            ray_origin
            + ray_direction * t
        )

        grid_point = Vector(
            (
                round(point.x / grid_size)
                * grid_size,

                0.0,

                round(point.z / grid_size)
                * grid_size
            )
        )

    # -------------------------------------
    # LEFT / RIGHT
    # YZ plane
    # -------------------------------------

    else:

        if abs(ray_direction.x) < 0.000001:
            return None

        t = (
            -ray_origin.x
            / ray_direction.x
        )

        point = (
            ray_origin
            + ray_direction * t
        )

        grid_point = Vector(
            (
                0.0,

                round(point.y / grid_size)
                * grid_size,

                round(point.z / grid_size)
                * grid_size
            )
        )

    # -------------------------------------
    # Check screen distance
    # -------------------------------------

    screen_pos = (
        view3d_utils.location_3d_to_region_2d(
            region,
            rv3d,
            grid_point
        )
    )

    if screen_pos is None:
        return None

    dx = screen_pos.x - mouse_x
    dy = screen_pos.y - mouse_y

    screen_distance = (
        dx * dx
        + dy * dy
    ) ** 0.5

    if screen_distance <= pixel_radius:

        return SnapResult(
            location=grid_point,
            snap_type="GRID",
            screen_distance=screen_distance,
        )

    return None

# -----------------------------------------------------------------------------
# Shared SketchTools inference resolver
# -----------------------------------------------------------------------------

SNAP_LABELS = {
    "ENDPOINT": "Endpoint",
    "MIDPOINT": "Midpoint",
    "CENTER": "Center",
    "EDGE": "On Edge",
    "FACE": "On Face",
    "ORIGIN": "Origin",
    "X_AXIS": "On X Axis",
    "Y_AXIS": "On Y Axis",
    "Z_AXIS": "On Z Axis",
    "GRID": "On Grid",
    "X_GRID": "On Grid · X Axis",
    "Y_GRID": "On Grid · Y Axis",
    "Z_GRID": "On Grid · Z Axis",
}


def snap_label(result):
    if result is None:
        return None
    return SNAP_LABELS.get(result.snap_type, result.snap_type.title())


def _screen_distance(context, world_pos, mouse_x, mouse_y):
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None
    screen = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(world_pos))
    if screen is None:
        return None
    return (screen - Vector((mouse_x, mouse_y))).length


def find_circle_center(context, mouse_x, mouse_y, pixel_radius=14, extra_centers=None):
    best = None
    best_distance = float(pixel_radius)
    candidates = []

    def add_regular_face_center(obj, verts_world, center_world):
        # A SketchTools circle can legitimately have only 3 segments, so
        # regular triangles and all higher regular polygons are valid centers.
        if len(verts_world) < 3:
            return
        center = Vector(center_world)
        radii = [(Vector(v) - center).length for v in verts_world]
        avg = sum(radii) / len(radii) if radii else 0.0
        if avg <= 1e-8:
            return
        if max(abs(r - avg) for r in radii) <= max(1e-4, avg * 0.03):
            candidates.append((center, obj))

    def add_regular_loop_center(obj, verts_world):
        """v177: derive Circle Center from a closed geometry loop itself.

        This deliberately does not depend on the construction grid or on the
        loop owning a face.  A circle drawn off-grid can therefore still offer
        a Center snap as long as its visible closed edge loop is circular.
        """
        if len(verts_world) < 3:
            return
        pts = [Vector(v) for v in verts_world]
        center = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
        radii = [(p - center).length for p in pts]
        avg = sum(radii) / len(radii) if radii else 0.0
        if avg <= 1e-8:
            return
        # Same tolerance as regular face detection.  This accepts SketchTools
        # 3-segment circles while rejecting ordinary irregular closed loops.
        if max(abs(r - avg) for r in radii) > max(1e-4, avg * 0.03):
            return
        # Guard against a non-planar equal-radius cloud.
        normal = Vector((0.0, 0.0, 0.0))
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            normal.x += (p.y - q.y) * (p.z + q.z)
            normal.y += (p.z - q.z) * (p.x + q.x)
            normal.z += (p.x - q.x) * (p.y + q.y)
        if normal.length <= 1e-9:
            return
        normal.normalize()
        if max(abs((p - center).dot(normal)) for p in pts) > max(1e-4, avg * 0.01):
            return
        candidates.append((center, obj))

    for obj in context.visible_objects:
        if obj.type != 'MESH':
            continue

        if obj.get("sketchtools_circle"):
            # The parametric circle is authored around local origin.  Transform
            # that origin to world space rather than assuming translation-only
            # placement; this remains correct if the object has been rotated,
            # parented or otherwise transformed.
            candidates.append((obj.matrix_world @ Vector((0.0, 0.0, 0.0)), obj))

        # In Edit Mode inspect the live BMesh.  obj.data.polygons can lag the
        # edit mesh, which previously made Center snapping disappear after a
        # circle was cut into a face.
        if context.mode == 'EDIT_MESH' and obj is context.edit_object:
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                for face in bm.faces:
                    if len(face.verts) < 3:
                        continue
                    center = obj.matrix_world @ face.calc_center_median()
                    verts = [obj.matrix_world @ v.co for v in face.verts]
                    add_regular_face_center(obj, verts, center)

                # v177 off-grid Center: also inspect isolated closed edge loops.
                # Use only degree-2 components so arbitrary connected modelling
                # networks are not mistaken for circles.
                unseen = set(e for e in bm.edges if e.is_valid)
                while unseen:
                    seed = unseen.pop()
                    comp_edges = {seed}
                    comp_verts = set(seed.verts)
                    stack = list(seed.verts)
                    while stack:
                        vv = stack.pop()
                        for ee in vv.link_edges:
                            if ee in unseen:
                                unseen.remove(ee)
                                comp_edges.add(ee)
                                for nv in ee.verts:
                                    if nv not in comp_verts:
                                        comp_verts.add(nv)
                                        stack.append(nv)
                    if 3 <= len(comp_verts) <= 256 and len(comp_edges) == len(comp_verts):
                        if all(sum(1 for ee in vtx.link_edges if ee in comp_edges) == 2 for vtx in comp_verts):
                            # Order the loop for the planarity/Newell test.
                            start = next(iter(comp_verts))
                            ordered = [start]
                            prev = None
                            cur = start
                            for _ in range(len(comp_verts) - 1):
                                nxts = []
                                for ee in cur.link_edges:
                                    if ee not in comp_edges:
                                        continue
                                    nv = ee.other_vert(cur)
                                    if nv is not prev:
                                        nxts.append(nv)
                                if not nxts:
                                    break
                                nxt = nxts[0]
                                ordered.append(nxt)
                                prev, cur = cur, nxt
                            if len(ordered) == len(comp_verts):
                                add_regular_loop_center(obj, [obj.matrix_world @ vtx.co for vtx in ordered])
            except Exception:
                pass
        else:
            mesh = obj.data
            for poly in mesh.polygons:
                if len(poly.vertices) < 3:
                    continue
                center = obj.matrix_world @ poly.center
                verts = [obj.matrix_world @ mesh.vertices[i].co for i in poly.vertices]
                add_regular_face_center(obj, verts, center)


            # v177: face-less/off-grid Circle objects still expose Center.
            adjacency = {i: [] for i in range(len(mesh.vertices))}
            for edge in mesh.edges:
                a, b = edge.vertices
                adjacency[a].append(b)
                adjacency[b].append(a)
            eligible = {i for i, nbrs in adjacency.items() if len(nbrs) == 2}
            seen = set()
            for seed in list(eligible):
                if seed in seen:
                    continue
                comp = []
                stack = [seed]
                seen.add(seed)
                while stack:
                    vi = stack.pop()
                    comp.append(vi)
                    for nj in adjacency[vi]:
                        if nj in eligible and nj not in seen:
                            seen.add(nj)
                            stack.append(nj)
                if not (3 <= len(comp) <= 256):
                    continue
                if not all(all(n in comp for n in adjacency[i]) for i in comp):
                    continue
                # Walk in cyclic order.
                ordered = [comp[0]]
                prev = None
                cur = comp[0]
                for _ in range(len(comp)-1):
                    nxts = [n for n in adjacency[cur] if n != prev and n not in ordered[:-1]]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    ordered.append(nxt)
                    prev, cur = cur, nxt
                if len(ordered) == len(comp):
                    add_regular_loop_center(obj, [obj.matrix_world @ mesh.vertices[i].co for i in ordered])

    for center in extra_centers or ():
        candidates.append((Vector(center), None))

    for center, obj in candidates:
        distance = _screen_distance(context, center, mouse_x, mouse_y)
        if distance is not None and distance <= best_distance:
            best_distance = distance
            best = SnapResult(center, "CENTER", obj=obj, element=None, screen_distance=distance)
    return best

def find_face_snap(context, mouse_x, mouse_y):
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None

    coord = Vector((mouse_x, mouse_y))
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
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
                local_origin, local_direction
            )
            if hit_location is not None and face_index is not None:
                return SnapResult(
                    obj.matrix_world @ hit_location,
                    "FACE",
                    obj=obj,
                    element=face_index,
                    screen_distance=0.0,
                )

    depsgraph = context.evaluated_depsgraph_get()
    hit, location, normal, face_index, obj, matrix = context.scene.ray_cast(
        depsgraph, ray_origin, ray_direction
    )

    if hit and obj is not None and obj.type == 'MESH':
        return SnapResult(location, "FACE", obj=obj, element=face_index, screen_distance=0.0)
    return None


def find_axis_inference(
    context,
    mouse_x,
    mouse_y,
    origin,
    plane_normal=None,
    pixel_radius=13,
    world_3d=False,
    preferred_axis=None,
):
    """Infer global X/Y/Z from their actual screen-projected directions.

    v108 keeps the inference deliberately visual: each signed world axis is
    projected from the current line anchor into the viewport and the mouse
    direction is compared with those projected rays.  A new axis is acquired
    only when it is clearly the best screen-space match.  Once acquired, a
    slightly wider release corridor provides hysteresis without making the
    inference feel sticky.  Manual X/Y/Z locks remain authoritative upstream.
    """
    from math import acos, degrees

    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None or origin is None:
        return None

    origin = Vector(origin)
    mouse = Vector((mouse_x, mouse_y))
    coord = Vector((mouse_x, mouse_y))

    normal = None
    if plane_normal is not None:
        normal = Vector(plane_normal)
        if normal.length > 1e-10:
            normal.normalize()
        else:
            normal = None

    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord).normalized()

    mouse_on_plane = None
    if not world_3d:
        if normal is None:
            return None
        denom = ray_direction.dot(normal)
        if abs(denom) <= 1e-8:
            return None
        t_plane = (origin - ray_origin).dot(normal) / denom
        if rv3d.view_perspective == 'PERSP' and t_plane < 0.0:
            return None
        mouse_on_plane = ray_origin + ray_direction * t_plane

    origin_screen = view3d_utils.location_3d_to_region_2d(region, rv3d, origin)
    if origin_screen is None:
        return None

    mouse_vec = mouse - origin_screen
    mouse_len = mouse_vec.length
    if mouse_len <= 3.0:
        return None
    mouse_unit = mouse_vec / mouse_len

    axes = (
        (Vector((1.0, 0.0, 0.0)), "X_AXIS", "X"),
        (Vector((0.0, 1.0, 0.0)), "Y_AXIS", "Y"),
        (Vector((0.0, 0.0, 1.0)), "Z_AXIS", "Z"),
    )

    # Tight acquisition plus a modest hold corridor gives "magnetic" axis
    # inference while avoiding v107's over-eager/sticky behavior.
    acquire_angle = 7.0
    release_angle = 12.0
    acquire_px = max(7.0, min(12.0, float(pixel_radius)))
    release_px = max(12.0, acquire_px * 1.65)
    local_span = max(float(getattr(rv3d, "view_distance", 10.0)) * 2.0, 2.0)

    raw = []

    for world_axis, snap_type, axis_letter in axes:
        if world_3d:
            base_direction = world_axis.copy()
        else:
            base_direction = world_axis - normal * world_axis.dot(normal)
            if base_direction.length <= 1e-8:
                continue
            base_direction.normalize()

        for sign in (1.0, -1.0):
            direction = base_direction * sign
            signed_key = axis_letter + ("+" if sign > 0.0 else "-")
            probe_screen = view3d_utils.location_3d_to_region_2d(
                region, rv3d, origin + direction * local_span
            )
            if probe_screen is None:
                continue

            screen_ray = probe_screen - origin_screen
            ray_len = screen_ray.length
            if ray_len <= 1e-6:
                continue
            screen_unit = screen_ray / ray_len

            dot = max(-1.0, min(1.0, mouse_unit.dot(screen_unit)))
            angle = degrees(acos(dot))
            if angle > 90.0:
                continue

            along = mouse_vec.dot(screen_unit)
            if along < 0.0:
                continue
            closest_screen = origin_screen + screen_unit * along
            pixel_distance = (mouse - closest_screen).length

            # v109: hysteresis is directional. Holding +Z must never make -Z
            # sticky merely because both belong to the Z family. Older callers
            # that pass X_AXIS/Y_AXIS/Z_AXIS still receive family-level hold.
            is_preferred = (signed_key == preferred_axis) or (snap_type == preferred_axis)
            angle_limit = release_angle if is_preferred else acquire_angle
            px_limit = release_px if is_preferred else acquire_px

            # Close to the anchor, angular measurements are noisy, so permit a
            # small pixel corridor. Farther away require both good direction
            # and reasonable perpendicular distance.
            if mouse_len < 36.0:
                aligned = (angle <= angle_limit) or (pixel_distance <= px_limit)
            else:
                aligned = (angle <= angle_limit) and (pixel_distance <= max(px_limit, mouse_len * 0.13))
            if not aligned:
                continue

            if world_3d:
                closest = intersect_line_line(
                    origin, origin + direction,
                    ray_origin, ray_origin + ray_direction,
                )
                if closest is None:
                    continue
                point_on_axis = Vector(closest[0])
                if (point_on_axis - origin).dot(direction) < -1e-7:
                    continue
            else:
                delta = mouse_on_plane - origin
                amount = delta.dot(direction)
                if amount < -1e-7:
                    continue
                point_on_axis = origin + direction * max(0.0, amount)

            # Screen-space score. Preferred axis gets only a small hold bonus;
            # it must still remain inside the release corridor.
            score = angle + (pixel_distance / max(mouse_len, 1.0)) * 10.0
            if is_preferred:
                score -= 1.0
            raw.append((score, angle, pixel_distance, snap_type, point_on_axis, signed_key))

    if not raw:
        return None

    raw.sort(key=lambda item: item[0])
    best = raw[0]

    # Ambiguity guard: when two different axis families project almost on top
    # of one another, do not guess unless one is the already-held axis. This
    # is especially important in perspective views looking nearly down an axis.
    preferred_matches_best = preferred_axis in {best[3], best[5]}
    if not preferred_matches_best:
        for other in raw[1:]:
            if other[3] != best[3]:
                if (other[0] - best[0]) < 1.75:
                    return None
                break

    result = SnapResult(best[4], best[3], screen_distance=best[2])
    result.axis_direction = best[5]
    return result


def find_axis_grid_snap(
    context,
    mouse_x,
    mouse_y,
    origin,
    plane_normal=None,
    pixel_radius=13,
    world_3d=False,
    preferred_axis=None,
):
    """Snap to a native grid crossing that lies on the currently inferred axis.

    v95: the ordinary grid finder rounds all coordinates on the active view
    grid.  That is useful for free grid snapping, but it can miss the grid
    crossing directly ahead when a Line is constrained to X/Y/Z.  This helper
    first resolves the same SketchTools axis inference, then quantizes only the
    active world-axis coordinate while keeping the result exactly collinear
    with the line's inference origin.
    """
    if origin is None:
        return None

    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None

    axis_result = find_axis_inference(
        context,
        mouse_x,
        mouse_y,
        origin,
        plane_normal=plane_normal,
        pixel_radius=max(14, pixel_radius),
        world_3d=world_3d,
        preferred_axis=preferred_axis,
    )
    if axis_result is None or not axis_result.valid:
        return None

    axis_index = {
        "X_AXIS": 0,
        "Y_AXIS": 1,
        "Z_AXIS": 2,
    }.get(axis_result.snap_type)
    if axis_index is None:
        return None

    space = context.space_data
    if space is not None and space.type == 'VIEW_3D':
        grid_size = float(space.overlay.grid_scale)
    else:
        grid_size = 1.0
    if grid_size <= 0.0:
        grid_size = 1.0

    origin = Vector(origin)
    axis_point = Vector(axis_result.location)
    delta = axis_point - origin
    if delta.length <= 1e-9:
        return None
    direction = delta.normalized()

    component = direction[axis_index]
    if abs(component) <= 1e-9:
        return None

    # Round the active world coordinate to the same native Blender grid used
    # by find_grid_snap(), then solve back along the inferred line.  The other
    # coordinates remain on the exact X/Y/Z inference ray from the start point.
    mouse_axis_coord = axis_point[axis_index]
    target_axis_coord = round(mouse_axis_coord / grid_size) * grid_size
    amount = (target_axis_coord - origin[axis_index]) / component

    # Six-ray inference is directional. Never snap behind the current anchor,
    # and avoid snapping straight back onto the anchor itself.
    if amount <= 1e-7:
        return None

    candidate = origin + direction * amount
    screen = view3d_utils.location_3d_to_region_2d(region, rv3d, candidate)
    if screen is None:
        return None

    distance = (screen - Vector((mouse_x, mouse_y))).length
    if distance > float(pixel_radius):
        return None

    grid_type = {
        "X_AXIS": "X_GRID",
        "Y_AXIS": "Y_GRID",
        "Z_AXIS": "Z_GRID",
    }[axis_result.snap_type]

    result = SnapResult(
        location=candidate,
        snap_type=grid_type,
        screen_distance=distance,
    )
    result.axis_direction = getattr(axis_result, "axis_direction", None)
    return result

def find_axis_snap(context, mouse_x, mouse_y, pixel_radius=9):
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None

    mouse = Vector((mouse_x, mouse_y))
    best = None
    best_distance = float(pixel_radius)
    axes = (
        (Vector((1.0, 0.0, 0.0)), "X_AXIS", "X"),
        (Vector((0.0, 1.0, 0.0)), "Y_AXIS", "Y"),
        (Vector((0.0, 0.0, 1.0)), "Z_AXIS", "Z"),
    )

    extent = 100000.0
    for axis, snap_type, _axis_letter in axes:
        a3 = -axis * extent
        b3 = axis * extent
        a2 = view3d_utils.location_3d_to_region_2d(region, rv3d, a3)
        b2 = view3d_utils.location_3d_to_region_2d(region, rv3d, b3)
        if a2 is None or b2 is None:
            continue

        edge = b2 - a2
        lsq = edge.length_squared
        if lsq <= 1e-12:
            continue

        t = max(0.0, min(1.0, (mouse - a2).dot(edge) / lsq))
        closest2 = a2 + edge * t
        distance = (mouse - closest2).length
        if distance <= best_distance:
            best_distance = distance
            best = SnapResult(a3.lerp(b3, t), snap_type, screen_distance=distance)
    return best


def _constrain_result_to_plane(
    result, plane_point, plane_normal, tolerance=1e-4, allow_off_plane_geometry=False
):
    if result is None or not result.valid or plane_point is None or plane_normal is None:
        return result

    normal = Vector(plane_normal).normalized()
    point = Vector(result.location)
    plane_point = Vector(plane_point)
    signed = (point - plane_point).dot(normal)

    precision_geometry = {"ENDPOINT", "MIDPOINT", "CENTER", "EDGE", "ORIGIN"}
    if allow_off_plane_geometry and result.snap_type in precision_geometry:
        # v100: when a 3D construction tool explicitly allows geometry snaps
        # outside its fallback drawing plane, the exact mesh coordinate is
        # authoritative.  Do not reject or project it onto the plane.
        result.location = point
        return result

    if result.snap_type in {
        "ENDPOINT", "MIDPOINT", "CENTER", "EDGE", "ORIGIN",
        "X_AXIS", "Y_AXIS", "Z_AXIS",
        "X_GRID", "Y_GRID", "Z_GRID"
    } and abs(signed) > tolerance:
        return None

    result.location = point - normal * signed
    return result


SNAP_PRIORITY = {
    # Precision geometry inference.
    "ENDPOINT": 110,
    "CENTER": 98,
    "MIDPOINT": 96,
    "ORIGIN": 94,

    # Continuous inference.
    # Axis-grid crossings are discrete precision targets and should beat the
    # continuous axis ray, while endpoints/centers/midpoints/origin remain
    # stronger. This is the v95 same-axis grid preference.
    "X_GRID": 90,
    "Y_GRID": 90,
    "Z_GRID": 90,
    "X_AXIS": 88,
    "Y_AXIS": 88,
    "Z_AXIS": 88,
    "EDGE": 92,

    # Surface/grid are intentionally fallback classes.
    "FACE": 30,
    "GRID": 20,
}


def _snap_score(result, previous_snap_type=None):
    """
    Higher is better. Priority dominates, but nearby candidates of similar
    semantic importance can win naturally instead of finder order deciding.

    v178: add a deliberately small same-class hysteresis bonus.  This does
    not let Edge beat Endpoint/Midpoint/Center; it only stops an acquired
    geometry snap from flickering to Face/Grid/axis inference as the cursor
    moves a few pixels or the viewport zoom changes.
    """
    if result is None or not result.valid:
        return None
    priority = SNAP_PRIORITY.get(result.snap_type, 0)
    distance = (
        float(result.screen_distance)
        if result.screen_distance is not None
        else 999.0
    )
    sticky_bonus = 45.0 if previous_snap_type == result.snap_type else 0.0
    return priority * 100.0 - distance + sticky_bonus



def _result_is_occluded(context, result):
    """Return True when a snap candidate lies behind the first visible mesh surface.

    SketchTools is screen-driven: a vertex/edge/grid point that projects near
    the cursor must not be allowed to snap through an opaque face in front of
    it. FACE results are already the first visible surface and are never
    rejected here.
    """
    if result is None or not result.valid or result.snap_type == "FACE":
        return False

    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return False

    candidate = Vector(result.location)
    screen = view3d_utils.location_3d_to_region_2d(region, rv3d, candidate)
    if screen is None:
        return False

    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, screen)
    ray_direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, screen).normalized()
    candidate_distance = (candidate - ray_origin).length
    epsilon = max(1e-4, candidate_distance * 1e-5)
    if result.snap_type == "ENDPOINT" and context.mode == 'EDIT_MESH':
        # v163: intersection-created endpoints can sit microscopically behind
        # the BVH hit due to edit-mesh/ray precision. Give endpoints a slightly
        # wider depth equality tolerance without disabling true occlusion.
        epsilon = max(epsilon, 1e-3, candidate_distance * 5e-5)

    # In Edit Mode use the live BMesh first; Scene.ray_cast can lag edit data.
    if context.mode == 'EDIT_MESH':
        obj = context.edit_object
        if obj is not None and obj.type == 'MESH':
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                inv = obj.matrix_world.inverted()
                local_origin = inv @ ray_origin
                local_direction = (inv.to_3x3() @ ray_direction).normalized()
                tree = BVHTree.FromBMesh(bm)
                hit_location, _normal, _face_index, _distance = tree.ray_cast(
                    local_origin, local_direction
                )
                if hit_location is not None:
                    hit_world = obj.matrix_world @ hit_location
                    # A center/vertex/edge lying on the visible face itself is
                    # not occluded.  This matters especially for vertical or
                    # steeply viewed discs where tiny ray/BVH error previously
                    # made Center inference flicker on and off.
                    if result.obj is obj and (hit_world - candidate).length <= max(1e-4, epsilon * 4.0):
                        return False
                    if (hit_world - ray_origin).length < candidate_distance - epsilon:
                        return True
            except Exception:
                pass

    try:
        depsgraph = context.evaluated_depsgraph_get()
        hit, location, _normal, _face_index, obj, _matrix = context.scene.ray_cast(
            depsgraph, ray_origin, ray_direction
        )
        if hit and obj is not None and obj.type == 'MESH':
            hit_world = Vector(location)
            if result.obj is obj and (hit_world - candidate).length <= max(1e-4, epsilon * 4.0):
                return False
            if (hit_world - ray_origin).length < candidate_distance - epsilon:
                return True
    except Exception:
        pass

    return False

def resolve_snap(
    context,
    mouse_x,
    mouse_y,
    plane_point=None,
    plane_normal=None,
    extra_centers=None,
    inference_origin=None,
    include_face=True,
    include_grid=True,
    world_axis_inference=False,
    preferred_axis=None,
    include_axis_grid=False,
    allow_off_plane_geometry=False,
    endpoint_pixel_radius=26,
    edge_pixel_radius=24,
    center_pixel_radius=22,
    midpoint_pixel_radius=22,
    previous_snap_type=None,
    snap_hysteresis_pixels=6,
):
    # v178 shared snapping refinement: strengthen every geometric snap class
    # and give the currently acquired class a small release margin.  Capture
    # remains screen-space based so it is visually consistent across zoom.
    # across every drawing tool.  Endpoint remains strongest; Edge now has a
    # wider screen-space capture zone (stable across zoom) and outranks continuous axis inference,
    # while Face/Grid remain fallbacks.
    sticky = previous_snap_type if previous_snap_type in {
        "ENDPOINT", "MIDPOINT", "CENTER", "EDGE"
    } else None
    endpoint_radius = endpoint_pixel_radius + (snap_hysteresis_pixels if sticky == "ENDPOINT" else 0)
    center_radius = center_pixel_radius + (snap_hysteresis_pixels if sticky == "CENTER" else 0)
    midpoint_radius = midpoint_pixel_radius + (snap_hysteresis_pixels if sticky == "MIDPOINT" else 0)
    edge_radius = edge_pixel_radius + (snap_hysteresis_pixels if sticky == "EDGE" else 0)

    finders = [
        lambda: find_nearest_endpoint(
            context, mouse_x, mouse_y, pixel_radius=endpoint_radius
        ),
        lambda: find_circle_center(
            context, mouse_x, mouse_y, pixel_radius=center_radius,
            extra_centers=extra_centers
        ),
        lambda: find_nearest_midpoint(
            context, mouse_x, mouse_y, pixel_radius=midpoint_radius
        ),
        lambda: find_origin_snap(
            context, mouse_x, mouse_y, pixel_radius=14
        ),
        lambda: find_nearest_edge(
            context, mouse_x, mouse_y, pixel_radius=edge_radius
        ),
        lambda: (
            find_axis_grid_snap(
                context,
                mouse_x,
                mouse_y,
                inference_origin,
                plane_normal=plane_normal,
                pixel_radius=13,
                world_3d=world_axis_inference,
                preferred_axis=preferred_axis,
            )
            if include_axis_grid and inference_origin is not None
            else None
        ),
        lambda: (
            find_axis_inference(
                context,
                mouse_x,
                mouse_y,
                inference_origin,
                plane_normal=plane_normal,
                pixel_radius=14,
                world_3d=world_axis_inference,
                preferred_axis=preferred_axis,
            )
            if inference_origin is not None
            else find_axis_snap(
                context,
                mouse_x,
                mouse_y,
                pixel_radius=9,
            )
        ),
    ]

    if include_face:
        finders.append(
            lambda: find_face_snap(context, mouse_x, mouse_y)
        )
    if include_grid:
        finders.append(
            lambda: find_grid_snap(
                context, mouse_x, mouse_y, pixel_radius=10
            )
        )

    candidates = []
    for finder in finders:
        result = finder()
        if not (
            world_axis_inference
            and result is not None
            and result.snap_type in {"X_AXIS", "Y_AXIS", "Z_AXIS", "X_GRID", "Y_GRID", "Z_GRID"}
        ):
            result = _constrain_result_to_plane(
                result,
                plane_point,
                plane_normal,
                allow_off_plane_geometry=allow_off_plane_geometry,
            )
        if result is not None and result.valid:
            # Never allow geometry/grid inference hidden by a nearer face.
            if _result_is_occluded(context, result):
                continue
            candidates.append(result)

    if not candidates:
        return None

    # Face is a fallback even though its projected distance is zero; semantic
    # priority prevents it stealing Endpoint/Midpoint/Center/Edge inference.
    return max(candidates, key=lambda result: _snap_score(result, previous_snap_type))
