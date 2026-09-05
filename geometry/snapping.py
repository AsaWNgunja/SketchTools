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

    for obj in context.visible_objects:

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

    for obj in context.visible_objects:

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

    for obj in context.visible_objects:

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
            t = (mouse - screen_a).dot(screen_edge) / length_squared
            t = max(0.0, min(1.0, t))

            closest_screen = screen_a + screen_edge * t
            distance = (mouse - closest_screen).length

            if distance <= best_distance:
                best_distance = distance

                # Use the same segment parameter in world space.  This keeps
                # the snapped point constrained exactly to the mesh edge.
                world_point = world_a.lerp(world_b, t)

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

    for obj in context.visible_objects:
        if obj.type != 'MESH':
            continue

        if obj.get("sketchtools_circle"):
            candidates.append((Vector(obj.matrix_world.translation), obj))

        # Edit Mode circles become part of the host mesh and no longer have a
        # dedicated object origin.  Recover regular polygon centers directly
        # from mesh faces so Center inference survives tool changes/restarts.
        mesh = obj.data
        for poly in mesh.polygons:
            if len(poly.vertices) < 8:
                continue
            center = obj.matrix_world @ poly.center
            verts = [obj.matrix_world @ mesh.vertices[i].co for i in poly.vertices]
            radii = [(v - center).length for v in verts]
            avg = sum(radii) / len(radii) if radii else 0.0
            if avg <= 1e-8:
                continue
            if max(abs(r - avg) for r in radii) <= max(1e-4, avg * 0.03):
                candidates.append((center, obj))

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
    pixel_radius=14,
    world_3d=False,
    preferred_axis=None,
):
    """
    Shared SketchUp-style Perspective X/Y/Z inference.

    All drawing tools use this same six-ray screen-space engine.  Each world
    axis is tested in both positive and negative directions from the current
    inference origin.  Line may keep the result in true 3D (world_3d=True);
    planar tools use the same inference choice but project the chosen world
    direction into their active drawing plane.

    ``preferred_axis`` provides hysteresis: the previously acquired X/Y/Z
    direction receives a slightly wider release corridor so normal hand
    movement does not make the inference flicker.
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
    if mouse_len <= 2.0:
        return None
    mouse_unit = mouse_vec / mouse_len

    axes = (
        (Vector((1.0, 0.0, 0.0)), "X_AXIS"),
        (Vector((0.0, 1.0, 0.0)), "Y_AXIS"),
        (Vector((0.0, 0.0, 1.0)), "Z_AXIS"),
    )

    # Separate acquire and hold thresholds approximate SketchUp's inference
    # latch.  Near the anchor a small pixel corridor helps because angular
    # measurements become noisy when the cursor has barely moved.
    acquire_angle = 6.0
    release_angle = 10.0
    near_anchor_band = float(pixel_radius)
    local_span = max(float(getattr(rv3d, "view_distance", 10.0)) * 2.0, 2.0)

    candidates = []

    for world_axis, snap_type in axes:
        if world_3d:
            base_direction = world_axis.copy()
        else:
            # Planar tools still infer the same global X/Y/Z families, but a
            # direction perpendicular to the active plane cannot be used to
            # construct planar geometry and is therefore ignored.
            base_direction = world_axis - normal * world_axis.dot(normal)
            if base_direction.length <= 1e-8:
                continue
            base_direction.normalize()

        for sign in (1.0, -1.0):
            direction = base_direction * sign

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

            # Directed comparison is the key six-ray behavior: +X and -X are
            # independent screen rays instead of one undirected line.
            dot = max(-1.0, min(1.0, mouse_unit.dot(screen_unit)))
            angle = degrees(acos(dot))
            if angle > 90.0:
                continue

            along = mouse_vec.dot(screen_unit)
            if along < 0.0:
                continue
            closest_screen = origin_screen + screen_unit * along
            pixel_distance = (mouse - closest_screen).length

            allowed_angle = release_angle if snap_type == preferred_axis else acquire_angle
            aligned = (
                angle <= allowed_angle
                or (mouse_len < 40.0 and pixel_distance <= near_anchor_band)
            )
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
                # A six-ray inference must not snap behind its anchor.
                if (point_on_axis - origin).dot(direction) < -1e-7:
                    continue
            else:
                delta = mouse_on_plane - origin
                amount = delta.dot(direction)
                if amount < -1e-7:
                    continue
                point_on_axis = origin + direction * max(0.0, amount)

            score = angle + (pixel_distance / max(mouse_len, 1.0)) * 2.0
            if snap_type == preferred_axis:
                score -= 1.25

            candidates.append((score, SnapResult(
                point_on_axis, snap_type, screen_distance=pixel_distance
            )))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]



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

    return SnapResult(
        location=candidate,
        snap_type=grid_type,
        screen_distance=distance,
    )

def find_axis_snap(context, mouse_x, mouse_y, pixel_radius=9):
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None

    mouse = Vector((mouse_x, mouse_y))
    best = None
    best_distance = float(pixel_radius)
    axes = (
        (Vector((1.0, 0.0, 0.0)), "X_AXIS"),
        (Vector((0.0, 1.0, 0.0)), "Y_AXIS"),
        (Vector((0.0, 0.0, 1.0)), "Z_AXIS"),
    )

    extent = 100000.0
    for axis, snap_type in axes:
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


def _constrain_result_to_plane(result, plane_point, plane_normal, tolerance=1e-4):
    if result is None or not result.valid or plane_point is None or plane_normal is None:
        return result

    normal = Vector(plane_normal).normalized()
    point = Vector(result.location)
    plane_point = Vector(plane_point)
    signed = (point - plane_point).dot(normal)

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
    "ENDPOINT": 100,
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
    "EDGE": 82,

    # Surface/grid are intentionally fallback classes.
    "FACE": 30,
    "GRID": 20,
}


def _snap_score(result):
    """
    Higher is better. Priority dominates, but nearby candidates of similar
    semantic importance can win naturally instead of finder order deciding.
    """
    if result is None or not result.valid:
        return None
    priority = SNAP_PRIORITY.get(result.snap_type, 0)
    distance = (
        float(result.screen_distance)
        if result.screen_distance is not None
        else 999.0
    )
    return priority * 100.0 - distance



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
            if (Vector(location) - ray_origin).length < candidate_distance - epsilon:
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
):
    # Keep radii intentionally modest. Endpoint/Center/Midpoint can be slightly
    # easier to acquire; continuous Edge/Axis inference should feel less sticky.
    finders = [
        lambda: find_nearest_endpoint(
            context, mouse_x, mouse_y, pixel_radius=14
        ),
        lambda: find_circle_center(
            context, mouse_x, mouse_y, pixel_radius=15,
            extra_centers=extra_centers
        ),
        lambda: find_nearest_midpoint(
            context, mouse_x, mouse_y, pixel_radius=14
        ),
        lambda: find_origin_snap(
            context, mouse_x, mouse_y, pixel_radius=14
        ),
        lambda: find_nearest_edge(
            context, mouse_x, mouse_y, pixel_radius=9
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
    return max(candidates, key=_snap_score)
