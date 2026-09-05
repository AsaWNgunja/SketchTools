from mathutils import Vector


AXIS_VECTORS = {
    "X": Vector((1.0, 0.0, 0.0)),
    "Y": Vector((0.0, 1.0, 0.0)),
    "Z": Vector((0.0, 0.0, 1.0)),
}

AXIS_SNAP_TYPES = {
    "X": "X_AXIS",
    "Y": "Y_AXIS",
    "Z": "Z_AXIS",
}

PLANE_LABELS = {
    "X": "YZ",
    "Y": "XZ",
    "Z": "XY",
}


def axis_vector(axis):
    return AXIS_VECTORS[str(axis).upper()].copy()


def closest_point_on_axis_to_view_ray(ray_origin, ray_direction, anchor, axis):
    """Return the point on a world axis through *anchor* nearest the view ray.

    This makes X/Y/Z locks independent of the current drawing plane, which is
    especially important for vertical Z drawing in Perspective view.
    """
    p = Vector(anchor)
    a = axis_vector(axis).normalized()
    q = Vector(ray_origin)
    d = Vector(ray_direction).normalized()

    # Minimize |(p + a*s) - (q + d*t)|.  With unit vectors the denominator is
    # 1-(a.d)^2.  When the camera is almost looking straight down the locked
    # axis, screen motion cannot determine a stable distance along that axis;
    # use the anchor rather than allowing a numerical jump.
    ad = a.dot(d)
    denom = 1.0 - ad * ad
    if abs(denom) < 1e-8:
        return p.copy()

    w0 = p - q
    s = (ad * d.dot(w0) - a.dot(w0)) / denom
    return p + a * s


def project_point_to_axis(point, anchor, axis):
    p = Vector(point)
    origin = Vector(anchor)
    a = axis_vector(axis).normalized()
    return origin + a * (p - origin).dot(a)


def set_world_axis_plane(tool, axis, anchor):
    """Set a stable world-aligned construction plane through *anchor*.

    X -> YZ plane, Y -> XZ plane, Z -> XY plane.  Returns (normal, u, v).
    """
    axis = str(axis).upper()
    normal = axis_vector(axis).normalized()

    # Choose a deterministic in-plane basis with familiar world directions.
    if axis == "X":
        u = Vector((0.0, 1.0, 0.0))
    elif axis == "Y":
        u = Vector((0.0, 0.0, 1.0))
    else:  # Z
        u = Vector((1.0, 0.0, 0.0))

    v = normal.cross(u).normalized()
    tool.drawing_plane_point = Vector(anchor).copy()
    tool.drawing_plane_normal = normal
    tool.axis_u = u
    tool.axis_v = v
    return normal, u, v
