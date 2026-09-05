from mathutils import Vector
from bpy_extras import view3d_utils


def get_view_ray(
    context,
    event,
):
    region = context.region
    rv3d = context.region_data

    coord = (
        event.mouse_region_x,
        event.mouse_region_y,
    )

    origin = view3d_utils.region_2d_to_origin_3d(
        region,
        rv3d,
        coord,
    )

    direction = view3d_utils.region_2d_to_vector_3d(
        region,
        rv3d,
        coord,
    )

    return origin, direction


def ray_plane_intersection(
    ray_origin,
    ray_direction,
    plane_point,
    plane_normal,
):
    plane_normal = Vector(
        plane_normal
    ).normalized()

    denominator = ray_direction.dot(
        plane_normal
    )

    if abs(denominator) < 1e-8:
        return None

    distance = (
        Vector(plane_point) - ray_origin
    ).dot(
        plane_normal
    ) / denominator

    if distance < 0:
        return None

    return (
        ray_origin +
        ray_direction * distance
    )


def mouse_to_plane(
    context,
    event,
    plane_point,
    plane_normal,
):
    ray_origin, ray_direction = get_view_ray(
        context,
        event
    )

    return ray_plane_intersection(
        ray_origin,
        ray_direction,
        plane_point,
        plane_normal
    )