import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .raycast import get_view_ray


def view_fallback_plane(context):
    """Return the SketchTools empty-space drawing plane.

    Perspective/User view always uses the flat world floor (XY, Z=0).
    Orthographic principal views use their natural world plane.
    """
    rv3d = context.region_data
    if rv3d is None:
        return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))

    if rv3d.view_perspective == 'PERSP':
        return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 1.0))

    view_direction = rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))
    ax, ay, az = abs(view_direction.x), abs(view_direction.y), abs(view_direction.z)

    if az >= ax and az >= ay:
        normal = Vector((0.0, 0.0, 1.0))       # Top / Bottom -> XY
    elif ay >= ax and ay >= az:
        normal = Vector((0.0, 1.0, 0.0))       # Front / Back -> XZ
    else:
        normal = Vector((1.0, 0.0, 0.0))       # Left / Right -> YZ

    return Vector((0.0, 0.0, 0.0)), normal


def object_or_floor_plane(context, event):
    """Use a hovered mesh face; otherwise use the shared world fallback plane."""
    ray_origin, ray_direction = get_view_ray(context, event)

    if context.mode == 'EDIT_MESH':
        obj = context.edit_object
        if obj is not None and obj.type == 'MESH':
            bm = bmesh.from_edit_mesh(obj.data)
            inv = obj.matrix_world.inverted()
            local_origin = inv @ ray_origin
            local_direction = (inv.to_3x3() @ ray_direction).normalized()
            tree = BVHTree.FromBMesh(bm)
            hit_location, hit_normal, face_index, distance = tree.ray_cast(
                local_origin, local_direction
            )
            if hit_location is not None and face_index is not None:
                world_point = obj.matrix_world @ hit_location
                bm.faces.ensure_lookup_table()
                face = bm.faces[face_index]
                world_normal = (obj.matrix_world.to_3x3() @ face.normal).normalized()
                return world_point, world_normal, obj, face_index

    depsgraph = context.evaluated_depsgraph_get()
    hit, location, normal, face_index, obj, matrix = context.scene.ray_cast(
        depsgraph, ray_origin, ray_direction
    )

    if hit and obj is not None and obj.type == 'MESH':
        return Vector(location), Vector(normal).normalized(), obj, face_index

    point, normal = view_fallback_plane(context)
    return point, normal, None, None


def project_to_plane(point, plane_point, plane_normal):
    point = Vector(point)
    plane_point = Vector(plane_point)
    plane_normal = Vector(plane_normal).normalized()
    delta = point - plane_point
    return point - plane_normal * delta.dot(plane_normal)
