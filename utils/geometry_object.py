import bpy


OBJECT_NAME = "SketchTools Geometry"


def get_or_create_geometry_object():

    obj = bpy.data.objects.get(
        OBJECT_NAME
    )

    if obj is not None:
        return obj


    mesh = bpy.data.meshes.new(
        "SketchTools Mesh"
    )


    obj = bpy.data.objects.new(
        OBJECT_NAME,
        mesh
    )


    collection = bpy.context.collection

    collection.objects.link(
        obj
    )


    return obj