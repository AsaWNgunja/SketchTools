from ..utils.logging import debug_print
import bmesh

from mathutils import Vector
from mathutils.geometry import intersect_line_line


VERTEX_TOLERANCE = 1e-5
INTERSECTION_TOLERANCE = 1e-5


def find_or_create_vertex(bm, point):
    point = Vector(point)

    for vert in bm.verts:
        if (vert.co - point).length < VERTEX_TOLERANCE:
            return vert

    return bm.verts.new(point)


def _segment_parameter(point, a, b):
    ab = b - a
    denom = ab.length_squared
    if denom <= 1e-16:
        return 0.0
    return (point - a).dot(ab) / denom


def _segment_intersection_3d(a, b, c, d):
    """Return (point, t_new, t_edge) for a true finite 3D crossing."""
    result = intersect_line_line(a, b, c, d)
    if result is None:
        return None

    p_new, p_edge = result

    if (p_new - p_edge).length > INTERSECTION_TOLERANCE:
        return None

    point = (p_new + p_edge) * 0.5
    t_new = _segment_parameter(point, a, b)
    t_edge = _segment_parameter(point, c, d)

    eps = 1e-7
    if t_new < -eps or t_new > 1.0 + eps:
        return None
    if t_edge < -eps or t_edge > 1.0 + eps:
        return None

    return point, max(0.0, min(1.0, t_new)), max(0.0, min(1.0, t_edge))


def _vertex_at_edge_parameter(bm, edge, t, point):
    if t <= 1e-6:
        return edge.verts[0]
    if t >= 1.0 - 1e-6:
        return edge.verts[1]

    # edge_split preserves faces and all radial topology around the edge.
    # The factor is measured from the vertex passed as the second argument.
    _new_edge, new_vert = bmesh.utils.edge_split(
        edge,
        edge.verts[0],
        t,
    )
    new_vert.co = Vector(point)
    return new_vert


def _autocut_edge_bmesh(bm, p1, p2):
    """Create a segment and split every truly intersected existing edge."""
    p1 = Vector(p1)
    p2 = Vector(p2)

    if (p1 - p2).length < 1e-6:
        return False

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Snapshot edges before mutation.  Newly created split edges must not be
    # reconsidered during this same insertion.
    existing_edges = list(bm.edges)
    hits = []

    for edge in existing_edges:
        if not edge.is_valid:
            continue

        a = edge.verts[0].co.copy()
        b = edge.verts[1].co.copy()
        hit = _segment_intersection_3d(p1, p2, a, b)
        if hit is None:
            continue

        point, t_new, t_edge = hit
        hits.append((t_new, edge, t_edge, point))

    # Split existing edges first.  Deduplicate crossings at shared vertices.
    cut_vertices = []

    for t_new, edge, t_edge, point in sorted(hits, key=lambda item: item[0]):
        # A previous split at the same topological location may already have
        # produced the required vertex.  Prefer it before touching the edge.
        vert = None
        for candidate in bm.verts:
            if (candidate.co - point).length < VERTEX_TOLERANCE:
                vert = candidate
                break

        if vert is None and edge.is_valid:
            vert = _vertex_at_edge_parameter(bm, edge, t_edge, point)

        if vert is not None:
            cut_vertices.append((t_new, vert))

    start_vert = find_or_create_vertex(bm, p1)
    end_vert = find_or_create_vertex(bm, p2)

    chain = [(0.0, start_vert)] + cut_vertices + [(1.0, end_vert)]
    chain.sort(key=lambda item: item[0])

    # Remove duplicate vertices/parameters from shared endpoints or multiple
    # edges meeting at the same crossing.
    unique = []
    for t, vert in chain:
        if unique and (
            vert == unique[-1][1]
            or (vert.co - unique[-1][1].co).length < VERTEX_TOLERANCE
        ):
            continue
        unique.append((t, vert))

    created = False
    for (_, v1), (_, v2) in zip(unique, unique[1:]):
        if v1 == v2:
            continue

        # If the two vertices lie on the boundary of the same face, this
        # segment must become part of that face's topology.  A plain
        # bm.edges.new() only lays a wire edge over the polygon and leaves the
        # face unsplit.  connect_vert_pair performs Blender's topological
        # connection and splits the shared face(s), which is the SketchUp-like
        # autocut behaviour we want.
        existing = bm.edges.get((v1, v2))
        if existing is not None:
            continue

        shared_faces = set(v1.link_faces).intersection(v2.link_faces)
        if shared_faces:
            try:
                result = bmesh.ops.connect_vert_pair(
                    bm,
                    verts=[v1, v2],
                )
                if result.get("edges"):
                    created = True
                    continue
            except (RuntimeError, ValueError):
                # Fall through to a wire edge only when Blender cannot make a
                # valid face split (for example non-manifold/degenerate input).
                pass

        # No shared face: this is either a free-space Line segment or a
        # segment travelling between already-separated surface regions.
        # Preserve the previous behaviour in that case.
        if bm.edges.get((v1, v2)) is None:
            bm.edges.new((v1, v2))
            created = True

    return created


def create_edge(obj, p1, p2):
    mesh = obj.data
    inv_matrix = obj.matrix_world.inverted()
    p1 = inv_matrix @ Vector(p1)
    p2 = inv_matrix @ Vector(p2)

    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(mesh)
        created = _autocut_edge_bmesh(bm, p1, p2)
        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        debug_print("Autocut edge created in Edit Mode" if created else "Edge already exists")
        return created

    bm = bmesh.new()
    bm.from_mesh(mesh)
    created = _autocut_edge_bmesh(bm, p1, p2)
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    obj.update_tag()
    debug_print("Autocut edge created in Object Mode" if created else "Edge already exists")
    return created


def _create_face_bmesh(bm, local_points):
    verts = []

    for point in local_points:
        vert = None
        for existing_vert in bm.verts:
            if (existing_vert.co - point).length < VERTEX_TOLERANCE:
                vert = existing_vert
                break
        if vert is None:
            debug_print("Face vertex not found:", point)
            return False
        verts.append(vert)

    if len(verts) > 1 and verts[0] == verts[-1]:
        verts.pop()

    if len(verts) < 3:
        return False

    for i in range(len(verts)):
        v1 = verts[i]
        v2 = verts[(i + 1) % len(verts)]
        if v1 != v2 and bm.edges.get((v1, v2)) is None:
            bm.edges.new((v1, v2))

    try:
        bm.faces.new(verts)
    except ValueError:
        debug_print("Face already exists or invalid polygon")
        return False

    return True


def create_face_from_points(obj, points):
    if len(points) < 3:
        debug_print("Not enough points for face")
        return False

    mesh = obj.data
    inv_matrix = obj.matrix_world.inverted()
    local_points = [inv_matrix @ Vector(point) for point in points]

    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(mesh)
        result = _create_face_bmesh(bm, local_points)
        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        if result:
            debug_print("Face Created in Edit Mode")
        return result

    bm = bmesh.new()
    bm.from_mesh(mesh)
    result = _create_face_bmesh(bm, local_points)
    if result:
        bm.to_mesh(mesh)
        mesh.update()
        obj.update_tag()
        debug_print("Face Created")
    bm.free()
    return result
