"""Safe disconnected-geometry creation for SketchTools Edit Mode.

v123 keeps the v120 native-navigation architecture untouched and refines only
free-space Edit Mode face intersections.  Blender's Exact Intersect is still
used to calculate robust crossing vertices, but v123 snapshots the pre-cut
boundary graph and removes solver-generated helper diagonals that are not part
of any original/new polygon boundary.  This produces much cleaner coplanar
SketchUp-style overlap topology.
"""
import bpy
import bmesh
from mathutils import Vector


_POINT_TOL = 2.0e-5
_LINE_TOL = 4.0e-5


def _point_on_segment(p, a, b, tol=_LINE_TOL):
    """True when p lies on finite segment a-b within tolerance."""
    ab = b - a
    l2 = ab.length_squared
    if l2 <= tol * tol:
        return (p - a).length <= tol
    t = (p - a).dot(ab) / l2
    if t < -1.0e-5 or t > 1.0 + 1.0e-5:
        return False
    q = a + ab * max(0.0, min(1.0, t))
    return (p - q).length <= tol


def _edge_lies_on_source_segment(edge, source_segments):
    """Keep only edges that are subdivisions of a pre-intersection boundary."""
    p = edge.verts[0].co
    q = edge.verts[1].co
    for a, b in source_segments:
        if _point_on_segment(p, a, b) and _point_on_segment(q, a, b):
            return True
    return False


def _snapshot_source_segments(bm):
    """Snapshot all current mesh edges before Blender's Exact solver mutates them."""
    return [(e.verts[0].co.copy(), e.verts[1].co.copy()) for e in bm.edges if e.is_valid]


def _cleanup_solver_helper_edges(obj, source_segments):
    """Dissolve Exact-Intersect diagonals that are not part of input boundaries.

    For coplanar polygon overlap every meaningful resulting boundary edge is a
    sub-segment of an edge that existed before the operation (either old mesh
    boundary or the newly-created SketchTools face boundary).  Blender may add
    triangulation/helper diagonals to support the solve; those edges do not lie
    on any source segment and can be safely dissolved back into clean n-gons.
    """
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    helpers = []
    for edge in list(bm.edges):
        if not edge.is_valid:
            continue
        # Wire edges are intentional geometry and must never be dissolved here.
        if len(edge.link_faces) == 0:
            continue
        if not _edge_lies_on_source_segment(edge, source_segments):
            helpers.append(edge)

    if not helpers:
        return

    try:
        bmesh.ops.dissolve_edges(
            bm,
            edges=helpers,
            use_verts=False,
            use_face_split=False,
        )
    except (RuntimeError, ValueError):
        # Cleanup is deliberately non-destructive on failure.
        return

    bm.normal_update()
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)




def _cleanup_solver_helper_vertices(obj, source_segments, source_endpoints):
    """Dissolve only solver-created collinear degree-2 vertices.

    Genuine crossings are junctions (normally degree 3/4+) and are preserved.
    This keeps the original segment count wherever no topological split is
    mathematically required, while removing Exact-solver sampling debris.
    """
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return
    bm = bmesh.from_edit_mesh(obj.data)
    candidates = []
    for v in list(bm.verts):
        if not v.is_valid or len(v.link_edges) != 2:
            continue
        # Never dissolve a vertex that existed as an input boundary endpoint.
        if any((v.co - p).length <= _POINT_TOL for p in source_endpoints):
            continue
        e1, e2 = tuple(v.link_edges)
        a = e1.other_vert(v).co
        b = e2.other_vert(v).co
        va = a - v.co
        vb = b - v.co
        if va.length <= _LINE_TOL or vb.length <= _LINE_TOL:
            continue
        # Collinear and opposite directions only.
        if abs(va.normalized().dot(vb.normalized()) + 1.0) > 1e-4:
            continue
        # Both incident pieces must belong to the same original boundary line.
        same_source = False
        for s0, s1 in source_segments:
            if (_point_on_segment(a, s0, s1) and
                _point_on_segment(v.co, s0, s1) and
                _point_on_segment(b, s0, s1)):
                same_source = True
                break
        if same_source:
            candidates.append(v)
    if candidates:
        try:
            bmesh.ops.dissolve_verts(
                bm, verts=candidates, use_face_split=False, use_boundary_tear=False
            )
        except (RuntimeError, ValueError):
            pass
        bm.normal_update()
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)


def _run_selected_face_intersection(context):
    """Intersect selected new face(s) against existing faces, then clean topology.

    v123 deliberately keeps Blender's robust Exact solver for locating crossings,
    but removes solver-only diagonals afterward so coplanar overlaps retain only
    the actual polygon boundaries and their split intersection vertices.
    """
    obj = context.edit_object
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return False

    bm = bmesh.from_edit_mesh(obj.data)
    source_segments = _snapshot_source_segments(bm)
    source_endpoints = [p.copy() for seg in source_segments for p in seg]

    try:
        result = bpy.ops.mesh.intersect(
            mode='SELECT_UNSELECT',
            separate_mode='CUT',
            threshold=1e-6,
            solver='EXACT',
        )
        finished = 'FINISHED' in result
    except (RuntimeError, TypeError, ValueError):
        return False
    except Exception:
        return False

    if finished:
        _cleanup_solver_helper_edges(obj, source_segments)
        _cleanup_solver_helper_vertices(obj, source_segments, source_endpoints)
    return finished


def _select_only_face(bm, face):
    """Prepare SELECT_UNSELECT so only the newly-created face is selected."""
    for f in bm.faces:
        f.select = False
    for e in bm.edges:
        e.select = False
    for v in bm.verts:
        v.select = False

    if face is None or not face.is_valid:
        return False

    face.select = True
    for e in face.edges:
        e.select = True
    for v in face.verts:
        v.select = True
    return True


def intersect_face_at_points(context, world_points):
    """Find the Edit Mode face matching ``world_points`` and intersect it."""
    obj = context.edit_object
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return False

    pts = [Vector(p) for p in world_points]
    if len(pts) > 1 and (pts[0] - pts[-1]).length < 1e-7:
        pts.pop()
    if len(pts) < 3:
        return False

    inv = obj.matrix_world.inverted()
    local = [inv @ p for p in pts]
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    target = None
    tol = 2e-5
    for face in reversed(list(bm.faces)):
        if len(face.verts) != len(local):
            continue
        face_cos = [v.co for v in face.verts]
        if all(any((co - p).length <= tol for co in face_cos) for p in local):
            target = face
            break

    if target is None or not _select_only_face(bm, target):
        return False

    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
    return _run_selected_face_intersection(context)


def create_disconnected_face(context, world_points):
    obj = context.edit_object
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return False
    pts = [Vector(p) for p in world_points]
    if len(pts) > 1 and (pts[0] - pts[-1]).length < 1e-7:
        pts.pop()
    if len(pts) < 3:
        return False
    inv = obj.matrix_world.inverted()
    bm = bmesh.from_edit_mesh(obj.data)
    verts = [bm.verts.new(inv @ p) for p in pts]
    try:
        face = bm.faces.new(verts)
    except (ValueError, RuntimeError):
        valid = [v for v in verts if v.is_valid]
        if valid:
            bmesh.ops.delete(bm, geom=valid, context='VERTS')
        return False

    bm.normal_update()
    _select_only_face(bm, face)
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    _run_selected_face_intersection(context)
    return True


def create_disconnected_chain(context, world_points, closed=False):
    obj = context.edit_object
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return False
    pts = [Vector(p) for p in world_points]
    if len(pts) < 2:
        return False
    if len(pts) > 1 and (pts[0] - pts[-1]).length < 1e-7:
        pts.pop()
        closed = True
    if len(pts) < 2:
        return False
    inv = obj.matrix_world.inverted()
    bm = bmesh.from_edit_mesh(obj.data)
    verts = [bm.verts.new(inv @ p) for p in pts]
    pairs = list(zip(verts, verts[1:]))
    if closed and len(verts) >= 3:
        pairs.append((verts[-1], verts[0]))
    try:
        for a, b in pairs:
            if a != b:
                bm.edges.new((a, b))
    except (ValueError, RuntimeError):
        valid = [v for v in verts if v.is_valid]
        if valid:
            bmesh.ops.delete(bm, geom=valid, context='VERTS')
        return False
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=True)
    return True
