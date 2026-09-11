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



def _component_wire_edges(seed_edge, wire_set):
    """Return one connected component of the current wire-edge graph."""
    stack = [seed_edge]
    component = set()
    while stack:
        edge = stack.pop()
        if edge in component or edge not in wire_set or not edge.is_valid:
            continue
        component.add(edge)
        for vert in edge.verts:
            for linked in vert.link_edges:
                if linked in wire_set and linked not in component:
                    stack.append(linked)
    return component


def _wire_component_basis(edges):
    """Return (origin, x_axis, y_axis) when a wire component is planar.

    v130 deliberately refuses to invent a face for a non-planar network.  A
    projected camera crossing must never become a twisted polygon merely
    because it looks closed from the current view.
    """
    verts = list({v for edge in edges for v in edge.verts if v.is_valid})
    if len(verts) < 3:
        return None

    origin = verts[0].co.copy()

    # Stable X direction: use the longest available component edge.
    longest = None
    longest_len = 0.0
    for edge in edges:
        vec = edge.verts[1].co - edge.verts[0].co
        if vec.length_squared > longest_len:
            longest = vec.copy()
            longest_len = vec.length_squared
    if longest is None or longest_len <= 1e-16:
        return None
    x_axis = longest.normalized()

    normal = None
    for vert in verts[1:]:
        vec = vert.co - origin
        candidate = x_axis.cross(vec)
        if candidate.length > 1e-7:
            normal = candidate.normalized()
            break
    if normal is None:
        return None

    # Reject components that are not truly coplanar.
    plane_tol = 2e-5
    for vert in verts:
        if abs((vert.co - origin).dot(normal)) > plane_tol:
            return None

    y_axis = normal.cross(x_axis)
    if y_axis.length <= 1e-12:
        return None
    y_axis.normalize()
    return origin, x_axis, y_axis


def _project_wire_vertex(vert, basis):
    origin, x_axis, y_axis = basis
    delta = vert.co - origin
    return (delta.dot(x_axis), delta.dot(y_axis))


def _signed_cycle_area(cycle, projected):
    area2 = 0.0
    for i, vert in enumerate(cycle):
        x1, y1 = projected[vert]
        x2, y2 = projected[cycle[(i + 1) % len(cycle)]]
        area2 += x1 * y2 - x2 * y1
    return 0.5 * area2


def _minimal_bounded_wire_cycles(edges):
    """Extract the minimal bounded cells of a planar wire graph.

    Each undirected mesh edge is treated as two directed half-edges.  At every
    vertex the outgoing half-edges are angle-sorted in the component plane.
    Walking the immediately-clockwise edge from the incoming reverse direction
    keeps the current cell on the left.  That visits each bounded planar cell
    exactly once and visits the unbounded exterior with the opposite winding.

    This is the key v130 difference from edgenet_fill: large/composite loops
    are not candidates at all, so a figure-8 can yield its two lobes without an
    external bow-tie/spanning face.
    """
    import math

    basis = _wire_component_basis(edges)
    if basis is None:
        return []

    verts = {v for edge in edges for v in edge.verts if v.is_valid}
    projected = {v: _project_wire_vertex(v, basis) for v in verts}

    adjacency = {v: set() for v in verts}
    for edge in edges:
        if not edge.is_valid:
            continue
        a, b = edge.verts
        adjacency[a].add(b)
        adjacency[b].add(a)

    ordered = {}
    for vert, neighbors in adjacency.items():
        vx, vy = projected[vert]
        ordered[vert] = sorted(
            neighbors,
            key=lambda n: math.atan2(projected[n][1] - vy, projected[n][0] - vx),
        )

    directed = []
    for edge in edges:
        a, b = edge.verts
        directed.append((a, b))
        directed.append((b, a))

    visited = set()
    cycles = []
    max_steps = max(16, len(directed) + 4)

    for start in directed:
        if start in visited:
            continue

        u, v = start
        cycle = [u]
        local_seen = set()
        closed = False

        for _ in range(max_steps):
            half = (u, v)
            if half in local_seen:
                break
            local_seen.add(half)
            visited.add(half)
            cycle.append(v)

            neighbors = ordered.get(v, [])
            if len(neighbors) < 2:
                break
            try:
                reverse_index = neighbors.index(u)
            except ValueError:
                break

            # Previous item in CCW order = immediately clockwise from the
            # incoming reverse direction, keeping the traversed cell on left.
            w = neighbors[(reverse_index - 1) % len(neighbors)]
            u, v = v, w
            if (u, v) == start:
                closed = True
                break

        if not closed or len(cycle) < 4:
            continue

        # Last vertex repeats the first after a successful closure.
        if cycle[-1] == cycle[0]:
            cycle.pop()
        if len(cycle) < 3:
            continue

        # Bridge/exterior walks can revisit a vertex.  They are not a bounded
        # mesh cell and must never become a face.
        if len(set(cycle)) != len(cycle):
            continue

        area = _signed_cycle_area(cycle, projected)
        if area <= 1e-10:
            # With the half-edge rule above, bounded cells are CCW/positive;
            # the unbounded exterior is CW/negative.
            continue

        cycles.append(cycle)

    return cycles


def _point_on_segment_2d(point, a, b, tol=1e-8):
    px, py = point
    ax, ay = a
    bx, by = b
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    cross = abs(abx * apy - aby * apx)
    scale = max(1.0, abs(abx) + abs(aby))
    if cross > tol * scale:
        return False
    dot = apx * abx + apy * aby
    if dot < -tol:
        return False
    sq = abx * abx + aby * aby
    if dot > sq + tol:
        return False
    return True


def _point_in_cycle_2d(point, cycle, projected, strict=True):
    """Return True when a projected point lies inside a simple cycle.

    In strict mode, points on the boundary are rejected.  That distinction is
    useful for v131: a neighbouring face sharing an edge with the candidate
    must not cause the candidate to be rejected, while a face genuinely
    enclosed by a larger/composite loop must.
    """
    poly = [projected[v] for v in cycle]
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        if _point_on_segment_2d(point, a, b):
            return not strict

    x, y = point
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            denom = (yj - yi)
            if abs(denom) > 1e-20:
                x_cross = (xj - xi) * (y - yi) / denom + xi
                if x < x_cross:
                    inside = not inside
        j = i
    return inside


def _cycle_encloses_existing_face(bm, cycle, basis):
    """Reject Edit-Mode composite/exterior cells that wrap existing faces.

    v130's half-edge walk used only *wire* edges.  Once a live bounded region
    became a face, its boundary edges stopped being wire edges and disappeared
    from that graph.  A later segment could therefore make the reduced wire
    graph see a larger outside/composite loop as a fresh cell.

    v131 preserves live face creation but filters such candidates: if the
    candidate strictly contains the centre of an already-existing coplanar
    face, it is not a minimal missing cell and must not be created.
    """
    origin, x_axis, y_axis = basis
    projected = {v: _project_wire_vertex(v, basis) for v in cycle}

    # Candidate normal for coplanarity tests.
    normal = x_axis.cross(y_axis)
    if normal.length <= 1e-12:
        return False
    normal.normalize()

    candidate_verts = set(cycle)
    for face in bm.faces:
        if not face.is_valid or len(face.verts) < 3:
            continue

        # An exact existing face with the same boundary will be rejected by
        # bm.faces.new anyway; skip it here so only *enclosed* prior regions
        # drive this v131 filter.
        if set(face.verts) == candidate_verts:
            continue

        centre = sum((v.co for v in face.verts), Vector((0.0, 0.0, 0.0))) / len(face.verts)
        if abs((centre - origin).dot(normal)) > 2e-5:
            continue

        d = centre - origin
        centre2 = (d.dot(x_axis), d.dot(y_axis))
        if _point_in_cycle_2d(centre2, cycle, projected, strict=True):
            return True

    return False



def _edge_component_all(seed_edge):
    """Return the full connected edge component containing *seed_edge*.

    v132 Edit Mode uses the complete planar graph, including boundaries that
    already belong to faces.  This is crucial: once a live cell becomes a
    face, its boundary must remain in the graph so a later Line cannot see a
    larger composite/exterior loop around it.
    """
    if seed_edge is None or not seed_edge.is_valid:
        return set()
    stack = [seed_edge]
    component = set()
    while stack:
        edge = stack.pop()
        if edge in component or not edge.is_valid:
            continue
        component.add(edge)
        for vert in edge.verts:
            for linked in vert.link_edges:
                if linked.is_valid and linked not in component:
                    stack.append(linked)
    return component


def _edges_on_segment(bm, p1, p2, tol=2e-5):
    """Return current mesh edges that are actual pieces of segment p1->p2."""
    p1 = Vector(p1)
    p2 = Vector(p2)
    direction = p2 - p1
    length2 = direction.length_squared
    if length2 <= 1e-16:
        return []

    result = []
    for edge in bm.edges:
        if not edge.is_valid:
            continue
        ok = True
        params = []
        for vert in edge.verts:
            rel = vert.co - p1
            t = rel.dot(direction) / length2
            closest = p1 + direction * t
            if (vert.co - closest).length > tol or t < -1e-6 or t > 1.0 + 1e-6:
                ok = False
                break
            params.append(t)
        if ok and abs(params[1] - params[0]) > 1e-8:
            result.append(edge)
    return result


def _adjacent_bounded_cycles(component, seed_edges):
    """Return true planar cells adjacent to the latest segment (v134).

    v134 uses a directed half-edge embedding for the *entire* connected
    component.  Every directed mesh edge is walked exactly as a planar face
    boundary by taking the immediately-clockwise neighbor at each vertex.
    The unbounded/exterior walk is identified globally for the component and
    removed before we select cells touching the newest segment.  This avoids
    treating a locally closed but exterior/composite walk as a live face.
    """
    import math

    if not component or not seed_edges:
        return []
    basis = _wire_component_basis(component)
    if basis is None:
        return []

    verts = {v for edge in component for v in edge.verts if v.is_valid}
    projected = {v: _project_wire_vertex(v, basis) for v in verts}
    adjacency = {v: set() for v in verts}
    for edge in component:
        if not edge.is_valid:
            continue
        a, b = edge.verts
        adjacency[a].add(b)
        adjacency[b].add(a)

    ordered = {}
    for vert, neighbors in adjacency.items():
        vx, vy = projected[vert]
        ordered[vert] = sorted(
            neighbors,
            key=lambda n: math.atan2(projected[n][1] - vy, projected[n][0] - vx),
        )

    def walk(start):
        u, v = start
        cycle = [u]
        visited = set()
        max_steps = max(16, len(component) * 2 + 8)
        for _ in range(max_steps):
            half = (u, v)
            if half in visited:
                return None
            visited.add(half)
            cycle.append(v)
            neighbors = ordered.get(v, [])
            if len(neighbors) < 2:
                return None
            try:
                reverse_index = neighbors.index(u)
            except ValueError:
                return None
            # Clockwise from the reverse/incoming direction: one consistent
            # face side for every directed half-edge in this embedding.
            w = neighbors[(reverse_index - 1) % len(neighbors)]
            u, v = v, w
            if (u, v) == start:
                if cycle[-1] == cycle[0]:
                    cycle.pop()
                if len(cycle) < 3 or len(set(cycle)) != len(cycle):
                    return None
                return cycle, visited
        return None

    # Enumerate the actual faces of the full planar embedding, not arbitrary
    # loops reachable from only the newest edge.
    unvisited = set()
    for edge in component:
        if edge.is_valid:
            a, b = edge.verts
            unvisited.add((a, b))
            unvisited.add((b, a))

    walks = []
    while unvisited:
        start = next(iter(unvisited))
        result = walk(start)
        if result is None:
            unvisited.discard(start)
            continue
        cycle, used = result
        unvisited.difference_update(used)
        area = _signed_cycle_area(cycle, projected)
        if abs(area) <= 1e-10:
            continue
        boundary = frozenset(
            frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))
            for i in range(len(cycle))
        )
        walks.append({"cycle": cycle, "area": area, "boundary": boundary})

    if not walks:
        return []

    # With a consistent half-edge turn rule the unbounded face is the walk
    # whose absolute signed area is greatest.  In the common one-cell tie the
    # exterior is the negative-winding copy.  This decision is made globally,
    # before latest-segment filtering, which is the key v134 change.
    max_abs = max(abs(item["area"]) for item in walks)
    exterior_candidates = [
        item for item in walks
        if abs(abs(item["area"]) - max_abs) <= max(1e-10, max_abs * 1e-9)
    ]
    exterior = min(exterior_candidates, key=lambda item: item["area"])

    seed_pairs = set()
    for edge in seed_edges:
        if edge in component and edge.is_valid:
            a, b = edge.verts
            seed_pairs.add(frozenset((a, b)))

    cycles = []
    seen = set()
    for item in walks:
        if item is exterior:
            continue
        if not (item["boundary"] & seed_pairs):
            continue
        # Bounded cells should have the opposite winding to the exterior.
        # Keep this as a secondary guard rather than relying on sign alone.
        if item["area"] * exterior["area"] >= 0.0:
            continue
        if item["boundary"] in seen:
            continue
        seen.add(item["boundary"])
        cycles.append((item["cycle"], basis))

    return cycles

def _cycle_contains_internal_graph_geometry(component, cycle, basis):
    """Reject a candidate that is not a minimal planar cell.

    v133 Edit Mode rule: a face candidate may contain only its own boundary.
    If any other vertex of the connected planar graph lies strictly inside the
    candidate, or if any non-boundary graph edge has a midpoint strictly inside
    it, then the walk is a composite/exterior region rather than the smallest
    bounded cell adjacent to the latest Line segment.

    This is intentionally Edit-Mode-only. Object Mode keeps v130 behavior.
    """
    if not component or len(cycle) < 3:
        return True

    projected = {
        v: _project_wire_vertex(v, basis)
        for edge in component if edge.is_valid
        for v in edge.verts if v.is_valid
    }
    cycle_set = set(cycle)

    # Any graph vertex strictly inside means the candidate encloses a finer
    # subdivision and therefore cannot itself be a minimal planar cell.
    for vert, point2 in projected.items():
        if vert in cycle_set:
            continue
        if _point_in_cycle_2d(point2, cycle, projected, strict=True):
            return True

    boundary_edges = {
        frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))
        for i in range(len(cycle))
    }

    # A chord/internal edge can subdivide a candidate even when both of its
    # endpoints happen to be boundary vertices, so also test edge midpoints.
    for edge in component:
        if not edge.is_valid:
            continue
        a, b = edge.verts
        if frozenset((a, b)) in boundary_edges:
            continue
        pa = projected.get(a)
        pb = projected.get(b)
        if pa is None or pb is None:
            continue
        midpoint = ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5)
        if _point_in_cycle_2d(midpoint, cycle, projected, strict=True):
            return True

    return False


def _cycle_has_existing_face_inside(bm, cycle, basis):
    """Return True when an already-existing coplanar face occupies the cell.

    This is a safety guard for Edit Mode.  With the full planar graph the walk
    should already be minimal; this check prevents a duplicate/composite face
    if Blender's existing topology represents the same region with a different
    face boundary.
    """
    origin, x_axis, y_axis = basis
    projected = {v: _project_wire_vertex(v, basis) for v in cycle}
    normal = x_axis.cross(y_axis)
    if normal.length <= 1e-12:
        return True
    normal.normalize()
    candidate_verts = set(cycle)

    for face in bm.faces:
        if not face.is_valid or len(face.verts) < 3:
            continue
        if set(face.verts) == candidate_verts:
            return True
        centre = sum((v.co for v in face.verts), Vector((0.0, 0.0, 0.0))) / len(face.verts)
        if abs((centre - origin).dot(normal)) > 2e-5:
            continue
        d = centre - origin
        centre2 = (d.dot(x_axis), d.dot(y_axis))
        if _point_in_cycle_2d(centre2, cycle, projected, strict=True):
            return True
    return False


def _autofill_cells_adjacent_to_new_segment(bm, segment_edges):
    """v133 Edit Mode live fill: create only minimal latest-segment cells.

    v132 limited walks to the two cells adjacent to each split piece of the
    newest Line. v133 adds a strict minimal-cell guard: a candidate is rejected
    if any other graph vertex or non-boundary edge lies inside it. This keeps
    live SketchUp-like face creation while preventing exterior/composite faces.
    """
    segment_edges = [e for e in segment_edges if e is not None and e.is_valid]
    if not segment_edges:
        return 0

    created = 0
    handled_components = set()
    for seed in segment_edges:
        if not seed.is_valid:
            continue
        component = _edge_component_all(seed)
        if len(component) < 3:
            continue
        component_key = frozenset(component)
        if component_key in handled_components:
            continue
        handled_components.add(component_key)

        local_seeds = [e for e in segment_edges if e in component and e.is_valid]
        for cycle, basis in _adjacent_bounded_cycles(component, local_seeds):
            # v133: only the smallest bounded cell adjacent to the latest
            # segment may become a face.  A larger/composite walk is rejected
            # whenever it encloses any other graph vertex or internal chord.
            if _cycle_contains_internal_graph_geometry(component, cycle, basis):
                debug_print(
                    "Line Auto-Face rejected non-minimal Edit Mode cell",
                    len(cycle), "verts"
                )
                continue
            if _cycle_has_existing_face_inside(bm, cycle, basis):
                continue
            try:
                face = bm.faces.new(cycle)
            except (ValueError, RuntimeError):
                continue
            if face is not None:
                created += 1

    if created:
        bm.normal_update()
    return created

def _autofill_closed_wire_regions(bm, reject_enclosing_existing_faces=False):
    """Live SketchUp-like fill using only minimal bounded planar cells.

    Faces still appear immediately after the segment that closes a region, as
    in v127.  Unlike v127/v129, no general-purpose loop solver is allowed to
    invent bridge edges or composite/external cycles.

    v131 adds an Edit-Mode-only exterior/composite guard.  Once an earlier
    live cell has become a face, its boundary is no longer part of the wire
    graph.  A later wire loop is therefore rejected when it strictly encloses
    an already-existing coplanar face.
    """
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    wire_edges = {e for e in bm.edges if e.is_valid and len(e.link_faces) == 0}
    if len(wire_edges) < 3:
        return 0

    components = []
    remaining = set(wire_edges)
    while remaining:
        seed = next(iter(remaining))
        component = _component_wire_edges(seed, wire_edges)
        components.append(component)
        remaining.difference_update(component)

    created = 0
    for component in components:
        if len(component) < 3:
            continue
        basis = _wire_component_basis(component)
        if basis is None:
            continue
        for cycle in _minimal_bounded_wire_cycles(component):
            # v131: in Edit Mode, never create a newly discovered loop that
            # wraps a face which already exists on this same plane.  Such a
            # loop is necessarily exterior/composite relative to the live
            # planar subdivision, not a new minimal cell.
            if reject_enclosing_existing_faces and _cycle_encloses_existing_face(
                bm, cycle, basis
            ):
                debug_print(
                    "Line Auto-Face rejected exterior/composite Edit Mode loop",
                    len(cycle), "verts"
                )
                continue

            # Every boundary edge must still be a wire edge.  A prior cycle
            # created during this pass may have claimed an edge; a second face
            # on the other side is valid only when Blender accepts it.
            try:
                face = bm.faces.new(cycle)
            except (ValueError, RuntimeError):
                continue
            if face is not None:
                created += 1

    if created:
        bm.normal_update()
    return created


def _remove_faces_affected_by_latest_segment(bm, segment_edges):
    """Remove existing Edit-Mode faces locally cut/touched by newest Line.

    v135 retopologizes the affected planar patch instead of layering new cells
    over faces that were created by earlier live Line steps.  Boundary edges
    and vertices are preserved; only the affected faces are removed, allowing
    the half-edge cell walker to rebuild the local subdivision cleanly.
    """
    segment_edges = [e for e in segment_edges if e is not None and e.is_valid]
    if not segment_edges:
        return 0
    affected = set()
    for edge in segment_edges:
        for face in edge.link_faces:
            if face.is_valid:
                affected.add(face)
    if not affected:
        return 0
    # Remove faces only.  Keep their boundary network so the planar-cell walk
    # can reconstruct the correct minimal cells around the newest segment.
    bmesh.ops.delete(bm, geom=list(affected), context='FACES_ONLY')
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return len(affected)


def _derive_planar_seed_basis(seed_edges):
    """Derive a stable drawing plane for an Edit-Mode seed segment.

    Prefer an existing linked face normal (best when drawing on a surface).
    For free-space drawing, use a second connected non-collinear edge.  This
    lets Edit Mode rebuild a clean planar graph without importing unrelated
    non-coplanar mesh topology from the rest of the object.
    """
    seed_edges = [e for e in seed_edges if e is not None and e.is_valid]
    if not seed_edges:
        return None

    seed = seed_edges[0]
    origin = seed.verts[0].co.copy()
    x = seed.verts[1].co - seed.verts[0].co
    if x.length <= 1e-12:
        return None
    x.normalize()

    # Surface drawing: Blender already knows the intended plane.
    for edge in seed_edges:
        for face in edge.link_faces:
            if face.is_valid and face.normal.length > 1e-12:
                normal = face.normal.normalized()
                # Guard against a pathological face normal parallel to seed.
                if abs(normal.dot(x)) < 0.999999:
                    y = normal.cross(x)
                    if y.length > 1e-12:
                        y.normalize()
                        return origin, x, y

    # Free-space drawing: find a connected non-collinear direction.
    seen_edges = set(seed_edges)
    stack = list(seed_edges)
    while stack:
        edge = stack.pop()
        for vert in edge.verts:
            for linked in vert.link_edges:
                if not linked.is_valid or linked in seen_edges:
                    continue
                seen_edges.add(linked)
                stack.append(linked)
                vec = linked.verts[1].co - linked.verts[0].co
                if vec.length <= 1e-12:
                    continue
                normal = x.cross(vec)
                if normal.length > 1e-7:
                    normal.normalize()
                    y = normal.cross(x)
                    if y.length > 1e-12:
                        y.normalize()
                        return origin, x, y
    return None


def _planar_component_from_seed_edges(seed_edges, basis, plane_tol=2e-5):
    """Return only connected edges lying on the seed drawing plane."""
    if basis is None:
        return set()
    origin, x_axis, y_axis = basis
    normal = x_axis.cross(y_axis)
    if normal.length <= 1e-12:
        return set()
    normal.normalize()

    def edge_on_plane(edge):
        if edge is None or not edge.is_valid:
            return False
        return all(abs((v.co - origin).dot(normal)) <= plane_tol for v in edge.verts)

    stack = [e for e in seed_edges if edge_on_plane(e)]
    component = set()
    while stack:
        edge = stack.pop()
        if edge in component or not edge_on_plane(edge):
            continue
        component.add(edge)
        for vert in edge.verts:
            for linked in vert.link_edges:
                if linked not in component and edge_on_plane(linked):
                    stack.append(linked)
    return component


def _all_bounded_planar_cells(component, basis):
    """Enumerate true bounded cells of a connected planar edge graph.

    v137 keeps *all* directed half-edge walks long enough to identify the
    unbounded exterior, including walks that revisit a bridge vertex.  Earlier
    builds discarded repeated-vertex walks too early; on graphs with open
    tails/bridges that can throw away the real exterior and misclassify a
    larger composite region as a fillable face.
    """
    import math

    if not component or basis is None:
        return []
    verts = {v for e in component for v in e.verts if v.is_valid}
    if len(verts) < 3:
        return []
    projected = {v: _project_wire_vertex(v, basis) for v in verts}
    adjacency = {v: set() for v in verts}
    for edge in component:
        if not edge.is_valid:
            continue
        a, b = edge.verts
        adjacency[a].add(b)
        adjacency[b].add(a)

    ordered = {}
    for vert, neighbors in adjacency.items():
        vx, vy = projected[vert]
        ordered[vert] = sorted(
            neighbors,
            key=lambda n: math.atan2(projected[n][1] - vy, projected[n][0] - vx),
        )

    unvisited = set()
    for edge in component:
        if edge.is_valid:
            a, b = edge.verts
            unvisited.add((a, b))
            unvisited.add((b, a))

    walks = []
    max_steps = max(32, len(unvisited) * 2 + 8)
    while unvisited:
        start = next(iter(unvisited))
        u, v = start
        cycle = [u]
        used = set()
        closed = False
        for _ in range(max_steps):
            half = (u, v)
            if half in used:
                break
            used.add(half)
            cycle.append(v)
            neighbors = ordered.get(v, [])
            if not neighbors:
                break
            try:
                reverse_index = neighbors.index(u)
            except ValueError:
                break
            # Walk the face on a consistent side of every directed edge.
            # Degree-1 bridge vertices simply send us back along the bridge;
            # that is valid for the exterior walk and must not be discarded.
            w = neighbors[(reverse_index - 1) % len(neighbors)]
            u, v = v, w
            if (u, v) == start:
                closed = True
                break
        unvisited.difference_update(used)
        if not closed:
            continue
        if cycle[-1] == cycle[0]:
            cycle.pop()
        if len(cycle) < 3:
            continue
        # Signed area remains meaningful for the exterior walk even when a
        # bridge causes a vertex to be visited more than once.
        area = _signed_cycle_area(cycle, projected)
        if abs(area) <= 1e-10:
            continue
        walks.append((cycle, area))

    if not walks:
        return []

    # With the turn rule above, bounded cells share one winding and the
    # unbounded walk has the opposite winding.  Determine exterior from the
    # most negative walk when available; otherwise use the largest-magnitude
    # walk as a robust fallback for mirrored bases.
    neg = [w for w in walks if w[1] < -1e-10]
    pos = [w for w in walks if w[1] > 1e-10]
    if neg and pos:
        # Usually the exterior is the negative-winding walk for this rule.
        exterior = min(neg, key=lambda item: item[1])
        bounded_sign = 1.0
        # If the negative set looks like several ordinary cells while one
        # positive walk dominates, the drawing basis is mirrored: flip.
        max_neg = max(abs(w[1]) for w in neg)
        max_pos = max(abs(w[1]) for w in pos)
        if max_pos > max_neg * 1.000001:
            exterior = max(pos, key=lambda item: item[1])
            bounded_sign = -1.0
    else:
        exterior = max(walks, key=lambda item: abs(item[1]))
        bounded_sign = -1.0 if exterior[1] > 0 else 1.0

    result = []
    seen = set()
    for cycle, area in walks:
        if (cycle, area) == exterior:
            continue
        if area * bounded_sign <= 1e-10:
            continue
        # A genuine mesh face must be a simple cell.  Repeated vertices are
        # allowed only in the exterior/bridge walk, never in a bounded cell.
        if len(set(cycle)) != len(cycle):
            continue
        boundary = frozenset(
            frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))
            for i in range(len(cycle))
        )
        if boundary in seen:
            continue
        seen.add(boundary)
        result.append(cycle)
    return result

def _retopologize_edit_planar_region(bm, segment_edges):
    """v137: rebuild the affected Edit-Mode plane as a clean graph.

    Rather than preserving historical faces and trying to patch them after each
    line, temporarily reduce the coplanar affected region to its edge graph,
    then recreate only the true bounded cells.  This mirrors the successful
    Object-Mode workflow while keeping every result inside the same edit object.
    """
    segment_edges = [e for e in segment_edges if e is not None and e.is_valid]
    if not segment_edges:
        return 0

    basis = _derive_planar_seed_basis(segment_edges)
    if basis is None:
        # An open free-space chain may not define a plane yet; no face is due.
        return 0
    component = _planar_component_from_seed_edges(segment_edges, basis)
    if len(component) < 3:
        return 0

    # Remove every face whose complete boundary belongs to this planar graph.
    # Keep edges/verts: they are the authoritative SketchTools drawing graph.
    component_edges = set(component)
    faces_to_remove = []
    for face in list(bm.faces):
        if not face.is_valid:
            continue
        if face.edges and all(edge in component_edges for edge in face.edges):
            faces_to_remove.append(face)
    if faces_to_remove:
        bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES_ONLY')
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

    created = 0
    for cycle in _all_bounded_planar_cells(component, basis):
        try:
            face = bm.faces.new(cycle)
        except (ValueError, RuntimeError):
            continue
        if face is not None:
            created += 1
    if created:
        bm.normal_update()
    return created




def _retopologize_edit_planar_region_local(bm, segment_edges, source_segments):
    """v151 local Rectangle retopology.

    Rebuild only the new boundary plus edges that belonged to faces/segments
    directly touched by the new rectangle before mutation.  This prevents a
    late rectangle from walking through a connected planar network and
    rebuilding unrelated distant geometry.
    """
    segment_edges = [e for e in segment_edges if e is not None and e.is_valid]
    if not segment_edges:
        return 0
    basis = _derive_planar_seed_basis(segment_edges)
    if basis is None:
        return 0

    allowed = set(segment_edges)
    # Reacquire split pieces of the pre-transaction edges belonging to the
    # directly affected local faces/segments.  No BMesh references are kept.
    for a, b in source_segments:
        for e in _edges_on_segment(bm, a, b):
            if e is not None and e.is_valid:
                allowed.add(e)
    if len(allowed) < 3:
        return 0

    # Delete only faces fully bounded by this local graph.  Unrelated faces in
    # the same coplanar connected component are deliberately untouched.
    faces_to_remove = []
    for face in list(bm.faces):
        if face.is_valid and face.edges and all(e in allowed for e in face.edges):
            faces_to_remove.append(face)
    if faces_to_remove:
        bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES_ONLY')
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table(); bm.faces.ensure_lookup_table()

    created = 0
    for cycle in _all_bounded_planar_cells(list(allowed), basis):
        if len(cycle) < 3 or any(v is None or not v.is_valid for v in cycle):
            continue
        try:
            face = bm.faces.new(cycle)
        except (ValueError, RuntimeError):
            continue
        if face is not None:
            created += 1
    if created:
        bm.normal_update()
    return created


def _complete_closed_boundary_faces_local(bm, local_points, source_segments):
    """v152 surgical local face completion for Rectangle crossings.

    Validate and complete only cells bounded by the new rectangle boundary and
    the pre-transaction edges/faces that were directly touched by it.  Unlike
    _complete_closed_boundary_faces(), this never walks the wider connected
    coplanar component, so a missing local face cannot recruit distant edges.
    It is non-destructive: no faces are deleted and no diagonals/helper edges
    are created.
    """
    pts = [Vector(p).copy() for p in local_points]
    if len(pts) < 3:
        return 0

    origin = pts[0]
    x_axis = None
    normal = None
    for i in range(1, len(pts)):
        vec = pts[i] - origin
        if vec.length > 1e-10:
            x_axis = vec.normalized()
            break
    if x_axis is None:
        return 0
    for i in range(1, len(pts)):
        a = pts[i] - origin
        for j in range(i + 1, len(pts)):
            b = pts[j] - origin
            n = a.cross(b)
            if n.length > 1e-8:
                normal = n.normalized()
                break
        if normal is not None:
            break
    if normal is None:
        return 0
    y_axis = normal.cross(x_axis)
    if y_axis.length <= 1e-10:
        return 0
    y_axis.normalize()
    basis = (origin, x_axis, y_axis)

    allowed = set()
    # Reacquire only the current split pieces of the new ordered boundary.
    for i, p1 in enumerate(pts):
        p2 = pts[(i + 1) % len(pts)]
        for edge in _edges_on_segment(bm, p1, p2):
            if edge is not None and edge.is_valid:
                allowed.add(edge)

    # Reacquire only split pieces of geometry that was directly touched by
    # the rectangle before mutation.  No historical BMesh references survive.
    for a, b in source_segments:
        for edge in _edges_on_segment(bm, a, b):
            if edge is not None and edge.is_valid:
                allowed.add(edge)

    if len(allowed) < 3:
        return 0

    created = 0
    for cycle in _all_bounded_planar_cells(list(allowed), basis):
        if len(cycle) < 3 or any(v is None or not v.is_valid for v in cycle):
            continue
        cycle_edges = set()
        ok = True
        for i, v1 in enumerate(cycle):
            v2 = cycle[(i + 1) % len(cycle)]
            edge = bm.edges.get((v1, v2))
            if edge is None or not edge.is_valid or edge not in allowed:
                ok = False
                break
            cycle_edges.add(edge)
        if not ok:
            continue
        # Do not duplicate an already-valid local face.
        if any(face.is_valid and set(face.edges) == cycle_edges for face in bm.faces):
            continue
        try:
            face = bm.faces.new(cycle)
        except (ValueError, RuntimeError):
            continue
        if face is not None:
            created += 1
    if created:
        bm.normal_update()
    return created

def _complete_closed_boundary_faces(bm, local_points):
    """v150 Rectangle face-completion pass.

    This pass is intentionally non-destructive: it never deletes faces or
    creates diagonals.  It derives the drawing plane directly from the ordered
    closed boundary, gathers the connected coplanar edge graph reached from
    that boundary, then creates only bounded cells that do not already have a
    face.  It exists to make Rectangle completion deterministic after the v149
    crossing transaction without changing the crossing/intersection engine.
    """
    pts = [Vector(p).copy() for p in local_points]
    if len(pts) < 3:
        return 0

    # Derive the plane from the user's ordered loop rather than linked faces;
    # this stays stable even when the crossing transaction temporarily leaves
    # the new boundary as wire edges only.
    origin = pts[0]
    x_axis = None
    normal = None
    for i in range(1, len(pts)):
        vec = pts[i] - origin
        if vec.length > 1e-10:
            x_axis = vec.normalized()
            break
    if x_axis is None:
        return 0
    for i in range(1, len(pts)):
        a = pts[i] - origin
        for j in range(i + 1, len(pts)):
            b = pts[j] - origin
            n = a.cross(b)
            if n.length > 1e-8:
                normal = n.normalized()
                break
        if normal is not None:
            break
    if normal is None:
        return 0
    y_axis = normal.cross(x_axis)
    if y_axis.length <= 1e-10:
        return 0
    y_axis.normalize()
    basis = (origin, x_axis, y_axis)

    # Reacquire the boundary pieces after every earlier BMesh mutation.
    seed_edges = []
    seen_edges = set()
    for i, p1 in enumerate(pts):
        p2 = pts[(i + 1) % len(pts)]
        for edge in _edges_on_segment(bm, p1, p2):
            if edge is not None and edge.is_valid and edge not in seen_edges:
                seen_edges.add(edge)
                seed_edges.append(edge)
    if not seed_edges:
        return 0

    component = _planar_component_from_seed_edges(seed_edges, basis)
    if len(component) < 3:
        return 0

    created = 0
    for cycle in _all_bounded_planar_cells(component, basis):
        if len(cycle) < 3 or any(v is None or not v.is_valid for v in cycle):
            continue
        cycle_edges = set()
        valid_cycle = True
        for i, v1 in enumerate(cycle):
            v2 = cycle[(i + 1) % len(cycle)]
            edge = bm.edges.get((v1, v2))
            if edge is None or not edge.is_valid:
                valid_cycle = False
                break
            cycle_edges.add(edge)
        if not valid_cycle:
            continue
        if any(face.is_valid and set(face.edges) == cycle_edges for face in bm.faces):
            continue
        try:
            face = bm.faces.new(cycle)
        except (ValueError, RuntimeError):
            continue
        if face is not None:
            created += 1
    if created:
        bm.normal_update()
    return created




def _closed_boundary_plane(local_points):
    """Return (origin, normal) for one ordered closed boundary, or None.

    Used only by Rectangle/Circle closed-boundary transactions.  This keeps
    the plane test local to the closed-shape path and does not change Line or
    Freehand autocut behaviour.
    """
    pts = [Vector(p) for p in local_points]
    if len(pts) < 3:
        return None
    origin = pts[0].copy()
    for i in range(1, len(pts)):
        a = pts[i] - origin
        if a.length <= 1e-10:
            continue
        for j in range(i + 1, len(pts)):
            n = a.cross(pts[j] - origin)
            if n.length > 1e-8:
                return origin, n.normalized()
    return None


def _point_plane_distance(point, origin, normal):
    return abs((Vector(point) - origin).dot(normal))


def _autocut_closed_boundary_segment_planar(bm, p1, p2, plane_origin, plane_normal, plane_tol=2e-5):
    """Closed-shape-only autocut restricted to the drawing plane.

    v155 safety gate: only existing edges whose BOTH endpoints lie on the new
    rectangle plane are eligible for splitting/connection.  This prevents a
    T-junction on a folded/non-coplanar face from recruiting geometry outside
    the rectangle plane.  The Line/Freehand autocut function remains untouched.
    """
    p1 = Vector(p1); p2 = Vector(p2)
    if (p1 - p2).length < 1e-6:
        return False
    bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
    existing_edges = list(bm.edges)
    hits = []
    for edge in existing_edges:
        if not edge.is_valid:
            continue
        a = edge.verts[0].co.copy(); b = edge.verts[1].co.copy()
        if _point_plane_distance(a, plane_origin, plane_normal) > plane_tol:
            continue
        if _point_plane_distance(b, plane_origin, plane_normal) > plane_tol:
            continue
        hit = _segment_intersection_3d(p1, p2, a, b)
        if hit is None:
            continue
        point, t_new, t_edge = hit
        if _point_plane_distance(point, plane_origin, plane_normal) > plane_tol:
            continue
        hits.append((t_new, edge, t_edge, point))

    cut_vertices = []
    for t_new, edge, t_edge, point in sorted(hits, key=lambda item: item[0]):
        vert = None
        for candidate in bm.verts:
            if (candidate.co - point).length < VERTEX_TOLERANCE:
                vert = candidate; break
        if vert is None and edge.is_valid:
            vert = _vertex_at_edge_parameter(bm, edge, t_edge, point)
        if vert is not None:
            cut_vertices.append((t_new, vert))

    start_vert = find_or_create_vertex(bm, p1); end_vert = find_or_create_vertex(bm, p2)
    chain = [(0.0, start_vert)] + cut_vertices + [(1.0, end_vert)]
    chain.sort(key=lambda item: item[0])
    unique = []
    for t, vert in chain:
        if unique and (vert == unique[-1][1] or (vert.co - unique[-1][1].co).length < VERTEX_TOLERANCE):
            continue
        unique.append((t, vert))

    created = False
    for (_, v1), (_, v2) in zip(unique, unique[1:]):
        if v1 == v2 or bm.edges.get((v1, v2)) is not None:
            continue
        # Only split shared faces that are themselves coplanar with the new
        # rectangle.  A shared folded/non-coplanar face is never recruited.
        shared_faces = set(v1.link_faces).intersection(v2.link_faces)
        planar_shared = []
        for face in shared_faces:
            if face.is_valid and all(_point_plane_distance(v.co, plane_origin, plane_normal) <= plane_tol for v in face.verts):
                planar_shared.append(face)
        if planar_shared:
            try:
                result = bmesh.ops.connect_vert_pair(bm, verts=[v1, v2])
                if result.get('edges'):
                    created = True; continue
            except (RuntimeError, ValueError):
                pass
        if bm.edges.get((v1, v2)) is None:
            bm.edges.new((v1, v2)); created = True
    return created

def _complete_new_boundary_cells_v153(bm, local_points):
    """v153 surgical validator for Rectangle crossings.

    Reacquires the planar component from the NEW rectangle boundary, but may
    fill only bounded cells whose centroid lies inside the new ordered loop
    and whose perimeter contains at least one split piece of that loop.  The
    wider component is therefore read for connectivity only; no face outside
    the new rectangle can be created or changed.  No vertices/edges are added.
    """
    pts = [Vector(p).copy() for p in local_points]
    if len(pts) < 3:
        return 0
    origin = pts[0]
    x_axis = None
    normal = None
    for i in range(1, len(pts)):
        v = pts[i] - origin
        if v.length > 1e-10:
            x_axis = v.normalized(); break
    if x_axis is None:
        return 0
    for i in range(1, len(pts)):
        a = pts[i] - origin
        for j in range(i + 1, len(pts)):
            n = a.cross(pts[j] - origin)
            if n.length > 1e-8:
                normal = n.normalized(); break
        if normal is not None:
            break
    if normal is None:
        return 0
    y_axis = normal.cross(x_axis)
    if y_axis.length <= 1e-10:
        return 0
    y_axis.normalize()
    basis = (origin, x_axis, y_axis)

    def uv(co):
        d = co - origin
        return (d.dot(x_axis), d.dot(y_axis))
    poly = [uv(p) for p in pts]
    def inside(x, y):
        # boundary-inclusive even/odd test
        hit = False
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
            dx=x2-x1; dy=y2-y1
            cross=(x-x1)*dy-(y-y1)*dx
            if abs(cross) <= 1e-7 and min(x1,x2)-1e-7 <= x <= max(x1,x2)+1e-7 and min(y1,y2)-1e-7 <= y <= max(y1,y2)+1e-7:
                return True
            if ((y1 > y) != (y2 > y)):
                xin = x1 + (y-y1)*(x2-x1)/(y2-y1)
                if x < xin:
                    hit = not hit
        return hit

    boundary_edges=set(); seeds=[]
    for i,p1 in enumerate(pts):
        p2=pts[(i+1)%len(pts)]
        for e in _edges_on_segment(bm,p1,p2):
            if e is not None and e.is_valid:
                boundary_edges.add(e); seeds.append(e)
    if not seeds:
        return 0
    component=_planar_component_from_seed_edges(seeds,basis)
    if len(component)<3:
        return 0
    created=0
    for cycle in _all_bounded_planar_cells(component,basis):
        if len(cycle)<3 or any(v is None or not v.is_valid for v in cycle):
            continue
        edges=[]; ok=True
        for i,v1 in enumerate(cycle):
            e=bm.edges.get((v1,cycle[(i+1)%len(cycle)]))
            if e is None or not e.is_valid:
                ok=False; break
            edges.append(e)
        if not ok or not any(e in boundary_edges for e in edges):
            continue
        cx=sum(uv(v.co)[0] for v in cycle)/len(cycle)
        cy=sum(uv(v.co)[1] for v in cycle)/len(cycle)
        if not inside(cx,cy):
            continue
        eset=set(edges)
        if any(f.is_valid and set(f.edges)==eset for f in bm.faces):
            continue
        try:
            f=bm.faces.new(cycle)
        except (ValueError,RuntimeError):
            continue
        if f is not None:
            created += 1
    if created:
        bm.normal_update()
    return created


def _complete_rectangle_overlap_cells_v180(bm, local_points, source_segments, source_polygons):
    """v180 Rectangle-only local overlap completion.

    Repair missing cells after a Rectangle x Rectangle crossing without walking
    the wider coplanar component.  The admissible graph is limited to split
    pieces of the new rectangle boundary plus boundaries of faces that were
    directly touched before mutation.  A candidate cell is accepted only when
    its centroid lies inside the new rectangle or inside one of those touched
    source faces.  No vertices, diagonals, or helper edges are created.
    """
    pts = [Vector(p).copy() for p in local_points]
    if len(pts) < 3:
        return 0

    plane = _closed_boundary_plane(pts)
    if plane is None:
        return 0
    origin, normal = plane

    # Stable in-plane basis from the new rectangle.
    x_axis = None
    for i in range(1, len(pts)):
        v = pts[i] - origin
        v = v - normal * v.dot(normal)
        if v.length > 1e-10:
            x_axis = v.normalized()
            break
    if x_axis is None:
        return 0
    y_axis = normal.cross(x_axis)
    if y_axis.length <= 1e-10:
        return 0
    y_axis.normalize()
    basis = (origin, x_axis, y_axis)

    def uv(co):
        d = Vector(co) - origin
        return Vector((d.dot(x_axis), d.dot(y_axis)))

    def point_in_poly(point, poly):
        if len(poly) < 3:
            return False
        px, py = point.x, point.y
        inside = False
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            ab = b - a
            ap = point - a
            cross = ab.x * ap.y - ab.y * ap.x
            if abs(cross) <= 1e-7:
                dot = ap.dot(ab)
                if -1e-7 <= dot <= ab.length_squared + 1e-7:
                    return True
            if (a.y > py) != (b.y > py):
                x_hit = (b.x - a.x) * (py - a.y) / (b.y - a.y) + a.x
                if px < x_hit:
                    inside = not inside
        return inside

    new_poly = [uv(p) for p in pts]
    touched_polys = [[uv(p) for p in poly] for poly in source_polygons if len(poly) >= 3]

    allowed = set()
    # Reacquire every split piece of the new rectangle boundary.
    for i, p1 in enumerate(pts):
        p2 = pts[(i + 1) % len(pts)]
        for edge in _edges_on_segment(bm, p1, p2):
            if edge is not None and edge.is_valid:
                allowed.add(edge)
    # Reacquire every split piece of directly touched source-face boundaries.
    for a, b in source_segments:
        for edge in _edges_on_segment(bm, a, b):
            if edge is not None and edge.is_valid:
                allowed.add(edge)

    if len(allowed) < 3:
        return 0

    created = 0
    for cycle in _all_bounded_planar_cells(list(allowed), basis):
        if len(cycle) < 3 or any(v is None or not v.is_valid for v in cycle):
            continue
        edges = []
        valid = True
        for i, v1 in enumerate(cycle):
            v2 = cycle[(i + 1) % len(cycle)]
            edge = bm.edges.get((v1, v2))
            if edge is None or not edge.is_valid or edge not in allowed:
                valid = False
                break
            edges.append(edge)
        if not valid:
            continue
        edge_set = set(edges)
        if any(face.is_valid and set(face.edges) == edge_set for face in bm.faces):
            continue

        center = Vector((0.0, 0.0))
        for v in cycle:
            center += uv(v.co)
        center /= len(cycle)

        # Stay local: only restore cells belonging to the new rectangle or to
        # a face that the new rectangle directly touched before the transaction.
        if not point_in_poly(center, new_poly) and not any(
            point_in_poly(center, poly) for poly in touched_polys
        ):
            continue

        try:
            face = bm.faces.new(cycle)
        except (ValueError, RuntimeError):
            continue
        if face is not None:
            created += 1

    if created:
        bm.normal_update()
    return created

def insert_closed_planar_boundary_edit(obj, world_points):
    """Insert one ordered closed boundary into Edit Mode.

    v155 keeps the proven v151-v154 local crossing/validation architecture
    but hardens the rare T-junction path: a topology contact is accepted only
    when the touched edge and any recruited face are coplanar with the new
    rectangle.  This prevents folded/external faces without changing the
    already-good Rectangle intersection/locality logic.
    standalone/non-crossing loops are committed directly; genuine crossings
    are delegated to the proven v137 planar transaction used by Line.  This
    removes v148's unsafe per-segment mutation path and gives Rectangle x
    Rectangle one topology authority instead of mixing connect_vert_pair and
    direct face creation in the same transaction.
    """
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return False
    pts = [Vector(p).copy() for p in world_points]
    if len(pts) > 1 and (pts[0] - pts[-1]).length <= VERTEX_TOLERANCE:
        pts.pop()
    if len(pts) < 3:
        return False

    mesh = obj.data
    inv = obj.matrix_world.inverted()
    local = [inv @ p for p in pts]
    bm = bmesh.from_edit_mesh(mesh)

    plane = _closed_boundary_plane(local)
    if plane is None:
        return False
    plane_origin, plane_normal = plane
    plane_tol = 2e-5

    # Read-only crossing classification BEFORE any BMesh mutation.
    existing_segments = [
        (e.verts[0].co.copy(), e.verts[1].co.copy())
        for e in bm.edges if e.is_valid
    ]
    # v151: snapshot only geometry directly touched by this rectangle.
    # Start with faces crossed by any new boundary segment; their complete
    # boundaries define the local reconstruction island.
    local_source_segments = []
    local_source_keys = set()
    # v180: preserve ordered boundaries of only the faces directly touched by
    # this rectangle.  Coordinates, not BMFace references, survive mutation.
    local_source_polygons = []
    local_source_polygon_keys = set()
    def _remember_segment(a, b):
        key = tuple(sorted((tuple(round(float(x), 7) for x in a), tuple(round(float(x), 7) for x in b))))
        if key not in local_source_keys:
            local_source_keys.add(key); local_source_segments.append((a.copy(), b.copy()))

    genuine_crossing = False
    eps = 1e-6
    for i, p1 in enumerate(local):
        p2 = local[(i + 1) % len(local)]
        for a, b in existing_segments:
            hit = _segment_intersection_3d(p1, p2, a, b)
            if hit is None:
                continue
            _point, t_new, t_edge = hit
            # v154 Stage 6: a topology-changing contact is not limited to
            # strict interior/interior crossings.  A rectangle corner landing
            # on the interior of an existing edge (or an existing endpoint
            # landing on the interior of the new rectangle side) is a valid
            # T-junction and must enter the same local crossing transaction.
            # Pure endpoint-to-endpoint contact remains non-destructive.
            new_interior = eps < t_new < 1.0 - eps
            edge_interior = eps < t_edge < 1.0 - eps
            topology_contact = (new_interior or edge_interior)
            # v155: T-junction completion is strictly planar.  Both endpoints
            # of the existing edge must lie on the rectangle plane; otherwise
            # this contact cannot participate in the local topology repair.
            coplanar_edge = (
                _point_plane_distance(a, plane_origin, plane_normal) <= plane_tol
                and _point_plane_distance(b, plane_origin, plane_normal) <= plane_tol
                and _point_plane_distance(_point, plane_origin, plane_normal) <= plane_tol
            )
            if topology_contact and coplanar_edge:
                genuine_crossing = True
                _remember_segment(a, b)
                # Snapshot only directly linked faces that are fully coplanar
                # with the new rectangle.  Non-coplanar/folded faces are never
                # allowed into the local reconstruction island.
                for edge in bm.edges:
                    if not edge.is_valid:
                        continue
                    ea, eb = edge.verts[0].co, edge.verts[1].co
                    if ((ea-a).length <= 1e-7 and (eb-b).length <= 1e-7) or ((ea-b).length <= 1e-7 and (eb-a).length <= 1e-7):
                        for face in edge.link_faces:
                            if face.is_valid and all(
                                _point_plane_distance(v.co, plane_origin, plane_normal) <= plane_tol
                                for v in face.verts
                            ):
                                poly = [v.co.copy() for v in face.verts]
                                poly_key = tuple(sorted(
                                    (round(float(v.x), 7), round(float(v.y), 7), round(float(v.z), 7))
                                    for v in poly
                                ))
                                if poly_key not in local_source_polygon_keys:
                                    local_source_polygon_keys.add(poly_key)
                                    local_source_polygons.append(poly)
                                for fe in face.edges:
                                    _remember_segment(fe.verts[0].co, fe.verts[1].co)
                        break

    if not genuine_crossing:
        ordered = [find_or_create_vertex(bm, p) for p in local]
        if len(set(ordered)) != len(ordered):
            return False
        for i, v1 in enumerate(ordered):
            v2 = ordered[(i + 1) % len(ordered)]
            if bm.edges.get((v1, v2)) is None:
                bm.edges.new((v1, v2))
        cycle_edges = {bm.edges.get((ordered[i], ordered[(i+1)%len(ordered)])) for i in range(len(ordered))}
        already = any(f.is_valid and set(f.edges) == cycle_edges for f in bm.faces)
        if not already:
            try:
                bm.faces.new(ordered)
            except (ValueError, RuntimeError):
                pass
        completed = _complete_closed_boundary_faces(bm, local)
        bm.normal_update()
        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        debug_print("Unified Edit boundary v150: ordered loop committed;", completed, "missing bounded face(s) completed")
        return True

    # Genuine crossing: use the v137 Line topology transaction.  Feed the
    # complete ordered boundary into BMesh first, collecting only coordinates
    # between calls; never retain BMVert/BMEdge references across autocuts.
    for i, p1 in enumerate(local):
        p2 = local[(i + 1) % len(local)]
        if (p2 - p1).length <= VERTEX_TOLERANCE:
            continue
        _autocut_closed_boundary_segment_planar(
            bm, p1, p2, plane_origin, plane_normal, plane_tol
        )

    # Reacquire all BMesh elements after the mutation phase.  Seed the same
    # bounded-cell retopology used by Line from the complete new boundary.
    seed_edges = []
    seen = set()
    for i, p1 in enumerate(local):
        p2 = local[(i + 1) % len(local)]
        for e in _edges_on_segment(bm, p1, p2):
            if e is not None and e.is_valid and e not in seen:
                seen.add(e)
                seed_edges.append(e)

    filled = _retopologize_edit_planar_region_local(bm, seed_edges, local_source_segments) if seed_edges else 0

    # v152 Stage 4: validate/complete only the local cells around this new
    # rectangle.  Never traverse the wider connected planar component here.
    completed = _complete_closed_boundary_faces_local(bm, local, local_source_segments)
    # v153 Stage 5: final validation may fill only missing cells that are
    # inside the new rectangle and actually use its boundary.
    validated = _complete_new_boundary_cells_v153(bm, local)
    # v180: one final rectangle-only exact-local repair.  This fills a missing
    # overlap cell only when it belongs to the new rectangle or to a source
    # face directly touched by it.  It never recruits distant coplanar geometry
    # and never creates helper edges.
    repaired = _complete_rectangle_overlap_cells_v180(
        bm, local, local_source_segments, local_source_polygons
    )
    bm.normal_update()
    bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
    debug_print("Unified Edit boundary v180 Rectangle overlap repair:", len(seed_edges), "pieces;", filled, "rebuilt cells;", completed, "local completed;", validated, "validated;", repaired, "repaired")
    return bool(seed_edges)


def insert_circle_planar_boundary_edit(obj, world_points):
    """v156 Circle-only ordered boundary transaction for Edit Mode.

    This is intentionally isolated from the frozen v155 Rectangle entry point.
    It starts from the same proven planar-safe/local transaction so Circle x
    Circle can now be developed and tested without changing Rectangle behavior.

    v155 keeps the proven v151-v154 local crossing/validation architecture
    but hardens the rare T-junction path: a topology contact is accepted only
    when the touched edge and any recruited face are coplanar with the new
    rectangle.  This prevents folded/external faces without changing the
    already-good Rectangle intersection/locality logic.
    standalone/non-crossing loops are committed directly; genuine crossings
    are delegated to the proven v137 planar transaction used by Line.  This
    removes v148's unsafe per-segment mutation path and gives Rectangle x
    Rectangle one topology authority instead of mixing connect_vert_pair and
    direct face creation in the same transaction.
    """
    if obj is None or obj.type != "MESH" or obj.mode != "EDIT":
        return False
    pts = [Vector(p).copy() for p in world_points]
    if len(pts) > 1 and (pts[0] - pts[-1]).length <= VERTEX_TOLERANCE:
        pts.pop()
    if len(pts) < 3:
        return False

    mesh = obj.data
    inv = obj.matrix_world.inverted()
    local = [inv @ p for p in pts]
    bm = bmesh.from_edit_mesh(mesh)

    plane = _closed_boundary_plane(local)
    if plane is None:
        return False
    plane_origin, plane_normal = plane
    plane_tol = 2e-5

    # Read-only crossing classification BEFORE any BMesh mutation.
    existing_segments = [
        (e.verts[0].co.copy(), e.verts[1].co.copy())
        for e in bm.edges if e.is_valid
    ]
    # v151: snapshot only geometry directly touched by this rectangle.
    # Start with faces crossed by any new boundary segment; their complete
    # boundaries define the local reconstruction island.
    local_source_segments = []
    local_source_keys = set()
    def _remember_segment(a, b):
        key = tuple(sorted((tuple(round(float(x), 7) for x in a), tuple(round(float(x), 7) for x in b))))
        if key not in local_source_keys:
            local_source_keys.add(key); local_source_segments.append((a.copy(), b.copy()))

    genuine_crossing = False
    eps = 1e-6
    for i, p1 in enumerate(local):
        p2 = local[(i + 1) % len(local)]
        for a, b in existing_segments:
            hit = _segment_intersection_3d(p1, p2, a, b)
            if hit is None:
                continue
            _point, t_new, t_edge = hit
            # v154 Stage 6: a topology-changing contact is not limited to
            # strict interior/interior crossings.  A rectangle corner landing
            # on the interior of an existing edge (or an existing endpoint
            # landing on the interior of the new rectangle side) is a valid
            # T-junction and must enter the same local crossing transaction.
            # Pure endpoint-to-endpoint contact remains non-destructive.
            new_interior = eps < t_new < 1.0 - eps
            edge_interior = eps < t_edge < 1.0 - eps
            topology_contact = (new_interior or edge_interior)
            # v155: T-junction completion is strictly planar.  Both endpoints
            # of the existing edge must lie on the rectangle plane; otherwise
            # this contact cannot participate in the local topology repair.
            coplanar_edge = (
                _point_plane_distance(a, plane_origin, plane_normal) <= plane_tol
                and _point_plane_distance(b, plane_origin, plane_normal) <= plane_tol
                and _point_plane_distance(_point, plane_origin, plane_normal) <= plane_tol
            )
            if topology_contact and coplanar_edge:
                genuine_crossing = True
                _remember_segment(a, b)
                # Snapshot only directly linked faces that are fully coplanar
                # with the new rectangle.  Non-coplanar/folded faces are never
                # allowed into the local reconstruction island.
                for edge in bm.edges:
                    if not edge.is_valid:
                        continue
                    ea, eb = edge.verts[0].co, edge.verts[1].co
                    if ((ea-a).length <= 1e-7 and (eb-b).length <= 1e-7) or ((ea-b).length <= 1e-7 and (eb-a).length <= 1e-7):
                        for face in edge.link_faces:
                            if face.is_valid and all(
                                _point_plane_distance(v.co, plane_origin, plane_normal) <= plane_tol
                                for v in face.verts
                            ):
                                for fe in face.edges:
                                    _remember_segment(fe.verts[0].co, fe.verts[1].co)
                        break

    if not genuine_crossing:
        ordered = [find_or_create_vertex(bm, p) for p in local]
        if len(set(ordered)) != len(ordered):
            return False
        for i, v1 in enumerate(ordered):
            v2 = ordered[(i + 1) % len(ordered)]
            if bm.edges.get((v1, v2)) is None:
                bm.edges.new((v1, v2))
        cycle_edges = {bm.edges.get((ordered[i], ordered[(i+1)%len(ordered)])) for i in range(len(ordered))}
        already = any(f.is_valid and set(f.edges) == cycle_edges for f in bm.faces)
        if not already:
            try:
                bm.faces.new(ordered)
            except (ValueError, RuntimeError):
                pass
        completed = _complete_closed_boundary_faces(bm, local)
        bm.normal_update()
        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        debug_print("Circle v156: ordered loop committed;", completed, "missing bounded face(s) completed")
        return True

    # Genuine crossing: use the v137 Line topology transaction.  Feed the
    # complete ordered boundary into BMesh first, collecting only coordinates
    # between calls; never retain BMVert/BMEdge references across autocuts.
    for i, p1 in enumerate(local):
        p2 = local[(i + 1) % len(local)]
        if (p2 - p1).length <= VERTEX_TOLERANCE:
            continue
        _autocut_closed_boundary_segment_planar(
            bm, p1, p2, plane_origin, plane_normal, plane_tol
        )

    # Reacquire all BMesh elements after the mutation phase.  Seed the same
    # bounded-cell retopology used by Line from the complete new boundary.
    seed_edges = []
    seen = set()
    for i, p1 in enumerate(local):
        p2 = local[(i + 1) % len(local)]
        for e in _edges_on_segment(bm, p1, p2):
            if e is not None and e.is_valid and e not in seen:
                seen.add(e)
                seed_edges.append(e)

    filled = _retopologize_edit_planar_region_local(bm, seed_edges, local_source_segments) if seed_edges else 0

    # v152 Stage 4: validate/complete only the local cells around this new
    # rectangle.  Never traverse the wider connected planar component here.
    completed = _complete_closed_boundary_faces_local(bm, local, local_source_segments)
    # v153 Stage 5: final validation may fill only missing cells that are
    # inside the new rectangle and actually use its boundary.
    validated = _complete_new_boundary_cells_v153(bm, local)
    bm.normal_update()
    bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
    debug_print("Circle v156 planar-safe crossing:", len(seed_edges), "pieces;", filled, "rebuilt cells;", completed, "local completed cells;", validated, "validated missing cells")
    return bool(seed_edges)

def create_edge(obj, p1, p2):
    mesh = obj.data
    inv_matrix = obj.matrix_world.inverted()
    p1 = inv_matrix @ Vector(p1)
    p2 = inv_matrix @ Vector(p2)

    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(mesh)
        created = _autocut_edge_bmesh(bm, p1, p2)
        segment_edges = _edges_on_segment(bm, p1, p2)

        # v137: Edit Mode now follows the same clean-graph principle that made
        # Object Mode reliable.  Historical faces in the affected coplanar
        # region are removed (edges/verts preserved), then the planar embedding
        # is rebuilt into only its true bounded cells.
        filled = _retopologize_edit_planar_region(bm, segment_edges)
        if filled:
            debug_print(
                "Line Auto-Face rebuilt", filled,
                "bounded planar cell(s) in Edit Mode"
            )
        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)
        debug_print("Autocut edge created in Edit Mode" if created else "Edge already exists")
        return created

    bm = bmesh.new()
    bm.from_mesh(mesh)
    created = _autocut_edge_bmesh(bm, p1, p2)
    filled = _autofill_closed_wire_regions(bm)
    if filled:
        debug_print("Line Auto-Face created", filled, "closed region(s) in Object Mode")
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
