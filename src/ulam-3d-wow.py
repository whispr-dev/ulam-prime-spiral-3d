#!/usr/bin/env python3
"""
3D Ulam-style prime visualizer with:
- Cubic-shell mapping
- Surface "walk-ish" cube traversal order
- Parallel-line detection for many directions
- Top-K prime-dense line rendering
- Export to PLY / OBJ / glTF

Dependencies: numpy, matplotlib

Usage examples:
  python ulam-3d-wow.py --N 50000 --mapping shell --topk 12 --export-prefix primes_shell
  python ulam-3d-wow.py --N 50000 --mapping surface --topk 16 --dir-max 2 --export-prefix primes_surface
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Set

import numpy as np
import matplotlib.pyplot as plt

Vec3 = Tuple[int, int, int]


# --------------------------
# Primes
# --------------------------

def sieve_primes(limit: int) -> np.ndarray:
    """Boolean array is_prime[0..limit]."""
    if limit < 2:
        return np.zeros(limit + 1, dtype=bool)

    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[p]:
            is_prime[p * p:limit + 1:p] = False
    return is_prime


# --------------------------
# Geometry helpers
# --------------------------

def cross(a: Vec3, b: Vec3) -> Vec3:
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

def dot(a: Vec3, b: Vec3) -> int:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def normalize_direction(d: Vec3) -> Vec3:
    """Reduce to primitive integer step; canonical sign."""
    dx, dy, dz = d
    g = math.gcd(abs(dx), math.gcd(abs(dy), abs(dz)))
    if g == 0:
        raise ValueError("Zero direction vector")
    dx //= g
    dy //= g
    dz //= g

    # Canonical sign: first nonzero positive
    for c in (dx, dy, dz):
        if c != 0:
            if c < 0:
                dx, dy, dz = -dx, -dy, -dz
            break
    return (dx, dy, dz)

def perpendicular_basis(d: Vec3) -> Tuple[Vec3, Vec3]:
    """
    Construct two integer vectors a, b such that:
      a·d = 0
      b·d = 0
    Using:
      a = d × e
      b = d × a
    """
    dx, dy, dz = d
    # choose e not parallel to d
    if (dx, dy, dz) != (1, 0, 0) and not (dy == 0 and dz == 0):
        e = (1, 0, 0)
    elif not (dx == 0 and dz == 0):
        e = (0, 1, 0)
    else:
        e = (0, 0, 1)

    a = cross(d, e)
    if a == (0, 0, 0):
        # fallback, should be rare
        e = (0, 1, 0) if e != (0, 1, 0) else (0, 0, 1)
        a = cross(d, e)

    b = cross(d, a)
    if b == (0, 0, 0):
        # extremely unlikely, but guard
        raise RuntimeError("Failed to build perpendicular basis")

    return a, b

def chebyshev_radius(p: Vec3) -> int:
    return max(abs(p[0]), abs(p[1]), abs(p[2]))


# --------------------------
# Mapping 1: Cubic shells (fast)
# --------------------------

def cubic_shell_points(r: int) -> Iterable[Vec3]:
    """Points on surface max(|x|,|y|,|z|)=r, lexicographic order."""
    if r == 0:
        yield (0, 0, 0)
        return
    rng = range(-r, r + 1)
    for x in rng:
        for y in rng:
            for z in rng:
                if max(abs(x), abs(y), abs(z)) == r:
                    yield (x, y, z)

def generate_cubic_shell_mapping(n: int) -> List[Vec3]:
    coords: List[Vec3] = []
    r = 0
    while len(coords) < n:
        for p in cubic_shell_points(r):
            coords.append(p)
            if len(coords) >= n:
                break
        r += 1
    return coords


# --------------------------
# Mapping 2: Surface "walk-ish" cube traversal
# --------------------------

def perimeter_square_xy(r: int, z: int) -> List[Vec3]:
    """
    Perimeter of square in x-y with Chebyshev radius r at fixed z:
      max(|x|,|y|)=r
    Ordered loop starting at (r,-r,z) going CCW.
    """
    pts: List[Vec3] = []
    # right edge: x=r, y from -r..r
    for y in range(-r, r + 1):
        pts.append((r, y, z))
    # top edge: y=r, x from r-1..-r
    for x in range(r - 1, -r - 1, -1):
        pts.append((x, r, z))
    # left edge: x=-r, y from r-1..-r
    for y in range(r - 1, -r - 1, -1):
        pts.append((-r, y, z))
    # bottom edge: y=-r, x from -r+1..r-1
    for x in range(-r + 1, r):
        pts.append((x, -r, z))
    # This forms a closed loop with repeated corners in standard perimeter builds,
    # but our sequence avoids duplicates by construction.
    return pts

def face_square_xy(r: int, z: int, reverse: bool = False) -> List[Vec3]:
    """
    Full square face at fixed z:
      |x|<=r, |y|<=r
    Serpentine (row-wise) order to encourage adjacency.
    """
    pts: List[Vec3] = []
    ys = list(range(-r, r + 1))
    if reverse:
        ys = ys[::-1]

    for row_idx, y in enumerate(ys):
        xs = list(range(-r, r + 1))
        # snake
        if (row_idx % 2) == 1:
            xs.reverse()
        for x in xs:
            pts.append((x, y, z))
    return pts

def shell_surface_path(r: int) -> List[Vec3]:
    """
    Ordered traversal of the cube surface at radius r.
    Designed to feel like a continuous sweep across faces and slice perimeters.

    Order:
      1) Bottom face z=-r (full square)
      2) Middle z slices: perimeter loops
      3) Top face z=+r (full square, reversed serpentine)
    """
    if r == 0:
        return [(0, 0, 0)]

    out: List[Vec3] = []

    # Bottom face
    out.extend(face_square_xy(r, z=-r, reverse=False))

    # Middle slices perimeters
    for z in range(-r + 1, r):
        out.extend(perimeter_square_xy(r, z))

    # Top face
    out.extend(face_square_xy(r, z=+r, reverse=True))

    # Deduplicate while preserving order (should be unnecessary, but safe)
    seen: Set[Vec3] = set()
    deduped: List[Vec3] = []
    for p in out:
        if p not in seen and chebyshev_radius(p) == r:
            seen.add(p)
            deduped.append(p)
    return deduped

def generate_surface_walk_mapping(n: int) -> List[Vec3]:
    """
    Generate integer->position mapping by concatenating ordered cube-surface paths.

    Prints an adjacency-break metric in main() to quantify "walk-ness".
    """
    coords: List[Vec3] = []
    r = 0
    while len(coords) < n:
        path = shell_surface_path(r)
        for p in path:
            coords.append(p)
            if len(coords) >= n:
                break
        r += 1
    return coords

def manhattan(a: Vec3, b: Vec3) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def adjacency_breaks(path: List[Vec3]) -> Tuple[int, int]:
    """Return (#breaks, #steps). A break is a step with manhattan != 1."""
    if len(path) < 2:
        return (0, 0)
    breaks = 0
    steps = len(path) - 1
    for i in range(1, len(path)):
        if manhattan(path[i-1], path[i]) != 1:
            breaks += 1
    return breaks, steps


# --------------------------
# Parallel line detection
# --------------------------

@dataclass(frozen=True)
class LineKey:
    inv1: int
    inv2: int

@dataclass
class LineRecord:
    direction: Vec3
    key: LineKey
    points: List[Vec3]  # prime points on this line

    @property
    def count(self) -> int:
        return len(self.points)

    def endpoints_by_projection(self) -> Tuple[Vec3, Vec3]:
        """Pick endpoints among the prime points using projection along direction."""
        d = self.direction
        projs = [dot(p, d) for p in self.points]
        i_min = int(np.argmin(np.array(projs)))
        i_max = int(np.argmax(np.array(projs)))
        return self.points[i_min], self.points[i_max]

def generate_directions(max_comp: int = 2) -> List[Vec3]:
    """
    Generate unique primitive directions with components in [-max_comp..max_comp],
    excluding zero vector.
    """
    dirs: Set[Vec3] = set()
    for dx in range(-max_comp, max_comp + 1):
        for dy in range(-max_comp, max_comp + 1):
            for dz in range(-max_comp, max_comp + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                d = normalize_direction((dx, dy, dz))
                dirs.add(d)
    return sorted(dirs)

def group_parallel_lines(
    prime_points: List[Vec3],
    directions: List[Vec3],
    min_primes_on_line: int = 6,
) -> List[LineRecord]:
    """
    For each direction d:
      - build perpendicular basis (a,b)
      - invariant pair key = (p·a, p·b)
    Points with same key lie on same line parallel to d.

    Returns all lines across all directions with >= min_primes_on_line.
    """
    all_lines: List[LineRecord] = []

    for d in directions:
        a, b = perpendicular_basis(d)
        buckets: Dict[LineKey, List[Vec3]] = {}

        for p in prime_points:
            key = LineKey(dot(p, a), dot(p, b))
            buckets.setdefault(key, []).append(p)

        for key, pts in buckets.items():
            if len(pts) >= min_primes_on_line:
                # Sort points along direction for nicer endpoints
                pts_sorted = sorted(pts, key=lambda q: dot(q, d))
                all_lines.append(LineRecord(direction=d, key=key, points=pts_sorted))

    # Sort by count descending
    all_lines.sort(key=lambda L: L.count, reverse=True)
    return all_lines

def dedupe_similar_lines(lines: List[LineRecord]) -> List[LineRecord]:
    """
    Very light dedupe:
    It's possible for the same geometric line to appear under different
    directions if those directions are collinear (shouldn't happen)
    or if direction generation includes duplicates (we normalize, so rare).
    We'll dedupe by (direction, key).
    """
    seen: Set[Tuple[Vec3, LineKey]] = set()
    out: List[LineRecord] = []
    for L in lines:
        k = (L.direction, L.key)
        if k not in seen:
            seen.add(k)
            out.append(L)
    return out


# --------------------------
# Plotting
# --------------------------

def plot_primes_and_lines(
    prime_points: List[Vec3],
    lines: List[LineRecord],
    topk: int,
    title: str,
    point_size: float = 6.0,
) -> None:
    prime_arr = np.array(prime_points, dtype=float)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(prime_arr[:, 0], prime_arr[:, 1], prime_arr[:, 2], s=point_size)

    # draw top-k lines
    for L in lines[:topk]:
        p0, p1 = L.endpoints_by_projection()
        xs = [p0[0], p1[0]]
        ys = [p0[1], p1[1]]
        zs = [p0[2], p1[2]]
        ax.plot(xs, ys, zs, linewidth=2)

    # equal-ish bounds
    max_extent = 0
    for x, y, z in prime_points:
        max_extent = max(max_extent, abs(x), abs(y), abs(z))
    lim = max_extent + 1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()


# --------------------------
# Export: OBJ / PLY
# --------------------------

def export_obj(
    path: str,
    prime_points: List[Vec3],
    lines: List[LineRecord],
    topk: int,
) -> None:
    """
    Export as OBJ:
      - vertices for all prime points
      - line elements 'l i j' for top-k lines (using endpoints)
    """
    # unique vertices
    unique_pts = list(dict.fromkeys(prime_points))  # preserves order
    index_map = {p: i + 1 for i, p in enumerate(unique_pts)}  # 1-based

    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D Ulam prime map (points + top lines)\n")
        for x, y, z in unique_pts:
            f.write(f"v {x} {y} {z}\n")

        f.write("\n# Lines (top-k prime-dense)\n")
        for L in lines[:topk]:
            p0, p1 = L.endpoints_by_projection()
            i0 = index_map[p0]
            i1 = index_map[p1]
            f.write(f"l {i0} {i1}\n")

def export_ply(
    path: str,
    prime_points: List[Vec3],
    lines: List[LineRecord],
    topk: int,
) -> None:
    """
    Export ASCII PLY with:
      - vertex list (prime points)
      - edge list (top-k line segments)
    """
    unique_pts = list(dict.fromkeys(prime_points))
    index_map = {p: i for i, p in enumerate(unique_pts)}  # 0-based

    edges: List[Tuple[int, int]] = []
    for L in lines[:topk]:
        p0, p1 = L.endpoints_by_projection()
        edges.append((index_map[p0], index_map[p1]))

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(unique_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")

        for x, y, z in unique_pts:
            f.write(f"{x} {y} {z}\n")

        for i0, i1 in edges:
            f.write(f"{i0} {i1}\n")


# --------------------------
# Export: minimal glTF (points + lines)
# --------------------------

def export_gltf(
    prefix: str,
    prime_points: List[Vec3],
    lines: List[LineRecord],
    topk: int,
) -> Tuple[str, str]:
    """
    Write:
      prefix.gltf
      prefix.bin

    Minimal glTF 2.0 with:
      - one buffer containing positions (float32)
      - one accessor for positions
      - one index accessor for line endpoints
      - two primitives:
           POINTS using positions
           LINES using positions + indices

    Note: Some viewers import POINTS/LINES differently.
    """
    unique_pts = list(dict.fromkeys(prime_points))
    index_map = {p: i for i, p in enumerate(unique_pts)}

    # Build position buffer
    pos_f32: List[float] = []
    for x, y, z in unique_pts:
        pos_f32.extend([float(x), float(y), float(z)])
    pos_bytes = struct.pack("<" + "f" * len(pos_f32), *pos_f32)

    # Build line index buffer (uint32)
    line_indices: List[int] = []
    for L in lines[:topk]:
        p0, p1 = L.endpoints_by_projection()
        line_indices.extend([index_map[p0], index_map[p1]])
    idx_bytes = struct.pack("<" + "I" * len(line_indices), *line_indices)

    # Concatenate buffers with 4-byte alignment
    def pad4(b: bytes) -> bytes:
        pad = (-len(b)) % 4
        if pad:
            return b + (b"\x00" * pad)
        return b

    pos_bytes_p = pad4(pos_bytes)
    idx_offset = len(pos_bytes_p)
    idx_bytes_p = pad4(idx_bytes)

    bin_blob = pos_bytes_p + idx_bytes_p

    bin_path = f"{prefix}.bin"
    gltf_path = f"{prefix}.gltf"

    with open(bin_path, "wb") as fb:
        fb.write(bin_blob)

    # Accessor bounds
    arr = np.array(unique_pts, dtype=np.float32)
    minv = arr.min(axis=0).tolist() if len(arr) else [0, 0, 0]
    maxv = arr.max(axis=0).tolist() if len(arr) else [0, 0, 0]

    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [
            {"uri": os.path.basename(bin_path), "byteLength": len(bin_blob)}
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(pos_bytes),
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": len(idx_bytes),
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": len(unique_pts),
                "type": "VEC3",
                "min": minv,
                "max": maxv
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(line_indices),
                "type": "SCALAR"
            }
        ],
        "meshes": [
            {
                "primitives": [
                    # POINTS
                    {
                        "attributes": {"POSITION": 0},
                        "mode": 0
                    },
                    # LINES
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 1
                    }
                ]
            }
        ],
        "nodes": [
            {"mesh": 0}
        ],
        "scenes": [
            {"nodes": [0]}
        ],
        "scene": 0
    }

    with open(gltf_path, "w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=2)

    return gltf_path, bin_path


# --------------------------
# Main pipeline
# --------------------------

def build_prime_points_from_mapping(coords: List[Vec3], is_prime: np.ndarray) -> List[Vec3]:
    prime_points: List[Vec3] = []
    for i, p in enumerate(coords, start=1):
        if is_prime[i]:
            prime_points.append(p)
    return prime_points

def main() -> None:
    parser = argparse.ArgumentParser(description="3D Ulam-style prime cube visualizer (wow edition).")
    parser.add_argument("--N", type=int, default=20000, help="Max integer to map.")
    parser.add_argument("--mapping", choices=["shell", "surface"], default="shell",
                        help="Mapping strategy: fast cubic shells or surface traversal order.")
    parser.add_argument("--min-line-primes", type=int, default=6,
                        help="Minimum primes required for a line to be considered.")
    parser.add_argument("--topk", type=int, default=12,
                        help="How many prime-dense lines to draw/export.")
    parser.add_argument("--dir-max", type=int, default=2,
                        help="Max absolute component for direction enumeration (>=1).")
    parser.add_argument("--point-size", type=float, default=6.0,
                        help="Matplotlib scatter point size.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting (useful for batch export).")
    parser.add_argument("--export-prefix", type=str, default="",
                        help="If set, exports PLY/OBJ/glTF using this prefix.")
    args = parser.parse_args()

    N = args.N
    if N < 1:
        raise SystemExit("N must be >= 1")
    if args.dir_max < 1:
        raise SystemExit("--dir-max must be >= 1")

    # Mapping
    if args.mapping == "shell":
        coords = generate_cubic_shell_mapping(N)
        mapping_name = "Cubic Shell Mapping"
    else:
        coords = generate_surface_walk_mapping(N)
        mapping_name = "Surface Traversal Mapping"
        # quantify adjacency breaks in the produced sequence
        br, st = adjacency_breaks(coords)
        ratio = (br / st) if st else 0.0
        print(f"[surface mapping] adjacency breaks: {br}/{st} ({ratio:.3%}) "
              f"(lower is more 'walk-true')")

    # Primes
    is_prime = sieve_primes(N)
    prime_points = build_prime_points_from_mapping(coords, is_prime)
    print(f"Mapped 1..{N} -> {len(coords)} positions")
    print(f"Prime points: {len(prime_points)}")

    # Directions and lines
    directions = generate_directions(args.dir_max)
    print(f"Directions (primitive, unique) with |comp|<= {args.dir_max}: {len(directions)}")

    all_lines = group_parallel_lines(
        prime_points,
        directions,
        min_primes_on_line=args.min_line_primes
    )
    all_lines = dedupe_similar_lines(all_lines)

    if all_lines:
        print("Top candidate lines:")
        for L in all_lines[:min(10, len(all_lines))]:
            p0, p1 = L.endpoints_by_projection()
            print(f"  dir={L.direction} primes={L.count} "
                  f"endpoints={p0}->{p1}")
    else:
        print("No lines met the threshold. Try lowering --min-line-primes or raising --N.")

    # Plot
    if not args.no_plot and prime_points:
        title = f"3D Ulam Prime Cube ({mapping_name}) | N={N}"
        plot_primes_and_lines(
            prime_points=prime_points,
            lines=all_lines,
            topk=args.topk,
            title=title,
            point_size=args.point_size
        )

    # Export
    if args.export_prefix:
        prefix = args.export_prefix

        ply_path = f"{prefix}.ply"
        obj_path = f"{prefix}.obj"

        export_ply(ply_path, prime_points, all_lines, args.topk)
        export_obj(obj_path, prime_points, all_lines, args.topk)

        print(f"Wrote {ply_path}")
        print(f"Wrote {obj_path}")

        # glTF
        gltf_path, bin_path = export_gltf(prefix, prime_points, all_lines, args.topk)
        print(f"Wrote {gltf_path}")
        print(f"Wrote {bin_path}")
        print("Note: glTF POINTS/LINES import support varies by tool/viewer.")

if __name__ == "__main__":
    main()
