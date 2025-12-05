#!/usr/bin/env python3
"""
3D Ulam-style "cubic shell" prime map with alignment lines.

This script:
- Assigns integers 1..N to integer lattice points in 3D
  by expanding cubic shells with Chebyshev radius r = max(|x|,|y|,|z|).
- Highlights primes.
- Draws prime-rich lines through the origin along common directions:
  axes, face diagonals, body diagonals.

No external deps beyond numpy + matplotlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


Vec3 = Tuple[int, int, int]


def sieve_primes(limit: int) -> np.ndarray:
    """Return boolean array is_prime[0..limit]."""
    if limit < 2:
        arr = np.zeros(limit + 1, dtype=bool)
        return arr

    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    # Standard sieve
    for p in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start:limit + 1:p] = False
    return is_prime


def cubic_shell_points(r: int) -> Iterable[Vec3]:
    """
    Yield all integer lattice points on the surface of the cube
    with Chebyshev radius r: max(|x|,|y|,|z|) == r.

    Deterministic order: lexicographic over (x, y, z).
    """
    if r == 0:
        yield (0, 0, 0)
        return

    rng = range(-r, r + 1)
    for x in rng:
        for y in rng:
            for z in rng:
                if max(abs(x), abs(y), abs(z)) == r:
                    yield (x, y, z)


def generate_cubic_mapping(n: int) -> List[Vec3]:
    """
    Map integers 1..n to 3D integer lattice positions by cubic shells.

    Returns list coords where coords[i-1] is position for integer i.
    """
    coords: List[Vec3] = []
    r = 0
    while len(coords) < n:
        for p in cubic_shell_points(r):
            coords.append(p)
            if len(coords) >= n:
                break
        r += 1
    return coords


def normalize_direction(d: Vec3) -> Vec3:
    """Reduce a direction vector to primitive integer step."""
    dx, dy, dz = d
    g = math.gcd(abs(dx), math.gcd(abs(dy), abs(dz)))
    if g == 0:
        raise ValueError("Zero direction vector")
    dx //= g
    dy //= g
    dz //= g
    # Canonical sign to avoid duplicates: make first nonzero positive.
    for c in (dx, dy, dz):
        if c != 0:
            if c < 0:
                dx, dy, dz = -dx, -dy, -dz
            break
    return (dx, dy, dz)


def is_on_origin_line(p: Vec3, d: Vec3) -> Tuple[bool, int]:
    """
    Check if point p lies on the integer line through origin with direction d.
    Returns (True, t) if p == t*d for some integer t.
    """
    x, y, z = p
    dx, dy, dz = d

    # Determine t using any nonzero component of d.
    t_candidates = []
    for comp_p, comp_d in ((x, dx), (y, dy), (z, dz)):
        if comp_d == 0:
            if comp_p != 0:
                return (False, 0)
        else:
            if comp_p % comp_d != 0:
                return (False, 0)
            t_candidates.append(comp_p // comp_d)

    if not t_candidates:
        # d shouldn't be zero, but handle gracefully
        return (False, 0)

    # All nonzero components must yield same t
    t = t_candidates[0]
    if any(tc != t for tc in t_candidates[1:]):
        return (False, 0)

    return (True, t)


@dataclass
class OriginLine:
    direction: Vec3
    ts: List[int]
    points: List[Vec3]

    @property
    def count(self) -> int:
        return len(self.ts)

    @property
    def t_min(self) -> int:
        return min(self.ts)

    @property
    def t_max(self) -> int:
        return max(self.ts)


def collect_prime_lines_through_origin(
    prime_points: List[Vec3],
    directions: List[Vec3],
    min_primes_on_line: int = 6,
) -> List[OriginLine]:
    """
    For each direction, collect prime points that lie on the line through origin.
    Keep lines with at least min_primes_on_line primes.
    """
    out: List[OriginLine] = []

    for d in directions:
        d = normalize_direction(d)
        ts: List[int] = []
        pts: List[Vec3] = []
        for p in prime_points:
            ok, t = is_on_origin_line(p, d)
            if ok:
                ts.append(t)
                pts.append(p)

        if len(ts) >= min_primes_on_line:
            # Sort by t for cleaner line segments
            order = np.argsort(np.array(ts))
            ts_sorted = [ts[i] for i in order]
            pts_sorted = [pts[i] for i in order]
            out.append(OriginLine(direction=d, ts=ts_sorted, points=pts_sorted))

    # Sort lines by density (count)
    out.sort(key=lambda L: L.count, reverse=True)
    return out


def plot_cubic_ulam_3d(
    N: int = 100000,
    min_primes_on_line: int = 6,
    point_size: float = 6.0,
) -> None:
    """
    Main visualization.
    """
    coords = generate_cubic_mapping(N)
    is_prime = sieve_primes(N)

    prime_points: List[Vec3] = []
    prime_ns: List[int] = []
    for i, p in enumerate(coords, start=1):
        if is_prime[i]:
            prime_points.append(p)
            prime_ns.append(i)

    prime_arr = np.array(prime_points, dtype=float)

    # Common alignment directions
    raw_dirs = [
        # axes
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        # face diagonals
        (1, 1, 0), (1, -1, 0),
        (1, 0, 1), (1, 0, -1),
        (0, 1, 1), (0, 1, -1),
        # body diagonals
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
    ]
    dirs = list({normalize_direction(d) for d in raw_dirs})

    lines = collect_prime_lines_through_origin(
        prime_points, dirs, min_primes_on_line=min_primes_on_line
    )

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter primes
    ax.scatter(
        prime_arr[:, 0], prime_arr[:, 1], prime_arr[:, 2],
        s=point_size
    )

    # Draw alignment lines
    # We'll span the line segment to cover the t-range that contains primes.
    for L in lines:
        dx, dy, dz = L.direction
        t0, t1 = L.t_min, L.t_max
        xs = [t0 * dx, t1 * dx]
        ys = [t0 * dy, t1 * dy]
        zs = [t0 * dz, t1 * dz]
        ax.plot(xs, ys, zs, linewidth=2)

    # Make the axes roughly equal scale
    if len(prime_points) > 0:
        max_extent = 0
        for x, y, z in prime_points:
            max_extent = max(max_extent, abs(x), abs(y), abs(z))
        lim = max_extent + 1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    ax.set_title(f"3D Cubic Ulam-Style Prime Map (N={N})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Optional: print a quick summary of detected origin lines
    if lines:
        print("Prime-rich origin lines found:")
        for L in lines:
            print(f"  dir={L.direction} primes={L.count} t_range=[{L.t_min}, {L.t_max}]")
    else:
        print("No origin lines met the threshold.")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Tweak these values for denser or faster plots.
    plot_cubic_ulam_3d(
        N=20000,
        min_primes_on_line=6,
        point_size=6.0,
    )
