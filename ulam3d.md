Ulam\_prime\_spiral.pdf

PDF

fren,  found this real cool article on ulam prime spirals - especially cool is the bit on mapping them as 3d cubes in a space and how they become all lined up long various axes n stuff. i was wondering if you could make me some code to produce a cubic mapping like that n the draw lines through the various alignments that form, please?

ChatGPT said:



Yep — I think I know the vibe you mean. That article briefly shows a 3D analog of the Ulam idea (a cube-ish scatter of primes) but doesn’t really spell out a canonical construction algorithm, so the most practical move is to pick a reasonable “cubic-shell mapping” and then overlay axis/diagonal alignments to see where primes cluster. The figure in the article is essentially that spirit: primes embedded in a 3D cube-like coordinate system to look for line structure. 



Ulam\_prime\_spiral



Below is a complete Python script that:



Maps integers 

1..

𝑁

1..N into concentric cubic shells around the origin:



shell radius 

𝑟

=

max

⁡

(

∣

𝑥

∣

,

∣

𝑦

∣

,

∣

𝑧

∣

)

r=max(∣x∣,∣y∣,∣z∣)



enumerates lattice points on each shell in a deterministic order



Marks primes with a sieve.



Plots the primes in 3D.



Detects and draws prime-rich lines through the origin along:



axes



face diagonals



body diagonals



This should give you a very similar “aligned along various axes” feel, and it’s a solid base if you want to extend to all parallel lines (not just those passing through the origin).



Python: 3D cubic Ulam-style mapping + alignment lines

\#!/usr/bin/env python3

"""

3D Ulam-style "cubic shell" prime map with alignment lines.



This script:

\- Assigns integers 1..N to integer lattice points in 3D

&nbsp; by expanding cubic shells with Chebyshev radius r = max(|x|,|y|,|z|).

\- Highlights primes.

\- Draws prime-rich lines through the origin along common directions:

&nbsp; axes, face diagonals, body diagonals.



No external deps beyond numpy + matplotlib.

"""



from \_\_future\_\_ import annotations



import math

from dataclasses import dataclass

from typing import Dict, Iterable, List, Tuple



import numpy as np

import matplotlib.pyplot as plt





Vec3 = Tuple\[int, int, int]





def sieve\_primes(limit: int) -> np.ndarray:

&nbsp;   """Return boolean array is\_prime\[0..limit]."""

&nbsp;   if limit < 2:

&nbsp;       arr = np.zeros(limit + 1, dtype=bool)

&nbsp;       return arr



&nbsp;   is\_prime = np.ones(limit + 1, dtype=bool)

&nbsp;   is\_prime\[0:2] = False

&nbsp;   # Standard sieve

&nbsp;   for p in range(2, int(math.isqrt(limit)) + 1):

&nbsp;       if is\_prime\[p]:

&nbsp;           start = p \* p

&nbsp;           is\_prime\[start:limit + 1:p] = False

&nbsp;   return is\_prime





def cubic\_shell\_points(r: int) -> Iterable\[Vec3]:

&nbsp;   """

&nbsp;   Yield all integer lattice points on the surface of the cube

&nbsp;   with Chebyshev radius r: max(|x|,|y|,|z|) == r.



&nbsp;   Deterministic order: lexicographic over (x, y, z).

&nbsp;   """

&nbsp;   if r == 0:

&nbsp;       yield (0, 0, 0)

&nbsp;       return



&nbsp;   rng = range(-r, r + 1)

&nbsp;   for x in rng:

&nbsp;       for y in rng:

&nbsp;           for z in rng:

&nbsp;               if max(abs(x), abs(y), abs(z)) == r:

&nbsp;                   yield (x, y, z)





def generate\_cubic\_mapping(n: int) -> List\[Vec3]:

&nbsp;   """

&nbsp;   Map integers 1..n to 3D integer lattice positions by cubic shells.



&nbsp;   Returns list coords where coords\[i-1] is position for integer i.

&nbsp;   """

&nbsp;   coords: List\[Vec3] = \[]

&nbsp;   r = 0

&nbsp;   while len(coords) < n:

&nbsp;       for p in cubic\_shell\_points(r):

&nbsp;           coords.append(p)

&nbsp;           if len(coords) >= n:

&nbsp;               break

&nbsp;       r += 1

&nbsp;   return coords





def normalize\_direction(d: Vec3) -> Vec3:

&nbsp;   """Reduce a direction vector to primitive integer step."""

&nbsp;   dx, dy, dz = d

&nbsp;   g = math.gcd(abs(dx), math.gcd(abs(dy), abs(dz)))

&nbsp;   if g == 0:

&nbsp;       raise ValueError("Zero direction vector")

&nbsp;   dx //= g

&nbsp;   dy //= g

&nbsp;   dz //= g

&nbsp;   # Canonical sign to avoid duplicates: make first nonzero positive.

&nbsp;   for c in (dx, dy, dz):

&nbsp;       if c != 0:

&nbsp;           if c < 0:

&nbsp;               dx, dy, dz = -dx, -dy, -dz

&nbsp;           break

&nbsp;   return (dx, dy, dz)





def is\_on\_origin\_line(p: Vec3, d: Vec3) -> Tuple\[bool, int]:

&nbsp;   """

&nbsp;   Check if point p lies on the integer line through origin with direction d.

&nbsp;   Returns (True, t) if p == t\*d for some integer t.

&nbsp;   """

&nbsp;   x, y, z = p

&nbsp;   dx, dy, dz = d



&nbsp;   # Determine t using any nonzero component of d.

&nbsp;   t\_candidates = \[]

&nbsp;   for comp\_p, comp\_d in ((x, dx), (y, dy), (z, dz)):

&nbsp;       if comp\_d == 0:

&nbsp;           if comp\_p != 0:

&nbsp;               return (False, 0)

&nbsp;       else:

&nbsp;           if comp\_p % comp\_d != 0:

&nbsp;               return (False, 0)

&nbsp;           t\_candidates.append(comp\_p // comp\_d)



&nbsp;   if not t\_candidates:

&nbsp;       # d shouldn't be zero, but handle gracefully

&nbsp;       return (False, 0)



&nbsp;   # All nonzero components must yield same t

&nbsp;   t = t\_candidates\[0]

&nbsp;   if any(tc != t for tc in t\_candidates\[1:]):

&nbsp;       return (False, 0)



&nbsp;   return (True, t)





@dataclass

class OriginLine:

&nbsp;   direction: Vec3

&nbsp;   ts: List\[int]

&nbsp;   points: List\[Vec3]



&nbsp;   @property

&nbsp;   def count(self) -> int:

&nbsp;       return len(self.ts)



&nbsp;   @property

&nbsp;   def t\_min(self) -> int:

&nbsp;       return min(self.ts)



&nbsp;   @property

&nbsp;   def t\_max(self) -> int:

&nbsp;       return max(self.ts)





def collect\_prime\_lines\_through\_origin(

&nbsp;   prime\_points: List\[Vec3],

&nbsp;   directions: List\[Vec3],

&nbsp;   min\_primes\_on\_line: int = 6,

) -> List\[OriginLine]:

&nbsp;   """

&nbsp;   For each direction, collect prime points that lie on the line through origin.

&nbsp;   Keep lines with at least min\_primes\_on\_line primes.

&nbsp;   """

&nbsp;   out: List\[OriginLine] = \[]



&nbsp;   for d in directions:

&nbsp;       d = normalize\_direction(d)

&nbsp;       ts: List\[int] = \[]

&nbsp;       pts: List\[Vec3] = \[]

&nbsp;       for p in prime\_points:

&nbsp;           ok, t = is\_on\_origin\_line(p, d)

&nbsp;           if ok:

&nbsp;               ts.append(t)

&nbsp;               pts.append(p)



&nbsp;       if len(ts) >= min\_primes\_on\_line:

&nbsp;           # Sort by t for cleaner line segments

&nbsp;           order = np.argsort(np.array(ts))

&nbsp;           ts\_sorted = \[ts\[i] for i in order]

&nbsp;           pts\_sorted = \[pts\[i] for i in order]

&nbsp;           out.append(OriginLine(direction=d, ts=ts\_sorted, points=pts\_sorted))



&nbsp;   # Sort lines by density (count)

&nbsp;   out.sort(key=lambda L: L.count, reverse=True)

&nbsp;   return out





def plot\_cubic\_ulam\_3d(

&nbsp;   N: int = 20000,

&nbsp;   min\_primes\_on\_line: int = 6,

&nbsp;   point\_size: float = 6.0,

) -> None:

&nbsp;   """

&nbsp;   Main visualization.

&nbsp;   """

&nbsp;   coords = generate\_cubic\_mapping(N)

&nbsp;   is\_prime = sieve\_primes(N)



&nbsp;   prime\_points: List\[Vec3] = \[]

&nbsp;   prime\_ns: List\[int] = \[]

&nbsp;   for i, p in enumerate(coords, start=1):

&nbsp;       if is\_prime\[i]:

&nbsp;           prime\_points.append(p)

&nbsp;           prime\_ns.append(i)



&nbsp;   prime\_arr = np.array(prime\_points, dtype=float)



&nbsp;   # Common alignment directions

&nbsp;   raw\_dirs = \[

&nbsp;       # axes

&nbsp;       (1, 0, 0), (0, 1, 0), (0, 0, 1),

&nbsp;       # face diagonals

&nbsp;       (1, 1, 0), (1, -1, 0),

&nbsp;       (1, 0, 1), (1, 0, -1),

&nbsp;       (0, 1, 1), (0, 1, -1),

&nbsp;       # body diagonals

&nbsp;       (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),

&nbsp;   ]

&nbsp;   dirs = list({normalize\_direction(d) for d in raw\_dirs})



&nbsp;   lines = collect\_prime\_lines\_through\_origin(

&nbsp;       prime\_points, dirs, min\_primes\_on\_line=min\_primes\_on\_line

&nbsp;   )



&nbsp;   # --- Plot ---

&nbsp;   fig = plt.figure(figsize=(10, 8))

&nbsp;   ax = fig.add\_subplot(111, projection="3d")



&nbsp;   # Scatter primes

&nbsp;   ax.scatter(

&nbsp;       prime\_arr\[:, 0], prime\_arr\[:, 1], prime\_arr\[:, 2],

&nbsp;       s=point\_size

&nbsp;   )



&nbsp;   # Draw alignment lines

&nbsp;   # We'll span the line segment to cover the t-range that contains primes.

&nbsp;   for L in lines:

&nbsp;       dx, dy, dz = L.direction

&nbsp;       t0, t1 = L.t\_min, L.t\_max

&nbsp;       xs = \[t0 \* dx, t1 \* dx]

&nbsp;       ys = \[t0 \* dy, t1 \* dy]

&nbsp;       zs = \[t0 \* dz, t1 \* dz]

&nbsp;       ax.plot(xs, ys, zs, linewidth=2)



&nbsp;   # Make the axes roughly equal scale

&nbsp;   if len(prime\_points) > 0:

&nbsp;       max\_extent = 0

&nbsp;       for x, y, z in prime\_points:

&nbsp;           max\_extent = max(max\_extent, abs(x), abs(y), abs(z))

&nbsp;       lim = max\_extent + 1

&nbsp;       ax.set\_xlim(-lim, lim)

&nbsp;       ax.set\_ylim(-lim, lim)

&nbsp;       ax.set\_zlim(-lim, lim)



&nbsp;   ax.set\_title(f"3D Cubic Ulam-Style Prime Map (N={N})")

&nbsp;   ax.set\_xlabel("x")

&nbsp;   ax.set\_ylabel("y")

&nbsp;   ax.set\_zlabel("z")



&nbsp;   # Optional: print a quick summary of detected origin lines

&nbsp;   if lines:

&nbsp;       print("Prime-rich origin lines found:")

&nbsp;       for L in lines:

&nbsp;           print(f"  dir={L.direction} primes={L.count} t\_range=\[{L.t\_min}, {L.t\_max}]")

&nbsp;   else:

&nbsp;       print("No origin lines met the threshold.")



&nbsp;   plt.tight\_layout()

&nbsp;   plt.show()





if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   # Tweak these values for denser or faster plots.

&nbsp;   plot\_cubic\_ulam\_3d(

&nbsp;       N=20000,

&nbsp;       min\_primes\_on\_line=6,

&nbsp;       point\_size=6.0,

&nbsp;   )



What this gives you (and how to crank it up)



N ~ 20,000 is a nice sweet spot for quick rendering.



Increase to 100,000 if your machine is happy.



Lower min\_primes\_on\_line to see more lines.



Raise it to highlight only the most striking alignments.



Easy next upgrades if you want more “wow”



If you want, I can extend this to:



All parallel lines, not just through the origin:



group prime points by line keys for each direction



draw the top k prime-dense lines anywhere in the cube



A true 3D “walk” spiral path (like the 2D square spiral but on cube surfaces).



Export to PLY/OBJ/GLTF so you can spin it in Blender or a WebGL viewer.



But this version should already give you that satisfying “primes snap into axes and diagonals in 3D space” feeling you’re after.



Sources

You said:

ok fren; let's do the easy next upgrades for more “wow” and extend this to:

\- All parallel lines, not just through the origin:

\- group prime points by line keys for each direction

\- draw the top k prime-dense lines anywhere in the cube

\- A true 3D “walk” spiral path (like the 2D square spiral but on cube surfaces).

\- Export to PLY/OBJ/GLTF so you can spin it in Blender or a WebGL viewer.

ChatGPT said:

Thought for 1m 29s



Awesome — let’s crank the wow knob. 😄



Below is a single, complete, runnable Python script that adds all the upgrades you listed:



What it does



Two 3D mappings



Cubic-shell mapping (fast, deterministic)



3D surface “walk-ish” spiral order that traverses cube surfaces in an ordered sweep (it’s designed to feel like a spiral-on-surfaces; it also prints an adjacency-break metric so you can judge how “walk-true” it is for your N)



All parallel lines, not just through origin:



For each direction, it builds two integer invariants using a perpendicular integer basis.



Groups primes by line key.



Ranks by prime density.



Draw top-k prime-dense lines anywhere in the cube



Export



PLY (point cloud + optional edges)



OBJ (vertices + l line elements)



glTF (.gltf + .bin) minimal points+lines (works in many viewers; Blender import of POINTS/LINES can be a little implementation-dependent, but this is standards-valid)



The code

\#!/usr/bin/env python3

"""

3D Ulam-style prime visualizer with:

\- Cubic-shell mapping

\- Surface "walk-ish" cube traversal order

\- Parallel-line detection for many directions

\- Top-K prime-dense line rendering

\- Export to PLY / OBJ / glTF



Dependencies: numpy, matplotlib



Usage examples:

&nbsp; python ulam3d\_wow.py --N 50000 --mapping shell --topk 12 --export-prefix primes\_shell

&nbsp; python ulam3d\_wow.py --N 50000 --mapping surface --topk 16 --dir-max 2 --export-prefix primes\_surface

"""



from \_\_future\_\_ import annotations



import argparse

import json

import math

import os

import struct

from dataclasses import dataclass

from typing import Dict, Iterable, List, Tuple, Optional, Set



import numpy as np

import matplotlib.pyplot as plt



Vec3 = Tuple\[int, int, int]





\# --------------------------

\# Primes

\# --------------------------



def sieve\_primes(limit: int) -> np.ndarray:

&nbsp;   """Boolean array is\_prime\[0..limit]."""

&nbsp;   if limit < 2:

&nbsp;       return np.zeros(limit + 1, dtype=bool)



&nbsp;   is\_prime = np.ones(limit + 1, dtype=bool)

&nbsp;   is\_prime\[0:2] = False

&nbsp;   for p in range(2, int(math.isqrt(limit)) + 1):

&nbsp;       if is\_prime\[p]:

&nbsp;           is\_prime\[p \* p:limit + 1:p] = False

&nbsp;   return is\_prime





\# --------------------------

\# Geometry helpers

\# --------------------------



def cross(a: Vec3, b: Vec3) -> Vec3:

&nbsp;   ax, ay, az = a

&nbsp;   bx, by, bz = b

&nbsp;   return (ay \* bz - az \* by, az \* bx - ax \* bz, ax \* by - ay \* bx)



def dot(a: Vec3, b: Vec3) -> int:

&nbsp;   return a\[0]\*b\[0] + a\[1]\*b\[1] + a\[2]\*b\[2]



def normalize\_direction(d: Vec3) -> Vec3:

&nbsp;   """Reduce to primitive integer step; canonical sign."""

&nbsp;   dx, dy, dz = d

&nbsp;   g = math.gcd(abs(dx), math.gcd(abs(dy), abs(dz)))

&nbsp;   if g == 0:

&nbsp;       raise ValueError("Zero direction vector")

&nbsp;   dx //= g

&nbsp;   dy //= g

&nbsp;   dz //= g



&nbsp;   # Canonical sign: first nonzero positive

&nbsp;   for c in (dx, dy, dz):

&nbsp;       if c != 0:

&nbsp;           if c < 0:

&nbsp;               dx, dy, dz = -dx, -dy, -dz

&nbsp;           break

&nbsp;   return (dx, dy, dz)



def perpendicular\_basis(d: Vec3) -> Tuple\[Vec3, Vec3]:

&nbsp;   """

&nbsp;   Construct two integer vectors a, b such that:

&nbsp;     a·d = 0

&nbsp;     b·d = 0

&nbsp;   Using:

&nbsp;     a = d × e

&nbsp;     b = d × a

&nbsp;   """

&nbsp;   dx, dy, dz = d

&nbsp;   # choose e not parallel to d

&nbsp;   if (dx, dy, dz) != (1, 0, 0) and not (dy == 0 and dz == 0):

&nbsp;       e = (1, 0, 0)

&nbsp;   elif not (dx == 0 and dz == 0):

&nbsp;       e = (0, 1, 0)

&nbsp;   else:

&nbsp;       e = (0, 0, 1)



&nbsp;   a = cross(d, e)

&nbsp;   if a == (0, 0, 0):

&nbsp;       # fallback, should be rare

&nbsp;       e = (0, 1, 0) if e != (0, 1, 0) else (0, 0, 1)

&nbsp;       a = cross(d, e)



&nbsp;   b = cross(d, a)

&nbsp;   if b == (0, 0, 0):

&nbsp;       # extremely unlikely, but guard

&nbsp;       raise RuntimeError("Failed to build perpendicular basis")



&nbsp;   return a, b



def chebyshev\_radius(p: Vec3) -> int:

&nbsp;   return max(abs(p\[0]), abs(p\[1]), abs(p\[2]))





\# --------------------------

\# Mapping 1: Cubic shells (fast)

\# --------------------------



def cubic\_shell\_points(r: int) -> Iterable\[Vec3]:

&nbsp;   """Points on surface max(|x|,|y|,|z|)=r, lexicographic order."""

&nbsp;   if r == 0:

&nbsp;       yield (0, 0, 0)

&nbsp;       return

&nbsp;   rng = range(-r, r + 1)

&nbsp;   for x in rng:

&nbsp;       for y in rng:

&nbsp;           for z in rng:

&nbsp;               if max(abs(x), abs(y), abs(z)) == r:

&nbsp;                   yield (x, y, z)



def generate\_cubic\_shell\_mapping(n: int) -> List\[Vec3]:

&nbsp;   coords: List\[Vec3] = \[]

&nbsp;   r = 0

&nbsp;   while len(coords) < n:

&nbsp;       for p in cubic\_shell\_points(r):

&nbsp;           coords.append(p)

&nbsp;           if len(coords) >= n:

&nbsp;               break

&nbsp;       r += 1

&nbsp;   return coords





\# --------------------------

\# Mapping 2: Surface "walk-ish" cube traversal

\# --------------------------



def perimeter\_square\_xy(r: int, z: int) -> List\[Vec3]:

&nbsp;   """

&nbsp;   Perimeter of square in x-y with Chebyshev radius r at fixed z:

&nbsp;     max(|x|,|y|)=r

&nbsp;   Ordered loop starting at (r,-r,z) going CCW.

&nbsp;   """

&nbsp;   pts: List\[Vec3] = \[]

&nbsp;   # right edge: x=r, y from -r..r

&nbsp;   for y in range(-r, r + 1):

&nbsp;       pts.append((r, y, z))

&nbsp;   # top edge: y=r, x from r-1..-r

&nbsp;   for x in range(r - 1, -r - 1, -1):

&nbsp;       pts.append((x, r, z))

&nbsp;   # left edge: x=-r, y from r-1..-r

&nbsp;   for y in range(r - 1, -r - 1, -1):

&nbsp;       pts.append((-r, y, z))

&nbsp;   # bottom edge: y=-r, x from -r+1..r-1

&nbsp;   for x in range(-r + 1, r):

&nbsp;       pts.append((x, -r, z))

&nbsp;   # This forms a closed loop with repeated corners in standard perimeter builds,

&nbsp;   # but our sequence avoids duplicates by construction.

&nbsp;   return pts



def face\_square\_xy(r: int, z: int, reverse: bool = False) -> List\[Vec3]:

&nbsp;   """

&nbsp;   Full square face at fixed z:

&nbsp;     |x|<=r, |y|<=r

&nbsp;   Serpentine (row-wise) order to encourage adjacency.

&nbsp;   """

&nbsp;   pts: List\[Vec3] = \[]

&nbsp;   ys = list(range(-r, r + 1))

&nbsp;   if reverse:

&nbsp;       ys = ys\[::-1]



&nbsp;   for row\_idx, y in enumerate(ys):

&nbsp;       xs = list(range(-r, r + 1))

&nbsp;       # snake

&nbsp;       if (row\_idx % 2) == 1:

&nbsp;           xs.reverse()

&nbsp;       for x in xs:

&nbsp;           pts.append((x, y, z))

&nbsp;   return pts



def shell\_surface\_path(r: int) -> List\[Vec3]:

&nbsp;   """

&nbsp;   Ordered traversal of the cube surface at radius r.

&nbsp;   Designed to feel like a continuous sweep across faces and slice perimeters.



&nbsp;   Order:

&nbsp;     1) Bottom face z=-r (full square)

&nbsp;     2) Middle z slices: perimeter loops

&nbsp;     3) Top face z=+r (full square, reversed serpentine)

&nbsp;   """

&nbsp;   if r == 0:

&nbsp;       return \[(0, 0, 0)]



&nbsp;   out: List\[Vec3] = \[]



&nbsp;   # Bottom face

&nbsp;   out.extend(face\_square\_xy(r, z=-r, reverse=False))



&nbsp;   # Middle slices perimeters

&nbsp;   for z in range(-r + 1, r):

&nbsp;       out.extend(perimeter\_square\_xy(r, z))



&nbsp;   # Top face

&nbsp;   out.extend(face\_square\_xy(r, z=+r, reverse=True))



&nbsp;   # Deduplicate while preserving order (should be unnecessary, but safe)

&nbsp;   seen: Set\[Vec3] = set()

&nbsp;   deduped: List\[Vec3] = \[]

&nbsp;   for p in out:

&nbsp;       if p not in seen and chebyshev\_radius(p) == r:

&nbsp;           seen.add(p)

&nbsp;           deduped.append(p)

&nbsp;   return deduped



def generate\_surface\_walk\_mapping(n: int) -> List\[Vec3]:

&nbsp;   """

&nbsp;   Generate integer->position mapping by concatenating ordered cube-surface paths.



&nbsp;   Prints an adjacency-break metric in main() to quantify "walk-ness".

&nbsp;   """

&nbsp;   coords: List\[Vec3] = \[]

&nbsp;   r = 0

&nbsp;   while len(coords) < n:

&nbsp;       path = shell\_surface\_path(r)

&nbsp;       for p in path:

&nbsp;           coords.append(p)

&nbsp;           if len(coords) >= n:

&nbsp;               break

&nbsp;       r += 1

&nbsp;   return coords



def manhattan(a: Vec3, b: Vec3) -> int:

&nbsp;   return abs(a\[0]-b\[0]) + abs(a\[1]-b\[1]) + abs(a\[2]-b\[2])



def adjacency\_breaks(path: List\[Vec3]) -> Tuple\[int, int]:

&nbsp;   """Return (#breaks, #steps). A break is a step with manhattan != 1."""

&nbsp;   if len(path) < 2:

&nbsp;       return (0, 0)

&nbsp;   breaks = 0

&nbsp;   steps = len(path) - 1

&nbsp;   for i in range(1, len(path)):

&nbsp;       if manhattan(path\[i-1], path\[i]) != 1:

&nbsp;           breaks += 1

&nbsp;   return breaks, steps





\# --------------------------

\# Parallel line detection

\# --------------------------



@dataclass(frozen=True)

class LineKey:

&nbsp;   inv1: int

&nbsp;   inv2: int



@dataclass

class LineRecord:

&nbsp;   direction: Vec3

&nbsp;   key: LineKey

&nbsp;   points: List\[Vec3]  # prime points on this line



&nbsp;   @property

&nbsp;   def count(self) -> int:

&nbsp;       return len(self.points)



&nbsp;   def endpoints\_by\_projection(self) -> Tuple\[Vec3, Vec3]:

&nbsp;       """Pick endpoints among the prime points using projection along direction."""

&nbsp;       d = self.direction

&nbsp;       projs = \[dot(p, d) for p in self.points]

&nbsp;       i\_min = int(np.argmin(np.array(projs)))

&nbsp;       i\_max = int(np.argmax(np.array(projs)))

&nbsp;       return self.points\[i\_min], self.points\[i\_max]



def generate\_directions(max\_comp: int = 2) -> List\[Vec3]:

&nbsp;   """

&nbsp;   Generate unique primitive directions with components in \[-max\_comp..max\_comp],

&nbsp;   excluding zero vector.

&nbsp;   """

&nbsp;   dirs: Set\[Vec3] = set()

&nbsp;   for dx in range(-max\_comp, max\_comp + 1):

&nbsp;       for dy in range(-max\_comp, max\_comp + 1):

&nbsp;           for dz in range(-max\_comp, max\_comp + 1):

&nbsp;               if dx == 0 and dy == 0 and dz == 0:

&nbsp;                   continue

&nbsp;               d = normalize\_direction((dx, dy, dz))

&nbsp;               dirs.add(d)

&nbsp;   return sorted(dirs)



def group\_parallel\_lines(

&nbsp;   prime\_points: List\[Vec3],

&nbsp;   directions: List\[Vec3],

&nbsp;   min\_primes\_on\_line: int = 6,

) -> List\[LineRecord]:

&nbsp;   """

&nbsp;   For each direction d:

&nbsp;     - build perpendicular basis (a,b)

&nbsp;     - invariant pair key = (p·a, p·b)

&nbsp;   Points with same key lie on same line parallel to d.



&nbsp;   Returns all lines across all directions with >= min\_primes\_on\_line.

&nbsp;   """

&nbsp;   all\_lines: List\[LineRecord] = \[]



&nbsp;   for d in directions:

&nbsp;       a, b = perpendicular\_basis(d)

&nbsp;       buckets: Dict\[LineKey, List\[Vec3]] = {}



&nbsp;       for p in prime\_points:

&nbsp;           key = LineKey(dot(p, a), dot(p, b))

&nbsp;           buckets.setdefault(key, \[]).append(p)



&nbsp;       for key, pts in buckets.items():

&nbsp;           if len(pts) >= min\_primes\_on\_line:

&nbsp;               # Sort points along direction for nicer endpoints

&nbsp;               pts\_sorted = sorted(pts, key=lambda q: dot(q, d))

&nbsp;               all\_lines.append(LineRecord(direction=d, key=key, points=pts\_sorted))



&nbsp;   # Sort by count descending

&nbsp;   all\_lines.sort(key=lambda L: L.count, reverse=True)

&nbsp;   return all\_lines



def dedupe\_similar\_lines(lines: List\[LineRecord]) -> List\[LineRecord]:

&nbsp;   """

&nbsp;   Very light dedupe:

&nbsp;   It's possible for the same geometric line to appear under different

&nbsp;   directions if those directions are collinear (shouldn't happen)

&nbsp;   or if direction generation includes duplicates (we normalize, so rare).

&nbsp;   We'll dedupe by (direction, key).

&nbsp;   """

&nbsp;   seen: Set\[Tuple\[Vec3, LineKey]] = set()

&nbsp;   out: List\[LineRecord] = \[]

&nbsp;   for L in lines:

&nbsp;       k = (L.direction, L.key)

&nbsp;       if k not in seen:

&nbsp;           seen.add(k)

&nbsp;           out.append(L)

&nbsp;   return out





\# --------------------------

\# Plotting

\# --------------------------



def plot\_primes\_and\_lines(

&nbsp;   prime\_points: List\[Vec3],

&nbsp;   lines: List\[LineRecord],

&nbsp;   topk: int,

&nbsp;   title: str,

&nbsp;   point\_size: float = 6.0,

) -> None:

&nbsp;   prime\_arr = np.array(prime\_points, dtype=float)



&nbsp;   fig = plt.figure(figsize=(11, 9))

&nbsp;   ax = fig.add\_subplot(111, projection="3d")



&nbsp;   ax.scatter(prime\_arr\[:, 0], prime\_arr\[:, 1], prime\_arr\[:, 2], s=point\_size)



&nbsp;   # draw top-k lines

&nbsp;   for L in lines\[:topk]:

&nbsp;       p0, p1 = L.endpoints\_by\_projection()

&nbsp;       xs = \[p0\[0], p1\[0]]

&nbsp;       ys = \[p0\[1], p1\[1]]

&nbsp;       zs = \[p0\[2], p1\[2]]

&nbsp;       ax.plot(xs, ys, zs, linewidth=2)



&nbsp;   # equal-ish bounds

&nbsp;   max\_extent = 0

&nbsp;   for x, y, z in prime\_points:

&nbsp;       max\_extent = max(max\_extent, abs(x), abs(y), abs(z))

&nbsp;   lim = max\_extent + 1

&nbsp;   ax.set\_xlim(-lim, lim)

&nbsp;   ax.set\_ylim(-lim, lim)

&nbsp;   ax.set\_zlim(-lim, lim)



&nbsp;   ax.set\_title(title)

&nbsp;   ax.set\_xlabel("x")

&nbsp;   ax.set\_ylabel("y")

&nbsp;   ax.set\_zlabel("z")



&nbsp;   plt.tight\_layout()

&nbsp;   plt.show()





\# --------------------------

\# Export: OBJ / PLY

\# --------------------------



def export\_obj(

&nbsp;   path: str,

&nbsp;   prime\_points: List\[Vec3],

&nbsp;   lines: List\[LineRecord],

&nbsp;   topk: int,

) -> None:

&nbsp;   """

&nbsp;   Export as OBJ:

&nbsp;     - vertices for all prime points

&nbsp;     - line elements 'l i j' for top-k lines (using endpoints)

&nbsp;   """

&nbsp;   # unique vertices

&nbsp;   unique\_pts = list(dict.fromkeys(prime\_points))  # preserves order

&nbsp;   index\_map = {p: i + 1 for i, p in enumerate(unique\_pts)}  # 1-based



&nbsp;   with open(path, "w", encoding="utf-8") as f:

&nbsp;       f.write("# 3D Ulam prime map (points + top lines)\\n")

&nbsp;       for x, y, z in unique\_pts:

&nbsp;           f.write(f"v {x} {y} {z}\\n")



&nbsp;       f.write("\\n# Lines (top-k prime-dense)\\n")

&nbsp;       for L in lines\[:topk]:

&nbsp;           p0, p1 = L.endpoints\_by\_projection()

&nbsp;           i0 = index\_map\[p0]

&nbsp;           i1 = index\_map\[p1]

&nbsp;           f.write(f"l {i0} {i1}\\n")



def export\_ply(

&nbsp;   path: str,

&nbsp;   prime\_points: List\[Vec3],

&nbsp;   lines: List\[LineRecord],

&nbsp;   topk: int,

) -> None:

&nbsp;   """

&nbsp;   Export ASCII PLY with:

&nbsp;     - vertex list (prime points)

&nbsp;     - edge list (top-k line segments)

&nbsp;   """

&nbsp;   unique\_pts = list(dict.fromkeys(prime\_points))

&nbsp;   index\_map = {p: i for i, p in enumerate(unique\_pts)}  # 0-based



&nbsp;   edges: List\[Tuple\[int, int]] = \[]

&nbsp;   for L in lines\[:topk]:

&nbsp;       p0, p1 = L.endpoints\_by\_projection()

&nbsp;       edges.append((index\_map\[p0], index\_map\[p1]))



&nbsp;   with open(path, "w", encoding="utf-8") as f:

&nbsp;       f.write("ply\\n")

&nbsp;       f.write("format ascii 1.0\\n")

&nbsp;       f.write(f"element vertex {len(unique\_pts)}\\n")

&nbsp;       f.write("property float x\\nproperty float y\\nproperty float z\\n")

&nbsp;       f.write(f"element edge {len(edges)}\\n")

&nbsp;       f.write("property int vertex1\\nproperty int vertex2\\n")

&nbsp;       f.write("end\_header\\n")



&nbsp;       for x, y, z in unique\_pts:

&nbsp;           f.write(f"{x} {y} {z}\\n")



&nbsp;       for i0, i1 in edges:

&nbsp;           f.write(f"{i0} {i1}\\n")





\# --------------------------

\# Export: minimal glTF (points + lines)

\# --------------------------



def export\_gltf(

&nbsp;   prefix: str,

&nbsp;   prime\_points: List\[Vec3],

&nbsp;   lines: List\[LineRecord],

&nbsp;   topk: int,

) -> Tuple\[str, str]:

&nbsp;   """

&nbsp;   Write:

&nbsp;     prefix.gltf

&nbsp;     prefix.bin



&nbsp;   Minimal glTF 2.0 with:

&nbsp;     - one buffer containing positions (float32)

&nbsp;     - one accessor for positions

&nbsp;     - one index accessor for line endpoints

&nbsp;     - two primitives:

&nbsp;          POINTS using positions

&nbsp;          LINES using positions + indices



&nbsp;   Note: Some viewers import POINTS/LINES differently.

&nbsp;   """

&nbsp;   unique\_pts = list(dict.fromkeys(prime\_points))

&nbsp;   index\_map = {p: i for i, p in enumerate(unique\_pts)}



&nbsp;   # Build position buffer

&nbsp;   pos\_f32: List\[float] = \[]

&nbsp;   for x, y, z in unique\_pts:

&nbsp;       pos\_f32.extend(\[float(x), float(y), float(z)])

&nbsp;   pos\_bytes = struct.pack("<" + "f" \* len(pos\_f32), \*pos\_f32)



&nbsp;   # Build line index buffer (uint32)

&nbsp;   line\_indices: List\[int] = \[]

&nbsp;   for L in lines\[:topk]:

&nbsp;       p0, p1 = L.endpoints\_by\_projection()

&nbsp;       line\_indices.extend(\[index\_map\[p0], index\_map\[p1]])

&nbsp;   idx\_bytes = struct.pack("<" + "I" \* len(line\_indices), \*line\_indices)



&nbsp;   # Concatenate buffers with 4-byte alignment

&nbsp;   def pad4(b: bytes) -> bytes:

&nbsp;       pad = (-len(b)) % 4

&nbsp;       if pad:

&nbsp;           return b + (b"\\x00" \* pad)

&nbsp;       return b



&nbsp;   pos\_bytes\_p = pad4(pos\_bytes)

&nbsp;   idx\_offset = len(pos\_bytes\_p)

&nbsp;   idx\_bytes\_p = pad4(idx\_bytes)



&nbsp;   bin\_blob = pos\_bytes\_p + idx\_bytes\_p



&nbsp;   bin\_path = f"{prefix}.bin"

&nbsp;   gltf\_path = f"{prefix}.gltf"



&nbsp;   with open(bin\_path, "wb") as fb:

&nbsp;       fb.write(bin\_blob)



&nbsp;   # Accessor bounds

&nbsp;   arr = np.array(unique\_pts, dtype=np.float32)

&nbsp;   minv = arr.min(axis=0).tolist() if len(arr) else \[0, 0, 0]

&nbsp;   maxv = arr.max(axis=0).tolist() if len(arr) else \[0, 0, 0]



&nbsp;   gltf = {

&nbsp;       "asset": {"version": "2.0"},

&nbsp;       "buffers": \[

&nbsp;           {"uri": os.path.basename(bin\_path), "byteLength": len(bin\_blob)}

&nbsp;       ],

&nbsp;       "bufferViews": \[

&nbsp;           {

&nbsp;               "buffer": 0,

&nbsp;               "byteOffset": 0,

&nbsp;               "byteLength": len(pos\_bytes),

&nbsp;               "target": 34962  # ARRAY\_BUFFER

&nbsp;           },

&nbsp;           {

&nbsp;               "buffer": 0,

&nbsp;               "byteOffset": idx\_offset,

&nbsp;               "byteLength": len(idx\_bytes),

&nbsp;               "target": 34963  # ELEMENT\_ARRAY\_BUFFER

&nbsp;           }

&nbsp;       ],

&nbsp;       "accessors": \[

&nbsp;           {

&nbsp;               "bufferView": 0,

&nbsp;               "byteOffset": 0,

&nbsp;               "componentType": 5126,  # FLOAT

&nbsp;               "count": len(unique\_pts),

&nbsp;               "type": "VEC3",

&nbsp;               "min": minv,

&nbsp;               "max": maxv

&nbsp;           },

&nbsp;           {

&nbsp;               "bufferView": 1,

&nbsp;               "byteOffset": 0,

&nbsp;               "componentType": 5125,  # UNSIGNED\_INT

&nbsp;               "count": len(line\_indices),

&nbsp;               "type": "SCALAR"

&nbsp;           }

&nbsp;       ],

&nbsp;       "meshes": \[

&nbsp;           {

&nbsp;               "primitives": \[

&nbsp;                   # POINTS

&nbsp;                   {

&nbsp;                       "attributes": {"POSITION": 0},

&nbsp;                       "mode": 0

&nbsp;                   },

&nbsp;                   # LINES

&nbsp;                   {

&nbsp;                       "attributes": {"POSITION": 0},

&nbsp;                       "indices": 1,

&nbsp;                       "mode": 1

&nbsp;                   }

&nbsp;               ]

&nbsp;           }

&nbsp;       ],

&nbsp;       "nodes": \[

&nbsp;           {"mesh": 0}

&nbsp;       ],

&nbsp;       "scenes": \[

&nbsp;           {"nodes": \[0]}

&nbsp;       ],

&nbsp;       "scene": 0

&nbsp;   }



&nbsp;   with open(gltf\_path, "w", encoding="utf-8") as f:

&nbsp;       json.dump(gltf, f, indent=2)



&nbsp;   return gltf\_path, bin\_path





\# --------------------------

\# Main pipeline

\# --------------------------



def build\_prime\_points\_from\_mapping(coords: List\[Vec3], is\_prime: np.ndarray) -> List\[Vec3]:

&nbsp;   prime\_points: List\[Vec3] = \[]

&nbsp;   for i, p in enumerate(coords, start=1):

&nbsp;       if is\_prime\[i]:

&nbsp;           prime\_points.append(p)

&nbsp;   return prime\_points



def main() -> None:

&nbsp;   parser = argparse.ArgumentParser(description="3D Ulam-style prime cube visualizer (wow edition).")

&nbsp;   parser.add\_argument("--N", type=int, default=20000, help="Max integer to map.")

&nbsp;   parser.add\_argument("--mapping", choices=\["shell", "surface"], default="shell",

&nbsp;                       help="Mapping strategy: fast cubic shells or surface traversal order.")

&nbsp;   parser.add\_argument("--min-line-primes", type=int, default=6,

&nbsp;                       help="Minimum primes required for a line to be considered.")

&nbsp;   parser.add\_argument("--topk", type=int, default=12,

&nbsp;                       help="How many prime-dense lines to draw/export.")

&nbsp;   parser.add\_argument("--dir-max", type=int, default=2,

&nbsp;                       help="Max absolute component for direction enumeration (>=1).")

&nbsp;   parser.add\_argument("--point-size", type=float, default=6.0,

&nbsp;                       help="Matplotlib scatter point size.")

&nbsp;   parser.add\_argument("--no-plot", action="store\_true",

&nbsp;                       help="Disable plotting (useful for batch export).")

&nbsp;   parser.add\_argument("--export-prefix", type=str, default="",

&nbsp;                       help="If set, exports PLY/OBJ/glTF using this prefix.")

&nbsp;   args = parser.parse\_args()



&nbsp;   N = args.N

&nbsp;   if N < 1:

&nbsp;       raise SystemExit("N must be >= 1")

&nbsp;   if args.dir\_max < 1:

&nbsp;       raise SystemExit("--dir-max must be >= 1")



&nbsp;   # Mapping

&nbsp;   if args.mapping == "shell":

&nbsp;       coords = generate\_cubic\_shell\_mapping(N)

&nbsp;       mapping\_name = "Cubic Shell Mapping"

&nbsp;   else:

&nbsp;       coords = generate\_surface\_walk\_mapping(N)

&nbsp;       mapping\_name = "Surface Traversal Mapping"

&nbsp;       # quantify adjacency breaks in the produced sequence

&nbsp;       br, st = adjacency\_breaks(coords)

&nbsp;       ratio = (br / st) if st else 0.0

&nbsp;       print(f"\[surface mapping] adjacency breaks: {br}/{st} ({ratio:.3%}) "

&nbsp;             f"(lower is more 'walk-true')")



&nbsp;   # Primes

&nbsp;   is\_prime = sieve\_primes(N)

&nbsp;   prime\_points = build\_prime\_points\_from\_mapping(coords, is\_prime)

&nbsp;   print(f"Mapped 1..{N} -> {len(coords)} positions")

&nbsp;   print(f"Prime points: {len(prime\_points)}")



&nbsp;   # Directions and lines

&nbsp;   directions = generate\_directions(args.dir\_max)

&nbsp;   print(f"Directions (primitive, unique) with |comp|<= {args.dir\_max}: {len(directions)}")



&nbsp;   all\_lines = group\_parallel\_lines(

&nbsp;       prime\_points,

&nbsp;       directions,

&nbsp;       min\_primes\_on\_line=args.min\_line\_primes

&nbsp;   )

&nbsp;   all\_lines = dedupe\_similar\_lines(all\_lines)



&nbsp;   if all\_lines:

&nbsp;       print("Top candidate lines:")

&nbsp;       for L in all\_lines\[:min(10, len(all\_lines))]:

&nbsp;           p0, p1 = L.endpoints\_by\_projection()

&nbsp;           print(f"  dir={L.direction} primes={L.count} "

&nbsp;                 f"endpoints={p0}->{p1}")

&nbsp;   else:

&nbsp;       print("No lines met the threshold. Try lowering --min-line-primes or raising --N.")



&nbsp;   # Plot

&nbsp;   if not args.no\_plot and prime\_points:

&nbsp;       title = f"3D Ulam Prime Cube ({mapping\_name}) | N={N}"

&nbsp;       plot\_primes\_and\_lines(

&nbsp;           prime\_points=prime\_points,

&nbsp;           lines=all\_lines,

&nbsp;           topk=args.topk,

&nbsp;           title=title,

&nbsp;           point\_size=args.point\_size

&nbsp;       )



&nbsp;   # Export

&nbsp;   if args.export\_prefix:

&nbsp;       prefix = args.export\_prefix



&nbsp;       ply\_path = f"{prefix}.ply"

&nbsp;       obj\_path = f"{prefix}.obj"



&nbsp;       export\_ply(ply\_path, prime\_points, all\_lines, args.topk)

&nbsp;       export\_obj(obj\_path, prime\_points, all\_lines, args.topk)



&nbsp;       print(f"Wrote {ply\_path}")

&nbsp;       print(f"Wrote {obj\_path}")



&nbsp;       # glTF

&nbsp;       gltf\_path, bin\_path = export\_gltf(prefix, prime\_points, all\_lines, args.topk)

&nbsp;       print(f"Wrote {gltf\_path}")

&nbsp;       print(f"Wrote {bin\_path}")

&nbsp;       print("Note: glTF POINTS/LINES import support varies by tool/viewer.")



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()



How to get the best-looking results fast



Try these:



1\) Big, dramatic line structure

python ulam3d\_wow.py --N 100000 --mapping shell --dir-max 2 --min-line-primes 7 --topk 20 --export-prefix ulam3d\_shell\_100k



2\) Surface traversal flavor

python ulam3d\_wow.py --N 60000 --mapping surface --dir-max 2 --min-line-primes 6 --topk 16 --export-prefix ulam3d\_surface\_60k





If the adjacency-break ratio bugs you aesthetically, just stick with shell for the cleanest algebraic structure visuals — the line-detection layer is independent of the “walkness”.



Why the parallel-line key works (quick intuition)



For a direction 

𝑑

d, we build two integer vectors 

𝑎

,

𝑏

a,b with 

𝑎

⋅

𝑑

=

0

a⋅d=0 and 

𝑏

⋅

𝑑

=

0

b⋅d=0.

Then for any point 

𝑝

p on a line parallel to 

𝑑

d,



(

𝑝

\+

𝑡

𝑑

)

⋅

𝑎

=

𝑝

⋅

𝑎

,

(

𝑝

\+

𝑡

𝑑

)

⋅

𝑏

=

𝑝

⋅

𝑏

(p+td)⋅a=p⋅a,(p+td)⋅b=p⋅b



So the pair 

(

𝑝

⋅

𝑎

,

𝑝

⋅

𝑏

)

(p⋅a,p⋅b) is a stable line ID for that direction.



That’s the whole magic that lets us group all parallel lines robustly using just integer arithmetic.



If you want the next-next upgrade



I can add:



Color-coding by line density (so clusters visually pop).



Export with per-line grouping (PLY with separate edge sets or multiple OBJ groups).



A more formal adjacency-preserving 3D spiral if you want a stricter “every consecutive integer is a unit step” construction — that one is doable but deserves a slightly more elaborate path rule.



But this script should already give you the “holy crap primes are forming beams in 3D” moment.

