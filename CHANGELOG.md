# Changelog

All notable changes to this project will be documented in this file.

This project follows a lightweight semantic versioning style:
- MAJOR: breaking CLI, export, or mapping behavior changes
- MINOR: new capabilities that do not break existing usage
- PATCH: bug fixes and performance improvements

---

## v0.1.0 - Initial Release (2025-12-05)

### Added
- 3D **cubic-shell mapping** of integers to lattice points.
- 3D **surface traversal (“walk-ish”) mapping** for cube-surface ordering.
- Prime detection using a fast sieve.
- **Parallel line grouping** for many integer directions using perpendicular
  integer invariants.
- Ranking and rendering of **top-K prime-dense lines** anywhere in the cube.
- 3D visualization using matplotlib.
- Export of prime points and top-K line segments to:
  - **PLY** (vertices + edges)
  - **OBJ** (vertices + `l` line elements)
  - **glTF 2.0** (`.gltf` + `.bin`) with POINTS + LINES primitives
- CLI controls for `--N`, `--mapping`, `--dir-max`, `--min-line-primes`,
  `--topk`, `--no-plot`, and export prefix.

### Notes
- glTF POINTS/LINES support varies across importers and viewers.
  PLY/OBJ are recommended for the most consistent Blender workflows.
- The surface traversal mapping is designed for visual exploration and may not
  preserve strict unit-step adjacency at all boundaries.

---