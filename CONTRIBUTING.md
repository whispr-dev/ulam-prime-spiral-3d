# Contributing to 3D Ulam Prime Cube

Thanks for your interest in improving this project!

This repo explores 3D Ulam-style prime structure visualizations, parallel-line
alignments, and practical exports for external 3D tools.

## What We Welcome

- Bug fixes
- Performance improvements (especially for line grouping and mapping)
- New direction sets or smarter ranking metrics
- Additional export formats (e.g., GLB)
- Documentation improvements and example galleries
- Validation tooling (consistency checks, small regression tests)

## Development Setup

Recommended environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install numpy matplotlib
If you add new dependencies, please keep them minimal and justify the need.

Code Style
Prefer clear, readable Python over clever tricks.

Use type hints where helpful.

Keep functions small and single-purpose.

Avoid unnecessary dependencies.

Keep exports deterministic where possible.

Optional but encouraged tooling:

bash
Copy code
pip install ruff black
ruff check .
black .
Testing
There is no large test suite yet, but PRs should include one or more of:

A short “manual test plan” in the PR description

A small deterministic check script

Before/after screenshots for visual changes

At minimum, please verify:

bash
Copy code
python ulam3d_wow.py --N 20000 --mapping shell --topk 8
python ulam3d_wow.py --N 20000 --mapping surface --topk 8
python ulam3d_wow.py --N 20000 --mapping shell --no-plot --export-prefix test_export
Then confirm the exported files open as expected.

Pull Request Workflow
Fork the repository.

Create a feature branch:

bash
Copy code
git checkout -b feature/your-feature
Make your changes.

Ensure the script runs without errors.

Open a Pull Request against main.

Commit Messages
Use clear, conventional messages when possible:

feat: add GLB export

fix: correct line grouping for dir-max=3

perf: speed up shell enumeration

docs: add Blender import notes

Communication
Open an Issue for:

Feature requests

Bug reports

Questions or clarifications

Thanks for helping push this prime-cube weirdness further into the fun zone.