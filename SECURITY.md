# Security Policy

This project is a mathematical visualization and export tool for exploring
prime alignments in 3D Ulam-style cube mappings.

Because it is not a network service and does not handle sensitive data by
default, the security surface is relatively small. That said, input size and
file export are still worth treating responsibly.

## Supported Versions

We support the latest release on the `main` branch.

| Version | Supported |
|--------:|:---------:|
| 0.1.x   | ✅ Active  |

## Reporting a Vulnerability

If you discover a potential vulnerability:

1. **Do not** open a public GitHub issue immediately.
2. Contact the maintainer privately via:
   - Email: security@whispr.dev  
   - Or GitHub’s “Report a vulnerability” feature.

Please include:
- A clear description of the issue
- Steps to reproduce
- Your environment (OS, Python version)
- Any proof-of-concept files/scripts (if safe to share)

We will aim to acknowledge your report within **72 hours**.

## Security Scope

### In scope
- Crashes or hangs triggered by malformed inputs or extreme CLI values
- Unsafe file writing behavior during export
- Dependency-related vulnerabilities that materially affect users

### Out of scope
- Attacks requiring a compromised local environment
- Issues in third-party viewers (e.g., Blender/WebGL import quirks)
- The mathematical properties of primes or pattern interpretation

## Safe Usage Guidance

- Treat `--N` as an untrusted input if you wrap this tool in services.
- Use reasonable limits in automated pipelines to avoid memory exhaustion.
- Prefer running in a virtual environment with pinned dependencies.

## Verification

Users are encouraged to review, audit, and run the script locally before using
it in any automated or shared environment.
