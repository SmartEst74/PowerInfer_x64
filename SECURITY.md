# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in PowerInfer_x64, please report it responsibly.

### How to Report

1. **Do NOT open a public issue** for security vulnerabilities
2. Email: security@smartest74.com (or create a private security advisory on GitHub)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment** within 48 hours
- **Assessment** within 1 week
- **Fix timeline** communicated based on severity
- **Credit** in the security advisory (if desired)

### Security Measures

- All dependencies audited with `cargo audit`
- GPU kernel code reviewed for memory safety
- Network endpoints use TLS
- No secrets in source code or configuration
- Regular dependency updates via Dependabot

## Scope

Security concerns include:
- Memory safety issues in Rust code
- GPU kernel vulnerabilities (out-of-bounds access)
- API authentication/authorization bypass
- Denial of service via malformed inputs
- Information disclosure through error messages
