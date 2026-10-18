# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-18

### Added
- Production-grade MarianMT English-to-Spanish translation service using CTranslate2 INT8 quantization.
- Pure SentencePiece tokenization integration with zero PyTorch runtime dependency.
- Dynamic request batching queue with AnyIO worker thread execution.
- Prometheus observability metrics exporter endpoint at `/metrics`.
- Liveness probe (`/healthz`) and readiness probe (`/readyz`).
- Centralized exception handler and correlation ID tracing middleware.
- Comprehensive security policy, Dependabot scanning, and pre-commit validation.
- Interactive HTML/JS Web UI with live side-by-side translation preview.
