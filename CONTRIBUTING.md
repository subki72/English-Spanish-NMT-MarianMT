# Contributing Guidelines

Thank you for your interest in contributing to the MarianMT English-Spanish NMT project.

## Development Workflow

1. Fork the repository and clone your fork locally.
2. Install the locked dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
4. Run tests and linting before submitting your changes:
   ```bash
   ruff check .
   ruff format --check .
   pytest tests/ -v
   ```
5. Commit using clear, descriptive commit messages adhering to Conventional Commits.
6. Submit a Pull Request targeting `main`.
