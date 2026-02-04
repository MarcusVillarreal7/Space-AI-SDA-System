# Contributing to Space AI

## Development Guidelines

### Code Style
- Follow PEP 8 style guide
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Type hints for all function signatures
- Docstrings for all public functions (Google style)

### Git Workflow
1. Create feature branch from `main`
2. Make atomic commits with clear messages
3. Write tests for new features
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: feat, fix, docs, style, refactor, test, chore

Example:
```
feat(tracking): Add Unscented Kalman Filter implementation

Implemented UKF for non-linear state estimation with configurable
sigma points. Includes unit tests and comparison with EKF.

Closes #42
```

### Testing
- Write unit tests for all new functions
- Maintain >80% code coverage
- Include integration tests for pipelines
- Add scenario tests for edge cases
- Use pytest fixtures for test data

### Documentation
- Update README.md for user-facing changes
- Update ARCHITECTURE.md for design changes
- Update DEVLOG.md with progress
- Add docstrings to all functions
- Include examples in docstrings

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No linter errors
- [ ] Type hints included
- [ ] Performance considered

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/space-ai.git
cd space-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/
```

## Project Structure Conventions

- `src/`: Production code only
- `tests/`: All test code
- `notebooks/`: Exploratory analysis (not production code)
- `scripts/`: Utility scripts for development
- `config/`: Configuration files (YAML)
- `docs/`: Documentation

## Questions?

Open an issue or contact the maintainer.
