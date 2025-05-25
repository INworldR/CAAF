# CAAF Setup Instructions

## Environment Setup Commands

### 1. UV Project Initialization
```bash
# In your CAAF directory
cd CAAF

# Initialize UV project
uv init --python 3.11

# Install all dependencies
uv sync --dev

# Activate virtual environment
source .venv/bin/activate
```

### 2. Configuration Setup
```bash
# Create config directory and copy example
mkdir -p config
cp config.example.yaml config/config.yaml

# Edit configuration with your API keys
# Set your Claude API key in config/config.yaml
```

### 3. Directory Structure Creation
```bash
# Create all necessary directories
mkdir -p {caaf/{core,models,agents,persistence,communication},examples,tests,docs,data/{conversations,agents},logs}

# Create __init__.py files
touch caaf/__init__.py
touch caaf/{core,models,agents,persistence,communication}/__init__.py
```

### 4. Verify Installation
```bash
# Check Python version
python --version  # Should show Python 3.11.x

# Test imports (after implementation)
python -c "import caaf; print('CAAF installed successfully')"

# Run tests (after implementation)
uv run pytest

# Check code quality
uv run black --check caaf/
uv run flake8 caaf/
uv run mypy caaf/
```

## Ollama Setup Verification
```bash
# Verify Ollama is working
ollama list

# Test with a simple query
ollama run llama3 "Hello, how are you?"

# Check API endpoint
curl http://localhost:11434/api/version
```

## Development Workflow

### Adding Dependencies
```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update all dependencies
uv sync
```

### Code Quality Checks
```bash
# Format code
uv run black caaf/

# Sort imports
uv run isort caaf/

# Lint code
uv run flake8 caaf/
uv run ruff caaf/

# Type checking
uv run mypy caaf/
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=caaf

# Run specific test file
uv run pytest tests/test_agents.py

# Run with verbose output
uv run pytest -v
```

## Cursor AI Configuration

The repository includes `.vscode/settings.json` for optimal Cursor AI integration:
- Python interpreter automatically detected
- Auto-formatting on save
- Integrated testing
- Type checking enabled
- Proper exclusions for generated files

## Environment Variables

Create a `.env` file in the project root:
```bash
# API Keys
ANTHROPIC_API_KEY=your-claude-api-key-here
OLLAMA_BASE_URL=http://localhost:11434

# Development Settings
CAAF_DEBUG=true
CAAF_LOG_LEVEL=DEBUG
```

## Troubleshooting

### Common Issues

1. **UV not found**: Install UV with `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. **Python version mismatch**: Use `uv python install 3.11` to install Python 3.11
3. **Ollama connection issues**: Ensure Ollama is running with `ollama serve`
4. **Permission errors**: Check file permissions in project directory

### Verification Commands
```bash
# Check UV installation
uv --version

# Check Python in virtual environment
.venv/bin/python --version

# Check installed packages
uv pip list

# Test configuration loading
python -c "import yaml; print(yaml.safe_load(open('config/config.yaml')))"
```

## Next Steps

After setup completion:
1. Review the configuration in `config/config.yaml`
2. Set up your Claude API key
3. Run the development server (once implemented)
4. Check the examples directory for usage patterns
5. Read the documentation in the `docs` directory

## File Cleanup

You can now remove these legacy files if they exist:
```bash
rm requirements.txt requirements-dev.txt requirements-test.txt
```

All dependencies are now managed through `pyproject.toml` and UV.
