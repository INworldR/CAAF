# Claude Code Integration Guide

This document provides specific instructions for using Claude Code with the CAAF project for automated development and maintenance tasks.

## Claude Code Overview

Claude Code is an agentic command-line tool that allows developers to delegate coding tasks directly from the terminal. This guide outlines how to effectively use Claude Code for CAAF development.

## Project Structure Context

When working with Claude Code on CAAF, provide this context:

```
CAAF is a lightweight agentic AI framework with the following key characteristics:
- Multi-model support (Ollama local + Claude API remote)
- JSON/JSONL-based persistence (no database dependencies)
- Clean architecture with modular design
- Python 3.8+ with PEP8 compliance
- Learning agents with conversation memory
- Inter-agent communication capabilities
```

## Development Guidelines for Claude Code

### Code Standards
- Follow PEP8 guidelines strictly
- Use modern Python with type hints
- Implement comprehensive docstrings
- Apply Clean Architecture principles
- Avoid unnecessary try-except blocks
- Maintain DRY principles

### Project Conventions
- All code and comments in English
- Use German for development discussions
- Start Python files with: `#!/usr/bin/env python3`
- Maximum 500 lines per code output
- Separate artifacts for code blocks

### Architecture Principles
- **Domain Layer**: Core business logic (agents, models, conversations)
- **Data Access Layer**: Persistence management (JSON/JSONL handlers)
- **Service Layer**: Orchestration and workflow management
- **Interface Layer**: CLI and API endpoints

## Common Claude Code Tasks

### 1. Creating New Agents
```bash
claude-code "Create a new specialized agent class for [TASK] following CAAF architecture patterns"
```

### 2. Implementing Model Integrations
```bash
claude-code "Implement Ollama API integration with error handling and model fallback"
```

### 3. Building Persistence Components
```bash
claude-code "Create ConversationManager class for JSONL-based conversation logging with search capabilities"
```

### 4. Testing and Quality Assurance
```bash
claude-code "Generate comprehensive tests for the AgentMemory class with pytest"
```

### 5. Documentation Generation
```bash
claude-code "Generate API documentation for all public methods in the models module"
```

## File Organization for Claude Code

### Core Modules Structure
```
caaf/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── agent_manager.py
│   └── exceptions.py
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── claude_model.py
│   └── ollama_model.py
├── persistence/
│   ├── __init__.py
│   ├── conversation_manager.py
│   ├── agent_memory.py
│   └── search_engine.py
├── agents/
│   ├── __init__.py
│   ├── code_agent.py
│   ├── research_agent.py
│   └── general_agent.py
└── communication/
    ├── __init__.py
    ├── message_bus.py
    └── agent_coordinator.py
```

## Development Workflow with Claude Code

### Phase 1: Foundation
1. Implement core base classes
2. Create model abstraction layer
3. Build basic persistence components
4. Establish testing framework

### Phase 2: Agent Development
1. Implement specialized agent classes
2. Add inter-agent communication
3. Create learning mechanisms
4. Build conversation management

### Phase 3: Integration & Polish
1. Integrate all components
2. Add CLI interface
3. Performance optimization
4. Documentation completion

## Specific Implementation Requests

### Base Classes
```bash
claude-code "Implement BaseAgent abstract class with conversation handling, memory integration, and model communication methods"
```

### Model Integration
```bash
claude-code "Create ClaudeModel class with async API calls, rate limiting, and error recovery"
```

### Persistence Layer
```bash
claude-code "Build ConversationManager with JSONL streaming, full-text search, and conversation indexing"
```

### Agent Memory
```bash
claude-code "Implement AgentMemory class for pattern learning, conversation summarization, and knowledge retention"
```

## Quality Assurance

### Code Review Checklist
- [ ] PEP8 compliance verified
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] Error handling appropriate
- [ ] Tests coverage adequate
- [ ] Architecture patterns followed

### Testing Strategy
- Unit tests for all core components
- Integration tests for model communications
- Performance tests for persistence layer
- End-to-end tests for agent interactions

## Configuration Management

### Development Config
```yaml
# config/development.yaml
debug: true
log_level: DEBUG
models:
  claude:
    api_key: "dev-key"
    timeout: 10
  ollama:
    base_url: "http://localhost:11434"
    timeout: 5
persistence:
  conversations_dir: "./dev_data/conversations"
  agents_dir: "./dev_data/agents"
```

### Testing Config
```yaml
# config/testing.yaml
debug: true
log_level: INFO
persistence:
  conversations_dir: "./test_data/conversations"
  agents_dir: "./test_data/agents"
  cleanup_after_tests: true
```

## Dependencies Management

### Core Dependencies
- `requests` - HTTP client for API calls
- `pyyaml` - Configuration file parsing
- `typing-extensions` - Enhanced type hints
- `pathlib` - Modern path handling

### Development Dependencies
- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `pytest-cov` - Coverage reporting

## Best Practices for Claude Code Usage

1. **Provide Clear Context**: Always specify the CAAF architecture context
2. **Incremental Development**: Request components in logical order
3. **Specify Standards**: Mention PEP8, type hints, and documentation requirements
4. **Request Tests**: Always ask for corresponding tests with implementations
5. **Review Integration**: Verify new components integrate with existing architecture

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure proper `__init__.py` files
- **Type Errors**: Use consistent type hints across modules
- **Configuration Issues**: Validate YAML syntax and structure
- **API Errors**: Implement proper error handling and retries

### Debug Commands
```bash
# Test specific components
python -m pytest tests/test_agents.py -v

# Check code quality
flake8 caaf/
black --check caaf/

# Type checking
mypy caaf/
```

---

*This guide ensures Claude Code can effectively contribute to CAAF development while maintaining code quality and architectural consistency.*
