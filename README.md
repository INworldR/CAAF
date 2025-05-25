# CAAF - Claude Agentic AI Framework

A lightweight, extensible framework for building agentic AI systems with support for both local Ollama models and Claude API integration.

## Overview

CAAF provides a clean, modular architecture for creating AI agents that can communicate, learn from conversations, and maintain persistent memory across sessions. Built with simplicity and extensibility in mind.

## Features

- **Multi-Model Support**: Seamless integration with Ollama (local) and Claude API (remote)
- **Agent Communication**: Inter-agent messaging and coordination
- **Persistent Memory**: JSON/JSONL-based conversation history and agent learning
- **Searchable History**: Full-text search across conversation logs
- **Learning Agents**: Pattern recognition and knowledge accumulation from past interactions
- **Clean Architecture**: Modular design following clean code principles
- **Zero Database Dependencies**: File-based persistence for simplicity

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/CAAF.git
cd CAAF

# Install dependencies
pip install -r requirements.txt

# Configure your API keys
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your Claude API key and Ollama settings

# Run a simple example
python examples/basic_conversation.py
```

## Architecture

```
CAAF/
├── caaf/
│   ├── agents/          # Agent implementations
│   ├── models/          # Model integrations (Ollama, Claude)
│   ├── persistence/     # JSON/JSONL data management
│   ├── communication/   # Inter-agent messaging
│   └── core/           # Core framework classes
├── examples/           # Usage examples
├── tests/             # Test suite
└── docs/              # Documentation
```

## Core Components

### Agents
- **BaseAgent**: Abstract base class for all agents
- **CodeAgent**: Specialized for code generation and analysis
- **ResearchAgent**: Optimized for information gathering and synthesis
- **GeneralAgent**: Multi-purpose conversational agent

### Models
- **OllamaModel**: Local model integration via Ollama API
- **ClaudeModel**: Claude API integration with fallback handling
- **ModelRouter**: Intelligent model selection and load balancing

### Persistence
- **ConversationManager**: JSONL-based conversation logging
- **AgentMemory**: Pattern learning and knowledge retention
- **SearchEngine**: Full-text search across conversation history

## Configuration

Create `config/config.yaml`:

```yaml
# Model Configuration
models:
  claude:
    api_key: "your-claude-api-key"
    model: "claude-3-sonnet-20240229"
    max_tokens: 4000
  
  ollama:
    base_url: "http://localhost:11434"
    models: ["llama3", "mistral", "codellama"]
    timeout: 30

# Agent Configuration
agents:
  default_model: "claude"
  enable_learning: true
  memory_retention_days: 30

# Persistence Configuration
persistence:
  conversations_dir: "./data/conversations"
  agents_dir: "./data/agents"
  enable_search_index: true
```

## Usage Examples

### Basic Conversation
```python
from caaf import AgentManager, ClaudeModel

# Initialize framework
manager = AgentManager()
model = ClaudeModel(api_key="your-key")

# Create and configure agent
agent = manager.create_agent("general", model=model)

# Start conversation
response = agent.chat("Explain quantum computing in simple terms")
print(response)
```

### Multi-Agent Interaction
```python
from caaf import AgentManager

manager = AgentManager()

# Create specialized agents
code_agent = manager.create_agent("code", model="ollama:codellama")
research_agent = manager.create_agent("research", model="claude")

# Coordinate between agents
research_result = research_agent.chat("Research best practices for REST APIs")
code_implementation = code_agent.chat(f"Implement these practices: {research_result}")
```

### Learning from History
```python
# Agent automatically learns from past conversations
agent = manager.get_agent("code")

# Agent remembers previous patterns and improves responses
response = agent.chat("Create a sorting algorithm")  # Uses learned preferences
```

## Development

### Requirements
- Python 3.8+
- Ollama (for local models)
- Claude API key (for Claude integration)

### Setup Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 caaf/
black caaf/
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Project Status

🚧 **Early Development** - This framework is in active development. APIs may change.

### Roadmap
- [x] Core architecture design
- [ ] Basic agent implementations
- [ ] Model integrations (Ollama + Claude)
- [ ] Persistence layer
- [ ] Inter-agent communication
- [ ] Learning and memory systems
- [ ] Documentation and examples
- [ ] Performance optimization
- [ ] Plugin system

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with inspiration from modern agentic AI research
- Designed for the Claude ecosystem
- Thanks to the Ollama community for local model support

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/CAAF/issues)
- **Documentation**: [Project Wiki](https://github.com/your-username/CAAF/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/CAAF/discussions)

---

*CAAF - Making agentic AI accessible and extensible*