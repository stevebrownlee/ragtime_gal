# Feedback-Driven RAG System

This project implements a comprehensive, self-improving document-based question answering system that learns from user feedback to continuously enhance its performance. The system evolves through five distinct phases, each building upon the previous to create a sophisticated, production-ready AI system.

## System Overview

The Feedback-Driven RAG System is a multi-phase architecture designed to:
- **Learn from User Feedback**: Continuously improve based on user ratings and comments
- **Optimize Retrieval**: Enhance document retrieval based on feedback patterns
- **Fine-tune Models**: Automatically improve embedding models using collected feedback
- **Ensure Reliability**: Comprehensive testing and validation for production deployment
- **Provide Observability**: Complete monitoring, documentation, and operational support

## Architecture Overview

```
Complete System Architecture:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Feedback-Driven RAG System                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Phase 5       │  │   Phase 4       │  │   Phase 3       │                │
│  │ Documentation   │  │ Testing &       │  │ Model Fine-     │                │
│  │ & Monitoring    │  │ Validation      │  │ tuning System   │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                                      │
│  │   Phase 2       │  │   Phase 1       │                                      │
│  │ Retrieval       │  │ Feedback        │                                      │
│  │ Optimization    │  │ Collection      │                                      │
│  └─────────────────┘  └─────────────────┘                                      │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                              Core System                                       │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Web Interface │  │   Flask API     │  │   Conversation  │                │
│  │   (HTML/JS)     │  │   Server        │  │   Management    │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   Vector DB     │  │   LLM Engine    │  │   ConPort       │                │
│  │   (ChromaDB)    │  │   (Ollama)      │  │   Memory        │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### Core RAG Capabilities
- **Vector-based conversation memory**: Uses embeddings to find semantically relevant previous interactions
- **Dynamic context management**: Intelligently selects and formats context based on query type
- **Conversation summarization**: Compresses older conversation turns to manage token usage
- **Sophisticated query classification**: Determines if a query is a follow-up using both regex and semantic similarity

### Feedback-Driven Improvements
- **User Feedback Collection**: Star ratings, detailed feedback, and usage analytics
- **Pattern Recognition**: Identifies successful query characteristics and common issues
- **Query Enhancement**: Suggests improvements based on feedback patterns
- **Model Fine-tuning**: Automatically improves embedding models using feedback data

### Production Features
- **Comprehensive Testing**: Integration, performance, validation, and regression testing
- **Real-time Monitoring**: System metrics, performance tracking, and alerting
- **Production Deployment**: Docker containerization, security, and scalability
- **Complete Documentation**: User guides, technical docs, and API references

## System Phases

### Phase 1: Feedback Collection System
**Purpose**: Collect and analyze user feedback to identify improvement opportunities.

**Key Components**:
- User feedback interface with star ratings and detailed feedback
- ConPort integration for structured feedback storage
- Feedback analysis and pattern recognition
- Usage analytics and trend identification

**Documentation**: [`PHASE1_README.md`](PHASE1_README.md)

### Phase 2: Retrieval Optimization
**Purpose**: Optimize document retrieval based on feedback patterns.

**Key Components**:
- Feedback pattern analysis and insight generation
- Query enhancement based on successful patterns
- Retrieval performance optimization
- A/B testing for optimization strategies

**Documentation**: [`PHASE2_README.md`](PHASE2_README.md)

### Phase 3: Model Fine-tuning System
**Purpose**: Fine-tune embedding models using collected feedback data.

**Key Components**:
- Training data generation from user feedback
- Sentence transformer model fine-tuning
- A/B testing framework for model comparison
- Enhanced embedding integration

**Documentation**: [`PHASE3_README.md`](PHASE3_README.md)

### Phase 4: Testing and Validation
**Purpose**: Comprehensive testing to ensure system reliability and effectiveness.

**Key Components**:
- Integration testing for end-to-end workflows
- Performance testing under load
- Validation testing for improvement effectiveness
- Regression testing for backward compatibility

**Documentation**: [`PHASE4_README.md`](PHASE4_README.md)

### Phase 5: Documentation and Monitoring
**Purpose**: Production-ready documentation, monitoring, and deployment infrastructure.

**Key Components**:
- Comprehensive user and technical documentation
- Real-time monitoring dashboard with alerting
- Production deployment guides and automation
- Backup, recovery, and maintenance procedures

**Documentation**: [`PHASE5_README.md`](PHASE5_README.md)

## Core Components

- **TemplateManager**: Manages prompt templates with dynamic context sections
- **ContextManager**: Handles context selection and formatting for different query types
- **QueryClassifier**: Classifies queries to determine appropriate context handling
- **ConversationEmbedder**: Embeds conversation interactions for vector-based retrieval
- **ConversationSummarizer**: Generates summaries of conversation history
- **EnhancedConversation**: Extends the base Conversation class with vector-based retrieval
- **FeedbackAnalyzer**: Analyzes user feedback patterns and generates insights
- **QueryEnhancer**: Improves query formulation based on feedback patterns
- **ModelFineTuner**: Fine-tunes embedding models using feedback data
- **MonitoringDashboard**: Real-time system monitoring and performance tracking

## Quick Start

### Development Setup

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd ragtime-gal
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

3. **Install Ollama and Models**:
   ```bash
   # Install Ollama (see https://ollama.ai)
   ollama pull mistral  # For embeddings
   ollama pull sixthwood  # For LLM inference
   ```

4. **Start the Application**:
   ```bash
   python app.py
   ```

5. **Access the System**:
   - Main Interface: `http://localhost:8084`
   - Monitoring Dashboard: `http://localhost:8084/monitoring`

### Production Deployment

1. **Docker Compose (Recommended)**:
   ```bash
   # Configure production environment
   cp .env.template .env
   # Edit .env with production settings

   # Deploy with monitoring stack
   docker-compose up -d

   # Initialize models
   docker exec ragtime-ollama ollama pull mistral
   docker exec ragtime-ollama ollama pull sixthwood
   ```

2. **Access Production Services**:
   - Application: `https://your-domain.com`
   - Monitoring: `https://your-domain.com/monitoring`
   - Grafana: `http://your-domain.com:3000`
   - Prometheus: `http://your-domain.com:9090`

### Using the System

#### Document Management
1. **Upload Documents**: Use the web interface to upload PDF or Markdown files
2. **Organize Collections**: Group related documents into named collections
3. **Monitor Processing**: Check the monitoring dashboard for embedding progress

#### Querying and Feedback
1. **Ask Questions**: Enter questions in the query interface
2. **Select Response Style**: Choose from Standard, Creative, or Sixth Wood
3. **Provide Feedback**: Rate responses and provide detailed feedback
4. **Track Improvements**: Monitor system learning through the dashboard

#### Advanced Features
- **Conversation Context**: System maintains context across multiple questions
- **Query Enhancement**: System suggests improved query formulations
- **Model Fine-tuning**: Automatic model improvements based on feedback
- **Performance Monitoring**: Real-time system performance tracking

## Configuration

### Environment Variables

The system can be configured through environment variables in `.env`:

```bash
# Core Configuration
LLM_MODEL=sixthwood                    # Language model to use
EMBEDDING_MODEL=mistral                # Embedding model to use
CHROMA_PERSIST_DIR=./chroma_db        # Vector database directory
OLLAMA_BASE_URL=http://localhost:11434 # Ollama API URL
RETRIEVAL_K=4                         # Number of documents to retrieve
TEMPERATURE=0.7                       # Response creativity
MAX_TOKENS=500                        # Maximum response length

# Feedback System
FEEDBACK_ENABLED=true                 # Enable feedback collection
FEEDBACK_ANALYTICS_ENABLED=true      # Enable feedback analytics
QUERY_ENHANCEMENT_ENABLED=true       # Enable query enhancement

# Model Fine-tuning
FINETUNING_ENABLED=true              # Enable model fine-tuning
AB_TESTING_ENABLED=true              # Enable A/B testing
TRAINING_DATA_MIN_SAMPLES=100        # Minimum samples for training

# Monitoring
MONITORING_ENABLED=true              # Enable monitoring dashboard
METRICS_COLLECTION_INTERVAL=30       # Metrics collection interval (seconds)
ALERT_EMAIL=admin@your-domain.com    # Alert email address

# Security
SECRET_KEY=your-secret-key-here      # Flask secret key
ALLOWED_HOSTS=localhost,your-domain.com # Allowed hosts
CORS_ORIGINS=https://your-domain.com # CORS origins
```

### Template Customization

The system uses a unified template structure defined in `prompt_templates.json`:

```json
{
  "base_templates": {
    "standard": "You are a helpful assistant...",
    "creative": "You are a creative and engaging assistant...",
    "sixthwood": "You are a specialized assistant..."
  },
  "system_instructions": {
    "standard": "Provide accurate and helpful responses...",
    "creative": "Be creative and engaging in your responses...",
    "sixthwood": "Follow the Sixth Wood methodology..."
  },
  "context_formats": {
    "initial": "Based on the following documents:\n{documents}",
    "follow_up": "Continuing our conversation:\n{conversation}\n\nNew documents:\n{documents}",
    "with_previous_content": "Previous context:\n{previous}\n\nNew information:\n{documents}"
  }
}
```

### Monitoring Configuration

Configure monitoring thresholds and settings:

```bash
# Alert Thresholds
ALERT_CPU_WARNING=80                 # CPU usage warning threshold (%)
ALERT_CPU_CRITICAL=90               # CPU usage critical threshold (%)
ALERT_MEMORY_WARNING=85             # Memory usage warning threshold (%)
ALERT_MEMORY_CRITICAL=95            # Memory usage critical threshold (%)
ALERT_ERROR_RATE_WARNING=5          # Error rate warning threshold (%)
ALERT_ERROR_RATE_CRITICAL=10        # Error rate critical threshold (%)
ALERT_RESPONSE_TIME_WARNING=5       # Response time warning threshold (seconds)
ALERT_RESPONSE_TIME_CRITICAL=10     # Response time critical threshold (seconds)

# Dashboard Settings
DASHBOARD_REFRESH_INTERVAL=30       # Dashboard refresh interval (seconds)
DASHBOARD_RETENTION_HOURS=168       # Data retention period (hours)
DASHBOARD_AUTH_ENABLED=true         # Enable dashboard authentication
```

## API Reference

### Core Endpoints

- `POST /embed` - Upload and embed documents
- `POST /query` - Submit queries to the system
- `POST /feedback` - Submit user feedback
- `GET /collections` - List document collections
- `GET /health` - System health check

### Monitoring Endpoints

- `GET /monitoring` - Monitoring dashboard
- `GET /monitoring/api/metrics` - System metrics
- `GET /monitoring/api/alerts` - Active alerts
- `GET /monitoring/api/health` - Detailed health status

### Phase 3 Endpoints (Model Fine-tuning)

- `POST /training/generate-data` - Generate training data
- `POST /training/fine-tune` - Start model fine-tuning
- `GET /training/status/{job_id}` - Check training status
- `POST /testing/ab-test/start` - Start A/B test
- `GET /testing/ab-test/{test_id}/results` - Get A/B test results

For complete API documentation, see [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md).

## Documentation

### User Documentation
- [`USER_GUIDE.md`](USER_GUIDE.md) - Comprehensive user guide with tutorials and troubleshooting
- [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md) - Complete REST API reference

### Technical Documentation
- [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) - Detailed system architecture and design
- [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) - Development setup, code organization, and extension guides
- [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) - Production deployment and configuration

### Phase Documentation
- [`PHASE3_README.md`](PHASE3_README.md) - Model fine-tuning system documentation
- [`PHASE4_README.md`](PHASE4_README.md) - Testing and validation documentation
- [`PHASE5_README.md`](PHASE5_README.md) - Documentation and monitoring system

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/performance/    # Performance tests

# Run with coverage
pytest --cov=. --cov-report=html

# Run tests in parallel
pytest -n auto
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load testing and benchmarking
- **Validation Tests**: System improvement effectiveness
- **Regression Tests**: Backward compatibility assurance

## Monitoring and Observability

### Real-time Monitoring
- System performance metrics (CPU, memory, disk usage)
- Application metrics (response times, error rates, throughput)
- Feedback system metrics (ratings, trends, improvement indicators)
- Model performance tracking (accuracy, relevance, user satisfaction)

### Alerting
- Configurable alert thresholds for all metrics
- Multiple alert levels (info, warning, error, critical)
- Email and webhook notifications
- Alert resolution tracking and escalation

### Analytics
- User feedback trends and patterns
- Query performance analytics
- Model fine-tuning effectiveness
- System usage statistics

## Security

### Authentication and Authorization
- Session-based authentication for web interface
- API key authentication for programmatic access
- Role-based access control for monitoring endpoints
- Secure secret management

### Data Protection
- Local data storage by default (no external data sharing)
- Optional encryption at rest
- Secure session management
- Input validation and sanitization
- GDPR compliance ready

### Network Security
- HTTPS/TLS encryption
- Rate limiting on API endpoints
- CORS configuration
- Firewall recommendations

## Performance and Scalability

### Performance Optimization
- Efficient vector database operations
- Conversation history caching
- Template pre-compilation
- Batch processing for embeddings
- Connection pooling and resource management

### Scalability Features
- Horizontal scaling with Docker Compose
- Load balancing with Nginx
- Database sharding support
- Distributed processing capabilities
- Auto-scaling policies

### Resource Requirements
- **Minimum**: 4 CPU cores, 8GB RAM, 50GB storage
- **Recommended**: 8+ CPU cores, 16GB+ RAM, 100GB+ SSD storage
- **GPU**: Optional for model fine-tuning acceleration

## Implementation Highlights

This system implements a modern approach to conversational RAG with:

1. **Feedback-Driven Learning**: Continuous improvement based on user feedback
2. **Multi-Phase Architecture**: Modular design with clear separation of concerns
3. **Production-Ready Monitoring**: Comprehensive observability and alerting
4. **Automated Model Fine-tuning**: Self-improving embedding models
5. **Comprehensive Testing**: Reliability assurance through extensive testing
6. **Complete Documentation**: User and technical documentation for all aspects
7. **Security-First Design**: Built-in security features and best practices
8. **Scalable Deployment**: Container-based deployment with orchestration support

## Contributing

### Development Workflow

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch for your changes
3. **Development**: Make your changes following the coding standards
4. **Testing**: Run the full test suite and add tests for new features
5. **Documentation**: Update documentation for any new features
6. **Pull Request**: Submit a pull request with a clear description

### Coding Standards

- **Python**: Follow PEP 8, use type hints, include docstrings
- **Testing**: Maintain >90% test coverage, include unit and integration tests
- **Documentation**: Update relevant documentation for all changes
- **Security**: Follow security best practices, validate all inputs

### Extension Points

The system is designed for extensibility:

- **Custom Feedback Analyzers**: Implement domain-specific feedback analysis
- **Custom Query Enhancers**: Add specialized query improvement logic
- **Custom Embedding Models**: Integrate new embedding models
- **Custom Monitoring Metrics**: Add application-specific metrics
- **Custom Templates**: Create specialized response templates

For detailed development information, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

## Troubleshooting

### Common Issues

**System Not Starting**:
- Check Ollama is running: `ollama list`
- Verify environment variables in `.env`
- Check port availability (8084, 11434)

**Poor Response Quality**:
- Ensure documents are properly embedded
- Check feedback patterns in monitoring dashboard
- Verify model configuration and templates

**Performance Issues**:
- Monitor system resources in dashboard
- Check database optimization settings
- Review alert thresholds and system limits

**Monitoring Dashboard Not Loading**:
- Verify monitoring is enabled in configuration
- Check authentication settings
- Review browser console for errors

For comprehensive troubleshooting, see [`USER_GUIDE.md`](USER_GUIDE.md#troubleshooting).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

### Getting Help

1. **Documentation**: Check the comprehensive documentation first
2. **Issues**: Search existing GitHub issues or create a new one
3. **Discussions**: Use GitHub Discussions for questions and ideas
4. **Community**: Join our community channels for support

### Reporting Issues

When reporting issues, please include:
- System information (OS, Python version, Docker version)
- Configuration details (anonymized)
- Steps to reproduce the issue
- Error messages and logs
- Expected vs actual behavior

### Feature Requests

We welcome feature requests! Please:
- Check existing issues and discussions first
- Provide clear use cases and requirements
- Consider contributing the implementation
- Follow the feature request template

## Roadmap

### Upcoming Features

- **Advanced Analytics**: Machine learning insights from feedback data
- **Multi-language Support**: Support for non-English documents and queries
- **Cloud Integration**: Native cloud provider integrations
- **Mobile Interface**: Mobile-optimized interface and apps
- **Advanced Security**: Enhanced authentication and authorization
- **Plugin Architecture**: Extensible plugin system for custom functionality

### Long-term Vision

- **Autonomous Learning**: Fully autonomous system improvement
- **Multi-modal Support**: Support for images, audio, and video
- **Federated Learning**: Distributed learning across multiple instances
- **Advanced Reasoning**: Integration with reasoning and planning capabilities

## Acknowledgments

This project builds upon excellent open-source technologies:

- **Ollama**: Local LLM inference engine
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: State-of-the-art embedding models
- **Flask**: Web framework for API and interface
- **Docker**: Containerization and deployment
- **Prometheus & Grafana**: Monitoring and observability
- **Chart.js**: Interactive dashboard charts

Special thanks to the open-source community for making this project possible.

---

*The Feedback-Driven RAG System represents a comprehensive approach to building self-improving AI systems that learn from user feedback and continuously enhance their performance. With complete documentation, monitoring, and production-ready deployment, it provides a solid foundation for building intelligent document-based question answering systems.*