
# Developer Guide: Feedback-Driven RAG System

## Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Code Organization](#code-organization)
3. [Module Structure](#module-structure)
4. [Extension Points](#extension-points)
5. [Customization Options](#customization-options)
6. [Testing Procedures](#testing-procedures)
7. [CI/CD Integration](#cicd-integration)
8. [Performance Optimization](#performance-optimization)
9. [Debugging Guide](#debugging-guide)
10. [Contributing Guidelines](#contributing-guidelines)

## Development Environment Setup

### Prerequisites

**System Requirements:**
- Python 3.8 or higher
- Node.js 16+ (for frontend development)
- Docker and Docker Compose
- Git
- 8GB+ RAM recommended
- 10GB+ free disk space

**Required Services:**
- Ollama (for local LLM inference)
- ChromaDB (embedded, no separate installation needed)

### Local Development Setup

1. **Clone the Repository:**
```bash
git clone <repository-url>
cd ragtime-gal
```

2. **Create Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
# or using Pipenv
pipenv install --dev
```

4. **Environment Configuration:**
```bash
cp .env.template .env
# Edit .env with your configuration
```

5. **Install Ollama and Models:**
```bash
# Install Ollama (see https://ollama.ai)
ollama pull mistral  # For embeddings
ollama pull sixthwood  # For LLM inference
```

6. **Initialize Database:**
```bash
python -c "from app import get_vector_db; get_vector_db()"
```

7. **Run Development Server:**
```bash
python app.py
```

### Docker Development Setup

1. **Using Docker Compose:**
```bash
docker-compose -f docker-compose.dev.yml up --build
```

2. **Development with Hot Reload:**
```bash
docker-compose -f docker-compose.dev.yml up --build --watch
```

### IDE Configuration

**VS Code Setup:**
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"]
}
```

**PyCharm Setup:**
- Set Python interpreter to virtual environment
- Enable pytest as test runner
- Configure code style to use Black formatter
- Set up run configurations for Flask app

## Code Organization

### Project Structure

```
ragtime-gal/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── Pipfile                     # Pipenv configuration
├── .env.template              # Environment template
├── .gitignore                 # Git ignore rules
├── docker-compose.yml         # Production Docker setup
├── docker-compose.dev.yml     # Development Docker setup
├── Dockerfile                 # Container definition
│
├── core/                      # Core system modules
│   ├── __init__.py
│   ├── embed.py              # Document embedding
│   ├── query.py              # Query processing
│   ├── conversation.py       # Basic conversation
│   └── enhanced_conversation.py  # Enhanced conversation
│
├── phase1/                    # Phase 1: Feedback Collection
│   ├── __init__.py
│   └── feedback_analyzer.py  # Feedback analysis
│
├── phase2/                    # Phase 2: Retrieval Optimization
│   ├── __init__.py
│   └── query_enhancer.py     # Query enhancement
│
├── phase3/                    # Phase 3: Model Fine-tuning
│   ├── __init__.py
│   ├── training_data_generator.py  # Training data creation
│   ├── model_finetuner.py    # Model fine-tuning
│   ├── ab_testing.py         # A/B testing framework
│   ├── embed_enhanced.py     # Enhanced embedding
│   └── phase3_integration.py # Phase 3 integration
│
├── phase4/                    # Phase 4: Testing & Validation
│   ├── __init__.py
│   └── test_*.py             # Test modules
│
├── phase5/                    # Phase 5: Documentation & Monitoring
│   ├── __init__.py
│   ├── monitoring_dashboard.py  # Monitoring system
│   └── deployment/           # Deployment scripts
│
├── templates/                 # HTML templates
│   ├── template.html         # Main UI template
│   └── monitoring.html       # Monitoring dashboard
│
├── static/                    # Static assets
│   ├── css/
│   ├── js/
│   └── images/
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Test configuration
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── performance/          # Performance tests
│
├── docs/                      # Documentation
│   ├── USER_GUIDE.md
│   ├── API_DOCUMENTATION.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── DEVELOPER_GUIDE.md
│   └── DEPLOYMENT_GUIDE.md
│
└── scripts/                   # Utility scripts
    ├── setup.sh              # Setup script
    ├── deploy.sh             # Deployment script
    └── backup.sh             # Backup script
```

### Module Dependencies

```
Dependency Graph:

app.py
├── core/
│   ├── embed.py
│   ├── query.py
│   ├── conversation.py
│   └── enhanced_conversation.py
├── phase1/
│   └── feedback_analyzer.py
├── phase2/
│   └── query_enhancer.py
├── phase3/
│   ├── training_data_generator.py
│   ├── model_finetuner.py
│   ├── ab_testing.py
│   ├── embed_enhanced.py
│   └── phase3_integration.py
└── phase5/
    └── monitoring_dashboard.py
```

## Module Structure

### Core Modules

#### `app.py` - Main Application
```python
"""
Main Flask application with all API endpoints.
Handles routing, session management, and error handling.
"""

Key Components:
- Flask app initialization
- Route definitions
- Session management
- Error handling
- Health checks
```

#### `embed.py` - Document Embedding
```python
"""
Document processing and embedding functionality.
Handles file upload, text extraction, chunking, and vector storage.
"""

Key Functions:
- embed(file, collection_name): Main embedding function
- process_document(file_path): Document processing
- create_chunks(text): Text chunking
- store_embeddings(chunks, collection): Vector storage
```

#### `query.py` - Query Processing
```python
"""
Query processing and response generation.
Handles retrieval, context building, and LLM interaction.
"""

Key Functions:
- query(query_text, **kwargs): Main query function
- retrieve_documents(query, k): Document retrieval
- build_context(documents, conversation): Context building
- generate_response(context, query): Response generation
```

#### `enhanced_conversation.py` - Conversation Management
```python
"""
Enhanced conversation handling with vector-based memory.
Provides context-aware multi-turn conversations.
"""

Key Classes:
- EnhancedConversation: Main conversation class
- ConversationEmbedder: Conversation embedding
- ConversationSummarizer: History summarization
- QueryClassifier: Query classification
```

### Phase-Specific Modules

#### Phase 1: `feedback_analyzer.py`
```python
"""
Feedback analysis and pattern recognition.
Processes user feedback to identify improvement opportunities.
"""

Key Classes:
- FeedbackAnalyzer: Main analysis class
- PatternRecognizer: Pattern identification
- FeedbackProcessor: Data processing
- InsightGenerator: Insight generation
```

#### Phase 2: `query_enhancer.py`
```python
"""
Query enhancement based on feedback patterns.
Improves query formulation for better retrieval.
"""

Key Classes:
- QueryEnhancer: Main enhancement class
- QueryAnalyzer: Query analysis
- EnhancementSuggester: Suggestion generation
- PerformanceTracker: Enhancement tracking
```

#### Phase 3: Model Fine-tuning Modules

**`training_data_generator.py`:**
```python
"""
Training data generation from user feedback.
Creates positive/negative pairs for model training.
"""

Key Classes:
- TrainingDataGenerator: Main generator class
- FeedbackProcessor: Feedback processing
- PairGenerator: Training pair creation
- DataValidator: Data quality validation
```

**`model_finetuner.py`:**
```python
"""
Model fine-tuning using sentence transformers.
Handles model training and optimization.
"""

Key Classes:
- ModelFineTuner: Main fine-tuning class
- TrainingManager: Training orchestration
- ModelEvaluator: Performance evaluation
- ModelSaver: Model persistence
```

**`ab_testing.py`:**
```python
"""
A/B testing framework for model comparison.
Manages experiments and statistical analysis.
"""

Key Classes:
- ABTestManager: Test management
- ExperimentRunner: Experiment execution
- StatisticalAnalyzer: Results analysis
- TestReporter: Report generation
```

## Extension Points

### Adding New Phases

1. **Create Phase Directory:**
```bash
mkdir phase6
touch phase6/__init__.py
```

2. **Define Phase Interface:**
```python
# phase6/base.py
from abc import ABC, abstractmethod

class PhaseInterface(ABC):
    @abstractmethod
    def initialize(self):
        """Initialize phase components"""
        pass

    @abstractmethod
    def process(self, data):
        """Process phase-specific data"""
        pass

    @abstractmethod
    def get_metrics(self):
        """Return phase metrics"""
        pass
```

3. **Implement Phase Logic:**
```python
# phase6/my_new_phase.py
from .base import PhaseInterface

class MyNewPhase(PhaseInterface):
    def initialize(self):
        # Phase initialization logic
        pass

    def process(self, data):
        # Phase processing logic
        return processed_data

    def get_metrics(self):
        # Return phase metrics
        return metrics
```

4. **Register Phase in Main App:**
```python
# app.py
from phase6.my_new_phase import MyNewPhase

# Initialize phase
new_phase = MyNewPhase()
new_phase.initialize()

# Add phase endpoints
@app.route('/phase6/endpoint', methods=['POST'])
def phase6_endpoint():
    return new_phase.process(request.get_json())
```

### Custom Embedding Models

1. **Create Model Interface:**
```python
# core/embedding_interface.py
from abc import ABC, abstractmethod

class EmbeddingInterface(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed single text"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch of texts"""
        pass
```

2. **Implement Custom Model:**
```python
# custom/my_embedding_model.py
from core.embedding_interface import EmbeddingInterface

class MyEmbeddingModel(EmbeddingInterface):
    def __init__(self, model_path: str):
        # Load your custom model
        self.model = load_model(model_path)

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)
```

3. **Register Model:**
```python
# app.py
from custom.my_embedding_model import MyEmbeddingModel

# Register custom model
embedding_models = {
    'default': OllamaEmbeddings(),
    'custom': MyEmbeddingModel('path/to/model')
}

def get_embedding_model(model_name='default'):
    return embedding_models.get(model_name, embedding_models['default'])
```

### Custom Query Processors

1. **Create Processor Interface:**
```python
# core/query_processor_interface.py
from abc import ABC, abstractmethod

class QueryProcessorInterface(ABC):
    @abstractmethod
    def process_query(self, query: str, context: dict) -> str:
        """Process query and return response"""
        pass
```

2. **Implement Custom Processor:**
```python
# custom/my_query_processor.py
from core.query_processor_interface import QueryProcessorInterface

class MyQueryProcessor(QueryProcessorInterface):
    def process_query(self, query: str, context: dict) -> str:
        # Custom query processing logic
        enhanced_query = self.enhance_query(query)
        response = self.generate_response(enhanced_query, context)
        return response
```

### Custom Feedback Analyzers

1. **Extend Base Analyzer:**
```python
# custom/my_feedback_analyzer.py
from phase1.feedback_analyzer import FeedbackAnalyzer

class MyFeedbackAnalyzer(FeedbackAnalyzer):
    def analyze_patterns(self, feedback_data):
        # Custom pattern analysis
        patterns = super().analyze_patterns(feedback_data)

        # Add custom analysis
        custom_patterns = self.custom_analysis(feedback_data)
        patterns.update(custom_patterns)

        return patterns

    def custom_analysis(self, feedback_data):
        # Your custom analysis logic
        return custom_patterns
```

## Customization Options

### Configuration Management

**Environment Variables:**
```bash
# .env file
LLM_MODEL=sixthwood
EMBEDDING_MODEL=mistral
CHROMA_PERSIST_DIR=./chroma_db
OLLAMA_BASE_URL=http://localhost:11434
RETRIEVAL_K=4
TEMPERATURE=0.7
MAX_TOKENS=500
DEBUG=false
LOG_LEVEL=INFO
```

**Configuration Class:**
```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class Config:
    llm_model: str = os.getenv('LLM_MODEL', 'sixthwood')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'mistral')
    chroma_persist_dir: str = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
    ollama_base_url: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    retrieval_k: int = int(os.getenv('RETRIEVAL_K', '4'))
    temperature: float = float(os.getenv('TEMPERATURE', '0.7'))
    max_tokens: int = int(os.getenv('MAX_TOKENS', '500'))
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

config = Config()
```

### Template Customization

**Custom Templates:**
```python
# custom/templates.py
CUSTOM_TEMPLATES = {
    "my_template": {
        "system_instruction": "You are a helpful assistant...",
        "base_template": "Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        "context_formats": {
            "initial": "Based on the following documents:\n{documents}",
            "follow_up": "Continuing our conversation:\n{conversation}\n\nNew documents:\n{documents}"
        }
    }
}

# Register templates
from template_manager import TemplateManager
template_manager = TemplateManager()
template_manager.register_templates(CUSTOM_TEMPLATES)
```

### Custom Metrics

**Define Custom Metrics:**
```python
# custom/metrics.py
from abc import ABC, abstractmethod

class MetricInterface(ABC):
    @abstractmethod
    def calculate(self, data: dict) -> float:
        pass

class CustomRelevanceMetric(MetricInterface):
    def calculate(self, data: dict) -> float:
        # Custom relevance calculation
        query = data['query']
        response = data['response']
        documents = data['documents']

        # Your custom logic here
        relevance_score = self.compute_relevance(query, response, documents)
        return relevance_score

# Register metric
from monitoring_dashboard import MetricsRegistry
metrics_registry = MetricsRegistry()
metrics_registry.register('custom_relevance', CustomRelevanceMetric())
```

## Testing Procedures

### Test Structure

```
tests/
├── conftest.py              # Test configuration
├── unit/                    # Unit tests
│   ├── test_embed.py
│   ├── test_query.py
│   ├── test_conversation.py
│   └── test_feedback.py
├── integration/             # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_workflow.py
│   └── test_phase_integration.py
├── performance/             # Performance tests
│   ├── test_load.py
│   ├── test_memory.py
│   └── test_response_time.py
└── fixtures/                # Test data
    ├── sample_documents/
    └── test_data.json
```

### Running Tests

**All Tests:**
```bash
pytest
```

**Specific Test Categories:**
```bash
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/performance/    # Performance tests only
```

**With Coverage:**
```bash
pytest --cov=. --cov-report=html
```

**Parallel Execution:**
```bash
pytest -n auto  # Auto-detect CPU cores
pytest -n 4     # Use 4 processes
```

### Writing Tests

**Unit Test Example:**
```python
# tests/unit/test_embed.py
import pytest
from unittest.mock import Mock, patch
from core.embed import embed

class TestEmbed:
    def test_embed_success(self, mock_file):
        """Test successful document embedding"""
        result = embed(mock_file, "test_collection")
        assert result is True

    def test_embed_invalid_file(self):
        """Test embedding with invalid file"""
        with pytest.raises(ValueError):
            embed(None, "test_collection")

    @patch('core.embed.get_vector_db')
    def test_embed_database_error(self, mock_db, mock_file):
        """Test embedding with database error"""
        mock_db.side_effect = Exception("Database error")
        result = embed(mock_file, "test_collection")
        assert result is False
```

**Integration Test Example:**
```python
# tests/integration/test_api_endpoints.py
import pytest
from app import app

class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json['status'] == 'healthy'

    def test_query_endpoint(self, client):
        """Test query endpoint"""
        data = {'query': 'test query'}
        response = client.post('/query', json=data)
        assert response.status_code == 200
        assert 'message' in response.json
```

### Test Configuration

**conftest.py:**
```python
# tests/conftest.py
import pytest
import tempfile
import os
from unittest.mock import Mock

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_file():
    """Mock file object for testing"""
    mock_file = Mock()
    mock_file.filename = 'test.pdf'
    mock_file.read.return_value = b'test content'
    return mock_file

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'TESTING': True,
        'CHROMA_PERSIST_DIR': ':memory:',
        'DEBUG': False
    }

@pytest.fixture(autouse=True)
def setup_test_env(test_config):
    """Setup test environment"""
    for key, value in test_config.items():
        os.environ[key] = str(value)
    yield
    # Cleanup
    for key in test_config.keys():
        os.environ.pop(key, None)
```

## CI/CD Integration

### GitHub Actions

**.github/workflows/ci.yml:**
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9

    - name: Install linting tools
      run: |
        pip install black flake8 pylint mypy

    - name: Run linting
      run: |
        black --check .
        flake8 .
        pylint **/*.py
        mypy .

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8

  build:
    needs: [test, lint, security]
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Build Docker image
      run: |
        docker build -t ragtime-gal:${{ github.sha }} .

    - name: Run container tests
      run: |
        docker run --rm ragtime-gal:${{ github.sha }} python -m pytest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to staging
      run: |
        echo "Deploy to staging environment"
        # Add deployment commands here
```

### Pre-commit Hooks

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
```

## Performance Optimization

### Profiling

**Memory Profiling:**
```python
# scripts/profile_memory.py
from memory_profiler import profile
from app import app

@profile
def test_query_memory():
    with app.test_client() as client:
        for i in range(100):
            response = client.post('/query', json={'query': f'test query {i}'})

if __name__ == '__main__':
    test_query_memory()
```

**Performance Profiling:**
```python
# scripts/profile_performance.py
import cProfile
import pstats
from app import app

def profile_app():
    with app.test_client() as client:
        for i in range(100):
            client.post('/query', json={'query': f'test query {i}'})

if __name__ == '__main__':
    cProfile.run('profile_app()', 'profile_stats')
    stats = pstats.Stats('profile_stats')
    stats.sort_stats('cumulative').print_stats(20)
```

### Optimization Techniques

**Caching:**
```python
# utils/cache.py
from functools import lru_cache
import redis

# Memory cache
@lru_cache(maxsize=1000)
def cached_embedding(text: str):
    return compute_embedding(text)

# Redis cache
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_query_result(query: str, ttl: int = 3600):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"query:{hash(query)}"
            cached_result = redis_client.get(cache_key)

            if cached_result:
                return json.loads(cached_result)

            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
```

**Async Processing:**
```python
# utils/async_processing.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncProcessor:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def process_documents_async(self, documents):
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.process_document, doc)
            for doc in documents
        ]
        return await asyncio.gather(*tasks)
```

**Database Optimization:**
```python
# utils/db_optimization.py
from contextlib import contextmanager

@contextmanager
def batch_operations(db, batch_size=100):
    """Context manager for batch database operations"""
    operations = []

    def add_operation(operation):
        operations.append(operation)
        if len(operations) >= batch_size:
            db.batch_execute(operations)
            operations.clear()

    yield add_operation

    # Execute remaining operations
    if operations:
        db.batch_execute(operations)
```

## Debugging Guide

### Logging Configuration

```python
# utils/logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(app):
    """Setup application logging"""

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = RotatingFileHandler(
        'logs/app.log', maxBytes=10485760, backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Configure app logger
    app.logger.addHandler(console_handler)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.DEBUG if app.debug else logging.INFO)

    # Configure other loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
```

### Debug Utilities

```python
# utils/debug.py
import functools
import time
import traceback
from flask import current_app

def debug_timer(func):
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            current_app.logger.debug(
                f"{func.__name__} executed in {execution_time:.4f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            current_app.logger.error(
                f"{func.__name__} failed after {execution_time:.4f} seconds: {str(e)}"
            )
            raise
    return wrapper

def debug_trace(func):
    """Decorator to trace function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        current_app.logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            current_app.logger.debug(f"{func.__name__} returned: {type(result)}")
            return result
        except Exception as e:
            current_app.logger.error(f"{func.__name__} raised {type(e).__name__}: {str(e)}")
            current_app.logger.debug(traceback.format_exc())
            raise
    return wrapper
```

### Common Issues and Solutions

**Issue: ChromaDB Connection Errors**
```python
# Solution: Add connection retry logic
import time
from functools import wraps

def retry_on_connection_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    current_app.logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
```

**Issue: Memory Leaks in Long-Running Processes**
```python
# Solution: Memory monitoring and cleanup
import gc
import psutil
import os

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    current_app.logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

    # Force garbage collection if memory usage is high
    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
        gc.collect()
        current_app.logger.info("Forced garbage collection")
```

## Contributing Guidelines

### Code Style

**Python Style Guide:**
- Follow PEP 8