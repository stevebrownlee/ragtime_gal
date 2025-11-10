# Current Module Inventory - Pre-Reorganization

**Date**: 2025-11-10
**Branch**: feature/project-maturity-reorganization
**Purpose**: Document all root-level modules before reorganization

## Root-Level Python Modules (25+)

### Core RAG Functionality
| Module | Purpose | Target Package | Priority |
|--------|---------|----------------|----------|
| `app.py` | Flask application entry point | `ragtime/app.py` | Critical |
| `embed.py` | Document embedding logic | `ragtime/core/embeddings.py` | High |
| `embed_enhanced.py` | Enhanced embedding with metadata | `ragtime/core/embeddings.py` (merge) | High |
| `query.py` | Query processing and response generation | `ragtime/core/query_processor.py` | Critical |

### Conversation Management
| Module | Purpose | Target Package | Priority |
|--------|---------|----------------|----------|
| `conversation.py` | Basic conversation management | `ragtime/services/conversation.py` | High |
| `enhanced_conversation.py` | Enhanced conversation with context | `ragtime/services/conversation.py` (merge) | High |
| `conversation_embedder.py` | Conversation embedding logic | `ragtime/services/conversation.py` (integrate) | Medium |
| `conversation_summarizer.py` | Conversation summarization | `ragtime/services/conversation.py` (integrate) | Medium |
| `context_manager.py` | Context management utilities | `ragtime/utils/context.py` | Medium |

### Feedback & Optimization (Phase 2/3)
| Module | Purpose | Target Package | Priority |
|--------|---------|----------------|----------|
| `feedback_analyzer.py` | Feedback pattern analysis | `ragtime/services/feedback_analyzer.py` | High |
| `query_enhancer.py` | Query enhancement with feedback | `ragtime/services/query_enhancer.py` | High |
| `training_data_generator.py` | Training data from feedback | `ragtime/services/training_data_gen.py` | Medium |
| `model_finetuner.py` | Model fine-tuning logic | `ragtime/services/model_finetuner.py` | Medium |

### Query Processing
| Module | Purpose | Target Package | Priority |
|--------|---------|----------------|----------|
| `query_classifier.py` | Query classification | `ragtime/core/query_processor.py` (integrate) | Medium |

### Templates & Prompts
| Module | Purpose | Target Package | Priority |
|--------|---------|----------------|----------|
| `prompts.py` | Prompt definitions | `ragtime/utils/templates.py` (merge) | Medium |
| `template_manager.py` | Template management | `ragtime/utils/templates.py` | Medium |
| `template.py` | Template utilities | `ragtime/utils/templates.py` (merge) | Low |

### Storage & Database
| Module | Purpose | Target Package | Priority |
|--------|---------|----------------|----------|
| `conport_client.py` | ConPort integration client | `ragtime/storage/conport_client.py` | Critical |

### Monitoring
| Module | Purpose | Target Package | Priority |
|--------|---------|----------------|----------|
| `monitoring_dashboard.py` | Monitoring dashboard | `ragtime/monitoring/dashboard.py` | Medium |

### Configuration Files
| File | Purpose | Status |
|------|---------|--------|
| `.env` | Environment variables | Keep in root |
| `.env.production` | Production config | Keep in root |
| `.env.template` | Config template | Keep in root |
| `Pipfile` | Dependencies | Keep in root |
| `Pipfile.lock` | Locked dependencies | Keep in root |

### Templates & Static Files
| File | Type | Target Location |
|------|------|----------------|
| `template.html` | HTML template | `ragtime/templates/` (Flask templates dir) |
| `prompt_templates.json` | Prompt templates | `ragtime/config/` or embed in code |

## Module Analysis Summary

**Total Root-Level Python Modules**: 19
**Total Configuration Files**: 5
**Total Template Files**: 2

### Modules to Merge
- `embed.py` + `embed_enhanced.py` → Single embeddings module
- `conversation.py` + `enhanced_conversation.py` + embedder + summarizer → Unified conversation service
- `prompts.py` + `template_manager.py` + `template.py` → Unified template utilities

### Modules to Keep Separate
- `feedback_analyzer.py` - Distinct service
- `query_enhancer.py` - Distinct service
- `training_data_generator.py` - Distinct service
- `model_finetuner.py` - Distinct service
- `conport_client.py` - Critical storage integration

### Critical Dependencies
1. **ConPort Integration**: `conport_client.py` must work with external MCP server
2. **Flask App**: `app.py` is the entry point - minimal refactoring
3. **Vector DB**: Chroma database operations in embed/query modules
4. **Feedback System**: Phase 2/3 modules are interconnected

## Migration Risks

### High Risk
- Breaking ConPort integration (external dependency)
- Breaking Flask routes and sessions
- Vector database connection issues

### Medium Risk
- Template path changes affecting Flask rendering
- Import path updates across all modules
- Conversation history compatibility

### Low Risk
- Utility function reorganization
- Configuration management updates
- Monitoring dashboard updates

## Testing Strategy

### Critical Tests
1. **ConPort Integration Tests**: Verify external MCP server communication
2. **Flask Route Tests**: Validate all API endpoints
3. **Vector DB Tests**: Ensure Chroma operations work
4. **Feedback Pipeline Tests**: Validate Phase 2/3 functionality

### Integration Tests
1. End-to-end query flow (upload → embed → query → response)
2. Conversation management (multi-turn queries)
3. Feedback collection and analysis
4. Training data generation

## Next Steps (Phase 1)

1. Create `ragtime/` package structure with `__init__.py` files
2. Implement Pydantic models in `ragtime/models/`
3. Create settings management in `ragtime/config/settings.py`
4. Set up structured logging configuration
5. Begin migrating storage layer (ConPort client first)