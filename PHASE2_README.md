# Phase 2: Retrieval Optimization

## Overview

Phase 2 implements intelligent query enhancement and document retrieval optimization based on user feedback patterns collected through Phase 1. This phase transforms feedback data into actionable insights that improve the relevance and quality of retrieved documents and generated responses.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                Phase 2: Retrieval Optimization System               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Feedback Analysis Pipeline                   │  │
│  │                                                                 │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐      │  │
│  │  │   ConPort   │───▶│   Feedback   │───▶│  Patterns   │      │  │
│  │  │  Feedback   │    │   Analyzer   │    │  & Insights │      │  │
│  │  │   Storage   │    │              │    │             │      │  │
│  │  └─────────────┘    └──────────────┘    └─────────────┘      │  │
│  │                                                   │             │  │
│  │                                                   ▼             │  │
│  │                                          ┌─────────────┐       │  │
│  │                                          │ Successful  │       │  │
│  │                                          │  Query      │       │  │
│  │                                          │ Patterns    │       │  │
│  │                                          └─────────────┘       │  │
│  │                                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Query Enhancement Pipeline                    │  │
│  │                                                                 │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐      │  │
│  │  │   Original  │───▶│    Query     │───▶│  Enhanced   │      │  │
│  │  │    Query    │    │   Enhancer   │    │    Query    │      │  │
│  │  │             │    │              │    │             │      │  │
│  │  └─────────────┘    └──────────────┘    └─────────────┘      │  │
│  │                            │                      │            │  │
│  │                            ▼                      ▼            │  │
│  │                    ┌──────────────┐    ┌─────────────┐       │  │
│  │                    │   Pattern    │    │   Query     │       │  │
│  │                    │   Analysis   │    │ Expansion   │       │  │
│  │                    └──────────────┘    └─────────────┘       │  │
│  │                                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                 Retrieval Optimization Pipeline                 │  │
│  │                                                                 │  │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐      │  │
│  │  │  Enhanced   │───▶│   Vector     │───▶│  Retrieved  │      │  │
│  │  │   Query     │    │  Search with │    │  Documents  │      │  │
│  │  │             │    │  Adaptive    │    │             │      │  │
│  │  └─────────────┘    │  Threshold   │    └─────────────┘      │  │
│  │                     └──────────────┘            │             │  │
│  │                                                  ▼             │  │
│  │                                         ┌─────────────┐       │  │
│  │                                         │  Document   │       │  │
│  │                                         │  Re-ranking │       │  │
│  │                                         │             │       │  │
│  │                                         └─────────────┘       │  │
│  │                                                  │             │  │
│  │                                                  ▼             │  │
│  │                                         ┌─────────────┐       │  │
│  │                                         │  Optimized  │       │  │
│  │                                         │   Results   │       │  │
│  │                                         └─────────────┘       │  │
│  │                                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Feedback Analyzer

**File**: [`feedback_analyzer.py`](feedback_analyzer.py)

Analyzes user feedback patterns to identify successful query characteristics and problematic patterns.

**Features**:
- Retrieves feedback data from ConPort's UserFeedback category
- Analyzes rating distributions and trends
- Identifies successful query patterns (high-rated responses)
- Identifies problematic query patterns (low-rated responses)
- Generates actionable insights and recommendations
- Provides feedback summaries and analytics

**Key Classes**:
- `FeedbackAnalyzer`: Main class for feedback pattern analysis
- Provides methods for rating analysis, pattern identification, and insight generation

**Usage Example**:
```python
from feedback_analyzer import create_feedback_analyzer
from conport_client import ConPortClient

# Initialize with ConPort client
conport_client = ConPortClient()
workspace_id = "/path/to/workspace"

analyzer = create_feedback_analyzer(
    conport_client=conport_client,
    workspace_id=workspace_id
)

# Get feedback data
feedback_data = analyzer.get_feedback_data(days_back=30, min_rating=4)

# Analyze rating patterns
analysis = analyzer.analyze_rating_patterns(feedback_data)
print(f"Average rating: {analysis['average_rating']}")
print(f"High-rated percentage: {analysis['high_rated_percentage']}%")

# Identify successful patterns
patterns = analyzer.identify_successful_patterns(days_back=30)
print(f"Successful query characteristics: {patterns['successful_query_characteristics']}")

# Get insights
insights = analyzer.generate_insights(days_back=30)
print(f"Recommendations: {insights['recommendations']}")
```

### 2. Query Enhancer

**File**: [`query_enhancer.py`](query_enhancer.py)

Enhances queries and optimizes retrieval based on feedback-driven insights from the FeedbackAnalyzer.

**Features**:
- Query expansion based on successful patterns
- Query reformatting using proven structures
- Adaptive similarity thresholds for retrieval
- Document re-ranking based on feedback patterns
- Enhancement suggestions for users
- Caching for performance optimization

**Key Classes**:
- `QueryEnhancer`: Main class for query enhancement and retrieval optimization

**Key Methods**:
- `enhance_query()`: Enhances a query using feedback patterns
- `rerank_documents()`: Re-ranks retrieved documents based on patterns
- `get_adaptive_similarity_threshold()`: Calculates optimal similarity threshold
- `get_enhancement_suggestions()`: Provides query improvement suggestions

**Usage Example**:
```python
from query_enhancer import create_query_enhancer
from feedback_analyzer import create_feedback_analyzer

# Initialize components
analyzer = create_feedback_analyzer(
    conport_client=conport_client,
    workspace_id=workspace_id
)
enhancer = create_query_enhancer(feedback_analyzer=analyzer)

# Enhance a query
original_query = "How do I implement authentication?"
result = enhancer.enhance_query(original_query, enhancement_mode="auto")

print(f"Original: {result['original_query']}")
print(f"Enhanced: {result['enhanced_query']}")
print(f"Enhancements: {result['enhancements_applied']}")
print(f"Confidence: {result['confidence']}")

# Get adaptive threshold for retrieval
threshold = enhancer.get_adaptive_similarity_threshold(result['enhanced_query'])
print(f"Recommended similarity threshold: {threshold}")

# Re-rank documents based on patterns
documents = [...]  # Retrieved documents
reranked_docs = enhancer.rerank_documents(documents, result['enhanced_query'])
```

### 3. Integration with Query Pipeline

**File**: [`query.py`](query.py)

The query enhancement system is fully integrated into the main query pipeline, activated by the `USE_FEEDBACK_OPTIMIZATION` environment variable.

**Integration Points**:
1. **Query Enhancement** (line ~110): Original query is enhanced before retrieval
2. **Adaptive Thresholds** (line ~153): Similarity thresholds adjusted based on patterns
3. **Document Re-ranking** (line ~179): Retrieved documents re-ranked for relevance
4. **Metadata Tracking**: Enhancement metadata included in responses

**Flow**:
```
User Query → Query Enhancement → Vector Search (with adaptive threshold)
→ Document Re-ranking → LLM Generation → Response
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Phase 2: Retrieval Optimization
USE_FEEDBACK_OPTIMIZATION=true          # Enable/disable optimization
QUERY_ENHANCEMENT_ENABLED=true          # Enable query enhancement
FEEDBACK_ANALYTICS_ENABLED=true         # Enable feedback analytics

# Optimization Parameters
MIN_FEEDBACK_SAMPLES=10                 # Minimum feedback samples needed
PATTERN_ANALYSIS_DAYS_BACK=30          # Days of feedback to analyze
QUERY_EXPANSION_MAX_TERMS=3            # Max terms to add in expansion
ADAPTIVE_THRESHOLD_MIN=0.5             # Minimum similarity threshold
ADAPTIVE_THRESHOLD_MAX=0.9             # Maximum similarity threshold
RERANK_TOP_K=10                        # Number of documents to re-rank

# Cache Settings
ENHANCEMENT_CACHE_TTL=3600             # Cache TTL in seconds (1 hour)
PATTERN_CACHE_TTL=1800                 # Pattern cache TTL (30 minutes)
```

### Programmatic Configuration

```python
import os

# Enable optimization features
os.environ['USE_FEEDBACK_OPTIMIZATION'] = 'true'
os.environ['QUERY_ENHANCEMENT_ENABLED'] = 'true'

# Configure analyzer
analyzer = create_feedback_analyzer(
    conport_client=conport_client,
    workspace_id=workspace_id
)

# Configure enhancer
enhancer = create_query_enhancer(feedback_analyzer=analyzer)
```

## API Usage

### Get Feedback Analytics

**Endpoint**: `GET /feedback/analytics?days_back=30&min_rating=4`

Retrieves comprehensive feedback analytics and insights.

**Response Example**:
```json
{
  "summary": {
    "total_feedback": 245,
    "average_rating": 4.2,
    "rating_distribution": {
      "5": 98,
      "4": 87,
      "3": 35,
      "2": 15,
      "1": 10
    },
    "high_rated_percentage": 75.5
  },
  "successful_patterns": {
    "common_terms": ["authentication", "security", "implementation"],
    "query_structures": ["How do I...", "What is the best way to..."],
    "avg_query_length": 12
  },
  "problematic_patterns": {
    "common_issues": ["vague queries", "too broad"],
    "low_rated_queries": [...]
  },
  "recommendations": {
    "query_formulation": "Use specific technical terms",
    "query_structure": "Start with 'How do I' for implementation questions",
    "query_length": "Aim for 10-15 words"
  }
}
```

### Get Enhancement Suggestions

This functionality is available through the `/feedback/analytics` endpoint and provides suggestions based on query patterns.

### Clear Optimization Caches

**Endpoint**: `POST /optimization/clear-cache`

Note: This endpoint would need to be added to clear enhancement and pattern caches.

## Integration with ConPort

Phase 2 deeply integrates with ConPort's feedback storage:

1. **Feedback Retrieval**: Uses `search_custom_data_value_fts` to query UserFeedback
2. **Pattern Storage**: Can store successful patterns as custom data
3. **Analytics Tracking**: Feedback analytics can be logged for monitoring

### Example ConPort Integration

```python
from conport_client import ConPortClient
from feedback_analyzer import create_feedback_analyzer
from query_enhancer import create_query_enhancer

# Initialize ConPort
conport_client = ConPortClient()
workspace_id = "/path/to/workspace"

# Create analyzer
analyzer = create_feedback_analyzer(
    conport_client=conport_client,
    workspace_id=workspace_id
)

# Analyze feedback
feedback_data = analyzer.get_feedback_data(days_back=30)
patterns = analyzer.identify_successful_patterns(days_back=30)

# Store pattern analysis results in ConPort
conport_client.log_custom_data({
    "workspace_id": workspace_id,
    "category": "OptimizationPatterns",
    "key": f"patterns_{datetime.now().strftime('%Y%m%d')}",
    "value": patterns
})
```

## Optimization Strategies

### 1. Query Expansion

Adds relevant terms to queries based on successful patterns:

```python
# Original: "implement auth"
# Enhanced: "implement authentication security best practices"
```

**When to Use**: Short queries, technical abbreviations, ambiguous terms

### 2. Query Reformatting

Restructures queries based on successful formulations:

```python
# Original: "auth broken"
# Enhanced: "How do I fix authentication issues?"
```

**When to Use**: Vague queries, informal language, incomplete questions

### 3. Adaptive Similarity Thresholds

Adjusts vector search thresholds based on query characteristics:

- **Technical queries**: Higher threshold (0.75-0.85) for precision
- **General queries**: Lower threshold (0.60-0.70) for recall
- **Follow-up queries**: Medium threshold (0.65-0.75)

### 4. Document Re-ranking

Re-orders retrieved documents based on successful patterns:

- **Boost**: Documents matching successful query patterns
- **Demote**: Documents associated with low-rated responses
- **Recency**: Recent, well-rated documents scored higher

## Performance Metrics

### Query Enhancement Metrics

- **Enhancement Rate**: Percentage of queries enhanced
- **Avg Confidence**: Average confidence score of enhancements
- **Enhancements Applied**: Types and frequency of enhancements

### Retrieval Optimization Metrics

- **Avg Similarity Threshold**: Average adaptive threshold used
- **Documents Re-ranked**: Percentage of results re-ranked
- **Re-ranking Impact**: Position changes from re-ranking

### User Impact Metrics

- **Rating Improvement**: Average rating change with optimization
- **User Satisfaction**: Satisfaction score trends
- **Query Success Rate**: Percentage of high-rated responses

## Best Practices

### 1. Data Collection

- Collect at least 50-100 feedback samples before enabling optimization
- Maintain balanced feedback (mix of ratings)
- Regularly refresh pattern analysis (daily or weekly)

### 2. Enhancement Configuration

- Start with `enhancement_mode="auto"` for balanced enhancement
- Use `"expand"` for short queries, `"rephrase"` for vague queries
- Monitor confidence scores; only apply high-confidence enhancements

### 3. Threshold Management

- Set `ADAPTIVE_THRESHOLD_MIN` based on your corpus quality
- Allow wider range for diverse query types
- Monitor retrieval precision and recall

### 4. Re-ranking Strategy

- Re-rank top-k results (10-20) for performance
- Consider both pattern matching and recency
- A/B test re-ranking impact

### 5. Cache Management

- Use appropriate TTLs based on feedback frequency
- Clear caches after pattern updates
- Monitor cache hit rates

## Troubleshooting

### Common Issues

**Issue**: "Insufficient feedback data"
```bash
Solution: Collect more user feedback before enabling optimization
Minimum recommended: 50-100 samples with varied ratings
```

**Issue**: "No patterns identified"
```bash
Solution:
- Check feedback quality (ratings and comments)
- Increase analysis window (days_back parameter)
- Ensure ConPort connection is working
```

**Issue**: "Enhanced queries too different from originals"
```bash
Solution:
- Use more conservative enhancement mode
- Adjust pattern matching thresholds
- Review successful pattern criteria
```

**Issue**: "Re-ranking degrades results"
```bash
Solution:
- Disable re-ranking temporarily
- Review re-ranking criteria
- Check if patterns are still valid
- Consider A/B testing impact
```

## Monitoring and Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('query_enhancer')
logger.setLevel(logging.DEBUG)
```

### Monitor Enhancement Impact

```python
# Track enhancement metadata in responses
metadata = response_metadata.get('optimization_metadata', {})
print(f"Query enhanced: {metadata.get('query_enhanced', False)}")
print(f"Enhancements: {metadata.get('enhancements_applied', [])}")
print(f"Documents re-ranked: {metadata.get('documents_reranked', False)}")
```

### Analyze Pattern Quality

```python
patterns = analyzer.identify_successful_patterns(days_back=30)
print(f"Successful queries: {len(patterns['successful_queries'])}")
print(f"Pattern confidence: {patterns.get('confidence', 'N/A')}")
```

## Future Enhancements

### Planned Features

1. **Semantic Query Understanding**: Use embeddings for deeper query analysis
2. **User-specific Optimization**: Personalize based on user feedback history
3. **Multi-language Support**: Pattern analysis for non-English queries
4. **Real-time Learning**: Update patterns in real-time as feedback arrives
5. **Advanced Re-ranking**: Machine learning-based document scoring

### Research Opportunities

1. **Query Intent Classification**: Categorize queries by intent for targeted optimization
2. **Contextual Enhancement**: Use conversation context for better enhancement
3. **Feedback Loop Analysis**: Study long-term impact of optimizations
4. **Cross-domain Patterns**: Identify patterns across different document types

## Integration with Phase 3

Phase 2 optimization provides training data for Phase 3 model fine-tuning:

- **Successful patterns** → Positive training examples
- **Problematic patterns** → Negative training examples
- **Enhanced queries** → Query reformulation training
- **Re-ranking scores** → Relevance training labels

This creates a feedback loop where Phase 2 insights improve Phase 3 models, which in turn enhance Phase 2 optimization.

## References

### Documentation
- [`README.md`](README.md) - Main system documentation
- [`PHASE1_README.md`](PHASE1_README.md) - Feedback collection system
- [`PHASE3_README.md`](PHASE3_README.md) - Model fine-tuning system
- [`docs/CONPORT_INTEGRATION.md`](docs/CONPORT_INTEGRATION.md) - ConPort integration guide

### Code References
- [`feedback_analyzer.py`](feedback_analyzer.py) - Feedback analysis implementation
- [`query_enhancer.py`](query_enhancer.py) - Query enhancement implementation
- [`query.py`](query.py) - Integration in query pipeline

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example code in [`query_enhancer.py`](query_enhancer.py)
3. Examine [`feedback_analyzer.py`](feedback_analyzer.py) implementation
4. Consult the main [`README.md`](README.md) for system overview