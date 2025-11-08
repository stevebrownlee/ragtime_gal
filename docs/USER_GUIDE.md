# User Guide: Feedback-Driven RAG System

## Table of Contents
1. [Getting Started](#getting-started)
2. [Using the Web Interface](#using-the-web-interface)
3. [Understanding the Feedback System](#understanding-the-feedback-system)
4. [Interpreting System Recommendations](#interpreting-system-recommendations)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Getting Started

### What is the Feedback-Driven RAG System?

This system is an intelligent document question-answering platform that learns from your feedback to provide better responses over time. It combines:

- **Document Embedding**: Upload PDFs and text files to create a searchable knowledge base
- **Conversational Memory**: Maintains context across multiple questions in a session
- **Feedback Learning**: Uses your ratings and feedback to improve future responses
- **Model Fine-tuning**: Automatically improves the underlying AI models based on usage patterns

### System Requirements

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection for initial setup
- Documents in PDF or Markdown format for uploading

## Using the Web Interface

### 1. Accessing the System

1. Open your web browser
2. Navigate to the system URL (typically `http://localhost:8084` for local installations)
3. You'll see the main interface with upload and query sections

### 2. Uploading Documents

**Step-by-Step Process:**

1. **Select Files**: Click "Choose Files" and select PDF or Markdown documents
2. **Collection Name**: (Optional) Enter a collection name to organize your documents
   - If left blank, documents go to the default "langchain" collection
   - Use descriptive names like "company-policies" or "technical-docs"
3. **Upload**: Click "Upload and Embed" to process your documents

**What Happens During Upload:**
- Documents are split into manageable chunks
- Each chunk is converted to vector embeddings
- Embeddings are stored in the searchable database
- You'll see a success message when complete

**Tips for Better Results:**
- Upload related documents together in the same collection
- Use clear, descriptive collection names
- Ensure documents are text-readable (not scanned images)

### 3. Asking Questions

**Basic Query Process:**

1. **Enter Your Question**: Type your question in the query field
2. **Select Response Style**: Choose from:
   - **Standard**: Balanced, informative responses
   - **Creative**: More engaging, conversational tone
   - **Sixth Wood**: Specialized style for specific use cases
3. **Adjust Temperature**: (Optional) Control response creativity
   - Lower values (0.1-0.3): More focused, factual responses
   - Higher values (0.7-0.9): More creative, varied responses
4. **Submit**: Click "Submit Query" to get your answer

**Understanding Responses:**

The system provides:
- **Main Answer**: Direct response to your question
- **Source Information**: Which documents were used
- **Conversation Context**: Whether this is a follow-up question
- **Confidence Indicators**: How well the system matched your query

### 4. Conversation Features

**Multi-Turn Conversations:**

The system remembers your conversation history within a session:

- **Follow-up Questions**: Ask related questions without repeating context
- **Clarifications**: Request more details or different perspectives
- **Context Awareness**: The system understands references to previous answers

**Example Conversation Flow:**
```
You: "What is our company's vacation policy?"
System: [Provides vacation policy details]

You: "How does that apply to part-time employees?"
System: [Understands "that" refers to vacation policy, provides specific info for part-time staff]

You: "What about sick leave?"
System: [Provides sick leave information, understanding the policy context]
```

**Managing Conversation History:**

- **View Status**: Check conversation length in the interface
- **Clear History**: Use "Clear History" button to start fresh
- **Session Persistence**: History is maintained during your browser session

## Understanding the Feedback System

### Why Feedback Matters

Your feedback directly improves the system's performance:

- **Response Quality**: Helps identify what makes a good answer
- **Relevance Tuning**: Improves document retrieval accuracy
- **Model Learning**: Trains the AI to better understand your domain
- **Personalization**: Adapts to your organization's specific needs

### How to Provide Effective Feedback

**Rating System (1-5 Stars):**

- ⭐ **1 Star**: Completely unhelpful or incorrect
- ⭐⭐ **2 Stars**: Mostly unhelpful, significant issues
- ⭐⭐⭐ **3 Stars**: Somewhat helpful, but missing key information
- ⭐⭐⭐⭐ **4 Stars**: Very helpful, minor improvements possible
- ⭐⭐⭐⭐⭐ **5 Stars**: Excellent, exactly what was needed

**Detailed Feedback Categories:**

1. **Relevance**: How well did the answer address your question?
   - "Completely relevant" to "Not relevant at all"

2. **Completeness**: Did the answer provide sufficient detail?
   - "Too brief" / "Just right" / "Too detailed"

3. **Length**: Was the response appropriately sized?
   - "Too short" / "Perfect length" / "Too long"

4. **Comments**: Free-text feedback for specific improvements
   - Mention missing information
   - Suggest better phrasing
   - Note factual errors

**Best Practices for Feedback:**

✅ **Do:**
- Provide feedback on both good and poor responses
- Be specific in comments about what was missing or wrong
- Rate consistently based on your actual needs
- Give feedback promptly while the context is fresh

❌ **Don't:**
- Rate based on personal preferences for writing style alone
- Give low ratings for correct but incomplete answers (use "completeness" instead)
- Provide feedback without reading the full response

### How the System Uses Your Feedback

**Immediate Effects:**
- Feedback is stored and analyzed for patterns
- Common issues are identified for quick fixes
- Query enhancement suggestions are generated

**Medium-term Improvements:**
- Document retrieval algorithms are optimized
- Response templates are refined
- Context selection is improved

**Long-term Learning:**
- AI models are fine-tuned on successful patterns
- New features are developed based on usage patterns
- System performance continuously improves

## Interpreting System Recommendations

### Query Enhancement Suggestions

The system may suggest improvements to your questions:

**Example Suggestions:**
- **Original**: "How do I get time off?"
- **Suggested**: "What is the process for requesting vacation time or personal leave?"

**Why Suggestions Help:**
- More specific queries get better answers
- Proper terminology matches document language
- Complete questions reduce ambiguity

### Response Quality Indicators

**Confidence Signals:**
- **High Confidence**: Multiple relevant documents found, clear answer
- **Medium Confidence**: Some relevant information, may need clarification
- **Low Confidence**: Limited relevant documents, answer may be incomplete

**Document Source Quality:**
- **Primary Sources**: Direct answers from authoritative documents
- **Supporting Sources**: Additional context from related documents
- **Inferred Information**: Logical conclusions drawn from available data

### Performance Metrics

**System Health Indicators:**
- **Response Time**: How quickly queries are processed
- **Accuracy Trends**: Whether responses are improving over time
- **User Satisfaction**: Average feedback ratings
- **Coverage**: Percentage of queries that find relevant documents

## Advanced Features

### Collection Management

**Organizing Documents:**
- Create topic-specific collections
- Query specific collections for focused results
- Manage document versions and updates

**Collection Strategies:**
- **By Department**: HR, Engineering, Sales, etc.
- **By Document Type**: Policies, Procedures, References
- **By Project**: Project-specific documentation
- **By Sensitivity**: Public, Internal, Confidential

### API Integration

For developers and advanced users, the system provides REST API endpoints:

**Key Endpoints:**
- `POST /embed`: Upload and embed documents
- `POST /query`: Submit questions programmatically
- `POST /feedback`: Submit feedback data
- `GET /collections`: List available document collections
- `GET /health`: Check system status

### Monitoring and Analytics

**Usage Analytics:**
- Query patterns and trends
- Popular document sections
- Response quality over time
- User engagement metrics

**Performance Monitoring:**
- System response times
- Resource utilization
- Error rates and types
- Feedback trends

## Troubleshooting

### Common Issues and Solutions

**Problem: "No relevant documents found"**

*Possible Causes:*
- Documents not uploaded to the correct collection
- Query terms don't match document language
- Documents failed to embed properly

*Solutions:*
1. Check that documents are uploaded and embedded successfully
2. Try different keywords or phrasing
3. Verify you're querying the correct collection
4. Check document format compatibility

**Problem: "Response is not relevant to my question"**

*Possible Causes:*
- Ambiguous question phrasing
- Missing context in follow-up questions
- Documents don't contain the requested information

*Solutions:*
1. Rephrase your question more specifically
2. Provide more context or background
3. Try breaking complex questions into parts
4. Verify the information exists in your uploaded documents

**Problem: "System is responding slowly"**

*Possible Causes:*
- High system load
- Large document collections
- Complex queries requiring extensive processing

*Solutions:*
1. Wait for current operations to complete
2. Try simpler, more focused queries
3. Clear conversation history if very long
4. Contact administrator if persistent

**Problem: "Conversation history seems incorrect"**

*Possible Causes:*
- Browser session issues
- Multiple tabs or windows open
- System restart during conversation

*Solutions:*
1. Clear conversation history and start fresh
2. Use only one browser tab for the system
3. Refresh the page and try again

### Error Messages

**"File upload failed"**
- Check file format (PDF or Markdown only)
- Ensure file is not corrupted
- Verify file size is reasonable
- Try uploading one file at a time

**"Query processing error"**
- Check internet connection
- Try a simpler query first
- Clear browser cache and cookies
- Contact support if error persists

**"Feedback submission failed"**
- Ensure you've provided a rating
- Check that the response is still visible
- Try refreshing the page
- Submit feedback immediately after receiving response

### Getting Help

**Self-Service Resources:**
1. Check this user guide first
2. Review the FAQ section below
3. Try the troubleshooting steps
4. Test with simple queries to isolate issues

**Contacting Support:**
When contacting support, please provide:
- Exact error message (if any)
- Steps you took before the issue occurred
- Browser type and version
- Screenshot of the problem (if applicable)
- Example query that's not working

## FAQ

### General Questions

**Q: How long does it take to embed documents?**
A: Embedding time depends on document size and system load. Typical times:
- Small documents (1-10 pages): 30 seconds - 2 minutes
- Medium documents (10-50 pages): 2-10 minutes
- Large documents (50+ pages): 10+ minutes

**Q: Can I upload the same document multiple times?**
A: Yes, but it will create duplicate entries. It's better to organize documents into collections and avoid re-uploading unless the content has changed significantly.

**Q: How many documents can I upload?**
A: There's no hard limit, but performance may decrease with very large collections (1000+ documents). Consider organizing into multiple focused collections.

**Q: Does the system work offline?**
A: No, the system requires an internet connection for AI processing and document embedding.

### Feedback and Learning

**Q: How quickly does the system learn from my feedback?**
A: Feedback effects vary:
- Immediate: Feedback is stored and contributes to analytics
- Short-term (hours-days): Query enhancement suggestions improve
- Long-term (weeks-months): Model fine-tuning incorporates patterns

**Q: Can I see how my feedback is being used?**
A: The system provides general analytics about feedback trends and improvements, but individual feedback items are kept private.

**Q: What happens if I give conflicting feedback?**
A: The system uses statistical analysis to identify patterns across all feedback. Occasional conflicting ratings are normal and don't harm the learning process.

### Privacy and Security

**Q: Is my data secure?**
A: Yes, the system includes:
- Local data storage options
- Session-based conversation management
- No external data sharing without explicit configuration
- Secure document handling

**Q: Can other users see my questions and documents?**
A: By default, no. Each session is isolated. However, administrators may configure shared collections for organizational use.

**Q: How long is my data stored?**
A: Data retention depends on system configuration:
- Conversation history: Session-based (cleared when browser closes)
- Documents: Persistent until manually removed
- Feedback: Stored for system improvement (anonymized)

### Technical Questions

**Q: What file formats are supported?**
A: Currently supported formats:
- PDF files (text-based, not scanned images)
- Markdown (.md) files
- Plain text files

**Q: Can I integrate this with other systems?**
A: Yes, the system provides REST API endpoints for integration with other applications. See the API documentation for details.

**Q: How do I backup my data?**
A: Contact your system administrator for backup procedures. The system stores data in standard formats that can be exported.

**Q: Can I customize the response styles?**
A: Response styles are configured by administrators. Users can select from available styles but cannot create custom ones through the interface.

---

*This user guide is regularly updated. For the latest version and additional resources, check the system documentation or contact your administrator.*