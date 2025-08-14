# Output Length Configuration Guide

## Overview
This system has been enhanced to support **much larger model outputs** through multiple configuration improvements.

## Key Changes Made

### 1. Token Limit Increased
- **Previous**: Limited to 4,096 tokens (~3,000 words)
- **New Default**: 16,384 tokens (~12,000 words)
- **Maximum Possible**: Can be set even higher if needed

### 2. New Environment Variables

Add these to your `.env` file for output length control:

```bash
# Output length settings
MAX_OUTPUT_TOKENS=16384    # Maximum tokens the model can generate
REPEAT_PENALTY=1.1         # Reduces repetition (1.0 = no penalty, higher = less repetition)
TOP_K=40                   # Controls diversity (lower = more focused)
TOP_P=0.9                  # Nucleus sampling (0.0-1.0, lower = more focused)
```

### 3. Enhanced Model Parameters

The system now includes:
- **num_ctx=32768**: Larger context window for processing more information
- **verbose=True**: Better logging for debugging
- **Configurable sampling parameters**: Fine-tune response quality

### 4. Updated Prompt Templates

All prompt templates now:
- Explicitly request "comprehensive, detailed" responses
- Encourage thorough explanations and multiple paragraphs
- Ask for structured responses with clear sections
- Include phrases like "provide extensive detail" and "explore the topic fully"

## Usage Examples

### For Very Long Responses
Set in your `.env`:
```bash
MAX_OUTPUT_TOKENS=32768    # ~24,000 words
PROMPT_TEMPLATE=creative   # Uses enhanced creative prompts
```

### For Moderate Length Responses  
```bash
MAX_OUTPUT_TOKENS=8192     # ~6,000 words
PROMPT_TEMPLATE=standard   # Uses enhanced standard prompts
```

### For Specific Use Cases
- **Sixthwood style**: Uses `PROMPT_TEMPLATE=sixthwood` for immersive narrative responses
- **Technical documentation**: Uses `PROMPT_TEMPLATE=standard` for factual detail
- **Creative writing**: Uses `PROMPT_TEMPLATE=creative` for engaging, detailed responses

## Performance Notes

- **Higher token limits**: May increase response time but provide much more comprehensive answers
- **Memory usage**: Larger context windows use more RAM
- **Quality vs Quantity**: Use `REPEAT_PENALTY=1.2` or higher if responses become repetitive

## Testing Your Setup

1. Update your `.env` file with the new settings
2. Restart your application
3. Ask a complex question that would benefit from a long response
4. Check the logs for: `"max_tokens: 16384"` to confirm the new limit is active

## Troubleshooting

If responses are still short:
1. Verify `.env` file has the new `MAX_OUTPUT_TOKENS` setting
2. Check that Ollama has sufficient memory allocated
3. Try using `PROMPT_TEMPLATE=creative` for more encouraging prompts
4. Increase `MAX_OUTPUT_TOKENS` further if needed (32768, 65536, etc.)