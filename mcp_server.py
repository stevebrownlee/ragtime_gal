#!/usr/bin/env python3
"""
MCP server for reading and searching through markdown book files.
Uses the official MCP Python SDK with enhanced NLP capabilities.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Enhanced imports for NLP analysis
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load spaCy model (will be loaded on first use)
    nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
    # Initialize LanguageTool (will be loaded on first use)
    language_tool = None
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    language_tool = None

# Initialize the MCP server
mcp = FastMCP("Book MCP Server")

# Default book directory - can be overridden by environment variable
BOOK_DIRECTORY = os.getenv("BOOK_DIRECTORY", ".")

@mcp.tool()
def list_chapters() -> List[Dict[str, Any]]:
    """
    List all chapter files with enhanced metadata including titles and word counts.

    Returns:
        List of dictionaries containing chapter information
    """
    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    # Find all markdown files recursively
    md_files = list(book_path.glob("**/*.md"))

    result = []
    for file_path in md_files:
        relative_path = file_path.relative_to(book_path)

        # Try to extract chapter title from first header
        chapter_title = "Untitled"
        word_count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()

                # Look for first markdown header
                for line in lines:
                    if line.strip().startswith('#'):
                        chapter_title = line.strip().lstrip('#').strip()
                        break

                # Count words (rough estimate)
                word_count = len(content.split())
        except (UnicodeDecodeError, PermissionError):
            pass

        result.append({
            "filename": file_path.name,
            "relative_path": str(relative_path),
            "absolute_path": str(file_path),
            "chapter_title": chapter_title,
            "word_count": word_count,
            "size_bytes": str(file_path.stat().st_size if file_path.exists() else 0)
        })

    # Sort by filename to maintain some order
    result.sort(key=lambda x: x["filename"])
    return result

@mcp.tool()
def extract_all_characters() -> Dict[str, Any]:
    """
    Automatically discover all character names in the book using advanced NLP.

    Returns:
        Dictionary with all discovered characters, their appearances, and context
    """
    if not SPACY_AVAILABLE:
        return {"error": "spaCy library not available. Install with: pipenv install spacy && python -m spacy download en_core_web_sm"}

    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return {"error": "spaCy English model not found. Run: python -m spacy download en_core_web_sm"}

    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    characters = {}
    chapter_count = 0

    for file_path in book_path.glob("**/*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chapter_count += 1

                # Get chapter title
                chapter_title = "Untitled"
                lines = content.splitlines()
                for line in lines:
                    if line.strip().startswith('#'):
                        chapter_title = line.strip().lstrip('#').strip()
                        break

                # Process with spaCy
                doc = nlp(content)

                # Find all person entities
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and len(ent.text.strip()) > 1:
                        name = ent.text.strip()
                        # Filter out common false positives
                        if name.lower() not in ['i', 'me', 'you', 'he', 'she', 'they', 'we', 'us']:
                            if name not in characters:
                                characters[name] = {
                                    "total_mentions": 0,
                                    "chapters": [],
                                    "contexts": [],
                                    "first_appearance": chapter_title
                                }

                            characters[name]["total_mentions"] += 1

                            # Add chapter if not already recorded
                            if chapter_title not in [ch["chapter"] for ch in characters[name]["chapters"]]:
                                characters[name]["chapters"].append({
                                    "chapter": chapter_title,
                                    "filename": file_path.name,
                                    "mentions_in_chapter": 1
                                })
                            else:
                                # Increment mentions in this chapter
                                for ch in characters[name]["chapters"]:
                                    if ch["chapter"] == chapter_title:
                                        ch["mentions_in_chapter"] += 1
                                        break

                            # Add context (limit to first 5 per character)
                            if len(characters[name]["contexts"]) < 5:
                                characters[name]["contexts"].append({
                                    "chapter": chapter_title,
                                    "sentence": ent.sent.text.strip(),
                                    "context_before": ent.sent.text[:ent.start_char - ent.sent.start_char].strip(),
                                    "context_after": ent.sent.text[ent.end_char - ent.sent.start_char:].strip()
                                })

        except (UnicodeDecodeError, PermissionError):
            continue

    # Sort characters by frequency
    sorted_characters = dict(sorted(characters.items(), key=lambda x: x[1]["total_mentions"], reverse=True))

    # Analysis
    main_characters = {name: data for name, data in sorted_characters.items() if data["total_mentions"] > 5}
    secondary_characters = {name: data for name, data in sorted_characters.items() if 2 <= data["total_mentions"] <= 5}
    minor_characters = {name: data for name, data in sorted_characters.items() if data["total_mentions"] == 1}

    return {
        "summary": {
            "total_characters_found": len(characters),
            "chapters_processed": chapter_count,
            "main_characters": len(main_characters),
            "secondary_characters": len(secondary_characters),
            "minor_characters": len(minor_characters)
        },
        "character_analysis": {
            "main_characters": dict(list(main_characters.items())[:10]),  # Top 10
            "secondary_characters": secondary_characters,
            "minor_characters": dict(list(minor_characters.items())[:20])  # First 20 minor
        },
        "insights": {
            "most_mentioned": list(sorted_characters.keys())[0] if sorted_characters else "None",
            "characters_in_multiple_chapters": sum(1 for char in characters.values() if len(char["chapters"]) > 1),
            "single_chapter_characters": sum(1 for char in characters.values() if len(char["chapters"]) == 1)
        },
        "all_characters": sorted_characters
    }

@mcp.tool()
def analyze_character_relationships() -> Dict[str, Any]:
    """
    Analyze which characters appear together in scenes and chapters.

    Returns:
        Dictionary with character relationship analysis and co-occurrence patterns
    """
    if not SPACY_AVAILABLE:
        return {"error": "spaCy library not available. Install with: pipenv install spacy && python -m spacy download en_core_web_sm"}

    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return {"error": "spaCy English model not found. Run: python -m spacy download en_core_web_sm"}

    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    # First, get all characters
    characters_data = extract_all_characters()
    if "error" in characters_data:
        return characters_data

    all_characters = list(characters_data["all_characters"].keys())
    relationships = {}
    scene_cooccurrences = {}

    for file_path in book_path.glob("**/*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Get chapter title
                chapter_title = "Untitled"
                lines = content.splitlines()
                for line in lines:
                    if line.strip().startswith('#'):
                        chapter_title = line.strip().lstrip('#').strip()
                        break

                # Split into scenes (rough heuristic: double line breaks or scene breaks)
                scenes = content.split('\n\n')

                for scene_idx, scene in enumerate(scenes):
                    if len(scene.strip()) < 100:  # Skip very short scenes
                        continue

                    # Find characters in this scene
                    doc = nlp(scene)
                    scene_characters = set()

                    for ent in doc.ents:
                        if ent.label_ == "PERSON" and ent.text.strip() in all_characters:
                            scene_characters.add(ent.text.strip())

                    # Record co-occurrences
                    scene_characters = list(scene_characters)
                    for i, char1 in enumerate(scene_characters):
                        for char2 in scene_characters[i+1:]:
                            # Create a sorted pair to avoid duplicates
                            pair = tuple(sorted([char1, char2]))

                            if pair not in relationships:
                                relationships[pair] = {
                                    "total_scenes": 0,
                                    "chapters": set(),
                                    "scene_contexts": []
                                }

                            relationships[pair]["total_scenes"] += 1
                            relationships[pair]["chapters"].add(chapter_title)

                            if len(relationships[pair]["scene_contexts"]) < 3:  # Limit examples
                                relationships[pair]["scene_contexts"].append({
                                    "chapter": chapter_title,
                                    "scene_preview": scene[:200] + "..." if len(scene) > 200 else scene
                                })

        except (UnicodeDecodeError, PermissionError):
            continue

    # Convert sets to lists for JSON serialization
    for pair_data in relationships.values():
        pair_data["chapters"] = list(pair_data["chapters"])

    # Sort by frequency
    sorted_relationships = dict(sorted(relationships.items(), key=lambda x: x[1]["total_scenes"], reverse=True))

    # Analyze relationship types
    strong_relationships = {pair: data for pair, data in sorted_relationships.items() if data["total_scenes"] > 5}
    moderate_relationships = {pair: data for pair, data in sorted_relationships.items() if 2 <= data["total_scenes"] <= 5}

    return {
        "summary": {
            "total_character_pairs": len(relationships),
            "strong_relationships": len(strong_relationships),
            "moderate_relationships": len(moderate_relationships),
            "characters_analyzed": len(all_characters)
        },
        "strong_relationships": {
            f"{pair[0]} & {pair[1]}": {
                "scenes_together": data["total_scenes"],
                "chapters_together": len(data["chapters"]),
                "relationship_strength": "Strong" if data["total_scenes"] > 10 else "Moderate",
                "chapters": data["chapters"],
                "example_scenes": data["scene_contexts"]
            }
            for pair, data in list(strong_relationships.items())[:10]
        },
        "moderate_relationships": {
            f"{pair[0]} & {pair[1]}": {
                "scenes_together": data["total_scenes"],
                "chapters_together": len(data["chapters"]),
                "chapters": data["chapters"]
            }
            for pair, data in list(moderate_relationships.items())[:10]
        },
        "insights": {
            "most_connected_pairs": [f"{pair[0]} & {pair[1]}" for pair in list(sorted_relationships.keys())[:5]],
            "isolated_characters": [char for char in all_characters
                                  if not any(char in pair for pair in relationships.keys())],
            "relationship_density": round(len(relationships) / (len(all_characters) * (len(all_characters) - 1) / 2) * 100, 2) if len(all_characters) > 1 else 0
        }
    }

@mcp.tool()
def analyze_readability(chapter_path: str) -> Dict[str, Any]:
    """
    Calculate comprehensive readability metrics for a specific chapter.

    Args:
        chapter_path: Path to the chapter file to analyze

    Returns:
        Dictionary with multiple readability scores and reading level analysis
    """
    if not TEXTSTAT_AVAILABLE:
        return {"error": "textstat library not available. Install with: pipenv install textstat"}

    # Read the chapter content
    file_info = read_markdown_file(chapter_path)
    content = file_info["content"]

    # Clean content for analysis (remove markdown syntax)
    import re
    clean_content = re.sub(r'#{1,6}\s+', '', content)  # Remove headers
    clean_content = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', clean_content)  # Remove bold/italic
    clean_content = re.sub(r'`(.*?)`', r'\1', clean_content)  # Remove code
    clean_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_content)  # Remove links, keep text

    if len(clean_content.strip()) < 100:
        return {"error": "Chapter too short for meaningful readability analysis"}

    try:
        # Calculate various readability metrics
        flesch_kincaid = textstat.flesch_kincaid().grade(clean_content)
        flesch_reading_ease = textstat.flesch_reading_ease(clean_content)
        gunning_fog = textstat.gunning_fog(clean_content)
        coleman_liau = textstat.coleman_liau_index(clean_content)
        automated_readability = textstat.automated_readability_index(clean_content)
        smog = textstat.smog_index(clean_content)

        # Basic text statistics
        sentence_count = textstat.sentence_count(clean_content)
        word_count = textstat.lexicon_count(clean_content)
        char_count = textstat.char_count(clean_content)
        syllable_count = textstat.syllable_count(clean_content)

        # Average metrics
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_syllables_per_word = syllable_count / word_count if word_count > 0 else 0

        # Determine reading level
        if flesch_reading_ease >= 90:
            reading_level = "Very Easy (5th grade)"
        elif flesch_reading_ease >= 80:
            reading_level = "Easy (6th grade)"
        elif flesch_reading_ease >= 70:
            reading_level = "Fairly Easy (7th grade)"
        elif flesch_reading_ease >= 60:
            reading_level = "Standard (8th-9th grade)"
        elif flesch_reading_ease >= 50:
            reading_level = "Fairly Difficult (10th-12th grade)"
        elif flesch_reading_ease >= 30:
            reading_level = "Difficult (College level)"
        else:
            reading_level = "Very Difficult (Graduate level)"

        # Readability assessment for fiction
        fiction_assessment = ""
        if flesch_reading_ease >= 70:
            fiction_assessment = "Excellent for general fiction - accessible to most readers"
        elif flesch_reading_ease >= 50:
            fiction_assessment = "Good for fiction - appropriate for teen/adult readers"
        elif flesch_reading_ease >= 30:
            fiction_assessment = "May be challenging - consider simplifying some sentences"
        else:
            fiction_assessment = "Very complex - may lose general fiction readers"

        return {
            "chapter_info": {
                "filename": file_info["filename"],
                "word_count": word_count,
                "sentence_count": sentence_count,
                "character_count": char_count,
                "syllable_count": syllable_count
            },
            "readability_scores": {
                "flesch_kincaid_grade": round(flesch_kincaid, 1),
                "flesch_reading_ease": round(flesch_reading_ease, 1),
                "gunning_fog_index": round(gunning_fog, 1),
                "coleman_liau_index": round(coleman_liau, 1),
                "automated_readability_index": round(automated_readability, 1),
                "smog_index": round(smog, 1)
            },
            "text_complexity": {
                "average_sentence_length": round(avg_sentence_length, 1),
                "average_syllables_per_word": round(avg_syllables_per_word, 2),
                "reading_level": reading_level,
                "fiction_assessment": fiction_assessment
            },
            "recommendations": {
                "sentence_length": "Good sentence variety" if 10 <= avg_sentence_length <= 20 else
                                 "Consider shorter sentences" if avg_sentence_length > 20 else
                                 "Consider varying sentence length",
                "syllable_complexity": "Appropriate word complexity" if avg_syllables_per_word <= 1.7 else
                                     "Consider simpler word choices",
                "overall_readability": "Excellent readability" if flesch_reading_ease >= 70 else
                                     "Good readability" if flesch_reading_ease >= 50 else
                                     "Consider simplifying for broader appeal"
            },
            "comparison_benchmarks": {
                "popular_fiction": "Most popular fiction scores 60-80 on Flesch Reading Ease",
                "young_adult": "YA fiction typically scores 70-85 on Flesch Reading Ease",
                "literary_fiction": "Literary fiction often scores 40-70 on Flesch Reading Ease",
                "your_score": round(flesch_reading_ease, 1)
            }
        }

    except Exception as e:
        return {"error": f"Error calculating readability metrics: {str(e)}"}

@mcp.tool()
def comprehensive_grammar_check(chapter_path: str) -> Dict[str, Any]:
    """
    Perform professional-level grammar and style checking on a chapter.

    Args:
        chapter_path: Path to the chapter file to analyze

    Returns:
        Dictionary with grammar errors, style suggestions, and corrections
    """
    if not LANGUAGE_TOOL_AVAILABLE:
        return {"error": "language-tool-python library not available. Install with: pipenv install language-tool-python"}

    global language_tool
    if language_tool is None:
        try:
            language_tool = language_tool_python.LanguageTool('en-US')
        except Exception as e:
            return {"error": f"Error initializing LanguageTool: {str(e)}"}

    # Read the chapter content
    file_info = read_markdown_file(chapter_path)
    content = file_info["content"]

    # Clean content for analysis (remove markdown syntax but preserve structure)
    import re
    clean_content = re.sub(r'#{1,6}\s+', '', content)  # Remove headers
    clean_content = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', clean_content)  # Remove bold/italic
    clean_content = re.sub(r'`(.*?)`', r'\1', clean_content)  # Remove code
    clean_content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', clean_content)  # Remove links, keep text

    if len(clean_content.strip()) < 50:
        return {"error": "Chapter too short for meaningful grammar analysis"}

    try:
        # Check grammar and style
        matches = language_tool.check(clean_content)

        # Categorize issues
        grammar_errors = []
        style_issues = []
        spelling_errors = []
        punctuation_issues = []
        other_issues = []

        for match in matches:
            issue = {
                "message": match.message,
                "context": match.context,
                "offset": match.offset,
                "length": match.errorLength,
                "suggestions": match.replacements[:3] if match.replacements else [],  # Top 3 suggestions
                "rule_id": match.ruleId,
                "category": match.category
            }

            # Categorize by rule type
            if 'GRAMMAR' in match.category.upper() or 'VERB' in match.category.upper():
                grammar_errors.append(issue)
            elif 'STYLE' in match.category.upper() or 'REDUNDANCY' in match.category.upper():
                style_issues.append(issue)
            elif 'TYPOS' in match.category.upper() or 'SPELLING' in match.category.upper():
                spelling_errors.append(issue)
            elif 'PUNCTUATION' in match.category.upper():
                punctuation_issues.append(issue)
            else:
                other_issues.append(issue)

        # Count issues by severity
        total_issues = len(matches)
        critical_issues = len([m for m in matches if 'error' in m.message.lower()])

        # Generate overall assessment
        if total_issues == 0:
            assessment = "Excellent - No grammar or style issues detected"
        elif total_issues <= 5:
            assessment = "Very Good - Minor issues that can be easily fixed"
        elif total_issues <= 15:
            assessment = "Good - Some issues to address for polish"
        elif total_issues <= 30:
            assessment = "Needs Work - Multiple issues requiring attention"
        else:
            assessment = "Significant Revision Needed - Many issues found"

        # Most common issue types
        issue_types = {}
        for match in matches:
            rule_type = match.ruleId.split('_')[0] if '_' in match.ruleId else match.ruleId
            issue_types[rule_type] = issue_types.get(rule_type, 0) + 1

        most_common_issues = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "chapter_info": {
                "filename": file_info["filename"],
                "word_count": len(clean_content.split()),
                "character_count": len(clean_content)
            },
            "summary": {
                "total_issues": total_issues,
                "grammar_errors": len(grammar_errors),
                "style_issues": len(style_issues),
                "spelling_errors": len(spelling_errors),
                "punctuation_issues": len(punctuation_issues),
                "overall_assessment": assessment,
                "issues_per_100_words": round(total_issues / len(clean_content.split()) * 100, 1) if clean_content.split() else 0
            },
            "issue_breakdown": {
                "grammar_errors": grammar_errors[:10],  # Top 10 of each type
                "style_issues": style_issues[:10],
                "spelling_errors": spelling_errors[:10],
                "punctuation_issues": punctuation_issues[:10],
                "other_issues": other_issues[:5]
            },
            "patterns": {
                "most_common_issue_types": [{"type": issue_type, "count": count} for issue_type, count in most_common_issues],
                "critical_errors": critical_issues,
                "suggestions_available": sum(1 for m in matches if m.replacements)
            },
            "recommendations": [
                "Focus on grammar errors first - these affect readability most",
                "Address style issues to improve flow and clarity",
                "Check spelling errors - these undermine credibility",
                "Review punctuation for proper dialogue formatting" if any("dialogue" in str(m).lower() for m in matches) else "Punctuation appears correct",
                "Consider using grammar suggestions provided" if any(m.replacements for m in matches) else "Manual review recommended for flagged issues"
            ]
        }

    except Exception as e:
        return {"error": f"Error performing grammar check: {str(e)}"}

@mcp.tool()
def get_schema() -> Dict[str, Any]:
    """
    List all available methods/tools in the book MCP server with descriptions and usage examples.

    Returns:
        Dictionary with all available methods, their purposes, and usage examples
    """
    methods = {
        "statistical_analysis": {
            "description": "Methods that provide statistical insights about your book",
            "methods": [
                {
                    "name": "list_markdown_files",
                    "purpose": "List all markdown files in the book directory",
                    "usage": "Get an overview of all files in your book project",
                    "returns": "File information including names, paths, and sizes",
                    "example_use": "What files are in my book project?"
                },
                {
                    "name": "list_chapters",
                    "purpose": "List chapters with titles and word counts",
                    "usage": "See all chapters with enhanced metadata",
                    "returns": "Chapter titles, word counts, and file information",
                    "example_use": "Show me all my chapters with their word counts"
                },
                {
                    "name": "get_book_stats",
                    "purpose": "Get overall statistics about the entire book project",
                    "usage": "Understand scope and progress of your book",
                    "returns": "Total chapters, words, size, and averages",
                    "example_use": "What are my overall book statistics?"
                },
                {
                    "name": "get_chapter_length_distribution",
                    "purpose": "Analyze chapter length consistency and distribution",
                    "usage": "Check if chapters are evenly paced",
                    "returns": "Length statistics, consistency analysis, and chapter categorization",
                    "example_use": "Are my chapters consistently sized?"
                },
                {
                    "name": "get_word_frequency",
                    "purpose": "Analyze word frequency across all chapters",
                    "usage": "Find most common words, themes, character mentions",
                    "parameters": "min_length (int), exclude_common (bool), top_n (int)",
                    "returns": "Word frequency analysis and vocabulary insights",
                    "example_use": "What are my most frequently used words?"
                },
                {
                    "name": "get_narrative_structure",
                    "purpose": "Analyze story structure, pacing, and narrative elements",
                    "usage": "Understand story arc and chapter flow",
                    "returns": "Story phases, pacing analysis, chapter endings, dialogue density",
                    "example_use": "How is my story structured across chapters?"
                }
            ]
        },
        "content_access": {
            "description": "Methods for reading and searching specific content",
            "methods": [
                {
                    "name": "read_markdown_file",
                    "purpose": "Read the complete content of a specific file",
                    "usage": "Access full text of chapters, notes, or research files",
                    "parameters": "file_path (str) - relative or absolute path",
                    "returns": "Complete file content with metadata",
                    "example_use": "Read Chapter 5 for me"
                },
                {
                    "name": "search_content",
                    "purpose": "Search for specific text across all markdown files",
                    "usage": "Find character mentions, themes, or specific passages",
                    "parameters": "query (str), case_sensitive (bool)",
                    "returns": "List of matches with file locations and line numbers",
                    "example_use": "Search for all mentions of 'dragon' in my book"
                },
                {
                    "name": "get_file_summary",
                    "purpose": "Get a preview and basic stats of a file",
                    "usage": "Quick overview before reading full content",
                    "parameters": "file_path (str), max_lines (int)",
                    "returns": "Preview lines, word count, headers, and file stats",
                    "example_use": "Give me a summary of Chapter 3"
                }
            ]
        },
        "advanced_analysis": {
            "description": "AI-powered analysis using advanced NLP libraries",
            "methods": [
                {
                    "name": "extract_all_characters",
                    "purpose": "Automatically discover all character names using NLP",
                    "usage": "Find characters without manually specifying names",
                    "requires": "spaCy library",
                    "returns": "All characters with frequency, contexts, and categorization",
                    "example_use": "Who are all the characters in my book?"
                },
                {
                    "name": "analyze_character_relationships",
                    "purpose": "Discover which characters appear together in scenes",
                    "usage": "Understand character dynamics and story structure",
                    "requires": "spaCy library",
                    "returns": "Character co-occurrence patterns and relationship analysis",
                    "example_use": "Which characters interact most frequently?"
                },
                {
                    "name": "analyze_readability",
                    "purpose": "Calculate professional readability metrics (Flesch-Kincaid, etc.)",
                    "usage": "Ensure your writing matches target audience reading level",
                    "requires": "textstat library",
                    "parameters": "chapter_path (str)",
                    "returns": "Multiple readability scores and recommendations",
                    "example_use": "Is Chapter 5 too complex for my target audience?"
                },
                {
                    "name": "comprehensive_grammar_check",
                    "purpose": "Professional-level grammar and style checking",
                    "usage": "Find grammar errors, style issues, and get specific corrections",
                    "requires": "language-tool-python library",
                    "parameters": "chapter_path (str)",
                    "returns": "Categorized grammar/style issues with specific suggestions",
                    "example_use": "What grammar errors are in Chapter 2?"
                }
            ]
        },
        "editorial": {
            "description": "Methods that provide actionable editorial feedback",
            "methods": [
                {
                    "name": "analyze_writing_issues",
                    "purpose": "Identify specific writing problems in a chapter",
                    "usage": "Find passive voice, weak verbs, repetition, and prose issues",
                    "parameters": "chapter_path (str) - path to chapter file",
                    "returns": "Detailed writing issues with specific suggestions for improvement",
                    "example_use": "What writing issues are in Chapter 2?"
                },
                {
                    "name": "track_character_consistency",
                    "purpose": "Check character descriptions and traits across all chapters",
                    "usage": "Ensure character consistency and catch contradictions",
                    "parameters": "character_name (str) - name of character to analyze",
                    "returns": "Character appearances, descriptions, traits, and consistency issues",
                    "example_use": "Check if Sarah is consistently described throughout my book"
                },
                {
                    "name": "check_show_vs_tell",
                    "purpose": "Analyze balance between showing (scenes) vs telling (summary)",
                    "usage": "Improve scene engagement and reduce exposition dumps",
                    "parameters": "chapter_path (str) - path to chapter file",
                    "returns": "Paragraph analysis with showing/telling ratios and specific suggestions",
                    "example_use": "Is Chapter 4 too heavy on exposition?"
                }
            ]
        }
    }

    # Count total methods
    total_methods = sum(len(category["methods"]) for category in methods.values())

    # Create a flat list for quick reference
    method_names = []
    for category in methods.values():
        for method in category["methods"]:
            method_names.append(method["name"])

    return {
        "server_info": {
            "name": "Book MCP Server",
            "purpose": "Comprehensive manuscript analysis and content access for book writing",
            "book_directory": BOOK_DIRECTORY,
            "total_methods": total_methods
        },
        "quick_reference": {
            "all_method_names": sorted(method_names),
            "most_useful_for_writing": [
                "analyze_writing_issues - Fix prose problems",
                "track_character_consistency - Ensure character continuity",
                "check_show_vs_tell - Improve scene engagement",
                "search_content - Find specific content",
                "get_narrative_structure - Analyze story flow"
            ]
        },
        "method_categories": methods,
        "usage_tips": [
            "Start with 'list_chapters' to see your book structure",
            "Use 'search_content' to find character mentions or themes",
            "Run 'analyze_writing_issues' on chapters you're revising",
            "Check 'track_character_consistency' for main characters",
            "Use 'check_show_vs_tell' to improve scene writing",
            "Get 'get_book_stats' for overall progress tracking"
        ],
        "example_workflows": [
            {
                "workflow": "Chapter Revision",
                "steps": [
                    "1. read_markdown_file('chapter3.md') - Read current version",
                    "2. analyze_writing_issues('chapter3.md') - Find prose problems",
                    "3. check_show_vs_tell('chapter3.md') - Check scene balance",
                    "4. Revise based on feedback"
                ]
            },
            {
                "workflow": "Character Development Check",
                "steps": [
                    "1. track_character_consistency('protagonist_name') - Check main character",
                    "2. search_content('protagonist_name') - Find all mentions",
                    "3. Review flagged inconsistencies",
                    "4. Update character descriptions as needed"
                ]
            },
            {
                "workflow": "Book Structure Analysis",
                "steps": [
                    "1. get_book_stats() - Overall project status",
                    "2. get_chapter_length_distribution() - Check pacing",
                    "3. get_narrative_structure() - Analyze story flow",
                    "4. Adjust structure based on insights"
                ]
            }
        ]
    }

@mcp.tool()
def list_markdown_files() -> List[Dict[str, str]]:
    """
    List all markdown files in the book directory.

    Returns:
        List of dictionaries containing file information
    """
    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    # Find all markdown files recursively
    md_files = list(book_path.glob("**/*.md"))

    result = []
    for file_path in md_files:
        relative_path = file_path.relative_to(book_path)
        result.append({
            "filename": file_path.name,
            "relative_path": str(relative_path),
            "absolute_path": str(file_path),
            "size_bytes": str(file_path.stat().st_size if file_path.exists() else 0)
        })

    return result

@mcp.tool()
def read_markdown_file(file_path: str) -> Dict[str, Any]:
    """
    Read the contents of a specific markdown file.

    Args:
        file_path: Path to the markdown file (relative to book directory or absolute)

    Returns:
        Dictionary containing file metadata and content
    """
    book_path = Path(BOOK_DIRECTORY)

    # Handle both relative and absolute paths
    if os.path.isabs(file_path):
        target_path = Path(file_path)
    else:
        target_path = book_path / file_path

    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {target_path}")

    if not target_path.suffix.lower() == '.md':
        raise ValueError(f"File is not a markdown file: {target_path}")

    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "filename": target_path.name,
            "path": str(target_path),
            "size_bytes": str(len(content.encode('utf-8'))),
            "content": content,
            "line_count": len(content.splitlines())
        }
    except UnicodeDecodeError:
        raise ValueError(f"Cannot read file as UTF-8: {target_path}")

@mcp.tool()
def search_content(query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
    """
    Search for text across all markdown files in the book directory.

    Args:
        query: Text to search for
        case_sensitive: Whether the search should be case sensitive

    Returns:
        List of matches with file and line information
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty")

    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    results = []
    search_query = query if case_sensitive else query.lower()

    # Search through all markdown files
    for md_file in book_path.glob("**/*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                search_line = line if case_sensitive else line.lower()
                if search_query in search_line:
                    results.append({
                        "filename": md_file.name,
                        "relative_path": str(md_file.relative_to(book_path)),
                        "line_number": line_num,
                        "line_content": line.strip(),
                        "match_query": query
                    })
        except (UnicodeDecodeError, PermissionError):
            # Skip files that can't be read
            continue

    return results

@mcp.tool()
def analyze_writing_issues(chapter_path: str) -> Dict[str, Any]:
    """
    Analyze a specific chapter for common writing issues like passive voice,
    weak verbs, repetition, and other prose problems.

    Args:
        chapter_path: Path to the chapter file to analyze

    Returns:
        Dictionary with identified writing issues and suggestions for improvement
    """
    import re

    # Read the chapter content
    file_info = read_markdown_file(chapter_path)
    content = file_info["content"]
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]

    issues = {
        "passive_voice": [],
        "weak_verbs": [],
        "repetitive_sentence_starts": [],
        "overused_words": [],
        "adverb_overuse": [],
        "filler_words": [],
        "long_sentences": [],
        "summary": {}
    }

    # Passive voice detection
    passive_indicators = [r'\b(was|were|is|are|been|be)\s+\w*ed\b', r'\bwas\s+being\b', r'\bwere\s+being\b']
    for i, sentence in enumerate(sentences):
        for pattern in passive_indicators:
            if re.search(pattern, sentence.lower()):
                issues["passive_voice"].append({
                    "sentence_number": i + 1,
                    "text": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                    "suggestion": "Consider rewriting in active voice for stronger prose"
                })
                break

    # Weak verbs (to be verbs and generic action verbs)
    weak_verbs = ['was', 'were', 'is', 'are', 'been', 'be', 'went', 'got', 'get', 'had', 'have', 'said', 'came', 'come', 'put']
    words = re.findall(r'\b\w+\b', content.lower())
    for verb in weak_verbs:
        count = words.count(verb)
        if count > 5:  # Threshold for concern
            issues["weak_verbs"].append({
                "verb": verb,
                "count": count,
                "suggestion": f"Consider replacing some instances of '{verb}' with more specific verbs"
            })

    # Repetitive sentence starts
    sentence_starts = {}
    for sentence in sentences:
        words = sentence.split()
        if words:
            first_word = words[0].lower()
            if len(first_word) > 2:  # Ignore short words like "I", "a"
                sentence_starts[first_word] = sentence_starts.get(first_word, 0) + 1

    for word, count in sentence_starts.items():
        if count > 3:  # More than 3 sentences starting the same way
            issues["repetitive_sentence_starts"].append({
                "word": word,
                "count": count,
                "suggestion": f"Vary sentence beginnings to improve flow"
            })

    # Overused words (excluding common words)
    word_freq = {}
    significant_words = [w for w in words if len(w) > 4 and w not in ['that', 'with', 'this', 'they', 'were', 'been', 'have', 'would', 'could', 'should']]
    for word in significant_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    for word, count in word_freq.items():
        if count > 8:  # Appears more than 8 times in one chapter
            issues["overused_words"].append({
                "word": word,
                "count": count,
                "suggestion": f"Consider synonyms or alternative phrasing to reduce repetition"
            })

    # Adverb overuse (words ending in -ly)
    adverbs = re.findall(r'\b\w+ly\b', content.lower())
    adverb_freq = {}
    for adverb in adverbs:
        adverb_freq[adverb] = adverb_freq.get(adverb, 0) + 1

    total_adverbs = len(adverbs)
    if total_adverbs > len(words) * 0.02:  # More than 2% adverbs
        for adverb, count in sorted(adverb_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
            issues["adverb_overuse"].append({
                "adverb": adverb,
                "count": count,
                "suggestion": f"Consider stronger verbs instead of '{adverb}'"
            })

    # Filler words
    filler_words = ['just', 'really', 'very', 'quite', 'rather', 'somewhat', 'actually', 'basically', 'literally', 'totally']
    for filler in filler_words:
        count = words.count(filler)
        if count > 2:
            issues["filler_words"].append({
                "word": filler,
                "count": count,
                "suggestion": f"Consider removing unnecessary '{filler}' for cleaner prose"
            })

    # Long sentences (might be hard to read)
    for i, sentence in enumerate(sentences):
        word_count = len(sentence.split())
        if word_count > 30:
            issues["long_sentences"].append({
                "sentence_number": i + 1,
                "word_count": word_count,
                "text": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "suggestion": "Consider breaking into shorter sentences for better readability"
            })

    # Summary
    total_issues = sum(len(issue_list) for key, issue_list in issues.items() if key != "summary")
    issues["summary"] = {
        "total_issues_found": total_issues,
        "chapter_name": file_info["filename"],
        "word_count": len(words),
        "sentence_count": len(sentences),
        "issues_per_100_words": round(total_issues / len(words) * 100, 1) if words else 0,
        "overall_assessment": "Needs significant revision" if total_issues > 20 else "Needs moderate revision" if total_issues > 10 else "Generally clean prose" if total_issues > 5 else "Excellent prose quality"
    }

    return issues

@mcp.tool()
def track_character_consistency(character_name: str) -> Dict[str, Any]:
    """
    Track how a character is described and portrayed across all chapters
    to identify potential consistency issues.

    Args:
        character_name: Name of the character to analyze

    Returns:
        Dictionary with character consistency analysis across chapters
    """
    import re

    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    character_appearances = []
    physical_descriptions = []
    personality_traits = []
    dialogue_examples = []
    actions_taken = []

    # Common descriptive words to look for near the character's name
    physical_words = ['hair', 'eyes', 'tall', 'short', 'thin', 'heavy', 'blonde', 'brown', 'blue', 'green', 'young', 'old', 'beautiful', 'handsome', 'pale', 'dark', 'fair', 'skin', 'face', 'smile', 'voice']
    personality_words = ['kind', 'cruel', 'gentle', 'harsh', 'smart', 'intelligent', 'stupid', 'funny', 'serious', 'shy', 'confident', 'brave', 'coward', 'honest', 'liar', 'loyal', 'betrayer', 'calm', 'angry', 'patient', 'impatient', 'generous', 'selfish', 'creative', 'boring']

    for file_path in book_path.glob("**/*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Find all mentions of the character
                character_pattern = r'\b' + re.escape(character_name) + r'\b'
                matches = list(re.finditer(character_pattern, content, re.IGNORECASE))

                if matches:
                    # Get chapter title
                    chapter_title = "Untitled"
                    lines = content.splitlines()
                    for line in lines:
                        if line.strip().startswith('#'):
                            chapter_title = line.strip().lstrip('#').strip()
                            break

                    # Analyze each mention
                    for match in matches:
                        start = max(0, match.start() - 200)
                        end = min(len(content), match.end() + 200)
                        context = content[start:end]

                        # Look for physical descriptions
                        for word in physical_words:
                            if word in context.lower():
                                # Extract the sentence containing the description
                                sentences = re.split(r'[.!?]+', context)
                                for sentence in sentences:
                                    if character_name.lower() in sentence.lower() and word in sentence.lower():
                                        physical_descriptions.append({
                                            "chapter": chapter_title,
                                            "description": sentence.strip(),
                                            "keyword": word
                                        })

                        # Look for personality traits
                        for word in personality_words:
                            if word in context.lower():
                                sentences = re.split(r'[.!?]+', context)
                                for sentence in sentences:
                                    if character_name.lower() in sentence.lower() and word in sentence.lower():
                                        personality_traits.append({
                                            "chapter": chapter_title,
                                            "trait": sentence.strip(),
                                            "keyword": word
                                        })

                        # Look for dialogue
                        dialogue_pattern = r'"[^"]*"'
                        dialogue_matches = re.findall(dialogue_pattern, context)
                        if dialogue_matches:
                            for dialogue in dialogue_matches:
                                if len(dialogue) > 10:  # Substantial dialogue
                                    dialogue_examples.append({
                                        "chapter": chapter_title,
                                        "dialogue": dialogue,
                                        "context": context[max(0, match.start()-50):match.end()+50]
                                    })

                    character_appearances.append({
                        "chapter": chapter_title,
                        "filename": file_path.name,
                        "mention_count": len(matches),
                        "context_examples": [content[max(0, m.start()-100):m.end()+100] for m in matches[:3]]  # First 3 examples
                    })

        except (UnicodeDecodeError, PermissionError):
            continue

    if not character_appearances:
        return {"error": f"Character '{character_name}' not found in any chapters"}

    # Analyze for consistency issues
    consistency_issues = []

    # Check for conflicting physical descriptions
    physical_keywords = [desc["keyword"] for desc in physical_descriptions]
    if "tall" in physical_keywords and "short" in physical_keywords:
        consistency_issues.append("Conflicting height descriptions found")
    if "young" in physical_keywords and "old" in physical_keywords:
        consistency_issues.append("Conflicting age descriptions found")

    # Check for conflicting personality traits
    personality_keywords = [trait["keyword"] for trait in personality_traits]
    conflicts = [
        ("shy", "confident"), ("brave", "coward"), ("kind", "cruel"),
        ("honest", "liar"), ("calm", "angry"), ("patient", "impatient"),
        ("generous", "selfish")
    ]
    for trait1, trait2 in conflicts:
        if trait1 in personality_keywords and trait2 in personality_keywords:
            consistency_issues.append(f"Conflicting personality traits: {trait1} vs {trait2}")

    return {
        "character_name": character_name,
        "summary": {
            "total_chapters_appearing": len(character_appearances),
            "total_mentions": sum(app["mention_count"] for app in character_appearances),
            "physical_descriptions_found": len(physical_descriptions),
            "personality_traits_found": len(personality_traits),
            "dialogue_examples_found": len(dialogue_examples),
            "consistency_issues": len(consistency_issues)
        },
        "appearances_by_chapter": character_appearances,
        "physical_descriptions": physical_descriptions[:10],  # Top 10
        "personality_traits": personality_traits[:10],  # Top 10
        "dialogue_examples": dialogue_examples[:5],  # Top 5
        "consistency_analysis": {
            "issues_found": consistency_issues,
            "recommendations": [
                "Review conflicting descriptions and choose consistent traits",
                "Consider character development arc vs. inconsistency",
                "Ensure physical descriptions remain constant unless story requires change"
            ] if consistency_issues else ["Character appears consistently portrayed across chapters"]
        }
    }

@mcp.tool()
def check_show_vs_tell(chapter_path: str) -> Dict[str, Any]:
    """
    Analyze the balance between 'showing' (scenes, dialogue, action) vs
    'telling' (summary, exposition) in a chapter.

    Args:
        chapter_path: Path to the chapter file to analyze

    Returns:
        Dictionary with show vs tell analysis and recommendations
    """
    import re

    # Read the chapter content
    file_info = read_markdown_file(chapter_path)
    content = file_info["content"]

    # Split into paragraphs for analysis
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    showing_indicators = {
        "dialogue": 0,
        "action_verbs": 0,
        "sensory_details": 0,
        "specific_details": 0,
        "scene_setting": 0
    }

    telling_indicators = {
        "summary_language": 0,
        "exposition": 0,
        "abstract_concepts": 0,
        "time_jumps": 0,
        "character_backstory": 0
    }

    paragraph_analysis = []

    # Analyze each paragraph
    for i, paragraph in enumerate(paragraphs):
        para_showing = 0
        para_telling = 0
        para_type = "neutral"
        issues = []
        suggestions = []

        # Dialogue detection (strong showing indicator)
        dialogue_count = len(re.findall(r'"[^"]*"', paragraph))
        if dialogue_count > 0:
            showing_indicators["dialogue"] += dialogue_count
            para_showing += dialogue_count * 2  # Weight dialogue heavily

        # Action verbs (showing)
        action_verbs = ['grabbed', 'ran', 'jumped', 'whispered', 'shouted', 'slammed', 'crept', 'dashed', 'stumbled', 'glanced', 'stared', 'frowned', 'smiled', 'laughed', 'cried', 'screamed', 'touched', 'reached', 'pulled', 'pushed']
        for verb in action_verbs:
            count = paragraph.lower().count(verb)
            if count > 0:
                showing_indicators["action_verbs"] += count
                para_showing += count

        # Sensory details (showing)
        sensory_words = ['saw', 'heard', 'felt', 'smelled', 'tasted', 'rough', 'smooth', 'cold', 'warm', 'bright', 'dark', 'loud', 'quiet', 'sweet', 'bitter', 'soft', 'hard']
        for word in sensory_words:
            count = paragraph.lower().count(word)
            if count > 0:
                showing_indicators["sensory_details"] += count
                para_showing += count

        # Specific details (showing)
        if re.search(r'\b\d+\b', paragraph):  # Numbers indicate specificity
            showing_indicators["specific_details"] += 1
            para_showing += 1

        # Summary/telling language
        telling_phrases = ['had always been', 'used to', 'would often', 'generally', 'usually', 'typically', 'for years', 'since childhood', 'explained', 'described', 'remembered', 'thought about', 'considered', 'realized', 'understood']
        for phrase in telling_phrases:
            if phrase in paragraph.lower():
                telling_indicators["summary_language"] += 1
                para_telling += 1

        # Exposition markers
        exposition_words = ['because', 'since', 'due to', 'as a result', 'therefore', 'consequently', 'background', 'history', 'past', 'previously']
        for word in exposition_words:
            count = paragraph.lower().count(word)
            if count > 0:
                telling_indicators["exposition"] += count
                para_telling += count

        # Abstract concepts (telling)
        abstract_words = ['love', 'hate', 'happiness', 'sadness', 'fear', 'anger', 'confusion', 'understanding', 'knowledge', 'wisdom', 'beauty', 'ugliness']
        for word in abstract_words:
            if word in paragraph.lower() and not re.search(r'"[^"]*' + word + r'[^"]*"', paragraph.lower()):  # Not in dialogue
                telling_indicators["abstract_concepts"] += 1
                para_telling += 1

        # Time jumps (telling)
        time_jump_phrases = ['later that day', 'the next morning', 'after a while', 'eventually', 'meanwhile', 'hours later', 'days passed', 'weeks went by']
        for phrase in time_jump_phrases:
            if phrase in paragraph.lower():
                telling_indicators["time_jumps"] += 1
                para_telling += 1

        # Determine paragraph type and provide feedback
        if para_showing > para_telling * 2:
            para_type = "showing"
        elif para_telling > para_showing * 2:
            para_type = "telling"
            if para_telling > 3:
                issues.append("Heavy exposition - consider breaking up with action or dialogue")
                suggestions.append("Add specific actions, dialogue, or sensory details")
        else:
            para_type = "mixed"

        # Additional analysis
        word_count = len(paragraph.split())
        if word_count > 150 and para_type == "telling":
            issues.append("Long expository paragraph - may lose reader interest")
            suggestions.append("Break into shorter paragraphs or add showing elements")

        if dialogue_count == 0 and para_showing == 0 and word_count > 100:
            issues.append("Pure telling with no showing elements")
            suggestions.append("Add character actions, dialogue, or sensory details")

        paragraph_analysis.append({
            "paragraph_number": i + 1,
            "type": para_type,
            "word_count": word_count,
            "showing_score": para_showing,
            "telling_score": para_telling,
            "preview": paragraph[:100] + "..." if len(paragraph) > 100 else paragraph,
            "issues": issues,
            "suggestions": suggestions
        })

    # Calculate overall ratios
    total_showing = sum(showing_indicators.values())
    total_telling = sum(telling_indicators.values())
    total_elements = total_showing + total_telling

    showing_percentage = (total_showing / total_elements * 100) if total_elements > 0 else 0
    telling_percentage = (total_telling / total_elements * 100) if total_elements > 0 else 0

    # Overall assessment
    if showing_percentage > 70:
        assessment = "Excellent - Strong showing with vivid scenes"
    elif showing_percentage > 50:
        assessment = "Good - Balanced mix favoring showing"
    elif showing_percentage > 30:
        assessment = "Needs improvement - Too much telling"
    else:
        assessment = "Revision needed - Mostly exposition and summary"

    return {
        "chapter_name": file_info["filename"],
        "overall_analysis": {
            "showing_percentage": round(showing_percentage, 1),
            "telling_percentage": round(telling_percentage, 1),
            "assessment": assessment,
            "total_paragraphs": len(paragraphs),
            "dialogue_paragraphs": sum(1 for p in paragraph_analysis if '"' in p["preview"]),
            "heavy_exposition_paragraphs": sum(1 for p in paragraph_analysis if p["type"] == "telling" and p["telling_score"] > 3)
        },
        "showing_elements": showing_indicators,
        "telling_elements": telling_indicators,
        "paragraph_breakdown": paragraph_analysis,
        "recommendations": [
            "Add more dialogue to bring scenes to life" if showing_indicators["dialogue"] < 3 else "Good use of dialogue",
            "Include more sensory details to immerse readers" if showing_indicators["sensory_details"] < 5 else "Good sensory writing",
            "Show character emotions through actions rather than stating them" if telling_indicators["abstract_concepts"] > 5 else "Good emotional showing",
            "Break up long expository sections with action or dialogue" if telling_indicators["exposition"] > 10 else "Well-paced exposition",
            "Consider showing backstory through scenes rather than summary" if telling_indicators["summary_language"] > 5 else "Good balance of backstory"
        ]
    }

@mcp.tool()
def get_narrative_structure() -> Dict[str, Any]:
    """
    Analyze the narrative structure and story elements across all chapters.

    Returns:
        Dictionary with narrative structure analysis including pacing, dialogue density,
        chapter transitions, and story arc elements
    """
    import re

    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    chapters = []
    md_files = sorted(list(book_path.glob("**/*.md")))

    if not md_files:
        return {"error": "No markdown files found"}

    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Extract chapter title
                chapter_title = "Untitled"
                lines = content.splitlines()
                for line in lines:
                    if line.strip().startswith('#'):
                        chapter_title = line.strip().lstrip('#').strip()
                        break

                # Basic metrics
                word_count = len(content.split())
                paragraph_count = len([p for p in content.split('\n\n') if p.strip()])

                # Dialogue analysis
                dialogue_lines = re.findall(r'"[^"]*"', content)
                dialogue_word_count = sum(len(line.split()) for line in dialogue_lines)
                dialogue_percentage = (dialogue_word_count / word_count * 100) if word_count > 0 else 0

                # Action indicators (simple heuristic)
                action_words = ['ran', 'jumped', 'fought', 'moved', 'rushed', 'grabbed', 'threw', 'hit', 'struck', 'fell', 'climbed', 'chased', 'escaped', 'attacked', 'defended']
                action_count = sum(content.lower().count(word) for word in action_words)
                action_density = action_count / word_count * 1000 if word_count > 0 else 0  # per 1000 words

                # Emotional intensity indicators
                emotion_words = ['love', 'hate', 'fear', 'angry', 'sad', 'happy', 'excited', 'nervous', 'worried', 'relieved', 'surprised', 'shocked', 'devastated', 'thrilled']
                emotion_count = sum(content.lower().count(word) for word in emotion_words)
                emotion_density = emotion_count / word_count * 1000 if word_count > 0 else 0

                # Tension indicators
                tension_words = ['suddenly', 'but', 'however', 'danger', 'threat', 'crisis', 'problem', 'conflict', 'struggle', 'challenge', 'urgent', 'desperate']
                tension_count = sum(content.lower().count(word) for word in tension_words)
                tension_density = tension_count / word_count * 1000 if word_count > 0 else 0

                # Chapter ending analysis
                last_sentences = content.strip().split('.')[-3:]  # Last few sentences
                last_text = ' '.join(last_sentences).lower()

                ending_type = "neutral"
                if any(word in last_text for word in ['?', 'what', 'how', 'why', 'who']):
                    ending_type = "question/mystery"
                elif any(word in last_text for word in ['suddenly', 'then', 'but', 'however']):
                    ending_type = "cliffhanger"
                elif any(word in last_text for word in ['smiled', 'laughed', 'happy', 'relief', 'safe']):
                    ending_type = "resolution"
                elif any(word in last_text for word in ['dark', 'shadow', 'danger', 'threat']):
                    ending_type = "ominous"

                # Time progression indicators
                time_words = ['yesterday', 'today', 'tomorrow', 'morning', 'evening', 'night', 'day', 'week', 'month', 'year', 'ago', 'later', 'before', 'after', 'meanwhile', 'suddenly']
                time_transitions = sum(content.lower().count(word) for word in time_words)

                chapters.append({
                    "filename": file_path.name,
                    "chapter_title": chapter_title,
                    "order": len(chapters) + 1,
                    "word_count": word_count,
                    "paragraph_count": paragraph_count,
                    "dialogue_percentage": round(dialogue_percentage, 1),
                    "action_density": round(action_density, 2),
                    "emotion_density": round(emotion_density, 2),
                    "tension_density": round(tension_density, 2),
                    "ending_type": ending_type,
                    "time_transitions": time_transitions,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })

        except (UnicodeDecodeError, PermissionError):
            continue

    if not chapters:
        return {"error": "No readable chapters found"}

    # Analyze overall structure
    total_chapters = len(chapters)
    avg_word_count = sum(ch["word_count"] for ch in chapters) / total_chapters

    # Story arc analysis (rough heuristic based on position and tension)
    story_phases = []
    for i, chapter in enumerate(chapters):
        position_ratio = i / (total_chapters - 1) if total_chapters > 1 else 0

        if position_ratio <= 0.25:
            phase = "Setup/Introduction"
        elif position_ratio <= 0.75:
            phase = "Rising Action"
        elif position_ratio <= 0.9:
            phase = "Climax"
        else:
            phase = "Resolution"

        story_phases.append(phase)
        chapter["story_phase"] = phase

    # Pacing analysis
    dialogue_trend = [ch["dialogue_percentage"] for ch in chapters]
    action_trend = [ch["action_density"] for ch in chapters]
    tension_trend = [ch["tension_density"] for ch in chapters]

    # Chapter ending distribution
    ending_types = {}
    for chapter in chapters:
        ending_type = chapter["ending_type"]
        ending_types[ending_type] = ending_types.get(ending_type, 0) + 1

    return {
        "overview": {
            "total_chapters": total_chapters,
            "average_chapter_length": round(avg_word_count, 0),
            "total_words": sum(ch["word_count"] for ch in chapters)
        },
        "story_arc": {
            "setup_chapters": sum(1 for phase in story_phases if phase == "Setup/Introduction"),
            "rising_action_chapters": sum(1 for phase in story_phases if phase == "Rising Action"),
            "climax_chapters": sum(1 for phase in story_phases if phase == "Climax"),
            "resolution_chapters": sum(1 for phase in story_phases if phase == "Resolution")
        },
        "pacing_analysis": {
            "average_dialogue_percentage": round(sum(dialogue_trend) / len(dialogue_trend), 1),
            "dialogue_range": f"{min(dialogue_trend):.1f}% - {max(dialogue_trend):.1f}%",
            "average_action_density": round(sum(action_trend) / len(action_trend), 2),
            "average_tension_density": round(sum(tension_trend) / len(tension_trend), 2),
            "high_tension_chapters": [ch["chapter_title"] for ch in chapters if ch["tension_density"] > sum(tension_trend) / len(tension_trend) * 1.5]
        },
        "chapter_endings": {
            "distribution": ending_types,
            "cliffhanger_percentage": round(ending_types.get("cliffhanger", 0) / total_chapters * 100, 1),
            "resolution_percentage": round(ending_types.get("resolution", 0) / total_chapters * 100, 1)
        },
        "narrative_flow": {
            "chapters_with_time_shifts": sum(1 for ch in chapters if ch["time_transitions"] > 3),
            "average_time_transitions_per_chapter": round(sum(ch["time_transitions"] for ch in chapters) / total_chapters, 1)
        },
        "chapter_details": chapters
    }

@mcp.tool()
def get_word_frequency(min_length: int = 3, exclude_common: bool = True, top_n: int = 50) -> Dict[str, Any]:
    """
    Analyze word frequency across all chapters in the book.

    Args:
        min_length: Minimum word length to include (default: 3)
        exclude_common: Whether to exclude common English words (default: True)
        top_n: Number of top words to return (default: 50)

    Returns:
        Dictionary with word frequency analysis
    """
    import re
    from collections import Counter

    # Common English words to exclude (basic stopwords)
    common_words = {
        'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as',
        'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but',
        'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she',
        'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these',
        'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'after',
        'words', 'first', 'been', 'who', 'oil', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get',
        'come', 'made', 'may', 'part', 'over', 'new', 'sound', 'take', 'only', 'little', 'work', 'know',
        'place', 'year', 'live', 'me', 'back', 'give', 'most', 'go', 'good', 'where', 'much', 'before',
        'move', 'right', 'boy', 'old', 'too', 'same', 'tell', 'does', 'set', 'three', 'want', 'air', 'well',
        'also', 'play', 'small', 'end', 'put', 'home', 'read', 'hand', 'port', 'large', 'spell', 'add',
        'even', 'land', 'here', 'must', 'big', 'high', 'such', 'follow', 'act', 'why', 'ask', 'men', 'change',
        'went', 'light', 'kind', 'off', 'need', 'house', 'picture', 'try', 'us', 'again', 'animal', 'point',
        'mother', 'world', 'near', 'build', 'self', 'earth', 'father', 'head', 'stand', 'own', 'page',
        'should', 'country', 'found', 'answer', 'school', 'grow', 'study', 'still', 'learn', 'plant',
        'cover', 'food', 'sun', 'four', 'between', 'state', 'keep', 'eye', 'never', 'last', 'let', 'thought',
        'city', 'tree', 'cross', 'farm', 'hard', 'start', 'might', 'story', 'saw', 'far', 'sea', 'draw',
        'left', 'late', 'run', 'don', 'while', 'press', 'close', 'night', 'real', 'life', 'few', 'north',
        'open', 'seem', 'together', 'next', 'white', 'children', 'begin', 'got', 'walk', 'example', 'ease',
        'paper', 'group', 'always', 'music', 'those', 'both', 'mark', 'often', 'letter', 'until', 'mile',
        'river', 'car', 'feet', 'care', 'second', 'book', 'carry', 'took', 'science', 'eat', 'room',
        'friend', 'began', 'idea', 'fish', 'mountain', 'stop', 'once', 'base', 'hear', 'horse', 'cut',
        'sure', 'watch', 'color', 'face', 'wood', 'main', 'enough', 'plain', 'girl', 'usual', 'young',
        'ready', 'above', 'ever', 'red', 'list', 'though', 'feel', 'talk', 'bird', 'soon', 'body',
        'dog', 'family', 'direct', 'leave', 'song', 'measure', 'door', 'product', 'black', 'short',
        'numeral', 'class', 'wind', 'question', 'happen', 'complete', 'ship', 'area', 'half', 'rock',
        'order', 'fire', 'south', 'problem', 'piece', 'told', 'knew', 'pass', 'since', 'top', 'whole',
        'king', 'space', 'heard', 'best', 'hour', 'better', 'during', 'hundred', 'five', 'remember',
        'step', 'early', 'hold', 'west', 'ground', 'interest', 'reach', 'fast', 'verb', 'sing', 'listen',
        'six', 'table', 'travel', 'less', 'morning', 'ten', 'simple', 'several', 'vowel', 'toward',
        'war', 'lay', 'against', 'pattern', 'slow', 'center', 'love', 'person', 'money', 'serve',
        'appear', 'road', 'map', 'rain', 'rule', 'govern', 'pull', 'cold', 'notice', 'voice', 'unit',
        'power', 'town', 'fine', 'certain', 'fly', 'fall', 'lead', 'cry', 'dark', 'machine', 'note',
        'wait', 'plan', 'figure', 'star', 'box', 'noun', 'field', 'rest', 'correct', 'able', 'pound',
        'done', 'beauty', 'drive', 'stood', 'contain', 'front', 'teach', 'week', 'final', 'gave', 'green',
        'oh', 'quick', 'develop', 'ocean', 'warm', 'free', 'minute', 'strong', 'special', 'mind', 'behind',
        'clear', 'tail', 'produce', 'fact', 'street', 'inch', 'multiply', 'nothing', 'course', 'stay',
        'wheel', 'full', 'force', 'blue', 'object', 'decide', 'surface', 'deep', 'moon', 'island', 'foot',
        'system', 'busy', 'test', 'record', 'boat', 'common', 'gold', 'possible', 'plane', 'stead',
        'dry', 'wonder', 'laugh', 'thousands', 'ago', 'ran', 'check', 'game', 'shape', 'equate', 'hot',
        'miss', 'brought', 'heat', 'snow', 'tire', 'bring', 'yes', 'distant', 'fill', 'east', 'paint',
        'language', 'among'
    } if exclude_common else set()

    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    all_words = []
    chapter_word_counts = {}
    total_chapters = 0

    # Process all markdown files
    for file_path in book_path.glob("**/*.md"):
        total_chapters += 1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Clean and extract words
                # Remove markdown syntax and normalize
                content = re.sub(r'#{1,6}\s+', '', content)  # Remove headers
                content = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', content)  # Remove bold/italic
                content = re.sub(r'`(.*?)`', r'\1', content)  # Remove code
                content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # Remove links, keep text

                # Extract words (alphabetic only, convert to lowercase)
                words = re.findall(r'\b[a-zA-Z]+\b', content.lower())

                # Filter words
                filtered_words = [
                    word for word in words
                    if len(word) >= min_length and (not exclude_common or word not in common_words)
                ]

                all_words.extend(filtered_words)
                chapter_word_counts[file_path.name] = len(filtered_words)

        except (UnicodeDecodeError, PermissionError):
            continue

    if not all_words:
        return {"error": "No words found matching criteria"}

    # Count word frequencies
    word_freq = Counter(all_words)
    total_unique_words = len(word_freq)
    total_words = len(all_words)

    # Get top N words
    top_words = word_freq.most_common(top_n)

    # Calculate some interesting statistics
    hapax_legomena = sum(1 for count in word_freq.values() if count == 1)  # Words that appear only once
    most_frequent_count = top_words[0][1] if top_words else 0

    return {
        "summary": {
            "total_words_analyzed": total_words,
            "unique_words": total_unique_words,
            "chapters_processed": total_chapters,
            "vocabulary_richness": round(total_unique_words / total_words * 100, 2) if total_words > 0 else 0,
            "hapax_legomena": hapax_legomena,
            "hapax_percentage": round(hapax_legomena / total_unique_words * 100, 2) if total_unique_words > 0 else 0
        },
        "filters_applied": {
            "minimum_word_length": min_length,
            "excluded_common_words": exclude_common,
            "requested_top_words": top_n
        },
        "top_words": [
            {
                "word": word,
                "frequency": count,
                "percentage": round(count / total_words * 100, 3) if total_words > 0 else 0
            }
            for word, count in top_words
        ],
        "frequency_distribution": {
            "words_appearing_once": hapax_legomena,
            "words_appearing_2-5_times": sum(1 for count in word_freq.values() if 2 <= count <= 5),
            "words_appearing_6-10_times": sum(1 for count in word_freq.values() if 6 <= count <= 10),
            "words_appearing_more_than_10_times": sum(1 for count in word_freq.values() if count > 10)
        },
        "chapter_word_distribution": dict(sorted(chapter_word_counts.items(), key=lambda x: x[1], reverse=True))
    }

@mcp.tool()
def get_chapter_length_distribution() -> Dict[str, Any]:
    """
    Analyze the length distribution of chapters in the book.

    Returns:
        Dictionary with chapter length statistics and distribution analysis
    """
    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    md_files = list(book_path.glob("**/*.md"))
    if not md_files:
        return {"error": "No markdown files found"}

    chapter_data = []
    word_counts = []

    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                word_count = len(content.split())

                # Extract chapter title
                chapter_title = "Untitled"
                lines = content.splitlines()
                for line in lines:
                    if line.strip().startswith('#'):
                        chapter_title = line.strip().lstrip('#').strip()
                        break

                chapter_data.append({
                    "filename": file_path.name,
                    "chapter_title": chapter_title,
                    "word_count": word_count
                })
                word_counts.append(word_count)
        except (UnicodeDecodeError, PermissionError):
            continue

    if not word_counts:
        return {"error": "No readable chapters found"}

    # Calculate statistics
    total_words = sum(word_counts)
    avg_words = total_words / len(word_counts)
    min_words = min(word_counts)
    max_words = max(word_counts)

    # Sort word counts for median calculation
    sorted_counts = sorted(word_counts)
    n = len(sorted_counts)
    median_words = (sorted_counts[n//2] + sorted_counts[(n-1)//2]) / 2 if n > 0 else 0

    # Categorize chapters by length
    short_chapters = [ch for ch in chapter_data if ch["word_count"] < avg_words * 0.7]
    medium_chapters = [ch for ch in chapter_data if avg_words * 0.7 <= ch["word_count"] <= avg_words * 1.3]
    long_chapters = [ch for ch in chapter_data if ch["word_count"] > avg_words * 1.3]

    # Calculate variance for consistency analysis
    variance = sum((count - avg_words) ** 2 for count in word_counts) / len(word_counts)
    std_deviation = variance ** 0.5

    return {
        "total_chapters": len(chapter_data),
        "statistics": {
            "total_words": total_words,
            "average_words": round(avg_words, 1),
            "median_words": round(median_words, 1),
            "min_words": min_words,
            "max_words": max_words,
            "standard_deviation": round(std_deviation, 1)
        },
        "distribution": {
            "short_chapters": {
                "count": len(short_chapters),
                "threshold": f"< {round(avg_words * 0.7)} words",
                "chapters": [{"title": ch["chapter_title"], "words": ch["word_count"]} for ch in short_chapters]
            },
            "medium_chapters": {
                "count": len(medium_chapters),
                "threshold": f"{round(avg_words * 0.7)}-{round(avg_words * 1.3)} words",
                "chapters": [{"title": ch["chapter_title"], "words": ch["word_count"]} for ch in medium_chapters]
            },
            "long_chapters": {
                "count": len(long_chapters),
                "threshold": f"> {round(avg_words * 1.3)} words",
                "chapters": [{"title": ch["chapter_title"], "words": ch["word_count"]} for ch in long_chapters]
            }
        },
        "consistency_analysis": {
            "is_consistent": std_deviation < avg_words * 0.3,
            "consistency_rating": "High" if std_deviation < avg_words * 0.2 else "Medium" if std_deviation < avg_words * 0.4 else "Low",
            "note": "Lower standard deviation indicates more consistent chapter lengths"
        },
        "chapter_details": sorted(chapter_data, key=lambda x: x["word_count"])
    }

@mcp.tool()
def get_book_stats() -> Dict[str, Any]:
    """
    Get overall statistics about the book project.

    Returns:
        Dictionary with book-wide statistics
    """
    book_path = Path(BOOK_DIRECTORY)
    if not book_path.exists():
        raise FileNotFoundError(f"Book directory not found: {BOOK_DIRECTORY}")

    md_files = list(book_path.glob("**/*.md"))
    total_words = 0
    total_chapters = len(md_files)
    total_size = 0

    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                total_words += len(content.split())
                total_size += len(content.encode('utf-8'))
        except (UnicodeDecodeError, PermissionError):
            continue

    return {
        "total_chapters": total_chapters,
        "total_words": total_words,
        "total_size_bytes": str(total_size),
        "average_words_per_chapter": total_words // total_chapters if total_chapters > 0 else 0,
        "book_directory": str(book_path)
    }

@mcp.tool()
def get_file_summary(file_path: str, max_lines: int = 10) -> Dict[str, Any]:
    """
    Get a summary of a markdown file including first few lines and basic stats.

    Args:
        file_path: Path to the markdown file
        max_lines: Maximum number of lines to include in preview

    Returns:
        Dictionary with file summary information
    """
    file_info = read_markdown_file(file_path)

    lines = file_info["content"].splitlines()
    preview_lines = lines[:max_lines]

    # Count headers
    header_count = sum(1 for line in lines if line.strip().startswith('#'))

    return {
        "filename": file_info["filename"],
        "path": file_info["path"],
        "total_lines": len(lines),
        "size_bytes": file_info["size_bytes"],  # This is already a string from read_markdown_file
        "header_count": header_count,
        "preview_lines": preview_lines,
        "truncated": len(lines) > max_lines
    }

def main():
    """Main entry point for the MCP server."""
    # Set the book directory if provided via environment variable
    global BOOK_DIRECTORY
    book_dir = os.getenv("BOOK_DIRECTORY")
    if book_dir:
        BOOK_DIRECTORY = book_dir
        print(f"Using book directory: {BOOK_DIRECTORY}")
    else:
        print(f"Using default book directory: {BOOK_DIRECTORY}")
        print("Set BOOK_DIRECTORY environment variable to specify a different directory")

    # Run the FastMCP server
    mcp.run()

if __name__ == "__main__":
    main()