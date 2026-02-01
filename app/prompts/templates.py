"""
Prompt Templates for Career Intelligence Assistant.

This module contains all prompt templates used to guide the LLM's behavior.
Prompt engineering is critical for RAG systems because it:
1. Defines the AI's persona and capabilities
2. Sets guardrails to keep responses on-topic
3. Provides structure for consistent, actionable responses
4. Ensures accurate analysis by emphasizing precision

The system prompt is particularly important - it establishes the assistant's
role as a career advisor and provides explicit instructions for handling
comparison queries, which require analyzing ALL job postings.
"""

# =============================================================================
# SYSTEM PROMPT - DEFINES AI PERSONA AND BEHAVIOR
# =============================================================================
# The system prompt is sent with every LLM request to establish context.
# It defines:
# - Who the AI is (Career Intelligence Assistant)
# - What it can do (analyze skills, assess fit, provide interview prep)
# - How it should behave (specific, actionable, professional)
# - Critical rules (analyze ALL jobs in comparisons, match skills precisely)
SYSTEM_PROMPT = """You are a Career Intelligence Assistant - an expert career advisor who analyzes resumes against job descriptions to provide personalized career guidance.

Your capabilities:
- Analyze skill gaps between a candidate's resume and job requirements
- Assess experience alignment and transferable skills
- Provide interview preparation advice tailored to specific roles
- Offer actionable recommendations for career development

Guidelines:
1. Be specific and actionable in your advice
2. Reference specific details from the resume and job postings when making points
3. Use a professional but encouraging tone
4. Acknowledge both strengths and areas for improvement
5. When comparing multiple jobs, you MUST analyze ALL job postings provided in the context - do not skip any
6. If information is missing or unclear, acknowledge it rather than making assumptions

CRITICAL - When comparing jobs or finding best fit:
- You MUST analyze EVERY job posting provided in the context
- List ALL jobs by name before making recommendations
- For each job, explicitly state the skill match percentage or fit level
- Match technical skills precisely (e.g., ".NET" matches ".NET jobs", not "Salesforce" jobs)
- The best fit is the job where the candidate has the MOST matching technical skills
- Do not recommend jobs where the candidate lacks the primary required skills

When analyzing fit:
- Consider both explicit requirements and implicit expectations
- Weigh "must-have" vs "nice-to-have" qualifications differently
- Look for transferable skills that might not be obvious matches
- Consider industry context and terminology differences
- Match programming languages and frameworks precisely (C#, .NET, Python, Java, etc.)

Always maintain focus on career-related topics. If asked about unrelated topics, politely redirect to career guidance."""


# =============================================================================
# QUESTION-ANSWERING PROMPT TEMPLATE
# =============================================================================
# This template is used for all user questions. It structures the context
# (retrieved document chunks) and chat history to help the LLM generate
# accurate, context-aware responses. The template emphasizes:
# - Using only provided context (prevents hallucination)
# - Acknowledging when information is missing (honesty)
# - Referencing specific document details (accuracy)
QA_PROMPT_TEMPLATE = """Use the following context from the candidate's resume and job postings to answer the question. If you don't have enough information to answer accurately, say so.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

Provide a helpful, specific answer based on the context provided. Reference specific details from the documents when relevant."""


# =============================================================================
# QUESTION CONDENSATION TEMPLATE
# =============================================================================
# This template helps convert follow-up questions into standalone questions
# that include necessary context. For example:
# - User: "What skills am I missing?"
# - Follow-up: "For the first one?"
# - Condensed: "What skills am I missing for Job #1?"
# 
# This ensures retrieval works correctly even when users use pronouns or
# references to previous messages.
CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that captures the full context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


# =============================================================================
# SPECIALIZED PROMPT TEMPLATES
# =============================================================================
# These templates are designed for specific use cases. While we currently
# use the general QA_PROMPT_TEMPLATE for all queries, these specialized
# prompts could be used in the future for more targeted analysis.

# Specialized prompt for skill gap analysis
# This would be used when users specifically ask about missing skills.
# It structures the response to clearly separate matches, gaps, and recommendations.
SKILL_GAP_PROMPT = """Analyze the skill gap between the candidate's resume and the specified job posting.

Resume Content:
{resume_context}

Job Posting:
{job_context}

Provide a structured analysis:

1. **Skills Match** - Skills the candidate has that align with the job requirements
2. **Skills Gap** - Required skills the candidate appears to be missing
3. **Transferable Skills** - Related skills that could partially fulfill requirements
4. **Recommendations** - Specific actions to address the gaps

Be specific about which skills from the job posting are matched or missing."""


# Specialized prompt for experience alignment
# This would be used for questions about how well the candidate's background
# matches the role requirements. It provides a structured framework for
# assessing years of experience, role relevance, and industry fit.
EXPERIENCE_ALIGNMENT_PROMPT = """Assess how well the candidate's experience aligns with the job requirements.

Resume Content:
{resume_context}

Job Posting:
{job_context}

Analyze:

1. **Years of Experience** - Does the candidate meet the experience requirements?
2. **Role Relevance** - How relevant are their past roles to this position?
3. **Industry Fit** - Is their industry background aligned or transferable?
4. **Key Achievements** - Which achievements are most relevant to highlight?
5. **Gaps to Address** - Experience areas where they may need to demonstrate transferability

Provide an overall alignment score estimate (Strong/Moderate/Weak) with justification."""


# Specialized prompt for interview preparation
# This would be used when users ask for interview prep help. It guides the
# LLM to provide specific, actionable advice including likely questions,
# strengths to highlight, and potential concerns to address.
INTERVIEW_PREP_PROMPT = """Based on the candidate's resume and the job posting, provide tailored interview preparation guidance.

Resume Content:
{resume_context}

Job Posting:
{job_context}

Provide:

1. **Likely Technical Questions** - Based on the job requirements and candidate's background
2. **Behavioral Questions** - Common questions for this type of role
3. **Strengths to Highlight** - Key points from their resume to emphasize
4. **Potential Concerns to Address** - Gaps or weaknesses to prepare responses for
5. **Questions to Ask** - Thoughtful questions the candidate could ask the interviewer

Make the advice specific to both the role and the candidate's background."""


# Template for comparing multiple job postings
# This template is designed for comparison queries. It instructs the LLM to:
# - Assess each job individually with a fit score
# - Identify strengths and gaps for each role
# - Provide a summary recommendation
# This ensures comprehensive analysis when users ask "which job is best for me?"
JOB_COMPARISON_PROMPT = """Compare the candidate's fit across multiple job postings.

Resume Content:
{resume_context}

Job Postings:
{jobs_context}

For each job, assess:
- Overall fit score (1-10)
- Key strengths for this role
- Main gaps or concerns
- Unique selling points the candidate brings

Then provide a summary recommendation on which role(s) the candidate is best suited for and why."""
