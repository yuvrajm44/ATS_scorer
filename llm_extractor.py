from groq import Groq
import json
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_entities_with_llm(text, text_type="JD"):
    """
    Extract skills and education from text using Groq LLM
    """
    
    system_prompt = """You are an expert NER model that extracts entities from job descriptions and resumes.

        SKILLS EXTRACTION RULES:
        - Extract technical skills including: programming languages, frameworks, tools, cloud platforms, libraries, and domain-specific techniques/methodologies !IMPORTANT!
        - Valid skills can be single words OR multi-word phrases that represent specific capabilities
        - Examples of valid multi-word skills: "Machine Learning", "Data Visualization", "Time Series Forecasting", "Computer Vision", "Deep Learning", "Demand Forecasting", "Statistical Analysis" !IMPORTANT!
        - Focus on extracting NOUN PHRASES (things you do/use/know) - not verb phrases or action descriptions
        - Use standardized names when possible: "Python", "AWS", "Docker", "TensorFlow", "SQL"
        - DO NOT extract: action phrases containing "experience in", "ability to", "knowledge of", "proficiency in"
        - DO NOT extract: soft skills, personality traits, job requirements, or company descriptions
        - Extract both concrete tools (Python, Docker) AND analytical techniques (Forecasting, Regression, Classification)

        EDUCATION EXTRACTION RULES:
        - Extract only degree names: "Bachelor's", "Master's", "PhD", "B.Tech", "M.Tech", "B.Sc", "M.Sc"
        - Extract field of study if mentioned: "Computer Science", "Engineering", "Data Science", "Statistics"
        - DO NOT extract full sentences or descriptions
        - Format as simple text: "Bachelor's in Computer Science" or "Master's degree"

        EXPERIENCE EXTRACTION RULES (FOR RESUMES ONLY) :
        - Calculate total years of professional work experience by analyzing work history dates  IMPORTANT!
        - Look for employment dates in formats like: "May 2023 - April 2024", "Sep 2024 - Present", "01/2025 - 07/2025"
        - Sum up all work experience periods 
        - "Present", "Current", "Now" means up to current date
        - Return as decimal number (e.g., 1.5 for 1 year 6 months, 0.9 for 11 months)
        - If no work experience found, return 0

        Return results in strict JSON format only."""

    if text_type == "Resume":
        user_prompt = f"""Extract all technical skills, education, contact info, and total work experience from this Resume:

    {text}

    Return JSON with this exact format:
    {{
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "+91 9876543210",
    "skills": ["Python", "AWS", "Docker"],
    "education": ["Bachelor's in Computer Science", "Master's degree"],
    "experience_years": 1.5
    }}

    IMPORTANT: Calculate experience_years by summing all job durations. Current date is October 2025."""
    else:
        user_prompt = f"""Extract all technical skills and education requirements from this {text_type}:

    {text}

    Return JSON with this exact format:
    {{
    "skills": ["Python", "AWS", "Docker"],
    "education": ["Bachelor's in Computer Science", "Master's degree"]
    }}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate and filter skills
        validated_skills = validate_skills(result.get("skills", []))
        validated_education = validate_education(result.get("education", []))
        experience_years = result.get("experience_years", 0) if text_type == "Resume" else 0
        
        
        return {
            "name": result.get("name", ""),
            "email": result.get("email", ""),
            "phone": result.get("phone", ""),
            "skills": validated_skills,
            "education": validated_education,
            "experience_years": experience_years
        }
    
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return {"skills": [], "education": []}


def validate_skills(skills):
    """Filter out invalid/generic skills"""
    
    # Blacklist: generic phrases that aren't specific skills
    blacklist = [
        'experience', 'knowledge', 'understanding', 'ability', 'working',
        'expertise', 'proficiency', 'familiarity', 'background',
         'machine learning experience', 'deep learning experience'
        , 'fine-tuning', 'cloud platforms'
    ]
    
    valid_skills = []
    for skill in skills:
        skill_lower = skill.lower().strip()
        
        # Skip if too long (likely a phrase, not a skill)
        if len(skill) > 30:
            continue
        
        # Skip if contains blacklisted terms
        if any(term in skill_lower for term in blacklist):
            continue
        
        # Skip if contains common stopwords that indicate it's a phrase
        if ' in ' in skill_lower or ' of ' in skill_lower or ' with ' in skill_lower:
            continue
        
        valid_skills.append(skill)
    
    return valid_skills


def validate_education(education_list):
    """Clean and format education entries"""
    
    valid_education = []
    for edu in education_list:
        # Remove dict-like strings
        if isinstance(edu, dict):
            # Extract just degree and field
            degree = edu.get('degree', '')
            field = edu.get('field', '')
            if degree:
                edu_text = f"{degree} {field}".strip() if field else degree
                valid_education.append(edu_text)
        elif isinstance(edu, str):
            # Remove surrounding quotes, braces, etc.
            edu_clean = edu.strip().strip("'\"{}").strip()
            
            # Skip if too long (likely a sentence)
            if len(edu_clean) > 60:
                continue
            
            # Skip if it's a stringified dict
            if edu_clean.startswith('{') or 'institution' in edu_clean.lower():
                continue
            
            valid_education.append(edu_clean)
    
    return valid_education


def cleanup_merged_entities(ner_skills, ner_education, llm_skills, llm_education, original_text):
    """
    Clean and validate merged entities from NER + LLM
    """
    
    system_prompt = """You are a data validator. Your job is to merge and deduplicate skills and education.

RULES:
- Merge similar skills (e.g., "Python" and "python" → "Python")
- Remove exact duplicates
- Keep only technical skills and degree names
- Remove company descriptions, junk text, soft skills
- For education: keep only degree types, not full sentences
- Return cleaned, deduplicated lists only"""

    user_prompt = f"""Merge and clean these extracted entities:

NER found:
- Skills: {ner_skills}
- Education: {ner_education}

LLM found:
- Skills: {llm_skills}
- Education: {llm_education}

Return JSON: {{"skills": [...], "education": [...]}}

Only keep valid technical skills and degree names. Remove duplicates and junk."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Additional validation
        final_skills = validate_skills(result.get("skills", []))
        final_education = validate_education(result.get("education", []))
        
        return {"skills": final_skills, "education": final_education}
    
    except Exception as e:
        print(f"Error in cleanup: {e}")
        # Fallback: manual merge and deduplicate
        all_skills = list(set([s for s in (ner_skills + llm_skills) if s]))
        all_education = list(set([e for e in (ner_education + llm_education) if e]))
        
        return {
            "skills": validate_skills(all_skills), 
            "education": validate_education(all_education)
        }