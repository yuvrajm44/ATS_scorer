from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
import numpy as np


def normalize_education(edu_list):
    """
    Normalize education to standard levels: PhD, Masters, Bachelors, Diploma
    Returns highest level found
    """
    if not edu_list:
        return None
    
    # Convert list to single string for easier matching
    edu_text = ' '.join(edu_list).lower()
    
    # Check for PhD (highest)
    if any(term in edu_text for term in ['phd', 'ph.d', 'doctorate', 'doctoral']):
        return 'PhD'
    
    # Check for Masters
    if any(term in edu_text for term in ['master', 'masters', 'm.tech', 'm.e.', 'm.s.', 'mba', 'postgraduate']):
        return 'Masters'
    
    # Check for Bachelors
    if any(term in edu_text for term in ['bachelor', 'bachelors', 'b.e.', 'b.tech', 'b.s.', 'undergraduate', 'graduate']):
        return 'Bachelors'
    
    # Check for Diploma
    if any(term in edu_text for term in ['diploma', 'associate']):
        return 'Diploma'
    
    return 'Bachelors'  # Default if unclear


def match_skills(resume_skills, jd_skills,model):
    """
    Calculate skill match percentage using exact + semantic matching
    Returns: score (0-100), matched_skills, missing_skills
    """
    print(f"DEBUG MATCHING - JD Skills: {jd_skills}")
    print(f"DEBUG MATCHING - Resume Skills: {resume_skills}")
    if not jd_skills:
        return 50, [], []  
    
    if not resume_skills:
        return 0, [], jd_skills  # No skills in resume = 0 score
    
    matched_skills = []
    missing_skills = []
    
    # Track which resume skills have been used (avoid double counting)
    used_resume_skills = set()
    
    # Process each JD skill
    for jd_skill in jd_skills:
        jd_skill_clean = jd_skill.lower().strip()
        found_match = False
        
        # Step 1: Try exact match first (fast)
        for resume_skill in resume_skills:
            resume_skill_clean = resume_skill.lower().strip()
            
            if resume_skill_clean in used_resume_skills:
                continue
                
            if jd_skill_clean == resume_skill_clean:
                matched_skills.append(f"{jd_skill} ✓")
                used_resume_skills.add(resume_skill_clean)
                found_match = True
                break
        
        # Step 2: If no exact match, try semantic matching
        if not found_match:
            jd_embedding = model.encode(jd_skill, convert_to_tensor=True)
            best_similarity = 0
            best_resume_skill = None
            
            for resume_skill in resume_skills:
                resume_skill_clean = resume_skill.lower().strip()
                
                if resume_skill_clean in used_resume_skills:
                    continue
                
                resume_embedding = model.encode(resume_skill, convert_to_tensor=True)
                similarity = util.cos_sim(jd_embedding, resume_embedding).item()
                
                # Keep track of best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_resume_skill = resume_skill
            
            # If best match exceeds threshold, consider it a match
            if best_similarity >= 0.75:
                matched_skills.append(f"{jd_skill} ≈ {best_resume_skill} ({int(best_similarity*100)}%)")
                used_resume_skills.add(best_resume_skill.lower().strip())
                found_match = True
        
        # If still no match, add to missing
        if not found_match:
            missing_skills.append(jd_skill)
    
    # Calculate percentage
    match_percentage = (len(matched_skills) / len(jd_skills)) * 100
    
    return round(match_percentage, 2), matched_skills, missing_skills

def match_education(resume_edu, jd_edu):
    """
    Check if resume education meets JD requirements
    Returns: score (0-100), explanation
    """
    # Normalize both
    resume_level = normalize_education(resume_edu)
    jd_level = normalize_education(jd_edu)
    
    if not jd_level:
        return 100, "No education requirement specified"
    
    if not resume_level:
        return 50, "Education not found in resume"
    
    # Education hierarchy
    levels = ['Diploma', 'Bachelors', 'Masters', 'PhD']
    
    try:
        resume_rank = levels.index(resume_level)
        jd_rank = levels.index(jd_level)
        
        if resume_rank >= jd_rank:
            return 100, f"{resume_level} meets {jd_level} requirement"
        elif resume_rank == jd_rank - 1:
            return 60, f"{resume_level} is one level below {jd_level} requirement"
        else:
            return 30, f" {resume_level} does not meet {jd_level} requirement"
    except ValueError:
        return 50, "Unable to compare education levels"
