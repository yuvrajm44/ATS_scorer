

import sys
import os

sys.path.append(r'D:\NER_trained')

from jd_ner import extract_yoe_from_jd, extract_skills_education_ner, extract_education_regex, extract_requirements_section
from ner_resume_hybrid import combined_extraction
from matcher import match_skills, match_education
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
import numpy as np

print("Loading Sentence Transformer model...")
model = SentenceTransformer('D:\\NER_trained\\sentence_transformer_model')
print("✅ Model loaded successfully!\n")




def calculate_ats_score(skill_match_pct, semantic_sim, jd_yoe, resume_yoe, edu_score):
    """
    Calculate final ATS score using weighted formula:
    - Skills Match: 40%
    - Experience Match: 30%
    - Semantic Similarity: 20%
    - Education Match: 10%
    
    Returns: final_score (0-100), breakdown dict
    """
    
    # Component 1: Skills Score (0-40 points)
    skills_score = (skill_match_pct / 100) * 40
    
    # Component 2: Experience Score (0-30 points)
    if jd_yoe == "Not Found" or resume_yoe == 0:
        yoe_score = 15  # Neutral when data missing
    else:
        jd_yoe_num = float(jd_yoe)
        
        if resume_yoe >= jd_yoe_num:
            yoe_score = 30  # Full score - meets requirement
        elif resume_yoe >= jd_yoe_num * 0.75:
            yoe_score = 25  # 75-99% of required
        elif resume_yoe >= jd_yoe_num * 0.5:
            yoe_score = 18  # 50-74% of required
        else:
            # Proportional for < 50%, minimum 8 points
            yoe_score = max(8, (resume_yoe / jd_yoe_num) * 30)
    
    # Component 3: Semantic Score (0-20 points)
    semantic_score = semantic_sim * 20
    
    # Component 4: Education Score (0-10 points)
    education_score = (edu_score / 100) * 10
    
    # Final score
    final_score = skills_score + yoe_score + semantic_score + education_score

    # Apply penalty if skill match is poor
    if skill_match_pct < 50:
        penalty = (50 - skill_match_pct) * 0.5
        final_score = max(final_score - penalty, 0)
        final_score = round(final_score, 2)
    
    # Breakdown for display
    breakdown = {
        'skills_score': round(skills_score, 2),
        'yoe_score': round(yoe_score, 2),
        'semantic_score': round(semantic_score, 2),
        'education_score': round(education_score, 2),
        'final_score': round(final_score, 2)
    }
    
    return round(final_score, 2), breakdown


def score_resume_for_jd(jd_text, resume_text):
    """
    Complete ATS scoring pipeline
    Returns: result dictionary with all details
    """
    
    print("\n" + "="*70)
    print("PROCESSING...")
    print("="*70 + "\n")
    
    # Step 1: Extract from JD
    jd_result = extract_yoe_from_jd(jd_text)
    print(f"DEBUG - YOE Extraction Result: {jd_result}")
    jd_yoe = jd_result['min_yoe']
    
    # Extract requirements section for better NER
    req_section = extract_requirements_section(jd_text)
    text_for_ner = req_section if req_section else jd_text
    
    jd_skills_raw, jd_edu_ner = extract_skills_education_ner(text_for_ner)
    jd_skills = [s.lower() for s in jd_skills_raw]
    print(f"DEBUG - JD Skills Extracted: {jd_skills}")
    jd_education = sorted(list(set(jd_edu_ner )))
    
    # Step 2: Extract from Resume
    resume_data = combined_extraction(resume_text)
    print(f"DEBUG - Resume Skills Extracted: {resume_data['skills']}") 
    resume_yoe = resume_data['experience_years']
    resume_skills = resume_data['skills']
    resume_education = resume_data['education']
    candidate_name = resume_data.get('name', ["Not Found"])[0] if resume_data.get('name') else "Not Found"
    candidate_email = resume_data.get('email', ["Not Found"])[0] if resume_data.get('email') else "Not Found"
    candidate_phone = resume_data.get('phone', ["Not Found"])[0] if resume_data.get('phone') else "Not Found"
    
    # Step 3: Match Skills
    skill_match_pct, matched_skills, missing_skills = match_skills(resume_skills, jd_skills,model)
    
    # Step 4: Match Education
    edu_score, edu_explanation = match_education(resume_education, jd_education)
    
    # Step 5: Calculate Semantic Similarity
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    semantic_similarity = cos_sim(jd_embedding, resume_embedding)[0][0].item()
    
    # Step 6: Calculate Final ATS Score
    final_score, breakdown = calculate_ats_score(
        skill_match_pct, semantic_similarity, jd_yoe, resume_yoe, edu_score
    )
    
    # Step 7: Determine decision
    # NEW:
    if final_score >= 65:
        decision = "FIT ✅"
        status = "SHORTLIST FOR INTERVIEW"
    else:
        decision = "UNFIT ❌"
        status = "REJECT"
    
    # Return all results
    return {
        'candidate_name': candidate_name,
        'candidate_email': candidate_email,
        'candidate_phone': candidate_phone,
        'jd_yoe': jd_yoe,
        'resume_yoe': resume_yoe,
        'jd_skills': jd_skills,
        'resume_skills': resume_skills,
        'skill_match_pct': skill_match_pct,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'jd_education': jd_education,
        'resume_education': resume_education,
        'edu_score': edu_score,
        'edu_explanation': edu_explanation,
        'semantic_similarity': round(semantic_similarity, 3),
        'breakdown': breakdown,
        'final_score': final_score,
        'decision': decision,
        'status': status
    }

