import spacy
import re
from datetime import datetime
from jd_ner import extract_skills_regex
from llm_extractor import extract_entities_with_llm, cleanup_merged_entities
from pathlib import Path

# Load models
BASE_DIR = Path(__file__).resolve().parent
nlp_trained = BASE_DIR / "pre-trained-ner-mode"
nlp_pretrained = spacy.load("en_core_web_sm")

def extract_total_experience(text):
    """Extract total years of experience from WORK EXPERIENCE section only"""
    
    current_year = datetime.now().year
    current_month = datetime.now().month
    total_months = 0
    
    months_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    # Extract only EXPERIENCE section
    experience_section = ""
    text_upper = text.upper()
    
    exp_keywords = ['EXPERIENCE', 'WORK EXPERIENCE', 'EMPLOYMENT HISTORY']
    exp_start = -1
    
    for keyword in exp_keywords:
        idx = text_upper.find(keyword)
        if idx != -1:
            exp_start = idx
            break
    
    if exp_start == -1:
        return 0
    
    # Find where experience section ends
    end_keywords = ['EDUCATION', 'PROJECTS', 'SKILLS', 'TECHNICAL SKILLS']
    exp_end = len(text)
    
    for keyword in end_keywords:
        idx = text_upper.find(keyword, exp_start + 10)
        if idx != -1 and idx < exp_end:
            exp_end = idx
    
    experience_section = text[exp_start:exp_end]
    
    # Extract dates from experience section only
    found_ranges = []
    
    # Pattern 1: "May 2023 – April 2024"
    pattern1 = r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\s*(?:to|–|-|till)\s*(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})'
    matches1 = re.finditer(pattern1, experience_section, re.IGNORECASE)
    
    for match in matches1:
        start_month_str, start_year, end_month_str, end_year = match.groups()
        start_month = months_map.get(start_month_str.lower(), 1)
        end_month = months_map.get(end_month_str.lower(), 12)
        
        months = (int(end_year) - int(start_year)) * 12 + (end_month - start_month)
        if months > 0:
            total_months += months
            found_ranges.append(match.span())
    
    # Pattern 2: "May 2023 - Present"
    pattern2 = r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\s*(?:to|–|-|till)\s*(?:Present|Current|Till Date|Now|Today)'
    matches2 = re.finditer(pattern2, experience_section, re.IGNORECASE)
    
    for match in matches2:
        start_month_str, start_year = match.groups()
        start_month = months_map.get(start_month_str.lower(), 1)
        months = (current_year - int(start_year)) * 12 + (current_month - start_month)
        if months > 0:
            total_months += months

    # Pattern 4: "2018 to Present" (year only, ongoing)
    pattern4 = r'(?<!\w)(\d{4})\s*(?:to|–|-|till)\s*(?:Present|Current|Till Date|Now|Today)'
    matches4 = re.finditer(pattern4, experience_section, re.IGNORECASE)

    for match in matches4:
        # Check if already counted (avoid overlap with Pattern 2)
        if any(match.start() >= r[0] and match.end() <= r[1] for r in found_ranges):
            continue
        
        start_year = match.group(1)
        months = (current_year - int(start_year)) * 12 + current_month
        if 0 < months <= 600:  # max 50 years
            total_months += months
    
    total_years = round(total_months / 12, 1) if total_months > 0 else 0

    
    
    if total_years > 50:
        total_years = 0
    
    return total_years


def combined_extraction(resume_text):
    """Combined extraction with experience calculation"""
    
    results = {
        'name': [],
        'email': [],
        'phone': [],
        'skills': [],
        'companies': [],
        'education': [],
        'designation': [],
        'location': [],
        'experience_years': 0
    }
    
    # 1. EXTRACT NAME from first line (improved)
    lines = [l.strip() for l in resume_text.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        # Name should be 2-4 words, capitalized, no technical terms
        words = first_line.split()
        if (2 <= len(words) <= 4 and 
            len(first_line) < 50 and 
            not any(c in first_line for c in ['@', 'http', '•', ':']) and
            not any(tech in first_line for tech in ['HTML', 'CSS', 'JavaScript', 'Python', 'Java']) and
            first_line[0].isupper()):
            results['name'].append(first_line)
    
    # 2. YOUR TRAINED MODEL
    doc_trained = nlp_trained(resume_text)
    for ent in doc_trained.ents:
        if ent.label_ == 'NAME':
            # Additional check: not a technical term
            if not any(tech in ent.text for tech in ['HTML', 'CSS', 'JS', 'Python', 'Java']):
                if not results['name'] or len(ent.text) > len(results['name'][0]):
                    results['name'] = [ent.text]
        elif ent.label_ == 'COMPANIES_WORKED_AT':
            if len(ent.text.split()) > 1 and 'and' not in ent.text.lower():
                if not any(word in ent.text for word in ['Engineer', 'Developer', 'Manager', 'Designer']):
                    results['companies'].append(ent.text)
        elif ent.label_ == 'DESIGNATION':
            results['designation'].append(ent.text)
        elif ent.label_ == 'COLLEGE_NAME':
            results['education'].append(ent.text)
        elif ent.label_ == 'SKILLS':
            if len(ent.text.split()) <= 5:
                results['skills'].append(ent.text)
        elif ent.label_ == 'EMAIL_ADDRESS':
            results['email'].append(ent.text)
        elif ent.label_ == 'LOCATION':
            results['location'].append(ent.text)
    
    # 3. PRETRAINED SPACY
    doc_pretrained = nlp_pretrained(resume_text)
    for ent in doc_pretrained.ents:
        if ent.label_ == 'PERSON' and not results['name'] and len(ent.text.split()) <= 4:
            # Must be in first 200 characters
            if resume_text.find(ent.text) < 200:
                results['name'].append(ent.text)
        
        elif ent.label_ == 'ORG':
            if any(kw in ent.text for kw in ['Microsoft', 'Google', 'Amazon', 'IBM', 'Oracle', 'Apple', 'Meta', 'Netflix', 'Tesla']):
                if ent.text not in ['Microsoft Azure', 'Google Cloud']:
                    results['companies'].append(ent.text)
        
        elif ent.label_ == 'GPE':
            real_cities = ['Nagpur', 'Maharashtra', 'Mumbai', 'Delhi', 'Bangalore', 'Pune', 
                          'Karnataka', 'Hyderabad', 'Chennai', 'Kolkata', 'India']
            if ent.text in real_cities:
                results['location'].append(ent.text)
    
    # 4. SECTION-BASED EXTRACTION
    text_upper = resume_text.upper()
    
    # Skills
    if 'SKILLS' in text_upper or 'TECHNICAL SKILLS' in text_upper:
        skill_idx = max(text_upper.find('SKILLS'), text_upper.find('TECHNICAL SKILLS'))
        if skill_idx != -1:
            skills_section = resume_text[skill_idx+6:skill_idx+600]
            
            skill_list = ['Machine Learning', 'Deep Learning', 'Natural Language Processing', 
                         'Big Data', 'Python', 'Java', 'JavaScript', 'SQL', 'React', 'Angular',
                         'HTML5', 'HTML', 'CSS3', 'CSS', 'Bootstrap', 'jQuery', 'Photoshop', 
                         'Docker', 'Kubernetes', 'AWS', 'Azure', 'MongoDB', 'PostgreSQL', 
                         'Git', 'Node.js', 'Django', 'Flask', 'Spring', 'TypeScript', 
                         'C++', 'C#', 'Ruby', 'PHP', 'SASS']
            
            for skill in skill_list:
                if re.search(r'\b' + re.escape(skill) + r'\b', skills_section, re.IGNORECASE):
                    results['skills'].append(skill)
    
    # Education
    if 'EDUCATION' in text_upper:
        edu_idx = text_upper.find('EDUCATION')
        edu_section = resume_text[edu_idx+9:edu_idx+500]
        
        college_pattern = r'([A-Z][A-Za-z\s\.]+(?:College|University|Institute)[A-Za-z\s]*?)(?:\s*[–-]\s*|$|\n)'
        colleges = re.findall(college_pattern, edu_section)
        for college in colleges:
            clean = college.strip()
            if len(clean) > 5:
                results['education'].append(clean)
        
        degrees = re.findall(r'\b(B\.E|B\.Tech|M\.Tech|MBA|MCA|B\.Sc|M\.Sc|Ph\.D|Bachelor|Master)\b', 
                            edu_section, re.IGNORECASE)
        results['education'].extend(degrees)
    
    # Work Experience / Companies
    if 'WORK EXPERIENCE' in text_upper or 'EXPERIENCE' in text_upper:
        exp_idx = max(text_upper.find('WORK EXPERIENCE'), text_upper.find('EXPERIENCE'))
        if exp_idx != -1:
            exp_section = resume_text[exp_idx:exp_idx+2000]
            
            company_pattern = r'\n([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s*[–-]\s*[A-Z]'
            companies = re.findall(company_pattern, exp_section)
            for comp in companies:
                if comp not in ['Software Engineer', 'Senior Developer', 'Web Designer', 'Junior Developer']:
                    results['companies'].append(comp)
    
    if 'COMPANY DETAILS' in text_upper:
        company_idx = text_upper.find('COMPANY DETAILS')
        company_section = resume_text[company_idx:company_idx+1500]
        
        company_matches = re.findall(r'company\s*[-:]\s*([A-Za-z0-9\s&\.]+?)(?:\n|description)', 
                                     company_section, re.IGNORECASE)
        for comp in company_matches:
            comp_clean = comp.strip()
            if 'www.' not in comp_clean and 'description' not in comp_clean and len(comp_clean) > 3:
                results['companies'].append(comp_clean)
    
    # Designation
    designation_pattern = r'\n((?:Senior |Junior |Lead |Staff |Principal )?(?:Software|Web|Data|Full Stack|Backend|Frontend|DevOps)\s+(?:Engineer|Developer|Designer|Analyst|Architect))'
    designations = re.findall(designation_pattern, resume_text, re.IGNORECASE)
    results['designation'].extend(designations)
    
    # 5. EXTRACT TOTAL EXPERIENCE
    results['experience_years'] = extract_total_experience(resume_text)
    
    # regex-based skills extraction
    skills_from_regex = extract_skills_regex(resume_text)
    results['skills'].extend(skills_from_regex)

    # 6. REGEX PATTERNS
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    results['email'].extend(emails)
    
    phones = re.findall(r'(?:\+91|0)?[6-9]\d{9}', resume_text)
    phones.extend(re.findall(r'\+?\d[\d\s\-\(\)]{9,}\d', resume_text))
    results['phone'].extend(phones)

    # LLM ENHANCEMENT FOR SKILLS AND EDUCATION
    llm_result = extract_entities_with_llm(resume_text, text_type="Resume")
    skills_llm = llm_result.get("skills", [])
    education_llm = llm_result.get("education", [])
    experience_llm = llm_result.get("experience_years", 0)
    name_llm = llm_result.get("name", "")
    email_llm = llm_result.get("email", "")
    phone_llm = llm_result.get("phone", "")

    # Always use LLM if available (more accurate than NER)
    if name_llm:
        results['name'] = [name_llm]
    if email_llm:
        results['email'] = [email_llm]
    if phone_llm:
        results['phone'] = [phone_llm]

    print(f"DEBUG - LLM Raw Result: {llm_result}")
    print(f"DEBUG - LLM Experience Extracted: {experience_llm}")

    # Cleanup and merge
    final_result = cleanup_merged_entities(
        results['skills'], results['education'],
        skills_llm, education_llm,
        resume_text
    )
    
    results['skills'] = final_result["skills"]
    results['education'] = final_result["education"]

    # USE LLM EXPERIENCE (with regex as fallback)
    if experience_llm > 0:
        results['experience_years'] = experience_llm
    else:
        # Fallback to regex if LLM returns 0
        results['experience_years'] = extract_total_experience(resume_text)
    
    # 7. CLEAN UP
    for key in results:
        if key == 'experience_years':
            continue
        
        seen = set()
        cleaned = []
        for item in results[key]:
            if not item:
                continue
            item_clean = str(item).strip()
            
            if item_clean.upper().startswith(('EDUCATION', 'SKILLS', 'WORK', 'COMPANY', 'EXPERIENCE')):
                continue
            
            item_lower = item_clean.lower()
            if item_lower not in seen and len(item_clean) > 1:
                seen.add(item_lower)
                cleaned.append(item_clean)
        
        results[key] = cleaned
    
    return results
