import re
import pandas as pd
import sys
import spacy
from pathlib import Path
from llm_extractor import extract_entities_with_llm, cleanup_merged_entities
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pre-trained-ner-mode"
nlp = spacy.load(MODEL_PATH)

def extract_requirements_section(text):
    """
    Extract only the Requirements/Qualifications/Experience section from JD
    to avoid picking up company history or other irrelevant YOE mentions
    """
    
    # Common section headers for requirements
    section_headers = [
        r'EDUCATION/EXPERIENCE:?',
        r'EDUCATION AND EXPERIENCE:?',
        r'YOU\'LL BRING THESE QUALIFICATIONS:?',
        r'QUALIFICATIONS:?',
        r'REQUIREMENTS:?',
        r'REQUIRED QUALIFICATIONS:?',
        r'MINIMUM QUALIFICATIONS:?',
        r'BASIC QUALIFICATIONS:?',
        r'EXPERIENCE:?',
        r'SKILLS, EXPERIENCE AND REQUIREMENTS:?',
        r'WHAT YOU\'LL BRING:?',
        r'REQUIRED SKILLS AND EXPERIENCE:?',
        r'MINIMUM REQUIREMENTS:?',
        r'REQUIRED SKILLS & EXPERIENCE:?',  # Add this
        r'REQUIRED SKILLS AND EXPERIENCE:?',  # Add this
        r'KEY RESPONSIBILITIES:?'
    ]
    
    # Try to find any of these headers
    for header in section_headers:
        # Case insensitive search
        match = re.search(header, text, re.IGNORECASE)
        if match:
            # Get text from this point onwards
            start_pos = match.start()
            
            # Try to find where requirements section ends (next major section)
            end_section_headers = [
                r'\n\n[A-Z][A-Z\s]{10,}:',  # All caps section header
                r'\n\n\*\*[A-Z][A-Z\s]{5,}\*\*',  # Bold section header
                r'PHYSICAL DEMANDS',
                r'PHYSICAL REQUIREMENTS',
                r'WHAT TO EXPECT',
                r'ABOUT THE JOB',
                r'WHAT WE OFFER',
                r'BENEFITS',
                r'ADDITIONAL INFORMATION',
                r'ADDITIONAL REQUIREMENTS',
                r'SUPERVISORY RESPONSIBILITIES',
            ]
            
            # Find the earliest end marker
            end_pos = len(text)
            for end_header in end_section_headers:
                end_match = re.search(end_header, text[start_pos:], re.IGNORECASE)
                if end_match:
                    potential_end = start_pos + end_match.start()
                    if potential_end < end_pos:
                        end_pos = potential_end
            
            # Extract the section
            requirements_text = text[start_pos:end_pos]
            
            # Return if we found a meaningful section (at least 100 chars)
            if len(requirements_text.strip()) > 100:
                return requirements_text
    
    # If no requirements section found, return None (will use full text)
    return None


def extract_yoe_from_jd(jd_text):
    """
    Extract Years of Experience from Job Description (IMPROVED VERSION)
    Returns: dict with min_yoe, max_yoe, is_range, is_plus
    """
    
    # Handle None or empty text
    if not jd_text or pd.isna(jd_text):
        return {
            'min_yoe': "Not Found",
            'max_yoe': "Not Found",
            'is_range': False,
            'is_plus': False,
            'found': False,
            'all_found': []
        }
    
    # Convert to string and unescape common escape sequences
    text = str(jd_text)
    text = text.replace('\\+', '+')      # Unescape \+
    text = text.replace('\\-', '-')      # Unescape \-
    text = text.replace("\\'", "'")      # Unescape \'
    text = text.replace('\\"', '"')      # Unescape \"
    text = text.replace('\\n', ' ')      # Replace \n with space
    text = text.replace('\\t', ' ')      # Replace \t with space
    
    # Extract only Requirements/Qualifications section to avoid company history
    requirements_section = extract_requirements_section(text)
    
    # If we found a requirements section, use it; otherwise use full text
    text_to_search = text

    print(f"\nDEBUG - Searching in text (first 500 chars):")
    print(text_to_search[:500])
    print(f"\nDEBUG - Looking for experience patterns...")
    
    # Convert to lowercase for easier matching
    text_to_search = text_to_search.lower()
    
    # Store all found YOE values
    found_yoe = []
    
    # Pattern 1: "X+ years" or "X years" - MORE FLEXIBLE
    # Catches: "5+ years", "5 years of experience", "5 years in Python", "5 years managing"
    pattern1 = r'(\d+)\+?\s*(?:to|\-|–|or)?\s*(\d+)?\s*years?\s*(?:of|in|with|managing|working|programming)?'
    matches1 = re.finditer(pattern1, text_to_search)
    for match in matches1:
        min_years = int(match.group(1))
        max_years = int(match.group(2)) if match.group(2) else None
        
        # Check if there's a + sign
        full_match = match.group(0)
        is_plus = '+' in full_match
        
        found_yoe.append({
            'min': min_years,
            'max': max_years,
            'is_range': max_years is not None,
            'is_plus': is_plus
        })
    
    # Pattern 2: "Minimum/At least X years" - FIXED FOR WORD NUMBERS
    pattern2 = r'(?:minimum|minimum\s+of|at\s+least|atleast)\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)\s*(?:\+)?\s*years?'
    matches2 = re.findall(pattern2, text_to_search)
    for match in matches2:
        # Convert word to number if needed
        word_to_num = {
            'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        years = word_to_num.get(match, int(match) if match.isdigit() else 0)
        if years > 0:
            found_yoe.append({
                'min': years,
                'max': None,
                'is_range': False,
                'is_plus': False
            })
    
    # Pattern 3: "X-Y years" or "X to Y years" (range) - MORE FLEXIBLE
    pattern3 = r'(\d+)\s*(?:\-|–|to)\s*(\d+)\s*years?'
    matches3 = re.findall(pattern3, text_to_search)
    for match in matches3:
        found_yoe.append({
            'min': int(match[0]),
            'max': int(match[1]),
            'is_range': True,
            'is_plus': False
        })
    
    # Pattern 4: Written numbers ANYWHERE - IMPROVED
    # Catches: "two years' experience", "five years of experience", "three years in"
    word_to_num = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    pattern4 = r'\b(two|three|four|five|six|seven|eight|nine|ten)\s+years?'
    matches4 = re.findall(pattern4, text_to_search)
    for match in matches4:
        found_yoe.append({
            'min': word_to_num[match],
            'max': None,
            'is_range': False,
            'is_plus': False
        })
    
    # Pattern 5: "X years (Required)" or "X years (Preferred)"
    pattern5 = r'(\d+)\s*years?\s*\((?:required|preferred)\)'
    matches5 = re.findall(pattern5, text_to_search)
    for match in matches5:
        found_yoe.append({
            'min': int(match),
            'max': None,
            'is_range': False,
            'is_plus': False
        })
    
    # If nothing found, return None
    if not found_yoe:
        return {
            'min_yoe': "Not Found",
            'max_yoe': "Not Found",
            'is_range': False,
            'is_plus': False,
            'found': False,
            'all_found': []
        }
    
    # Return the entry with HIGHEST minimum (most conservative requirement)
    best_match = max(found_yoe, key=lambda x: x['min'])
    
    return {
        'min_yoe': best_match['min'],
        'max_yoe': best_match['max'] if best_match['max'] else "Not Found",
        'is_range': best_match['is_range'],
        'is_plus': best_match['is_plus'],
        'found': True,
        'all_found': [f"{x['min']}-{x['max']}" if x['is_range'] else f"{x['min']}+" if x['is_plus'] else str(x['min']) for x in found_yoe]
    }



# Extract skills and education using NER
def extract_skills_education_ner(text):
    """Extract SKILLS and EDUCATION using NER + LLM hybrid"""
    
    # Step 1: Get NER skills
    doc = nlp(text)
    skills_ner = []
    education_ner = []
    
    for ent in doc.ents:
        if ent.label_ == "SKILLS":
            skills_ner.append(ent.text.strip())
        elif ent.label_ == "EDUCATION":
            education_ner.append(ent.text.strip())
    
    # Add regex skills to NER
    skills_regex = extract_skills_regex(text)
    skills_ner.extend(skills_regex)
    skills_ner = list(set(skills_ner))  # Remove duplicates
    
    # Step 2: Get LLM extraction
    llm_result = extract_entities_with_llm(text, text_type="JD")
    skills_llm = llm_result.get("skills", [])
    education_llm = llm_result.get("education", [])
    
    # Step 3: Cleanup and merge
    final_result = cleanup_merged_entities(
        skills_ner, education_ner, 
        skills_llm, education_llm, 
        text
    )
    
    return final_result["skills"], final_result["education"]

def extract_skills_regex(text):
    """Extract skills using regex patterns and keyword matching"""
    
    # Comprehensive skill list
    skill_keywords = [
        # Programming Languages
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'Ruby', 'PHP', 'Scala', 'R', 'Julia',
        
        # Data Engineering & Big Data
        'PySpark', 'Apache Spark', 'Hadoop', 'Hive', 'Kafka', 'Apache Kafka', 'Airflow', 'Apache Airflow', 
        'Databricks', 'Snowflake', 'BigQuery', 'Redshift', 'Presto', 'Trino', 'Flink', 'Storm',
        
        # Databases & Data Stores
        'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch', 'DynamoDB',
        'Oracle', 'SQL Server', 'MariaDB', 'Neo4j', 'Couchbase',
        
        # Cloud Platforms & Services
        'AWS', 'Azure', 'GCP', 'Google Cloud', 'S3', 'EC2', 'Lambda', 'Glue', 'EMR', 'Athena', 
        'Step Functions', 'CloudFormation', 'Terraform', 'SageMaker', 'Kinesis', 'CloudWatch',
        'Azure Data Factory', 'Azure Databricks', 'Google Cloud Storage', 'Cloud Functions',
        
        # ML/AI Frameworks & Tools
        'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'XGBoost', 'LightGBM', 'Hugging Face',
        'NLTK', 'spaCy', 'OpenCV', 'MLflow', 'Kubeflow', 'LangChain', 'LlamaIndex',
        
        # Data Science & Analytics
        'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly', 'Tableau', 'Power BI', 'Looker',
        'Excel', 'Jupyter', 'SAS', 'SPSS',
        
        # DevOps & Containers
        'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'GitHub Actions', 'CircleCI', 'ArgoCD',
        
        # Version Control & Collaboration
        'Git', 'GitHub', 'GitLab', 'Bitbucket', 'SVN',
        
        # Web Frameworks
        'Flask', 'Django', 'FastAPI', 'Spring', 'Node.js', 'React', 'Angular', 'Vue.js',
        'Express', 'Streamlit', 'Gradio',
        
        # ETL Tools
        'Talend', 'Informatica', 'SSIS', 'Apache NiFi', 'Pentaho', 'dbt',
        
        # Other Technical Skills
        'REST API', 'GraphQL', 'Microservices', 'CI/CD', 'Agile', 'Scrum', 'JIRA',
        'Linux', 'Unix', 'Bash', 'Shell Scripting', 'PowerShell'
    ]
    
    found_skills = set()
    text_lower = text.lower()
    
    # Match each skill (case-insensitive with word boundaries)
    for skill in skill_keywords:
        # Use word boundary for exact matching
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    return sorted(list(found_skills))

def extract_education_regex(text):
    """Extract education requirements from JD using regex patterns"""
    education = set()
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Pattern 1: Degree names with variations
    degree_patterns = [
        r'\b(bachelor\'?s?|bachelors?|b\.?e\.?|b\.?tech\.?|b\.?s\.?c?\.?|undergraduate)\s*(?:degree)?\s*(?:in)?\s*([a-z\s]+)?',
        r'\b(master\'?s?|masters?|m\.?e\.?|m\.?tech\.?|m\.?s\.?c?\.?|graduate)\s*(?:degree)?\s*(?:in)?\s*([a-z\s]+)?',
        r'\b(phd|ph\.?d\.?|doctorate|doctoral)\s*(?:degree)?\s*(?:in)?\s*([a-z\s]+)?',
        r'\b(mba|m\.?b\.?a\.?)\b',
    ]
    
    for pattern in degree_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            degree = match.group(0).strip()
            # Clean up and capitalize properly
            degree_clean = ' '.join(word.capitalize() for word in degree.split())
            education.add(degree_clean)
    
    # Pattern 2: Common fields of study
    field_patterns = [
        r'(?:degree|bachelor|master|phd)\s+(?:in|of)\s+(computer science|cs|information technology|it|engineering|statistics|mathematics|data science|ai|artificial intelligence)',
    ]
    
    for pattern in field_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            education.add(match.group(0).capitalize())
    
    return sorted(list(education))


if __name__ == "__main__":
    print("=== JD EXTRACTION TESTER ===")
    print("Paste your JD text below (press Ctrl+Z then Enter on Windows, or Ctrl+D on Mac/Linux when done):")
    print("-" * 50)
    
    jd_text = sys.stdin.read()
    
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)
    
    # Extract YOE
    yoe_result = extract_yoe_from_jd(jd_text)
    print(f"\n1. Years of Experience: {yoe_result['min_yoe']}")
    if yoe_result['max_yoe'] != "Not Found":
        print(f"   Range: {yoe_result['min_yoe']}-{yoe_result['max_yoe']}")
    
    # Extract Skills & Education (NER)
    skills_ner, edu_ner = extract_skills_education_ner(jd_text)
    print(f"\n2. Skills (NER): {skills_ner}")
    print(f"\n3. Education (NER): {edu_ner}")
    
    # Extract Education (Regex)
    edu_regex = extract_education_regex(jd_text)
    print(f"\n4. Education (Regex): {edu_regex}")