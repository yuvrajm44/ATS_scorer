# router.py
# FastAPI routes for ATS Resume Scoring System

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tempfile
import os
from pathlib import Path
import time


from parser import DocumentParser
from scorer import score_resume_for_jd

# Create router
router = APIRouter(prefix="/api", tags=["ATS Scoring"])

# Initialize parser (OCR off by default for speed)
parser = DocumentParser(use_ocr=False, use_table_structure=True)


@router.post("/score-single")
async def score_single_resume(
    jd_file: UploadFile = File(..., description="Job Description file (PDF/DOCX)"),
    resume_file: UploadFile = File(..., description="Resume file (PDF/DOCX)")
):
    """
    Score a single resume against a job description
    
    Args:
        jd_file: Job description file
        resume_file: Resume file
        
    Returns:
        JSON with ATS score and detailed breakdown
    """
    
    start_time = time.time()
    
    try:
        # Validate file formats
        allowed_formats = ['.pdf', '.docx', '.doc']
        jd_ext = Path(jd_file.filename).suffix.lower()
        resume_ext = Path(resume_file.filename).suffix.lower()

        
        
        if jd_ext not in allowed_formats:
            raise HTTPException(status_code=400, detail=f"JD format not supported: {jd_ext}")
        if resume_ext not in allowed_formats:
            raise HTTPException(status_code=400, detail=f"Resume format not supported: {resume_ext}")
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save JD file
            jd_path = os.path.join(temp_dir, jd_file.filename)
            with open(jd_path, "wb") as f:
                f.write(await jd_file.read())
            
            # Save Resume file
            resume_path = os.path.join(temp_dir, resume_file.filename)
            with open(resume_path, "wb") as f:
                f.write(await resume_file.read())
            
            # Parse documents using Docling
            jd_result = parser.parse_job_description(jd_path)
            resume_result = parser.parse_resume(resume_path)
            
            # Check parsing success
            if not jd_result['success']:
                raise HTTPException(status_code=500, detail=f"JD parsing failed: {jd_result['error']}")
            if not resume_result['success']:
                raise HTTPException(status_code=500, detail=f"Resume parsing failed: {resume_result['error']}")
            
            # Get extracted text
            jd_text = jd_result['text']
            resume_text = resume_result['text']

            # Debug: Print full parsed text
            print("\n" + "="*70)
            print("PARSED JD TEXT:")
            print("="*70)
            print(jd_text)
            print("\n" + "="*70)
            print("PARSED RESUME TEXT:")
            print("="*70)
            print(resume_text)
            print("="*70 + "\n")
            
            # Calculate ATS score
            score_result = score_resume_for_jd(jd_text, resume_text)
            
            # Calculate processing time
            processing_time = round(time.time() - start_time, 2)
            
            # Build response
            response = {
                "success": True,
                "jd_filename": jd_file.filename,
                "resume_filename": resume_file.filename,
                "ats_score": score_result['final_score'],
                "decision": score_result['decision'],
                "status": score_result['status'],
                "breakdown": {
                    "skills_score": score_result['breakdown']['skills_score'],
                    "experience_score": score_result['breakdown']['yoe_score'],
                    "semantic_score": score_result['breakdown']['semantic_score'],
                    "education_score": score_result['breakdown']['education_score']
                },
                "details": {
                    "skills": {
                        "match_percentage": score_result['skill_match_pct'],
                        "matched_skills": score_result['matched_skills'],
                        "missing_skills": score_result['missing_skills']
                    },
                    "experience": {
                        "required_years": score_result['jd_yoe'],
                        "candidate_years": score_result['resume_yoe']
                    },
                    "education": {
                        "required": score_result['jd_education'],
                        "candidate": score_result['resume_education'],
                        "score": score_result['edu_score'],
                        "explanation": score_result['edu_explanation']
                    },
                    "semantic_similarity": score_result['semantic_similarity']
                },
                "processing_time_seconds": processing_time
            }
            
            return JSONResponse(content=response)
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.post("/score-batch")
async def score_batch_resumes(
    jd_file: UploadFile = File(..., description="Job Description file (PDF/DOCX)"),
    resume_files: List[UploadFile] = File(..., description="Multiple resume files (PDF/DOCX)")
):
    """
    Score multiple resumes against a single job description
    
    Args:
        jd_file: Job description file
        resume_files: List of resume files (max 50)
        
    Returns:
        JSON with array of ATS scores for each resume
    """
    
    start_time = time.time()
    
    try:
        # Validate number of resumes
        if len(resume_files) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 resumes allowed per batch")
        
        if len(resume_files) == 0:
            raise HTTPException(status_code=400, detail="At least one resume file required")
        
        # Validate JD format
        allowed_formats = ['.pdf', '.docx', '.doc']
        jd_ext = Path(jd_file.filename).suffix.lower()
        if jd_ext not in allowed_formats:
            raise HTTPException(status_code=400, detail=f"JD format not supported: {jd_ext}")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save and parse JD file
            jd_path = os.path.join(temp_dir, jd_file.filename)
            with open(jd_path, "wb") as f:
                f.write(await jd_file.read())
            
            jd_result = parser.parse_job_description(jd_path)
            
            if not jd_result['success']:
                raise HTTPException(status_code=500, detail=f"JD parsing failed: {jd_result['error']}")
            
            jd_text = jd_result['text']

            print("\n" + "="*70)
            print("PARSED JD TEXT (FULL):")
            print("="*70)
            print(jd_text)
            print(f"\n[Length: {len(jd_text)} characters]")
            print("="*70 + "\n")
            
            # Process all resumes
            results = []
            successful_count = 0
            failed_count = 0
            
            for i, resume_file in enumerate(resume_files, 1):
                try:
                    # Validate resume format
                    resume_ext = Path(resume_file.filename).suffix.lower()
                    if resume_ext not in allowed_formats:
                        results.append({
                            "resume_filename": resume_file.filename,
                            "success": False,
                            "error": f"Unsupported format: {resume_ext}"
                        })
                        failed_count += 1
                        continue
                    
                    # Save resume file
                    resume_path = os.path.join(temp_dir, f"resume_{i}_{resume_file.filename}")
                    with open(resume_path, "wb") as f:
                        f.write(await resume_file.read())
                    
                    # Parse resume
                    resume_result = parser.parse_resume(resume_path)
                    
                    if not resume_result['success']:
                        results.append({
                            "resume_filename": resume_file.filename,
                            "success": False,
                            "error": f"Parsing failed: {resume_result['error']}"
                        })
                        failed_count += 1
                        continue
                    
                    resume_text = resume_result['text']

                    print("\n" + "="*70)
                    print(f"PARSED RESUME TEXT (FULL): {resume_file.filename}")
                    print("="*70)
                    print(resume_text)
                    print(f"\n[Length: {len(resume_text)} characters]")
                    print("="*70 + "\n")
                    
                    # Calculate ATS score
                    score_result = score_resume_for_jd(jd_text, resume_text)
                    
                    # Add to results
                    results.append({
                        "resume_filename": resume_file.filename,
                        "success": True,
                        "ats_score": score_result['final_score'],
                        "decision": score_result['decision'],
                        "status": score_result['status'],
                        "breakdown": {
                            "skills_score": score_result['breakdown']['skills_score'],
                            "experience_score": score_result['breakdown']['yoe_score'],
                            "semantic_score": score_result['breakdown']['semantic_score'],
                            "education_score": score_result['breakdown']['education_score']
                        },
                        "summary": {
                            "skill_match_pct": score_result['skill_match_pct'],
                            "required_experience": score_result['jd_yoe'],
                            "candidate_experience": score_result['resume_yoe'],
                            "education_score": score_result['edu_score']
                        }
                    })
                    successful_count += 1
                
                except Exception as e:
                    results.append({
                        "resume_filename": resume_file.filename,
                        "success": False,
                        "error": str(e)
                    })
                    failed_count += 1
            
            # Sort results by ATS score (highest first)
            successful_results = [r for r in results if r['success']]
            failed_results = [r for r in results if not r['success']]
            
            successful_results.sort(key=lambda x: x['ats_score'], reverse=True)
            
            # Combine: successful first (sorted), then failed
            sorted_results = successful_results + failed_results
            
            # Calculate processing time
            processing_time = round(time.time() - start_time, 2)
            
            # Build response
            response = {
                "success": True,
                "jd_filename": jd_file.filename,
                "total_resumes": len(resume_files),
                "successful": successful_count,
                "failed": failed_count,
                "processing_time_seconds": processing_time,
                "results": sorted_results
            }
            
            return JSONResponse(content=response)
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "ATS Resume Scoring API",
        "version": "1.0.0"
    }