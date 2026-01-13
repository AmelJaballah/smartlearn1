"""
========================================================================
EXERCISE GENERATOR - FastAPI Server v2.0
========================================================================
Simple, powerful API for generating math exercises.

Endpoints:
- GET  /health          - Health check
- GET  /subjects        - List available subjects  
- POST /generate        - Generate exercises (1-10 at a time)
- POST /check-answer    - Verify student answer

Author: SmartLearn Team
========================================================================
"""

import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import core functions from RAG module
from exercice_generator_rag import (
    generate_exercise,
    verify_student_answer,
    collection,
    device,
)

# Constants
OLLAMA_MODEL = "llama3.1"

# Default subjects (since collection has millions of records)
DEFAULT_SUBJECTS = [
    "Algebra", "Calculus", "Trigonometry", "Geometry",
    "Statistics", "Probability", "Linear Algebra",
    "Differential Equations", "Number Theory", "Mathematics",
    "Quadratic Equations", "Polynomials", "Logarithms",
    "Integrals", "Derivatives", "Limits", "Vectors"
]


# CONFIGURATION

PORT = int(os.getenv("EXERCISE_PORT", 5001))
HOST = "0.0.0.0"


# FASTAPI APP

app = FastAPI(
    title="📚 Exercise Generator API",
    description="Generate math exercises using RAG + LLM",
    version="2.0.0"
)

# CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST/RESPONSE MODELS (Clean & Simple)

class GenerateRequest(BaseModel):
    """Request to generate exercises"""
    subject: str = Field(..., description="Topic (e.g., algebra, calculus)")
    difficulty: str = Field("medium", description="Level: easy, medium, hard")
    count: int = Field(1, ge=1, le=10, description="Number of exercises (1-10)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject": "quadratic equations",
                "difficulty": "medium",
                "count": 3
            }
        }


class ExerciseItem(BaseModel):
    """Single exercise in response"""
    id: int
    question: str
    solution: str
    answer: str
    difficulty: str
    subject: str
    hint: str


class GenerateResponse(BaseModel):
    """Response with generated exercises"""
    success: bool
    exercises: List[ExerciseItem]
    total: int
    subject: str
    difficulty: str


class CheckAnswerRequest(BaseModel):
    """Request to check student answer"""
    expected: str = Field(..., description="Correct answer")
    student: str = Field(..., description="Student's answer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "expected": "x = 5",
                "student": "5"
            }
        }


class CheckAnswerResponse(BaseModel):
    """Response for answer check"""
    correct: bool
    expected: str
    student: str


class SubjectsResponse(BaseModel):
    """Response with available subjects"""
    subjects: List[str]
    total: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    device: str
    model: str
    collection: str


# ENDPOINTS

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the service is running.
    Returns device info and model configuration.
    """
    return HealthResponse(
        status="healthy",
        device=device,
        model=OLLAMA_MODEL,
        collection="exercise_bank"
    )


@app.get("/subjects", response_model=SubjectsResponse)
async def list_subjects():
    """
    Get list of available subjects from the exercise database.
    """
    return SubjectsResponse(
        subjects=DEFAULT_SUBJECTS,
        total=len(DEFAULT_SUBJECTS)
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """
    Generate one or more exercises.
    
    - **subject**: Topic to generate exercises for (e.g., "algebra")
    - **difficulty**: Level of difficulty (easy, medium, hard)
    - **count**: Number of exercises to generate (1-10)
    
    Example:
    ```json
    {
        "subject": "quadratic equations",
        "difficulty": "medium",
        "count": 3
    }
    ```
    """
    try:
        # Validate difficulty
        valid_difficulties = ["easy", "medium", "hard"]
        if request.difficulty.lower() not in valid_difficulties:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid difficulty. Choose from: {valid_difficulties}"
            )
        
        # Generate exercises one by one
        exercises = []
        for i in range(request.count):
            try:
                ex = generate_exercise(request.subject, request.difficulty.lower())
                exercises.append(ExerciseItem(
                    id=i + 1,
                    question=ex.question,
                    solution=ex.solution,
                    answer=ex.final_answer,
                    difficulty=request.difficulty,
                    subject=request.subject,
                    hint=ex.hint
                ))
            except Exception as e:
                print(f"⚠️ Failed to generate exercise {i+1}: {e}")
                continue
        
        if not exercises:
            raise HTTPException(status_code=500, detail="Failed to generate any exercises")
        
        return GenerateResponse(
            success=True,
            exercises=exercises,
            total=len(exercises),
            subject=request.subject,
            difficulty=request.difficulty
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating exercises: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-answer", response_model=CheckAnswerResponse)
def check_answer(request: CheckAnswerRequest):
    """
    Check if student's answer is correct.
    Uses mathematical comparison for expressions.
    
    Example:
    ```json
    {
        "expected": "x = 5",
        "student": "5"
    }
    ```
    """
    try:
        is_correct = verify_student_answer(request.expected, request.student)
        
        return CheckAnswerResponse(
            correct=is_correct,
            expected=request.expected,
            student=request.student
        )
    except Exception:
        # Fallback to string comparison
        is_correct = request.expected.strip().lower() == request.student.strip().lower()
        return CheckAnswerResponse(
            correct=is_correct,
            expected=request.expected,
            student=request.student
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# LEGACY ENDPOINTS (for backward compatibility)

class LegacyExerciseRequest(BaseModel):
    """Legacy request format"""
    subject: str
    difficulty: Optional[str] = "medium"
    exerciseType: Optional[str] = "multiple-choice"
    additionalContext: Optional[str] = ""


@app.post("/generate-exercise")
async def generate_exercise_legacy(request: LegacyExerciseRequest):
    """
    Legacy endpoint - generates a single exercise.
    Use POST /generate instead for new code.
    """
    try:
        ex = generate_exercise(
            subject=request.subject,
            difficulty=request.difficulty.lower() if request.difficulty else "medium"
        )
        
        return {
            "success": True,
            "exercise": {
                "question": ex.question,
                "hint": f"Think about {request.subject} concepts",
                "answer": ex.final_answer,
                "explanation": ex.solution,
                "difficulty": request.difficulty,
                "subject": request.subject,
                "type": request.exerciseType or "problem-solving",
                "retrievedSources": 0
            },
            "retrievedDocs": 0
        }
        
    except Exception as e:
        print(f"❌ Error in legacy endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# MAIN

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Exercise Generator API v2.0")
    print("=" * 60)
    print(f"📍 URL:        http://localhost:{PORT}")
    print(f"📚 Docs:       http://localhost:{PORT}/docs")
    print(f"🔧 Device:     {device}")
    print(f"🤖 Model:      {OLLAMA_MODEL}")
    print(f"💾 Collection: exercise_bank")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health         - Health check")
    print("  GET  /subjects       - List subjects")
    print("  POST /generate       - Generate exercises (1-10)")
    print("  POST /check-answer   - Verify student answer")
    print("  POST /generate-exercise - Legacy (single exercise)")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
