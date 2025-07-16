from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from dotenv import load_dotenv
import os
import csv
import asyncio
import google.generativeai as genai
from typing import Optional, List, Dict
from fastapi import HTTPException

# browser_use imports
from browser_use.llm import ChatGoogle
from browser_use import Agent
from browser_use.llm.messages import UserMessage

app = FastAPI()

# Allow CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class ScrapeRequest(BaseModel):
    url: str

class ProductData(BaseModel):
    images: list[str] = Field(default_factory=list)
    description: str = ""
    features: list[str] = Field(default_factory=list)
    reviews: list[str] = Field(default_factory=list)

class AuthenticityResult(BaseModel):
    images: dict = Field(default_factory=dict)
    description: dict = Field(default_factory=dict)
    features: dict = Field(default_factory=dict)
    reviews: dict = Field(default_factory=dict)

class ProductURL(BaseModel):
    url: HttpUrl
    name: Optional[str] = Field(None, description="Optional product name override")
    description: Optional[str] = Field(None, description="Optional description override")

class ProductAnalysisResponse(BaseModel):
    analysis: Dict
    product_data: Dict

class ScrapeError(Exception):
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

# Copy or import scrape_product_data and analyze_product_authenticity from main.py
# For now, import them if possible:
from backend.main import scrape_product_data, analyze_product_authenticity

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "healthy"})

@app.post("/scrape_product", response_model=ProductAnalysisResponse)
async def scrape_product(product: ProductURL):
    try:
        # Scrape product data
        product_data = await scrape_product_data(str(product.url))
        
        # Override with provided data if any
        if product.name:
            product_data['name'] = product.name
        if product.description:
            product_data['description'] = product.description
            
        # Add URL to product data
        product_data['url'] = str(product.url)
        
        # Analyze authenticity
        result = analyze_product_authenticity(product_data)
        
        # --- FIX: Wrap result in 'analysis' key as required by response_model ---
        return {
            "analysis": {
                "verdict": result.get("verdict"),
                "confidence": result.get("confidence"),
                "full_analysis": result.get("full_analysis"),
            },
            "product_data": result.get("product_data", {})
        }
        
    except ScrapeError as e:
        raise HTTPException(
            status_code=400,
            detail={"message": e.message, "details": e.details}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_gemini_models")
def list_gemini_models():
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        models = genai.list_models()
        model_names = [m.name for m in models]
        return JSONResponse(content={"models": model_names})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)