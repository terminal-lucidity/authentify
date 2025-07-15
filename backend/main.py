from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict
import google.generativeai as genai
from bs4 import BeautifulSoup, Tag
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import re
from urllib.parse import urlparse
import base64
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import time
from browser_use import Agent

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-pro-vision')

# Initialize FastAPI app
app = FastAPI(
    title="Product Authenticity API",
    description="API for verifying product authenticity using web scraping and AI analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting settings
MAX_REQUESTS_PER_MINUTE = 60
REQUEST_WINDOW = 60  # seconds

class RateLimiter:
    def __init__(self, max_requests: int, window: int):
        self.max_requests = max_requests
        self.window = window
        self.requests = []

    async def check_rate_limit(self):
        now = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window]
        
        if len(self.requests) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window} seconds."
            )
        
        self.requests.append(now)

rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, REQUEST_WINDOW)

class ScrapingPatterns:
    """Enhanced scraping patterns for different e-commerce sites."""
    
    AMAZON = {
        'name': ['#productTitle', '#title', '.product-title-word-break'],
        'description': ['#productDescription', '#feature-bullets', '.a-spacing-mini', '#dpx-product-description'],
        'price': ['#priceblock_ourprice', '#priceblock_dealprice', '.a-price', '#price_inside_buybox'],
        'reviews': ['.review-text', '.review-content', '.review-text-content'],
        'seller': ['#merchant-info', '#sellerProfileTriggerId', '#bylineInfo'],
        'features': ['#feature-bullets', '.a-spacing-mini', '#productDetails_feature_div']
    }
    
    EBAY = {
        'name': ['h1.x-item-title__mainTitle', '.it-ttl', '.product-title'],
        'description': ['#ds_div', '.item-desc', '.product-description'],
        'price': ['.x-price-primary', '.vi-price', '.display-price'],
        'reviews': ['.ebay-review-section', '.review-item', '.reviews-stream'],
        'seller': ['.mbg-nw', '.si-inner', '.user-info'],
        'features': ['.ux-labels-values', '.itemAttr', '.product-specs']
    }
    
    WALMART = {
        'name': ['.prod-ProductTitle', '.heading-5', '.w-product-title'],
        'description': ['.about-desc', '.product-description-content', '.product-detail-content'],
        'price': ['.price-characteristic', '.price-group', '.product-price'],
        'reviews': ['.review-text', '.review-content', '.customer-review'],
        'seller': ['.seller-name', '.seller-info', '.marketplace-seller'],
        'features': ['.about-desc', '.specifications', '.product-specifications']
    }
    
    ALIEXPRESS = {
        'name': ['.product-title', '.product-name', '.title'],
        'description': ['.product-description', '.detail-desc', '.description'],
        'price': ['.product-price-value', '.uniform-banner-box-price', '.price'],
        'reviews': ['.feedback-item', '.review-item', '.customer-feedback'],
        'seller': ['.shop-name', '.store-name', '.seller-info'],
        'features': ['.product-property', '.product-specs', '.specifications']
    }
    
    BESTBUY = {
        'name': ['.heading-5', '.sku-title', '.product-title'],
        'description': ['.product-description', '.long-description', '.description-text'],
        'price': ['.priceView-customer-price', '.price-box', '.current-price'],
        'reviews': ['.review-text', '.ugc-review', '.user-review'],
        'seller': ['.marketplace-seller', '.seller-info', '.vendor-name'],
        'features': ['.feature-list', '.specifications', '.product-data']
    }
    
    TARGET = {
        'name': ['.h-text-bold', '.product-name', '.title'],
        'description': ['.h-text-md', '.product-description', '.description'],
        'price': ['.style-price', '.price-box', '.current-price'],
        'reviews': ['.h-text-sm', '.review-content', '.user-review'],
        'seller': ['.seller-info', '.vendor-name', '.sold-by'],
        'features': ['.product-features', '.specifications', '.details']
    }
    
    NEWEGG = {
        'name': ['.product-title', '.item-title', '.title'],
        'description': ['.product-description', '.item-info', '.description'],
        'price': ['.price-current', '.product-price', '.price'],
        'reviews': ['.comments-content', '.review-content', '.feedback'],
        'seller': ['.seller-name', '.product-seller', '.sold-by'],
        'features': ['.product-specs', '.features', '.specifications']
    }

class ProductURL(BaseModel):
    url: HttpUrl
    name: Optional[str] = Field(None, description="Optional product name override")
    description: Optional[str] = Field(None, description="Optional description override")

class ProductAnalysisResponse(BaseModel):
    analysis: Dict
    product_data: Dict

class ScrapeError(Exception):
    """Custom exception for scraping errors."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

def get_site_patterns(url: str) -> Optional[dict]:
    """Get scraping patterns based on the URL domain."""
    domain = urlparse(url).netloc.lower()
    
    patterns = {
        'amazon': ScrapingPatterns.AMAZON,
        'ebay': ScrapingPatterns.EBAY,
        'walmart': ScrapingPatterns.WALMART,
        'aliexpress': ScrapingPatterns.ALIEXPRESS,
        'bestbuy': ScrapingPatterns.BESTBUY,
        'target': ScrapingPatterns.TARGET,
        'newegg': ScrapingPatterns.NEWEGG
    }
    
    for site, pattern in patterns.items():
        if site in domain:
            return pattern
    
    # Return Amazon patterns as fallback for unknown sites
    return ScrapingPatterns.AMAZON

@lru_cache(maxsize=100)
def get_cached_response(url: str) -> str:
    """Cache HTTP responses to avoid repeated requests."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

async def analyze_multiple_images(image_urls: List[str]) -> List[Dict]:
    """Analyze multiple images concurrently."""
    async def analyze_single_image(url: str) -> Dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return {"url": url, "analysis": f"Failed to fetch image: HTTP {response.status}"}
                    
                    image_data = await response.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    prompt = f"""
                    Analyze this product image (base64: {image_base64}) for authenticity indicators. Consider:
                    1. Image quality and resolution
                    2. Product details and craftsmanship
                    3. Branding and logo accuracy
                    4. Packaging quality (if visible)
                    5. Any visible red flags for counterfeits
                    
                    Provide a brief analysis focusing on authenticity indicators.
                    """
                    
                    response = model.generate_content(prompt)
                    return {"url": url, "analysis": response.text}
        except Exception as e:
            return {"url": url, "analysis": f"Failed to analyze image: {str(e)}"}
    
    tasks = [analyze_single_image(url) for url in image_urls]
    return await asyncio.gather(*tasks)

def clean_text(text: str) -> str:
    """Enhanced text cleaning."""
    if not text:
        return ""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove multiple punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    return text.strip()

def extract_price(text: str) -> Optional[float]:
    """Enhanced price extraction."""
    if not text:
        return None
    # Look for price patterns
    price_patterns = [
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $XX.XX or $X,XXX.XX
        r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars)',  # XX.XX USD
        r'(?:USD|dollars)\s*\d+(?:,\d{3})*(?:\.\d{2})?'  # USD XX.XX
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            price_str = re.sub(r'[^\d.]', '', match.group())
            try:
                return float(price_str)
            except ValueError:
                continue
    return None

def validate_product_data(data: dict) -> dict:
    """Enhanced validation with detailed rules."""
    # Required fields validation
    required_fields = {
        'name': "Product name is required",
        'description': "Product description is required",
        'price': "Price information is required for authenticity assessment"
    }
    
    missing_fields = []
    for field, message in required_fields.items():
        if not data.get(field):
            missing_fields.append({"field": field, "message": message})
    
    if missing_fields:
        raise ScrapeError(
            "Missing required product information",
            {"missing_fields": missing_fields}
        )
    
    # Content quality validation
    quality_rules = {
        'name': {
            'min_length': 3,
            'max_length': 200,
            'pattern': r'^[\w\s\-.,&\'()]+$'
        },
        'description': {
            'min_length': 20,
            'max_words': 1000
        },
        'price': {
            'min_value': 0.01,
            'max_value': 1000000
        }
    }
    
    validation_errors = []
    
    # Name validation
    name = data['name']
    if len(name) < quality_rules['name']['min_length']:
        validation_errors.append({
            'field': 'name',
            'error': f"Product name too short (minimum {quality_rules['name']['min_length']} characters)"
        })
    elif len(name) > quality_rules['name']['max_length']:
        validation_errors.append({
            'field': 'name',
            'error': f"Product name too long (maximum {quality_rules['name']['max_length']} characters)"
        })
    elif not re.match(quality_rules['name']['pattern'], name):
        validation_errors.append({
            'field': 'name',
            'error': "Product name contains invalid characters"
        })
    
    # Description validation
    description = data['description']
    if len(description) < quality_rules['description']['min_length']:
        validation_errors.append({
            'field': 'description',
            'error': f"Description too short (minimum {quality_rules['description']['min_length']} characters)"
        })
    word_count = len(description.split())
    if word_count > quality_rules['description']['max_words']:
        validation_errors.append({
            'field': 'description',
            'error': f"Description too long (maximum {quality_rules['description']['max_words']} words)"
        })
    
    # Price validation
    if data['price']:
        try:
            price = float(data['price'])
            if price < quality_rules['price']['min_value']:
                validation_errors.append({
                    'field': 'price',
                    'error': f"Price too low (minimum ${quality_rules['price']['min_value']:.2f})"
                })
            elif price > quality_rules['price']['max_value']:
                validation_errors.append({
                    'field': 'price',
                    'error': f"Price too high (maximum ${quality_rules['price']['max_value']:.2f})"
                })
        except (ValueError, TypeError):
            validation_errors.append({
                'field': 'price',
                'error': "Invalid price format"
            })
    
    # Image validation
    if not data['images']:
        validation_errors.append({
            'field': 'images',
            'error': "No product images found"
        })
    
    # Review validation
    if data['reviews']:
        suspicious_patterns = [
            r'\b(100%|perfect|best|amazing)\b',
            r'(?i)(verified|genuine|authentic|real|original)',
            r'(?i)(free|discount|coupon|promo)',
            r'(?i)(website|link|click|order|buy)',
            r'\b[A-Z]{2,}\b'  # Excessive caps
        ]
        
        suspicious_reviews = []
        for review in data['reviews']:
            for pattern in suspicious_patterns:
                if re.search(pattern, review, re.IGNORECASE):
                    suspicious_reviews.append(review)
                    break
        
        if suspicious_reviews:
            data['suspicious_reviews'] = suspicious_reviews
    
    if validation_errors:
        raise ScrapeError(
            "Product data validation failed",
            {"validation_errors": validation_errors}
        )
    
    return data

async def scrape_product_data(url: str) -> dict:
    try:
        import os
        # Use Gemini (ChatGoogle) if GOOGLE_API_KEY is set, else fallback to OpenAI
        if os.getenv('GOOGLE_API_KEY'):
            print('[DEBUG] Using Gemini (ChatGoogle) LLM for browser_use agent')
            try:
                from browser_use.llm import ChatGoogle
            except ImportError:
                raise ScrapeError('ChatGoogle is not available. Please ensure browser_use supports Gemini and is up to date.', {'url': url})
            llm = ChatGoogle(model='gemini-2.0-flash-exp')
        else:
            print('[DEBUG] GOOGLE_API_KEY not set, falling back to OpenAI')
            from browser_use.llm import ChatOpenAI
            llm = ChatOpenAI(model='gpt-4o')
        task = (
            f"Go to this product page: {url}\n"
            "Extract the following as a JSON object with these exact keys:\n"
            "{\n"
            "  \"name\": string,\n"
            "  \"price\": number or string,\n"
            "  \"images\": [list of image URLs],\n"
            "  \"description\": string,\n"
            "  \"reviews\": [list of up to 5 customer review texts],\n"
            "  \"seller\": string,\n"
            "  \"features\": [list of product features or bullet points]\n"
            "}\n"
            "If any field is missing, set it to null or an empty list as appropriate."
        )
        agent = Agent(
            task=task,
            llm=llm,
            use_vision=True
        )
        result = await agent.run()
        extracted = result.final_result()
        print(f"[DEBUG] browser_use agent output for {url}: {extracted}")
        import json
        if isinstance(extracted, str):
            try:
                extracted = json.loads(extracted)
            except Exception as e:
                print(f"[DEBUG] Failed to parse agent output as JSON: {e}")
                raise ScrapeError(f"Agent did not return valid JSON: {extracted}", {"url": url})
        if not isinstance(extracted, dict):
            raise ScrapeError(f"Agent did not return a dict: {extracted}", {"url": url})
        for key in ['name', 'price', 'images', 'description', 'reviews', 'seller', 'features']:
            if key not in extracted:
                extracted[key] = None if key in ['name', 'price', 'description', 'seller'] else []
        extracted['image_analysis'] = []
        # Loosen validation: require only name and description
        if not extracted.get('name') or not extracted.get('description'):
            raise ScrapeError("Missing required product name or description", {"url": url, "output": extracted})
        return extracted
    except Exception as e:
        raise ScrapeError(f"browser_use agent error: {str(e)}", {"url": url})

def analyze_product_authenticity(product_data: dict) -> dict:
    """Enhanced product authenticity analysis with more detailed prompts."""
    try:
        # Prepare a more comprehensive prompt
        prompt = f"""
        Analyze this product listing for authenticity. Consider all available data points to determine if this appears to be a genuine product or potentially counterfeit.

        PRODUCT INFORMATION:
        Name: {product_data['name']}
        Price: ${product_data['price'] if product_data['price'] else 'N/A'}
        Seller: {product_data['seller'] if product_data['seller'] else 'Unknown'}
        
        DETAILED DESCRIPTION:
        {product_data['description']}
        
        PRODUCT FEATURES:
        {json.dumps(product_data['features'], indent=2)}
        
        CUSTOMER REVIEWS:
        {json.dumps(product_data['reviews'], indent=2)}
        
        SUSPICIOUS REVIEW PATTERNS:
        {json.dumps(product_data.get('suspicious_reviews', []), indent=2)}
        
        IMAGE ANALYSIS RESULTS:
        {json.dumps([analysis['analysis'] for analysis in product_data['image_analysis']], indent=2)}

        Please provide a comprehensive authenticity analysis with the following sections:

        1. AUTHENTICITY VERDICT:
           - Overall assessment (Likely Authentic / Suspicious / Likely Counterfeit)
           - Confidence level (percentage)
           - Key factors influencing the verdict
        
        2. PRICE ANALYSIS:
           - Price point evaluation
           - Comparison with market expectations
           - Suspicious pricing patterns (if any)
        
        3. SELLER ASSESSMENT:
           - Seller credibility indicators
           - Red flags in seller information
           - Marketplace reputation (if available)
        
        4. PRODUCT DESCRIPTION EVALUATION:
           - Quality and accuracy of description
           - Technical specifications accuracy
           - Brand consistency
           - Language and formatting professionalism
        
        5. REVIEW ANALYSIS:
           - Review authenticity patterns
           - Common themes in reviews
           - Suspicious review indicators
           - Review consistency with product claims
        
        6. IMAGE VERIFICATION:
           - Image quality assessment
           - Brand/logo accuracy
           - Packaging authenticity
           - Product detail consistency
           - Common counterfeit indicators
        
        7. RED FLAGS:
           Major Concerns:
           - List critical authenticity issues
           - Severity rating for each issue
           
           Minor Concerns:
           - List potential minor issues
           - Impact on authenticity assessment
        
        8. POSITIVE INDICATORS:
           - List factors supporting authenticity
           - Strength of each positive indicator
        
        9. BUYER RECOMMENDATIONS:
           Verification Steps:
           - Specific actions to verify authenticity
           - Key details to check upon receipt
           
           Precautions:
           - Payment and purchase protection
           - Return policy verification
           - Additional security measures
        
        10. CONFIDENCE ASSESSMENT:
            - Explain confidence level
            - List factors affecting confidence
            - Note any limitations in analysis
        """
        
        response = model.generate_content(prompt)
        
        # Save verification data with enhanced metadata
        df = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'product_name': product_data['name'],
            'url': product_data.get('url', ''),
            'analysis': response.text,
            'price': product_data.get('price', ''),
            'seller': product_data.get('seller', ''),
            'image_count': len(product_data['images']),
            'review_count': len(product_data['reviews']),
            'suspicious_reviews': len(product_data.get('suspicious_reviews', [])),
            'feature_count': len(product_data['features'])
        }])
        
        df.to_csv('product_verifications.csv', mode='a', header=False, index=False)
        
        # Parse and structure the response
        try:
            # Extract key information using regex patterns
            verdict_pattern = r"Overall assessment: (Likely Authentic|Suspicious|Likely Counterfeit)"
            confidence_pattern = r"Confidence level: (\d+)%"
            
            verdict_match = re.search(verdict_pattern, response.text)
            confidence_match = re.search(confidence_pattern, response.text)
            
            structured_response = {
                'verdict': verdict_match.group(1) if verdict_match else "Unknown",
                'confidence': int(confidence_match.group(1)) if confidence_match else 0,
                'full_analysis': response.text,
                'product_data': {
                    'name': product_data['name'],
                    'price': product_data['price'],
                    'seller': product_data['seller'],
                    'images': product_data['images'],
                    'features': product_data['features'][:5],
                    'reviews': product_data['reviews'][:3],
                    'suspicious_reviews': product_data.get('suspicious_reviews', [])
                }
            }
            
            return structured_response
            
        except Exception as e:
            # Fallback to raw response if parsing fails
            return {
                'analysis': response.text,
                'product_data': product_data
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze product: {str(e)}")

@app.post("/scrape_product", response_model=ProductAnalysisResponse)
async def scrape_product(product: ProductURL):
    """Enhanced endpoint with validation and error handling."""
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
        
        return result
        
    except ScrapeError as e:
        raise HTTPException(
            status_code=400,
            detail={"message": e.message, "details": e.details}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 