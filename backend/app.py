from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import csv
import asyncio

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

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "healthy"})

@app.post("/scrape_product")
async def scrape_product(request: ScrapeRequest):
    url = request.url
    print(f"Received request to scrape: {url}")

    # Step 1: Initialize LLM (Gemini 2.0 for scraping)
    print("Initializing ChatGoogle (Gemini Pro) model...")
    # Standard Gemini model initialization for backend
    llm = ChatGoogle(model='gemini-2.5-pro', api_key=GOOGLE_API_KEY)

    # Step 2: Create Agent with scraping task
    task = f"Scrape the product page at {url} and extract: product images (urls or files), product description, all available features, and consumer reviews. Return as a JSON object with keys: images (list of urls/paths), description (string), features (list of strings), reviews (list of strings)."
    print("Creating Agent for scraping...")
    agent = Agent(task=task, llm=llm)

    # Step 3: Run the agent to scrape the product page
    print("Running agent to scrape product page...")
    try:
        result = await agent.run(max_steps=10)
    except Exception as e:
        print(f"Error during scraping: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # Step 4: Extract relevant info from agent result as JSON
    print("Extracting scraped data from agent result...")
    scraped = None
    for h in result.history:
        if h.model_output and hasattr(h.model_output, 'memory'):
            try:
                # Try to parse as JSON
                import json
                mem = h.model_output.memory
                if isinstance(mem, str):
                    scraped_json = json.loads(mem)
                else:
                    scraped_json = mem
                scraped = ProductData(**scraped_json)
                break
            except Exception as e:
                print(f"Error parsing agent output as JSON: {e}")
    if not scraped:
        print("No valid JSON scraped data found.")
        scraped = ProductData()

    print(f"Scraped data: {scraped.dict()}")

    # Step 5: Save to CSV
    print("Saving scraped data to CSV...")
    csv_file = "scraped_products.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["url", "images", "description", "features", "reviews"])
        writer.writerow([
            url,
            ";".join(scraped.images),
            scraped.description or "",
            ";".join(scraped.features),
            ";".join(scraped.reviews)
        ])
    print(f"Saved data to {csv_file}")

    # Step 6: Use Gemini 1.0 Pro to check for fake/real (direct prompt, not Agent)
    print("Initializing ChatGoogle (Gemini 1.0 Pro) for authenticity check...")
    llm_check = ChatGoogle(model='gemini-1.0-pro', api_key=GOOGLE_API_KEY)
    prompt = (
        "Given the following product data, analyze and determine if any of the images, content, or reviews appear fake or AI-generated. "
        "Return a JSON object with keys: images, description, features, reviews. "
        "Each key should have a value that is a dictionary with keys: 'verdict' (either 'real' or 'fake') and 'explanation' (a short explanation). "
        f"Product data: images: {scraped.images}, description: {scraped.description}, features: {scraped.features}, reviews: {scraped.reviews}"
    )
    from browser_use.llm.messages import UserMessage
    user_message = UserMessage(content=prompt)
    print("Sending prompt to Gemini 2.5 Pro...")
    try:
        response = await llm_check.ainvoke([user_message])
        import json
        # Fix: extract string from response.completion correctly
        completion_str = None
        if hasattr(response, 'completion'):
            completion = response.completion
            if isinstance(completion, str):
                completion_str = completion
            elif hasattr(completion, 'content'):
                completion_str = completion.content
            else:
                completion_str = str(completion)
        else:
            completion_str = str(response)
        authenticity_json = json.loads(completion_str)
        authenticity = AuthenticityResult(**authenticity_json)
    except Exception as e:
        print(f"Error during authenticity check: {e}")
        # Always return a dict for authenticity
        authenticity = AuthenticityResult(
            images={"verdict": "error", "explanation": str(e)},
            description={"verdict": "error", "explanation": str(e)},
            features={"verdict": "error", "explanation": str(e)},
            reviews={"verdict": "error", "explanation": str(e)}
        )

    print(f"Authenticity analysis: {authenticity.dict()}")

    return JSONResponse(content={
        "scraped": scraped.dict(),
        "authenticity": authenticity.dict()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)