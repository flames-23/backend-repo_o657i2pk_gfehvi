import os
import base64
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageJob(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(default=None)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=768, ge=64, le=2048)
    cfg_scale: float = Field(default=6.5)
    steps: int = Field(default=30, ge=10, le=150)
    samples: int = Field(default=1, ge=1, le=4)


class GenerateImagesRequest(BaseModel):
    jobs: List[ImageJob]
    model: str = Field(default="stable-diffusion-xl-1024-v1-0")


class GenerateImagesResponse(BaseModel):
    images: List[str]


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


STABILITY_HOST = os.getenv("STABILITY_HOST", "https://api.stability.ai")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")


def _stability_generate(job: ImageJob, model: str) -> List[str]:
    if not STABILITY_API_KEY:
        raise HTTPException(status_code=400, detail="STABILITY_API_KEY non configurée côté serveur.")

    url = f"{STABILITY_HOST}/v1/generation/{model}/text-to-image"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "text_prompts": [
            {"text": job.prompt, "weight": 1.0},
        ],
        "cfg_scale": job.cfg_scale,
        "clip_guidance_preset": "FAST_BLUE",
        "height": job.height,
        "width": job.width,
        "samples": job.samples,
        "steps": job.steps,
    }
    if job.negative_prompt:
        payload["text_prompts"].append({"text": job.negative_prompt, "weight": -1.0})

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Erreur de connexion à Stability: {e}")

    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = {"error": resp.text}
        raise HTTPException(status_code=resp.status_code, detail=err)

    data = resp.json()
    images_data_urls: List[str] = []
    for art in data.get("artifacts", []):
        b64 = art.get("base64")
        if not b64:
            continue
        # Stability returns PNG by défaut
        images_data_urls.append(f"data:image/png;base64,{b64}")
    return images_data_urls


@app.post("/api/generate-images", response_model=GenerateImagesResponse)
def generate_images(req: GenerateImagesRequest):
    images: List[str] = []
    for job in req.jobs:
        out = _stability_generate(job, req.model)
        # Ne conserver qu'un échantillon par job
        if out:
            images.append(out[0])
    return {"images": images}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
