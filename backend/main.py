from fastapi import FastAPI
from backend.config import settings

app = FastAPI(title="Scientific Computing Chatbox API")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}

@app.on_event("startup")
async def startup_event():
    print(f"Starting up with model {settings.MODEL_NAME}...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
