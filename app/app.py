from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.helper import predict

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html on root
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return FileResponse("static/index.html")

# Prediction endpoint
@app.post("/get_status")
async def get_status(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        with open("temp.jpg", "wb") as f:
            f.write(image_bytes)
        result = predict("temp.jpg")
        return result
    except Exception as e:
        return {"error": str(e)}
