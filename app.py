from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from helper import predict

app = FastAPI()

# CORS (optional since we serve frontend from same app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount the static folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("app/static/index.html")

# Prediction API
@app.post("/get_status")
async def get_status(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_path = "temp.jpg"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        prediction = predict(image_path)
        return prediction
    except Exception as e:
        return {"error": str(e)}
