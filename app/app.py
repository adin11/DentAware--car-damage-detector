from fastapi import FastAPI, File, UploadFile
from .helper import predict
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("app/static/index.html")

@app.post("/get_status")
async def get_status(file: UploadFile = File(...)):
    try:
        image_path = "/tmp/temp.jpg"
        with open(image_path, "wb") as f:
            f.write(await file.read())
        return predict(image_path)
    except Exception as e:
        return {"error": str(e)}
