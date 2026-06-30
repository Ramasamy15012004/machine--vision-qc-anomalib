# api.py
import os
import json
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from backend import inspection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve config.json absolute path from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

@app.post("/start")
def start():
    inspection.start_system()
    return {"status": "started"}

@app.post("/stop")
def stop():
    inspection.stop_system()
    return {"status": "stopped"}

@app.get("/live_process")
def live_process():
    def generate():
        while True:
            with inspection.JPEG_LOCK:
                frame = inspection.LATEST_JPEG_PROCESS
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame + b"\r\n"
                )
            time.sleep(0.1)
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )

@app.get("/status")
def status():
    return {"running": inspection.SYSTEM_RUNNING}

@app.get("/detected")
def detected():
    with inspection.DETECTION_LOCK:
        return inspection.LATEST_DETECTION or {}

@app.get("/config")
def get_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

@app.put("/config")
def save_config(data: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return {"status": "saved"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

