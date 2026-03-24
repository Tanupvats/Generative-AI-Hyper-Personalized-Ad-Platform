import os
import shutil
import uuid
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import torch

# Assuming your existing codebase is in the same directory
from models import Wav2Lip
import face_detection
from services.inference_service import run_inference
from services.train_service import run_hq_training

# --- Global State for Models ---
app_state = {
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "wav2lip_model": None,
    "face_detector": None
}

def load_wav2lip_model(path, device):
    model = Wav2Lip()
    print(f"Loading Wav2Lip checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=device)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Runs on startup to preload heavy ML models into GPU memory.
    Prevents loading overhead on every API request.
    """
    checkpoint_path = os.getenv("WAV2LIP_CHECKPOINT", "checkpoints/wav2lip_gan.pth")
    
    print(f"Starting API on device: {app_state['device']}")
    
    if os.path.exists(checkpoint_path):
        app_state["wav2lip_model"] = load_wav2lip_model(checkpoint_path, app_state["device"])
        app_state["face_detector"] = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, 
            flip_input=False, 
            device=app_state["device"]
        )
        print("Models loaded successfully.")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}. Inference endpoint will fail.")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down, clearing GPU memory...")
    app_state["wav2lip_model"] = None
    app_state["face_detector"] = None
    torch.cuda.empty_cache()

app = FastAPI(title="Wav2Lip API", lifespan=lifespan)

@app.post("/infer")
async def infer_video(
    background_tasks: BackgroundTasks,
    face_video: UploadFile = File(...), 
    audio_file: UploadFile = File(...)
):
    """
    Accepts a video and an audio file, performs lip-syncing, and returns the synced video.
    Uses UUID-based temp directories to prevent concurrent request race conditions.
    """
    if app_state["wav2lip_model"] is None:
        raise HTTPException(status_code=503, detail="Wav2Lip model not loaded.")

    # Create an isolated temporary directory for this specific request
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"wav2lip_{req_id}")
    os.makedirs(temp_dir, exist_ok=True)

    input_video_path = os.path.join(temp_dir, f"input_{face_video.filename}")
    input_audio_path = os.path.join(temp_dir, f"input_{audio_file.filename}")
    output_video_path = os.path.join(temp_dir, "output.mp4")

    # Save uploaded files
    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(face_video.file, buffer)
    with open(input_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        # Run the heavy inference block
        # In a highly concurrent environment, consider using run_in_threadpool here
        run_inference(
            video_path=input_video_path,
            audio_path=input_audio_path,
            outfile_path=output_video_path,
            model=app_state["wav2lip_model"],
            detector=app_state["face_detector"],
            device=app_state["device"]
        )
        
        # Schedule the temporary directory cleanup AFTER the file is returned
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        
        return FileResponse(
            path=output_video_path, 
            media_type="video/mp4", 
            filename=f"synced_{req_id}.mp4"
        )
        
    except Exception as e:
        # Cleanup immediately on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/train")
async def start_training(
    data_root: str,
    checkpoint_dir: str = "checkpoints/",
    syncnet_checkpoint: str = "checkpoints/expert_disc.pth",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Initiates the HQ Wav2Lip training loop. 
    Because training blocks the event loop for hours/days, it's sent to a BackgroundTask.
    """
    if not os.path.exists(data_root):
        raise HTTPException(status_code=400, detail="data_root path does not exist.")
        
    # Spin up training in the background
    background_tasks.add_task(
        run_hq_training, 
        data_root=data_root, 
        checkpoint_dir=checkpoint_dir, 
        syncnet_checkpoint=syncnet_checkpoint,
        device=app_state["device"]
    )
    
    return {"status": "Training started in background", "data_root": data_root}