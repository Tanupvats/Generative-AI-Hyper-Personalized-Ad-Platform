import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import soundfile as sf

from core.engine import RVCEngine
from core.uvr5_separator import VocalSeparator
from core.preprocess import DatasetPreprocessor
from train_pipeline import trigger_training, train_faiss_index

app = FastAPI(title="Headless RVC API", description="Production Voice Conversion & Training Backend")

# Initialize required directories on startup
DIRS = [
    "./temp", 
    "./models", 
    "./indices", 
    "./datasets", 
    "./assets/hubert", 
    "./assets/rmvpe", 
    "./assets/pretrained", 
    "./assets/uvr5"
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# Global engine initialization to keep HuBERT model persistent in memory
rvc_engine = RVCEngine()
hubert_path = "./assets/hubert/hubert_base.pt"
if os.path.exists(hubert_path):
    print("Loading global HuBERT model...")
    rvc_engine.load_hubert(hubert_path)
else:
    print(f"Warning: {hubert_path} missing. Download it before running inference/training.")

@app.post("/isolate")
async def isolate_vocals(audio_file: UploadFile = File(...)):
    """
    Removes background music from a mixed audio file.
    """
    file_id = str(uuid.uuid4())
    in_path = f"./temp/{file_id}_in.wav"
    vocal_out = f"./temp/{file_id}_vocal.wav"
    inst_out = f"./temp/{file_id}_inst.wav"
    
    with open(in_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
        
    try:
        separator = VocalSeparator(model_path="./assets/uvr5/HP2_all_vocals.pth")
        separator.process_audio(in_path, vocal_out, inst_out)
        return FileResponse(vocal_out, media_type="audio/wav", filename="isolated_vocals.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
async def inference(
    audio_file: UploadFile = File(...),
    model_name: str = Form(..., description="Name of the .pth model in ./models/"),
    pitch_shift: int = Form(0, description="Pitch shift (e.g., +12 for M to F, -12 for F to M)"),
    f0_method: str = Form("rmvpe", description="Pitch extraction method: rmvpe, harvest, dio, pm"),
    index_rate: float = Form(0.75, description="Timbre feature mix rate (0.0 to 1.0)")
):
    """
    Converts the uploaded audio file to the target voice model.
    """
    file_id = str(uuid.uuid4())
    in_path = f"./temp/{file_id}_in.wav"
    out_path = f"./temp/{file_id}_out.wav"
    
    with open(in_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        model_path = f"./models/{model_name}.pth"
        index_path = f"./indices/{model_name}.index"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in ./models/")
            
        rvc_engine.load_vits(model_path)
        if os.path.exists(index_path) and index_rate > 0:
            rvc_engine.load_faiss_index(index_path)
            
        audio_opt, tgt_sr = rvc_engine.infer(
            audio_path=in_path,
            f0_up_key=pitch_shift,
            f0_method=f0_method,
            index_rate=index_rate
        )
        
        sf.write(out_path, audio_opt, tgt_sr)
        return FileResponse(out_path, media_type="audio/wav", filename=f"converted_{model_name}.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def background_training_worker(dataset_name: str, model_name: str, epochs: int):
    """
    Asynchronous background task to process dataset and trigger the DDP training loop.
    """
    print(f"--- Started Training Job for {model_name} ---")
    raw_audio_dir = f"./datasets/{dataset_name}/raw"
    processed_dir = f"./datasets/{dataset_name}/processed"
    
    try:
        print("Step 1: Preprocessing Dataset...")
        preprocessor = DatasetPreprocessor(sr=40000)
        preprocessor.process_dataset(
            input_dir=raw_audio_dir, 
            output_dir=processed_dir, 
            hubert_model=rvc_engine.hubert_model, 
            f0_method="rmvpe"
        )
        
        print("Step 2: Training Faiss Index...")
        features_dir = os.path.join(processed_dir, "3_feature256")
        index_out = f"./indices/{model_name}.index"
        train_faiss_index(features_dir, index_out)
        
        print("Step 3: Triggering VITS Model Training...")
        filelist_path = os.path.join(processed_dir, "filelist.txt") 
        trigger_training(dataset_filelist=filelist_path, model_save_dir="./models", epochs=epochs)
        
        print(f"--- Training Job {model_name} Completed ---")
    except Exception as e:
        print(f"--- Training Failed: {e} ---")

@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    dataset_name: str = Form(..., description="Folder in ./datasets/ containing a 'raw' subfolder"),
    model_name: str = Form(..., description="Name for the output .pth and .index files"),
    epochs: int = Form(50, description="Number of training epochs")
):
    """
    Triggers an asynchronous training pipeline for a new voice model.
    """
    raw_dir = f"./datasets/{dataset_name}/raw"
    if not os.path.exists(raw_dir):
        raise HTTPException(status_code=400, detail=f"Dataset not found at {raw_dir}. Please place wavs there.")
        
    background_tasks.add_task(background_training_worker, dataset_name, model_name, epochs)
    
    return {
        "status": "success", 
        "message": f"Training pipeline for {model_name} started. Monitor terminal output for progress."
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, workers=1)