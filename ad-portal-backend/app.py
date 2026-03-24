import os
import uuid
import shutil
import httpx
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS

app = FastAPI(title="GenAI Ad Pipeline API Orchestrator")

# Allow CORS for the React Canvas frontend (Vite typically runs on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RVC_API_URL = os.getenv("RVC_API_URL", "http://localhost:8001")
WAV2LIP_API_URL = os.getenv("WAV2LIP_API_URL", "http://localhost:8002")

OUTPUT_DIR = os.path.abspath("./temp_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/api/generate-ad")
async def generate_ad(
    name: str = Form(...),
    designation: str = Form(...),
    dialog: str = Form(...),
    video: UploadFile = File(...)
):
    # Unique ID for this specific generation job
    session_id = str(uuid.uuid4())[:8]
    print(f"\n--- [Session {session_id}] Starting Ad Generation Pipeline ---")
    
    # Track file paths for this session
    input_video_path = os.path.join(OUTPUT_DIR, f"{session_id}_input.mp4")
    tts_audio_path = os.path.join(OUTPUT_DIR, f"{session_id}_tts.wav")
    rvc_audio_path = os.path.join(OUTPUT_DIR, f"{session_id}_rvc.wav")
    final_video_path = os.path.join(OUTPUT_DIR, f"{session_id}_final.mp4")
    
    try:
        # 1. Save uploaded video from frontend
        with open(input_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        print(f"[{session_id}] Source video saved.")

        # 2. Text to Speech (gTTS)
        full_script = f"Hello, I am {name}, {designation}. {dialog}"
        print(f"[{session_id}] Generating base speech via gTTS...")
        tts = gTTS(text=full_script, lang='en', slow=False)
        tts.save(tts_audio_path)

        # Use an async HTTP client to communicate with our microservices
        # Note: Set a long timeout since inference can take minutes
        async with httpx.AsyncClient(timeout=600.0) as client:
            
            # 3. Voice Style Transfer (via RVC Microservice)
            print(f"[{session_id}] Calling RVC API at {RVC_API_URL}/infer ...")
            with open(tts_audio_path, "rb") as f:
                rvc_files = {"audio_file": (os.path.basename(tts_audio_path), f, "audio/wav")}
                rvc_data = {
                    "model_name": "brand_ambassador", # Ensure this model exists in your RVC backend ./models folder
                    "pitch_shift": 0,
                    "f0_method": "rmvpe",
                    "index_rate": 0.75
                }
                
                rvc_response = await client.post(f"{RVC_API_URL}/infer", files=rvc_files, data=rvc_data)
                
                if rvc_response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"RVC API Error: {rvc_response.text}")
                
                # Save the converted audio returned by the API
                with open(rvc_audio_path, "wb") as out_f:
                    out_f.write(rvc_response.content)

            # 4. Lip Synchronization (via Wav2Lip Microservice)
            print(f"[{session_id}] Calling Wav2Lip API at {WAV2LIP_API_URL}/infer ...")
            with open(input_video_path, "rb") as vid_f, open(rvc_audio_path, "rb") as aud_f:
                w2l_files = {
                    "face_video": (os.path.basename(input_video_path), vid_f, "video/mp4"),
                    "audio_file": (os.path.basename(rvc_audio_path), aud_f, "audio/wav")
                }
                
                w2l_response = await client.post(f"{WAV2LIP_API_URL}/infer", files=w2l_files)
                
                if w2l_response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"Wav2Lip API Error: {w2l_response.text}")
                
                # Save the final synced video returned by the API
                with open(final_video_path, "wb") as out_f:
                    out_f.write(w2l_response.content)

        print(f"[{session_id}] Pipeline complete! Returning final video to frontend.")

        # 5. Return Output Video back to React App
        return FileResponse(
            path=final_video_path, 
            media_type="video/mp4", 
            filename=f"personalized_ad_{name.replace(' ', '_')}.mp4"
        )

    except httpx.RequestError as e:
        print(f"[{session_id}] Microservice Connection Failed: {e}")
        raise HTTPException(status_code=503, detail="Unable to connect to ML microservices. Are they running?")
    except Exception as e:
        print(f"[{session_id}] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Optional: Clean up input files immediately after successful request to save disk space
        pass

if __name__ == "__main__":
    import uvicorn
    # Run the orchestrator on port 8000
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=8000, reload=True)