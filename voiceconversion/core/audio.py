import platform
import os
import re
import traceback
import ffmpeg
import numpy as np
import av

def clean_path(path_str):
    """
    Cleans up file paths to prevent crashes caused by invisible Unicode 
    control characters, accidental spaces, or OS-specific slashes.
    """
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(r'[\u202a\u202b\u202c\u202d\u202e]', '', path_str)  # Remove Unicode control chars
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


def load_audio(file, sr):
    """
    Robustly loads audio using an FFmpeg subprocess.
    Automatically down-mixes to mono and resamples to the requested sample rate (sr).
    """
    try:
        file = clean_path(file)
        if not os.path.exists(file):
            raise RuntimeError(
                f"Audio path does not exist: {file}"
            )
            
        # Launches FFmpeg to decode the audio, outputting raw float32 bytes
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        traceback.print_exc()
        error_message = e.stderr.decode('utf8') if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg failed to load audio: {error_message}")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

    # Convert the raw bytes to a flat NumPy array
    return np.frombuffer(out, np.float32).flatten()


def wav2(i, o, format):
    """
    Utility function for transcoding audio formats using PyAV.
    Useful for saving final converted files into formats other than standard wav.
    """
    inp = av.open(i, "rb")
    
    if format == "m4a":
        format = "mp4"
        
    out = av.open(o, "wb", format=format)
    
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()