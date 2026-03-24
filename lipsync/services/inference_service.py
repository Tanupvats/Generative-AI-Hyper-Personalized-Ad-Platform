import os
import subprocess
import platform
import numpy as np
import cv2
import torch
from tqdm import tqdm

import audio

# Standard hyperparams extracted from the original inference.py
IMG_SIZE = 96
MEL_STEP_SIZE = 16
WAV2LIP_BATCH_SIZE = 128
FACE_DET_BATCH_SIZE = 16
PADS = [0, 10, 0, 0]

def get_smoothened_boxes(boxes, T=5):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, detector):
    batch_size = FACE_DET_BATCH_SIZE
    while True:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please resize.')
            batch_size //= 2
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = PADS
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected! Ensure the video contains a face in all frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    return results 

def datagen(frames, mels, face_det_results):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, m in enumerate(mels):
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= WAV2LIP_BATCH_SIZE:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, IMG_SIZE//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, IMG_SIZE//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


def run_inference(video_path, audio_path, outfile_path, model, detector, device):
    """
    Main isolated inference pipeline.
    Replaces the original global inference.py main() loop to be thread-safe.
    """
    temp_dir = os.path.dirname(outfile_path)
    temp_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
    temp_result_avi = os.path.join(temp_dir, "temp_result.avi")

    # 1. Read Video
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        full_frames.append(frame)

    if not full_frames:
        raise ValueError("No frames could be read from the video.")

    # 2. Extract Audio
    command = f'ffmpeg -y -i "{audio_path}" -strict -2 "{temp_audio_path}"'
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    wav = audio.load_wav(temp_audio_path, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Add a small epsilon noise to the audio file and try again.')

    # 3. Chunk Mel Spectrograms
    mel_chunks = []
    mel_idx_multiplier = 80. / fps 
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
        i += 1

    full_frames = full_frames[:len(mel_chunks)]
    
    # 4. Face Detection
    face_det_results = face_detect(full_frames, detector)

    # 5. Generation
    gen = datagen(full_frames.copy(), mel_chunks, face_det_results)

    out = None
    for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(temp_result_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)

    if out:
        out.release()

    # 6. Combine Audio and Video via FFmpeg
    command = f'ffmpeg -y -i "{temp_audio_path}" -i "{temp_result_avi}" -strict -2 -q:v 1 "{outfile_path}"'
    subprocess.call(command, shell=platform.system() != 'Windows', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)