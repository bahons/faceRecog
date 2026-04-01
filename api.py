import os
import sys
import cv2
import numpy as np
import httpx
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from insightface.app import FaceAnalysis

# --- Настройка ---
api_app = FastAPI()
CSHARP_BACKEND_URL = os.getenv("CSHARP_BACKEND_URL", "https://localhost:7094/api/attendance/process-faces")

# TensorRT провайдеры
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 2147483648,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': '/app/trt_cache'
    }),
    'CUDAExecutionProvider'
]

face_app = FaceAnalysis(name='buffalo_l', providers=providers)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Глобальный клиент для переиспользования соединений (быстрее)
client = httpx.AsyncClient(verify=False, timeout=30.0)

async def send_to_backend(payload: dict):
    """Фоновая задача для отправки данных на C#"""
    try:
        # Мы не ждем ответа, просто отправляем
        await client.post(CSHARP_BACKEND_URL, json=payload)
    except Exception as e:
        print(f"--- FAILED TO SEND TO C# ---: {e}")

@api_app.post("/analyze-audience")
async def analyze_audience(
    file: UploadFile = File(...), 
    room_id: str = Form(...),
    timestamp: str = Form(...)
):
    # 1. Чтение и детекция (на GPU)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    faces = face_app.get(img)
    
    # 2. Подготовка данных (векторы)
    face_data_list = [
        {
            "bbox": [float(x) for x in face.bbox.tolist()],
            "embedding": face.embedding.tolist()
        } for face in faces
    ]

    payload = {
        "room_id": room_id,
        "timestamp": timestamp,
        "faces": face_data_list
    }

    # 3. МАГИЯ: Запускаем отправку в фоне и НЕ ЖДЕМ её завершения
    asyncio.create_task(send_to_backend(payload))

    # 4. Моментальный ответ API
    return {
        "status": "processing_started",
        "faces_detected": len(faces),
        "room": room_id
    }

@api_app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)