import os
import sys
import cv2
import numpy as np
import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException
from psycopg2.extras import execute_values

# В самом начале api.py замени блок импорта на этот:
try:
    import numpy
    print(f"--- NumPy version: {numpy.__version__} ---")
    import onnxruntime as ort
    print(f"--- ONNX Runtime loaded. Version: {ort.__version__} ---")
    print(f"--- Available Providers: {ort.get_available_providers()} ---")
except Exception as e:
    import traceback
    print("--- CRITICAL ERROR DURING STARTUP ---")
    print(traceback.format_exc()) # Это покажет полную ошибку, а не просто пустую строку
    sys.exit(1)

from insightface.app import FaceAnalysis

# --- ШАГ 2: Конфигурация FastAPI ---
api_app = FastAPI()

# --- ШАГ 3: Настройка AI моделей ---
available_providers = ort.get_available_providers()

if 'TensorrtExecutionProvider' in available_providers:
    # Важно: папка /app/trt_cache должна существовать
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
    print("--- Using TensorRT Provider ---")
else:
    providers = ['CUDAExecutionProvider']
    print("--- Using CUDA Provider ---")

# Инициализируем FaceAnalysis (называем face_app, чтобы не путать с api_app)
face_app = FaceAnalysis(name='buffalo_l', providers=providers)
face_app.prepare(ctx_id=0, det_size=(1280, 1280))

# --- ШАГ 4: Работа с БД ---
DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:123456@db:5432/facerecog")

def get_db_connection():
    return psycopg2.connect(DB_URL)

# --- ШАГ 5: Эндпоинты ---

@api_app.post("/analyze-audience")
async def analyze_audience(file: UploadFile = File(...)):
    # 1. Чтение изображения
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 2. Инференс на GPU (RTX 3050)
    # max_num=0 найдет всех студентов на фото
    faces = face_app.get(img)
    
    found_students = []
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        for face in faces:
            embedding = face.embedding.tolist()
            
            # 3. Поиск в pgvector (Косинусное расстояние <=> )
            # Используем твои названия колонок из БД (embedding и platonus_id)
            cur.execute("""
                SELECT full_name, platonus_id, (embedding <=> %s::vector) as distance
                FROM students
                WHERE (embedding <=> %s::vector) < 0.5
                ORDER BY distance ASC
                LIMIT 1
            """, (embedding, embedding))
            
            res = cur.fetchone()
            if res:
                found_students.append({
                    "name": res[0],
                    "external_id": res[1],
                    "confidence": round(1 - res[2], 2),
                    "bbox": [round(float(x), 2) for x in face.bbox.tolist()]
                })
            else:
                found_students.append({
                    "name": "Unknown",
                    "bbox": [round(float(x), 2) for x in face.bbox.tolist()]
                })
        
        cur.close()
        conn.close()

    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

    return {
        "total_faces": len(faces),
        "identified_students": len([s for s in found_students if s['name'] != "Unknown"]),
        "results": found_students
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)