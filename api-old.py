### -------------------- CUDA ------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
import psycopg2
from psycopg2.extras import execute_values
import io

# app = FastAPI()

# # Инициализация моделей (на GPU)
# face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_app.prepare(ctx_id=0, det_size=(1280, 1280))
### -------------------- CUDA end -------------------

### -------------------- Tenser ------------------------------

# Проверяем, в каком контейнере мы находимся (можно через переменную окружения или просто пробовать)
# Для TensorRT версии:
# use_trt = os.path.exists('/usr/lib/x86_64-linux-gnu/libnvinfer.so.10')

# if use_trt:
#     providers = [
#         ('TensorrtExecutionProvider', {
#             'device_id': 0,
#             'trt_max_workspace_size': 2147483648, # 2GB
#             'trt_fp16_enable': True,              # Включаем ускорение FP16
#             'trt_engine_cache_enable': True,
#             'trt_engine_cache_path': '/app/trt_cache'
#         }),
#         'CUDAExecutionProvider'
#     ]
#     print("Инициализация с TensorRT (FP16)...")
# else:
#     providers = ['CUDAExecutionProvider']
#     print("Инициализация со стандартным CUDA...")

# face_app = FaceAnalysis(name='buffalo_l', providers=providers)
# --------------- Tensor end -----------------------------

# В api.py измени проверку на более простую:
import os
import sys

# ШАГ 1: Сначала импортируем onnxruntime
try:
    import onnxruntime as ort
    print(f"--- ONNX Runtime loaded. Version: {ort.__version__} ---")
    print(f"--- Available Providers: {ort.get_available_providers()} ---")
except Exception as e:
    print(f"--- FAILED TO LOAD ONNXRUNTIME: {e} ---")
    sys.exit(1)

# ШАГ 2: Только ПОТОМ импортируем insightface
from insightface.app import FaceAnalysis

# Настройка провайдеров
if 'TensorrtExecutionProvider' in ort.get_available_providers():
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

app = FaceAnalysis(name='buffalo_l', providers=providers)

face_app = FaceAnalysis(name='buffalo_l', providers=active_providers)
face_app.prepare(ctx_id=0, det_size=(1280, 1280))

# Настройки БД из docker-compose
DB_URL = "postgresql://admin:123456@db:5432/facerecog"

def get_db_connection():
    return psycopg2.connect(DB_URL)

@app.post("/analyze-audience")
async def analyze_audience(file: UploadFile = File(...)):
    # 1. Загрузка изображения
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 2. Детекция и извлечение векторов (Inference на RTX 3050)
    faces = face_app.get(img, max_num=0)
    
    found_students = []
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        for face in faces:
            embedding = face.embedding.tolist()
            
            # 3. Поиск в pgvector (Косинусное сходство)
            # Мы ищем студента, чей вектор ближе всего (порог 0.5 для уверенности)
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
                    "bbox": face.bbox.tolist()
                })
            else:
                found_students.append({
                    "name": "Unknown",
                    "bbox": face.bbox.tolist()
                })

    finally:
        cur.close()
        conn.close()

    return {
        "total_faces": len(faces),
        "identified_students": len([s for s in found_students if s['name'] != "Unknown"]),
        "results": found_students
    }