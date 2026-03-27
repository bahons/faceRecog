import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time

# 1. Инициализация (теперь без ошибок!)
print("Инициализация InsightFace на RTX 3050...")
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1280, 1280))

# 2. Загрузка фото
image_path = "group50.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Ошибка: файл {image_path} не найден")
else:
    # 3. Тот самый замер времени
    print("Начинаю магию GPU...")
    start_inference = time.time()
    faces = app.get(img)
    end_inference = time.time()

    print(f"Найдено лиц: {len(faces)}")

    # 4. Визуализация
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    cv2.imwrite("result_final_gpu.jpg", img)
    
    print("\n--- Финальный отчет ---")
    print(f"Время обработки на GPU: {end_inference - start_inference:.2f} сек.")
    print(f"Ускорение по сравнению с CPU: {65 / (end_inference - start_inference):.1f}x раз")
    print("-----------------------")