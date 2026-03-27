import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time

# Фиксируем общее время старта программы
start_total = time.time()

# 1. Инициализация модели
print("Загрузка моделей...")
start_init = time.time()
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1280, 1280))
end_init = time.time()

# 2. Загрузка фото
image_path = "group50.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Ошибка: не удалось загрузить файл {image_path}")
else:
    # 3. Детекция и анализ лиц (самый ресурсозатратный этап)
    print(f"Начинаю анализ изображения: {image_path}...")
    start_analysis = time.time()
    faces = app.get(img)
    end_analysis = time.time()

    print(f"Найдено лиц: {len(faces)}")

    # 4. Визуализация результатов
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        score = face.det_score
        cv2.putText(img, f"{score:.2f}", (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite("result_detection.jpg", img)
    
    # Итоговые метрики
    end_total = time.time()
    
    print("\n--- Отчет по времени ---")
    print(f"Инициализация моделей: {end_init - start_init:.2f} сек.")
    print(f"Чистая обработка фото (Inference): {end_analysis - start_analysis:.2f} сек.")
    print(f"Общее время выполнения скрипта: {end_total - start_total:.2f} сек.")
    print("------------------------")
    print("Результат сохранен в result_detection.jpg")