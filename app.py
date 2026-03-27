import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Инициализация модели (используем только CPU)
# 'buffalo_l' — это качественная модель для детекции и распознавания
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1280, 1280))

# 2. Загрузка вашего тестового фото
image_path = "group50.jpg"  # Путь к твоему фото с 10-15 лицами
img = cv2.imread(image_path)

# 3. Детекция и анализ лиц
# Система найдет все лица в кадре и извлечет их признаки 
faces = app.get(img)

print(f"Найдено лиц: {len(faces)}")

# 4. Визуализация результатов для проверки MVP
for face in faces:
    # Отрисовка рамки вокруг лица [cite: 45]
    bbox = face.bbox.astype(int)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # В будущем здесь будет сравнение с базой Platonus [cite: 35]
    # Пока просто выводим вероятность, что это лицо
    score = face.det_score
    cv2.putText(img, f"{score:.2f}", (bbox[0], bbox[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Сохранение результата теста
cv2.imwrite("result_detection.jpg", img)
print("Результат сохранен в result_detection.jpg")