import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Инициализация. Используем только CPU и модель 'buffalo_l' для точности
# buffalo_l — самая мощная модель в пакете, подходит для базы в 5000 студентов
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])

# det_size=(640, 640) — стандарт, но для 10-15 человек на 4K фото можно увеличить до (1024, 1024)
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Загрузка фото (укажи имя своего файла)
image_path = "test_students.jpg" 
img = cv2.imread(image_path)

if img is None:
    print(f"Ошибка: Не удалось найти файл {image_path}")
else:
    # 3. Запуск анализа (Детекция + Эмбеддинги)
    faces = app.get(img)

    print(f"--- Результаты FaceRecog ---")
    print(f"Найдено студентов в кадре: {len(faces)}")

    # 4. Рисуем рамки и выводим уверенность (Score)
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        # Зеленая рамка для каждого лица
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Выводим точность детекции (score)
        score = face.det_score
        cv2.putText(img, f"ID:{i} {score:.2f}", (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 5. Сохраняем результат для проверки
    output_path = "result_mvp.jpg"
    cv2.imwrite(output_path, img)
    print(f"Готово! Проверь файл {output_path}")