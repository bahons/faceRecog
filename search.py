import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

known_faces = [] # Сюда сохраним эмбеддинги
known_names = [] # Сюда — имена

base_path = "students_base"

print("--- Создание базы лиц ---")
for file in os.listdir(base_path):
    if file.endswith((".jpg", ".png")):
        img = cv2.imread(os.path.join(base_path, file))
        faces = app.get(img)
        if len(faces) > 0:
            # Берем первое найденное лицо на фото и его вектор (embedding)
            known_faces.append(faces[0].normed_embedding)
            known_names.append(os.path.splitext(file)[0])
            print(f"Добавлен: {file}")

# Сохраняем в память для текущего теста
known_faces = np.array(known_faces)

# Загружаем групповое фото (где 30 человек)
group_img = cv2.imread("students_test.jpg")
detected_faces = app.get(group_img)

print(f"\n--- Сверка со списком группы ---")
present_students = []

for face in detected_faces:
    embedding = face.normed_embedding
    # Считаем схожесть со всей базой сразу
    scores = np.dot(known_faces, embedding)
    
    max_idx = np.argmax(scores)
    max_score = scores[max_idx]

    # Порог схожести (обычно 0.4 - 0.5 для ArcFace)
    if max_score > 0.45:
        name = known_names[max_idx]
        present_students.append(name)
        print(f"Распознан: {name} (Сходство: {max_score:.2f})")
    else:
        # Если не нашли в базе — статус «Неизвестное лицо» по ТЗ [cite: 14]
        pass

print(f"\nИтог: Присутствуют {len(present_students)} из {len(known_names)} студентов базы.")