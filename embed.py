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