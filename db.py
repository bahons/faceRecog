import os
import cv2
import psycopg2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Настройка подключения к Docker-PostgreSQL
DB_CONFIG = {
    "dbname": "facerecog",
    "user": "admin",
    "password": "123456",
    "host": "localhost",
    "port": "5433"
}

# 2. Инициализация AI-ядра (только CPU)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def upload_students_base(folder_path):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print(f"--- Начало обработки папки: {folder_path} ---")
    
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Предполагаем формат имени файла: "ID_Имя_Фамилия.jpg"
            # Если ID нет, можно использовать имя файла как ID для теста
            name_parts = os.path.splitext(filename)[0].split('_')
            platonus_id = name_parts[0]
            full_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else name_parts[0]
            
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Ошибка: Не удалось прочитать {filename}")
                continue
                
            # Извлекаем лицо
            faces = app.get(img)
            
            if len(faces) > 0:
                # Берем самое четкое лицо (первое в списке)
                # Нам нужен именно normed_embedding для косинусного сравнения
                embedding = faces[0].normed_embedding.tolist()
                
                try:
                    cur.execute(
                        "INSERT INTO students (platonus_id, full_name, embedding) VALUES (%s, %s, %s) "
                        "ON CONFLICT (platonus_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                        (platonus_id, full_name, embedding)
                    )
                    conn.commit()
                    print(f"Успешно добавлен: {full_name} (ID: {platonus_id})")
                except Exception as e:
                    print(f"Ошибка БД при добавлении {full_name}: {e}")
                    conn.rollback()
            else:
                print(f"Предупреждение: Лицо не найдено на фото {filename}")

    cur.close()
    conn.close()
    print("--- Загрузка базы завершена ---")

# Запуск
if __name__ == "__main__":
    upload_students_base("students_base2")