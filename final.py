import cv2
import psycopg2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Настройки
DB_CONFIG = {"dbname": "facerecog", "user": "admin", "password": "123456", "host": "localhost", "port": "5433"}
GROUP_PHOTO = "group3.jpg"
THRESHOLD = 0.45  # Порог схожести (настраивается под освещение)

# 2. Инициализация ИИ
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1600, 1600))

def get_attendance():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Загружаем фото и ищем лица
    img = cv2.imread(GROUP_PHOTO)
    detected_faces = app.get(img)
    print(f"Обнаружено лиц в аудитории: {len(detected_faces)}")

    present_ids = []
    unknown_count = 0

    # 3. Сверка каждого лица с базой через SQL (pgvector)
    # # Исправленный блок в final.py
    # for face in detected_faces:
    #     # Превращаем массив в список, а затем в строку формата '[0.1, 0.2, ...]'
    #     # Это гарантирует, что pgvector корректно примет данные
    #     emb_str = "[" + ",".join(map(str, face.normed_embedding.tolist())) + "]"
        
    #     cur.execute("""
    #         SELECT platonus_id, full_name, embedding <=> %s::vector AS distance 
    #         FROM students 
    #         ORDER BY distance ASC LIMIT 1
    #     """, (emb_str,)) 
        
    #     res = cur.fetchone()
    #     # ... логика обработки результата (THRESHOLD)

    # Замени цикл в функции get_attendance на этот:
    print("\n--- Процесс идентификации ---")
    for i, face in enumerate(detected_faces):
        # Преобразуем в строку для pgvector
        emb_list = face.normed_embedding.tolist()
        emb_str = "[" + ",".join(map(str, emb_list)) + "]"
        
        try:
            cur.execute("""
                SELECT platonus_id, full_name, embedding <=> %s::vector AS distance 
                FROM students 
                ORDER BY distance ASC LIMIT 1
            """, (emb_str,))
            
            res = cur.fetchone()
            
            if res:
                name, p_id, dist = res[1], res[0], res[2]
                # ВЫВОДИМ ОТЛАДКУ: какое расстояние до самого похожего студента
                if i % 5 == 0: # выводим каждое 5-е лицо, чтобы не спамить
                    print(f"Лицо #{i}: ближайший {name}, дистанция: {dist:.4f}")
                
                if dist < THRESHOLD:
                    present_ids.append((p_id, name))
                else:
                    unknown_count += 1
            else:
                unknown_count += 1
                
        except Exception as e:
            print(f"Ошибка на лице #{i}: {e}")
            conn.rollback() # важно сбросить транзакцию при ошибке

    # 4. Логика определения отсутствующих (согласно ТЗ 4.4)
    cur.execute("SELECT platonus_id, full_name FROM students")
    all_students = cur.fetchall()
    
    present_set = {p[0] for p in present_ids}
    absent_students = [s for s in all_students if s[0] not in present_set]

    # 5. Вывод статистики (ТЗ 14)
    print("\n" + "="*30)
    print(f"ИТОГИ ПОСЕЩАЕМОСТИ:")
    print(f"Всего в группе: {len(all_students)}")
    print(f"Присутствуют:  {len(present_ids)}")
    print(f"Отсутствуют:   {len(absent_students)}")
    print(f"Неизвестные:   {unknown_count}")
    print(f"Процент явки:  {len(present_ids)/len(all_students)*100:.1f}%")
    print("="*30)

    print("\nСПИСОК ПРИСУТСТВУЮЩИХ:")
    for pid, name in present_ids: print(f"[✓] {name} ({pid})")

    # print("\nСПИСОК ОТСУТСТВУЮЩИХ:")
    # for pid, name in absent_students: print(f"[x] {name} ({pid})")

    cur.close()
    conn.close()

if __name__ == "__main__":
    get_attendance()