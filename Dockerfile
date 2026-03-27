# # Базовый образ от NVIDIA с CUDA 12 и cuDNN
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# # Установка Python и системных библиотек для работы с видео/фото
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-dev \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Устанавливаем библиотеки
# # onnxruntime-gpu внутри этого контейнера автоматически увидит библиотеки CUDA
# RUN pip3 install --no-cache-dir \
#     numpy \
#     opencv-python \
#     insightface \
#     onnxruntime-gpu \
#     psycopg2-binary

# COPY . .

# # Команда для запуска (пока просто наш тестовый скрипт)
# CMD ["python3", "appcuda.py"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    numpy opencv-python insightface onnxruntime-gpu \
    fastapi uvicorn python-multipart psycopg2-binary

COPY . .

CMD ["uvicorn", "api:api_app", "--host", "0.0.0.0", "--port", "8000"]