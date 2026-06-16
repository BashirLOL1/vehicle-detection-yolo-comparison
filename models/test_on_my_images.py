from ultralytics import YOLO
import cv2
import os

# Загружаем обученную модель
model = YOLO(r"B:\piton\Proektny praktikum\runs\detect\yolov8_danang\weights\best.pt")

# Путь к твоим картинкам (оригинальные 226 кадров)
test_folder = r"B:\piton\Proektny praktikum\data\raw"

# Создаём папку для результатов
output_folder = r"B:\piton\Proektny praktikum\test_results"
os.makedirs(output_folder, exist_ok=True)

# Берём первые 20 картинок для теста
test_images = [f for f in os.listdir(test_folder) if f.endswith('.jpg')][:20]

print(f"Найдено картинок: {len(test_images)}")

for img_name in test_images:
    img_path = os.path.join(test_folder, img_name)
    results = model(img_path)

    # Сохраняем результат с рамками
    result_img = results[0].plot()
    cv2.imwrite(os.path.join(output_folder, f"result_{img_name}"), result_img)
    print(f"Обработано: {img_name}")

print(f"✅ Готово! Результаты в {output_folder}")