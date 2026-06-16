from ultralytics import YOLO
import cv2
import os

# Проверяем пути
model_path = "runs/detect/yolo11s_danang/weights/best.pt"
if not os.path.exists(model_path):
    print(f"❌ Модель не найдена: {model_path}")
    exit(1)

# Используем val вместо test (в датасете есть только train и val)
val_folder = r"B:\piton\Proektny praktikum\data\Vehicle_Danang_2025\merged_dataset\images\val"
if not os.path.exists(val_folder):
    print(f"❌ Папка с валидационными изображениями не найдена: {val_folder}")
    exit(1)

# Загружаем модель
print(f"✅ Модель найдена: {model_path}")
model = YOLO(model_path)

# Создаём папку
os.makedirs("report_images", exist_ok=True)

# Получаем список изображений из val
val_images = [f for f in os.listdir(val_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"📊 Найдено изображений в val: {len(val_images)}")

if len(val_images) == 0:
    print("❌ Нет изображений в папке val")
    exit(1)

# Берём первые 10
val_images = val_images[:10]
print(f"🖼️ Обрабатываем первые 10...")

for i, img_name in enumerate(val_images):
    img_path = os.path.join(val_folder, img_name)
    print(f"   {i + 1}/10: {img_name}...", end=" ", flush=True)

    # Детекция
    results = model(img_path)
    img = results[0].plot()

    # Сохраняем
    save_path = f"report_images/{img_name}"
    success = cv2.imwrite(save_path, img)

    if success:
        print("✅")
    else:
        print("❌ ошибка сохранения")

print(f"\n✅ Готово! Проверь папку report_images")
print(f"📁 Полный путь: {os.path.abspath('report_images')}")