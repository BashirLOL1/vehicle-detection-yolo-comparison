from ultralytics import YOLO
import os

def main():
    # Путь к data.yaml
    DATA_YAML = r"B:\piton\Proektny praktikum\data\Vehicle_Danang_2025\merged_dataset\data.yaml"

    # Проверяем, существует ли файл
    if not os.path.exists(DATA_YAML):
        print(f"❌ Файл не найден: {DATA_YAML}")
        print("Проверь путь к data.yaml")
        return

    print(f"✅ Конфиг найден: {DATA_YAML}")

    # Загружаем предобученную модель
    print("Загрузка YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # Запускаем обучение
    print("Начинаем обучение...")
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,  # ← ВАЖНО! Ставим 0, чтобы избежать проблем с мультипроцессингом на Windows
        patience=10,
        name="yolov8_danang",
        exist_ok=True
    )

    print("✅ Обучение завершено!")
    print(f"📁 Результаты в: runs/detect/yolov8_danang/")
    print(f"🏆 Лучшие веса: runs/detect/yolov8_danang/weights/best.pt")

if __name__ == '__main__':
    main()