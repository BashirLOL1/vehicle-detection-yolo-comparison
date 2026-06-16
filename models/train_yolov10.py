from ultralytics import YOLO
import os


def main():
    DATA_YAML = r"B:\piton\Proektny praktikum\data\Vehicle_Danang_2025\merged_dataset\data.yaml"

    if not os.path.exists(DATA_YAML):
        print(f"❌ Файл не найден: {DATA_YAML}")
        return

    print(f"✅ Конфиг найден: {DATA_YAML}")
    print("Загрузка YOLOv10n...")
    model = YOLO("yolov10n.pt")  # веса скачаются автоматически

    print("Начинаем обучение YOLOv10n...")
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=10,
        name="yolov10n_danang",
        exist_ok=True
    )

    print("✅ Обучение YOLOv10n завершено!")


if __name__ == '__main__':
    main()