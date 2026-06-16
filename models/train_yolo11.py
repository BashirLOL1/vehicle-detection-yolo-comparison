from ultralytics import YOLO
import os


def main():
    DATA_YAML = r"B:\piton\Proektny praktikum\data\Vehicle_Danang_2025\merged_dataset\data.yaml"

    if not os.path.exists(DATA_YAML):
        print(f"❌ Файл не найден: {DATA_YAML}")
        return

    print(f"✅ Конфиг найден: {DATA_YAML}")
    print("Загрузка YOLOv11n...")
    model = YOLO("yolo11n.pt")

    print("Начинаем обучение YOLOv11n...")
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=10,
        name="yolo11n_danang",
        exist_ok=True
    )

    print("✅ Обучение YOLOv11n завершено!")


if __name__ == '__main__':
    main()