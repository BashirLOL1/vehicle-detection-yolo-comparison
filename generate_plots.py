from ultralytics import YOLO
import multiprocessing as mp

def main():
    # Путь к твоей модели YOLOv11n
    model_path = "runs/detect/yolo11n_danang/weights/best.pt"

    print(f"🔍 Загружаем модель: {model_path}")
    model = YOLO(model_path)

    print("📊 Запускаем валидацию с генерацией графиков...")
    metrics = model.val(plots=True, workers=0)  # workers=0 отключает многопроцессность

    print("\n✅ Готово! Графики сохранены.")
    print("📁 Ищи их в свежей папке runs/detect/val")

if __name__ == '__main__':
    mp.freeze_support()
    main()