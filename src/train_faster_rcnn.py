import torch
import torchvision
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import cv2
import sys

# ===== ВСТАВИТЬ СЮДА =====
sys.stdout.reconfigure(line_buffering=True)
print("DEBUG: скрипт запущен", flush=True)

# ========== КОНФИГУРАЦИЯ ==========
TRAIN_IMG_PATH = r"B:\piton\Proektny praktikum\data\Vehicle_Danang_2025\merged_dataset\images\train"
NUM_CLASSES = 4
NUM_EPOCHS = 25
BATCH_SIZE = 4
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Используем устройство: {DEVICE}")


# Dataset
class VehicleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.labels_dir = root_dir.replace('images', 'labels')
        print(f"📊 Найдено изображений: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Ошибка загрузки: {img_path}")
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]

        label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        x_min = int((x_center - width / 2) * img_width)
                        y_min = int((y_center - height / 2) * img_height)
                        x_max = int((x_center + width / 2) * img_width)
                        y_max = int((y_center + height / 2) * img_height)
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(int(class_id))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        img = torchvision.transforms.functional.to_tensor(img)

        return img, target


# Создаём датасет
print("🔄 Загрузка датасета...")
dataset = VehicleDataset(TRAIN_IMG_PATH)

print("🔄 Создание DataLoader...")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
print(f"📊 Количество батчей: {len(dataloader)}")

print("🔄 Загрузка предобученной модели...")
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)

model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

print(f"🚀 Начинаем обучение на {len(dataset)} изображениях...")
print(f"⏱️ Ожидаемое время: ~8-10 часов")
print("=" * 50)

for epoch in range(NUM_EPOCHS):
    print(f"\n🔥 ЭПОХА {epoch + 1}/{NUM_EPOCHS}")
    model.train()
    epoch_loss = 0

    for i, (images, targets) in enumerate(dataloader):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        if (i + 1) % 10 == 0:
            print(f"   Батч {i + 1}/{len(dataloader)}, Loss: {losses.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"✅ Эпоха {epoch + 1}/{NUM_EPOCHS} завершена, Средний Loss: {avg_loss:.4f}")

os.makedirs("runs/detect/faster_rcnn", exist_ok=True)
torch.save(model.state_dict(), "runs/detect/faster_rcnn/best.pth")
print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО! 🎉")
print("📁 Модель сохранена в runs/detect/faster_rcnn/best.pth")