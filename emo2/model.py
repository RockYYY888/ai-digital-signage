import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from PIL import Image
import numpy as np
from map import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dir = 'fer2013/train'
val_dir = 'fer2013/test'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(EmotionClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)  # 修改最后一层
        )
        

    def forward(self, x):
        return self.base_model(x)

# 初始化模型
num_classes = 4  # class of the emotion
model = EmotionClassifier(num_classes=num_classes)
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

# 训练代码
def train(model, dataloader, criterion, optimizer, epoch, print_freq=100):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % print_freq == 0:
            print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / total_samples
    print(f'Epoch [{epoch+1}] Training Loss: {epoch_loss:.4f}')
    return epoch_loss

# 验证代码
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return epoch_loss, accuracy

# 统一预测方法
def predict(model, image_path, transform=val_transform):
    model.eval()
    if type(image_path) == str:
        image = Image.open(image_path).convert('L')
        image = transform(image).unsqueeze(0).to(device)
    else:
        image = transform(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, emo_pred = torch.max(outputs, 1)
        return emo_pred.item()

# 训练和验证过程
if __name__ == '__main__':
    num_epochs = 10
    best_val_acc = 0.0
    start_epoch = 0  

    last_checkpoint = 'best_project_model.pth'
    best_model_path = 'best_project_model.pth'

    # 加载检查点（如果存在）
    if os.path.exists(last_checkpoint):
        print('Loading checkpoint...')
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = checkpoint['epoch'] + 1  # 下一轮开始的epoch
        print(f'Resuming training from epoch {start_epoch}')
        print(f'Current Best Validation Accuracy: {best_val_acc}')
    else:
        print('No checkpoint found, starting training from scratch.')

    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, print_freq=100)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f'\n\nEpoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}')
        print(f' - Validation Accuracy: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }
            torch.save(checkpoint, best_model_path)
            print('Best model updated.')

    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }
    torch.save(checkpoint, 'final_checkpoint.pth')
    print("Final model saved")
