from torch import nn, optim
from CV.UTKFaceDataset import *
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 72
csv_file = get_resource_path('CV/utk_dataset_metadata.csv')
img_dir = get_resource_path('CV/UTKFace')

# data augmentation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
])

val_transform2 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_dir = get_resource_path('CV/fer2013/train')
val_dir = get_resource_path('CV/fer2013/test')

dataset = UTKFaceDataset(
    csv_file=csv_file,
    img_dir=img_dir,
    transform=None,
    gender_mapping=gender_mapping,
    race_mapping=race_mapping
)

# train/val/test : 0.8/0.05/0.15
total_size = len(dataset)

train_size = int(0.9 * total_size)
val_size = total_size - train_size

train_indices, val_indices = random_split(
    range(total_size), [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
train_dataset2 = datasets.ImageFolder(root=train_dir, transform=transform2)
val_dataset2 = datasets.ImageFolder(root=val_dir, transform=val_transform2)

train_dataset = Subset(UTKFaceDataset(csv_file=csv_file, img_dir=img_dir, transform=train_transforms,
                                      gender_mapping=gender_mapping, race_mapping=race_mapping), train_indices)
val_dataset = Subset(UTKFaceDataset(csv_file=csv_file, img_dir=img_dir, transform=transform,
                                    gender_mapping=gender_mapping, race_mapping=race_mapping), val_indices)

train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
val_loader2 = DataLoader(val_dataset2, batch_size=batch_size, shuffle=False)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class FaceAttributeModel(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes, num_race_classes, *args, **kwargs):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.age_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_age_classes)  #output age
        )


        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_gender_classes)  #output gender
        )

        self.race_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_race_classes)  # output race
        )

    def forward(self, x):
        shared_features = self.base_model(x)

        age_output = self.age_head(shared_features)
        gender_output = self.gender_head(shared_features)
        race_output = self.race_head(shared_features)
        return {'age': age_output, 'gender': gender_output, 'race': race_output}

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(EmotionClassifier, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)  
        )
        

    def forward(self, x):
        return self.base_model(x)


num_age_classes = 4
num_gender_classes = len(gender_mapping)
num_race_classes = len(race_mapping)
num_classes = 4  
model2 = EmotionClassifier(num_classes=num_classes)
model = FaceAttributeModel(num_age_classes, num_gender_classes, num_race_classes)
model = model.to(device)
model2 = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)

def train(model, dataloader, criterion, optimizer, epoch, print_freq=100):
    model.train()
    running_loss = 0.0
    total_samples = 0

    # Initialize variables to keep track of individual losses
    running_loss_age = 0.0
    running_loss_gender = 0.0
    running_loss_race = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()

        outputs = model(images)

        loss_age = criterion(outputs['age'], labels['age'])
        loss_gender = criterion(outputs['gender'], labels['gender'])
        loss_race = criterion(outputs['race'], labels['race'])

        loss = loss_age + loss_gender + loss_race

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Accumulate individual losses
        running_loss_age += loss_age.item() * batch_size
        running_loss_gender += loss_gender.item() * batch_size
        running_loss_race += loss_race.item() * batch_size

        # Print training information every 'print_freq' batches
        if (batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == len(dataloader):
            print(f'Epoch [{epoch+1}], Batch [{(batch_idx+1)*batch_size}/{(len(dataloader)*batch_size)}], ')
            print(f'Batch Loss: {loss.item():.4f} (Age: {loss_age.item():.4f}, Gender: {loss_gender.item():.4f}, Race: {loss_race.item():.4f})')

    epoch_loss = running_loss / total_samples
    epoch_loss_age = running_loss_age / total_samples
    epoch_loss_gender = running_loss_gender / total_samples
    epoch_loss_race = running_loss_race / total_samples

    print(f'Epoch [{epoch+1}] Training Loss: {epoch_loss:.4f}')
    print(f' - Age Loss: {epoch_loss_age:.4f}, Gender Loss: {epoch_loss_gender:.4f}, Race Loss: {epoch_loss_race:.4f}')

    return epoch_loss

def train2(model2, dataloader, criterion, optimizer2, epoch, print_freq=100):
    model2.train()
    running_loss = 0.0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer2.zero_grad()

        outputs = model2(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % print_freq == 0:
            print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / total_samples
    print(f'Epoch [{epoch+1}] Training Loss: {epoch_loss:.4f}')
    return epoch_loss

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    correct_age = 0
    correct_gender = 0
    correct_race = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            outputs = model(images)

            loss_age = criterion(outputs['age'], labels['age'])
            loss_gender = criterion(outputs['gender'], labels['gender'])
            loss_race = criterion(outputs['race'], labels['race'])

            loss = loss_age + loss_gender + loss_race

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            _, age_preds = torch.max(outputs['age'], 1)
            _, gender_preds = torch.max(outputs['gender'], 1)
            _, race_preds = torch.max(outputs['race'], 1)

            correct_age += (age_preds == labels['age']).sum().item()
            correct_gender += (gender_preds == labels['gender']).sum().item()
            correct_race += (race_preds == labels['race']).sum().item()

    epoch_loss = running_loss / total_samples

    age_acc = correct_age / total_samples
    gender_acc = correct_gender / total_samples
    race_acc = correct_race / total_samples

    return epoch_loss, age_acc, gender_acc, race_acc

def validate2(model2, dataloader, criterion):
    model2.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model2(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

def predict(model, image_path, transform=transform):
    model.eval()
    if type(image_path) == "str":
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
    else:
        image = transform(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

        _, age_pred = torch.max(outputs['age'], 1)
        _, gender_pred = torch.max(outputs['gender'], 1)
        _, race_pred = torch.max(outputs['race'], 1)

    age_group = age_pred.item()
    gender = gender_pred.item()
    race = race_pred.item()

    age_label = age_group_transform(age_group)
    gender_label = [k for k, v in gender_mapping.items() if v == gender][0]
    race_label = [k for k, v in race_mapping.items() if v == race][0]

    return age_label, gender_label, race_label

def predict2(model2, image_path, transform2=val_transform2):
    model2.eval()
    if type(image_path) == str:
        image2 = Image.open(image_path).convert('L')
        image2 = transform2(image2).unsqueeze(0).to(device)
    else:
        image2 = transform2(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model2(image2)
        _, emo_pred = torch.max(outputs, 1)
        return emo_pred.item()

if __name__ == '__main__':
    num_epochs = 50
    best_val_acc = 0.0  # Initialize best validation accuracy
    start_epoch = 0  # Starting epoch

    last_checkpoint = get_resource_path('CV/best_face_attribute_model.pth')
    best_model_path = get_resource_path('CV/best_face_attribute_model.pth')

    if os.path.exists(last_checkpoint):
        print('Loading checkpoint...')
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = checkpoint['epoch'] + 1  # the next epoch to train
        print(f'Resuming training from epoch {start_epoch}')
        print(f'Current Avg Acc: {best_val_acc}')
    else:
        print('No checkpoint found, starting training from scratch.')

    # NOTICE: num_epochs is an accumulative value, represents for total number of epochs
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, print_freq=1)
        val_loss, age_acc, gender_acc, race_acc = validate(model, val_loader, criterion)

        # Calculate average validation accuracy
        avg_val_acc = (age_acc + gender_acc + race_acc) / 3

        print(f'\n\nEpoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}\n\n')
        print(f' - Age Acc: {age_acc:.4f}, Gender Acc: {gender_acc:.4f}, Race Acc: {race_acc:.4f}')
        print(f' - Average Validation Accuracy: {avg_val_acc:.4f}')

        # Check if this is the best model so far
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }
            # Save the best model
            torch.save(checkpoint, best_model_path)
            print('Best model updated.')

    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }
    # Save the final model after training
    torch.save(checkpoint, 'final_checkpoint.pth')
    print("Final model saved")