import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
# Load the data
X_train = np.load('x_train_HN.npy')
X_val = np.load('x_val_HN.npy')
X_test = np.load('x_test_HN.npy')
y_train = np.load('y_train_HN.npy')
y_val = np.load('y_val_HN.npy')
y_test = np.load('y_test_HN.npy')

# Encoding the labels
# encoder = LabelEncoder()
# y_train_encoded = encoder.fit_transform(y_train)
# y_val_encoded = encoder.transform(y_val)
# y_test_encoded = encoder.transform(y_test)
# 

y_train_encoded = y_train
y_val_encoded =y_val
y_test_encoded =y_test


# Converting data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# Creating data loaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#changed suffle to false
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc = nn.Linear(32*32*40, 5)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # add a channel dimension
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)  # flatten the tensor
#         x = self.fc(x)
#         return x
# Define a larger CNN model with additional convolutional layers and more neurons in the fully connected layer
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # add a channel dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)  # ResNet-18 like architecture (4 layers with 2 blocks each)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 800
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
save_every = 50
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_correct = 0
    val_correct = 0
    train_preds = []
    val_preds = []

    model.train()  # Set the model to training mode
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move the data to the device that is used

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

        _, pred = torch.max(y_pred, dim=1)
        train_correct += torch.sum(pred == y_batch.data)
        train_preds.extend(pred.cpu().numpy())  # Collecting train predictions for confusion matrix

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradient to speed up computation
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)  # Move the data to the device that is used

            y_val_pred = model(X_val)
            loss = criterion(y_val_pred, y_val)
            val_loss += loss.item() * X_val.size(0)

            _, val_pred = torch.max(y_val_pred, dim=1)
            val_correct += torch.sum(val_pred == y_val.data)
            val_preds.extend(val_pred.cpu().numpy())  # Collecting validation predictions for confusion matrix

    # Calculate average losses and accuracy
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    train_acc = train_correct.double() / len(train_loader.dataset)
    val_acc = val_correct.double() / len(val_loader.dataset)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} \tTraining Acc: {:.4f} \tValidation Acc: {:.4f}'.format(
        epoch + 1, train_loss, val_loss, train_acc, val_acc))
    if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
        model_path = f"/home/satyajeet/Desktop/Classifier Testing/models/trained_resnet_model_epoch{epoch + 1}.pt"
        torch.save(model.state_dict(), model_path)
        print("Trained model saved at:", model_path)


model_path = "/home/satyajeet/Desktop/Classifier Testing/trained_resnet_model.pt"
torch.save(model.state_dict(), model_path)
print("Trained model saved at:", model_path)
# Calculate test predictions and metrics
model.eval()  # Set the model to evaluation mode
y_test_pred = torch.empty(0, dtype=torch.long).to(device)
with torch.no_grad():  # Do not calculate gradient to speed up computation
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)  # Move the data to the device that is used

        y_pred = model(X_test)
        _, pred = torch.max(y_pred, dim=1)
        y_test_pred = torch.cat([y_test_pred, pred])

y_test_pred = y_test_pred.cpu().numpy()

# Metrics for test dataset
print('Accuracy: ', accuracy_score(y_test_encoded, y_test_pred))
print('Precision: ', precision_score(y_test_encoded, y_test_pred, average='weighted'))
print('F1-score: ', f1_score(y_test_encoded, y_test_pred, average='weighted'))

# Confusion Matrix for test dataset
conf_mat_test = confusion_matrix(y_test_encoded, y_test_pred)
print('Test Confusion Matrix:\n', conf_mat_test)

# Print the final accuracy for train and validation
final_train_accuracy = accuracy_score(y_train_encoded, train_preds)
final_val_accuracy = accuracy_score(y_val_encoded, val_preds)
print('Final Train Accuracy:', final_train_accuracy)
print('Final Validation Accuracy:', final_val_accuracy)

# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.plot(history['train_acc'], label='Train Accuracy')
# plt.plot(history['val_acc'], label='Validation Accuracy')
# plt.title("Training and Validation Accuracy")
# plt.legend()
#
# plt.subplot(1,2,2)
# plt.plot(history['train_loss'], label='Train Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.title("Training and Validation Loss")
# plt.legend()
#
# plt.tight_layout()
# plt.show()
