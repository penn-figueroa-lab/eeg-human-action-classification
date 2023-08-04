# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import json
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
def plot_confusion_matrix(cm, classes, save_path):
    """
    Plots the confusion matrix and saves it to an image file.
    """
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # Save the confusion matrix as an image
    plt.savefig(save_path)
    plt.close()


# Function to flatten the 3D data to 2D
def flatten_data(data):
    return data.reshape(data.shape[0], -1)

X_train = np.load('x_train_HN.npy')
X_val = np.load('x_val_HN.npy')
X_test = np.load('x_test_HN.npy')
y_train = np.load('y_train_HN.npy')
y_val = np.load('y_val_HN.npy')
y_test = np.load('y_test_HN.npy')

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(flatten_data(X_train), dtype=torch.float32)
X_val_tensor = torch.tensor(flatten_data(X_val), dtype=torch.float32)
X_test_tensor = torch.tensor(flatten_data(X_test), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
batch_size = 64
train_dataset = TensorDataset(torch.cat((X_train_tensor, X_val_tensor)), torch.cat((y_train_tensor, y_val_tensor)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Random Forest classifier with best hyperparameters

random_forest_params = {    'n_estimators': 100}

# random_forest_params = {
#     'bootstrap': True,
#     'ccp_alpha': 0.0,
#     'class_weight': None,
#     'criterion': 'gini',
#     'max_depth': 10,
#     'max_features': 'sqrt',
#     'max_leaf_nodes': None,
#     'max_samples': None,
#     'min_impurity_decrease': 0.0,
#     'min_samples_leaf': 2,
#     'min_samples_split': 2,
#     'min_weight_fraction_leaf': 0.0,
#     'n_estimators': 100,
#     'n_jobs': None,
#     'oob_score': False,
#     'random_state': None,
#     'verbose': 0,
#     'warm_start': False
# }

random_forest_classifier = RandomForestClassifier(**random_forest_params)

# Train the Random Forest classifier
print("Training Random Forest...")
start_time = time.time()
random_forest_classifier.fit(flatten_data(X_train), y_train)
end_time = time.time()
print(f"Time taken for training Random Forest: {end_time - start_time} seconds")

# Predict using the trained Random Forest model
y_train_pred = random_forest_classifier.predict(flatten_data(X_train))
y_test_pred = random_forest_classifier.predict(flatten_data(X_test))

# Evaluate the Random Forest model
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("Random Forest - Confusion Matrix (Train):")
print(cm_train)

print("Random Forest - Confusion Matrix (Test):")
print(cm_test)

precision_train = precision_score(y_train, y_train_pred, average='weighted')
precision_test = precision_score(y_test, y_test_pred, average='weighted')

recall_train = recall_score(y_train, y_train_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print("Random Forest - Precision (Train):", precision_train)
print("Random Forest - Precision (Test):", precision_test)

print("Random Forest - Recall (Train):", recall_train)
print("Random Forest - Recall (Test):", recall_test)

print("Random Forest - Accuracy (Train):", accuracy_train)
print("Random Forest - Accuracy (Test):", accuracy_test)

plt.figure()
plot_confusion_matrix(cm_test, classes=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'],
                      save_path="RandomForest_confusion_matrix.png")

# Store the test case predictions and ground truth in the dictionary
test_results = {
    'RandomForest': {
        'predictions': y_test_pred.tolist(),
        'ground_truth': y_test.tolist()
    }
}

with open('test_results.json', 'w') as json_file:
    json.dump(test_results, json_file)
