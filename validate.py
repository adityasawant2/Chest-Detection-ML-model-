import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from preprocess import get_dataset  # Updated to import get_dataset
import os

# Define test directory and batch size
test_dir = "data/chest_xray/test/"
batch_size = 32

# Load test dataset using the generator-based approach
test_dataset, class_names, num_test_samples = get_dataset(test_dir, batch_size=batch_size)
num_classes = len(class_names)
print(f"Loaded {num_test_samples} test images for classes: {class_names}")

# Load the trained model
model = tf.keras.models.load_model("models/mobilenet_pneumonia.keras")

# Accumulate predictions and true labels from the test dataset
y_true = []
y_pred = []
predictions_all = []

for batch_images, batch_labels in test_dataset:
    preds = model.predict(batch_images)
    predictions_all.append(preds)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(batch_labels.numpy())

# Concatenate predictions from all batches
predictions_all = np.concatenate(predictions_all, axis=0)
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ---------------------------
# Generate Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# Generate ROC Curves
# ---------------------------
# Binarize the true labels for ROC computation
y_true_bin = label_binarize(y_true, classes=range(num_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions_all[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2,
             label=f"ROC curve for {class_names[i]} (area = {roc_auc[i]:0.2f})")

# Plot the random chance line
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.legend(loc="lower right")
plt.show()
