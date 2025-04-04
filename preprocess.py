import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

def preprocess_uploaded_image(uploaded_file, target_size=(224, 224)):
    """
    Preprocess an image uploaded via a web interface:
    - Convert to grayscale, resize, convert back to RGB, and normalize.
    """
    try:
        image = Image.open(uploaded_file).convert("L")
    except Exception as e:
        print(f"Error opening uploaded image: {e}")
        return None
    img_array = np.array(image)
    try:
        img_array = cv2.resize(img_array, target_size)
    except Exception as e:
        print(f"Error resizing uploaded image: {e}")
        return None
    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_array = img_array.astype(np.float32) / 255.0
    return img_array

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image from a file path:
    - Read the image in grayscale, resize it, apply CLAHE, convert to RGB, and normalize.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image at {image_path}")
        return None
    
    try:
        img = cv2.resize(img, target_size)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    except Exception as e:
        print(f"Error applying CLAHE to image {image_path}: {e}")
        return None

    try:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        print(f"Error converting image {image_path} to RGB: {e}")
        return None

    img = img.astype(np.float32) / 255.0
    return img

def get_dataset(data_dir, target_size=(224, 224), batch_size=32):
    """
    Creates a tf.data.Dataset from images stored in a directory.
    The directory should have one subdirectory per class.
    Returns:
      - dataset: a tf.data.Dataset yielding (image, label) pairs
      - class_names: sorted list of class names
      - total_samples: total number of image samples
    """
    # Get sorted class names
    class_names = sorted([cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))])
    class_indices = {cls: i for i, cls in enumerate(class_names)}
    
    # Build a list of (file_path, label) pairs
    file_list = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_dir):
            file_list.append((os.path.join(cls_dir, img_name), class_indices[cls]))
    
    def generator():
        for file_path, label in file_list:
            img = preprocess_image(file_path, target_size)
            if img is not None:
                yield img, label

    output_signature = (
        tf.TensorSpec(shape=(target_size[0], target_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.shuffle(buffer_size=len(file_list))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, class_names, len(file_list)

def compute_class_weights_from_directory(data_dir):
    """
    Computes class weights based on the number of images per class in the directory.
    Returns a dictionary mapping class index to weight.
    """
    class_names = sorted([cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))])
    total = 0
    counts = {}
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        count = len(os.listdir(cls_dir))
        counts[cls] = count
        total += count
    class_weights = {}
    for i, cls in enumerate(class_names):
        # Weight = total_samples / (num_classes * samples_in_class)
        class_weights[i] = total / (len(class_names) * counts[cls])
    return class_weights

if __name__ == "__main__":
    # Example usage for testing:
    train_dir = "data/chest_xray/train/"
    dataset, class_names, num_samples = get_dataset(train_dir)
    print(f"Loaded dataset with {num_samples} images for classes: {class_names}")
    for images, labels in dataset.take(1):
        print("Batch images shape:", images.shape)
        print("Batch labels shape:", labels.shape)
