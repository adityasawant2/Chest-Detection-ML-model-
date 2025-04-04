import tensorflow as tf
import math
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from preprocess import get_dataset, compute_class_weights_from_directory
import numpy as np

# --- Custom Rotation Function (No TensorFlow Addons) ---
def rotate_image(image, angle):
    """
    Rotate an image by a given angle (in radians) using tf.raw_ops.ImageProjectiveTransformV2.
    """
    # Get image shape (assuming image is rank 3: [height, width, channels])
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)
    
    # Compute the center of the image
    center_y = height / 2.0
    center_x = width / 2.0

    # Calculate rotation matrix components
    cos_val = tf.math.cos(angle)
    sin_val = tf.math.sin(angle)
    
    # Build the transformation matrix in flattened format (8 elements)
    a0 = cos_val
    a1 = -sin_val
    a2 = center_x - center_x * cos_val + center_y * sin_val
    a3 = sin_val
    a4 = cos_val
    a5 = center_y - center_x * sin_val - center_y * cos_val
    transform = [a0, a1, a2, a3, a4, a5, 0.0, 0.0]
    
    # Add a batch dimension: now shape becomes [1, height, width, channels]
    image_batch = tf.expand_dims(image, 0)
    
    # Apply the transformation using tf.raw_ops.ImageProjectiveTransformV2.
    output = tf.raw_ops.ImageProjectiveTransformV2(
        images=image_batch,
        transforms=[transform],
        output_shape=[image_shape[0], image_shape[1]],
        interpolation="BILINEAR"
    )
    # Remove the batch dimension
    output = tf.squeeze(output, 0)
    return output
# --- End of Custom Rotation Function ---

# Define directories and parameters
train_dir = "data/chest_xray/train/"
test_dir = "data/chest_xray/test/"
batch_size = 32

# Load datasets using the data generator to avoid memory overload
print("Loading training dataset...")
train_dataset, class_names, num_train_samples = get_dataset(train_dir, batch_size=batch_size)
print("Loading test dataset...")
test_dataset, _, num_test_samples = get_dataset(test_dir, batch_size=batch_size)

num_classes = len(class_names)
print("Classes:", class_names)

# Compute class weights from the training directory
class_weights = compute_class_weights_from_directory(train_dir)
print("Computed class weights before adjustment:", class_weights)

# Determine the index of the "NORMAL" class from the sorted list
try:
    normal_index = class_names.index("NORMAL")
except ValueError:
    normal_index = None
    print("Warning: 'NORMAL' class not found in class_names. Conditional augmentation and weight adjustment will not be applied.")

# Increase the weight for "NORMAL" by 50% if found
if normal_index is not None:
    class_weights[normal_index] *= 1.5
print("Adjusted class weights:", class_weights)

# Define augmentation function for training data with conditional logic
def augment(image, label):
    # If the image is labeled as "NORMAL", apply a lighter augmentation
    def light_augmentation():
        img = tf.image.random_flip_left_right(image)
        angle = tf.random.uniform([], -5, 5) * (math.pi / 180)
        img = rotate_image(img, angle)
        img = tf.image.random_brightness(img, max_delta=0.05)
        return img

    # Otherwise, apply full augmentation
    def heavy_augmentation():
        img = tf.image.random_flip_left_right(image)
        angle = tf.random.uniform([], -10, 10) * (math.pi / 180)
        img = rotate_image(img, angle)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img

    if normal_index is not None:
        image = tf.cond(
            tf.equal(label, normal_index),
            light_augmentation,
            heavy_augmentation
        )
    else:
        image = heavy_augmentation()

    return image, label

# Since get_dataset returns a batched dataset, unbatch before mapping and then re-batch.
train_dataset = train_dataset.unbatch().map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

# Build the model using MobileNet as the base
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model for initial training

# Add custom classification layers on top with dropout for regularization
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout to reduce overfitting
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set up callbacks for better training control
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint("models/best_mobilenet_pneumonia.keras", monitor='val_loss',
                    save_best_only=True, verbose=1)
]

# Initial training phase
print("Starting initial training...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Fine-tuning: Unfreeze more layers for better adaptation
print("Starting fine-tuning...")
base_model.trainable = True
# Unfreeze the latter 60% of the layers for deeper fine-tuning
for layer in base_model.layers[:int(len(base_model.layers) * 0.4)]:
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tuning training phase
history_fine = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

# Save the final model
model.save("models/mobilenet_pneumonia.keras")
