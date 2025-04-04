import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image = np.uint8(255 * image)
    superimposed_img = heatmap * alpha + image
    return superimposed_img

if __name__ == "__main__":
    from preprocess import preprocess_image
    model = load_model("models/mobilenet_pneumonia.keras")  # Updated extension
    img_path = "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
    img_array = preprocess_image(img_path)[np.newaxis, ...]
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv_pw_13")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    superimposed_img = overlay_heatmap(heatmap, img)
    cv2.imwrite("gradcam_output.jpg", superimposed_img)