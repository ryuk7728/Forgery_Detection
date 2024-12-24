from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os
import cv2
import tensorflow as tf
from ultralytics import YOLO

MODEL_PATH = 'models\CNN.keras'
YOLO_MODEL_PATH = 'models\YOLO.pt'

try:
    forgery_model = tf.keras.models.load_model(MODEL_PATH)
    print("Forgery detection model loaded successfully.")
except Exception as e:
    print(f"Error loading forgery detection model: {e}")
    forgery_model = None

yolo_model = YOLO(YOLO_MODEL_PATH)


def perform_ela(image_path, output_path, quality=90):
    try:
        original_image = Image.open(image_path)
        temp_image_path = 'temp_image.jpg'
        original_image.save(temp_image_path, 'JPEG', quality=quality)
        compressed_image = Image.open(temp_image_path)

        if original_image.size != compressed_image.size:
            compressed_image = compressed_image.resize(original_image.size)
        if original_image.mode != compressed_image.mode:
            compressed_image = compressed_image.convert(original_image.mode)

        ela_image = ImageChops.difference(original_image, compressed_image)
        extrema = ela_image.getextrema()
        max_diff = max(max(extrema, key=lambda x: x if isinstance(x, tuple) else [x]))
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        ela_image.save(output_path)
        os.remove(temp_image_path)
        return output_path
    except Exception as e:
        print(f"Error in ELA processing: {e}")
        return None


def perform_forgery_inference(ela_image_path):
    try:
        img = cv2.imread(ela_image_path)
        resized = tf.image.resize(img, (256, 256))
        input_data = np.expand_dims(resized / 255.0, axis=0)
        yhat = forgery_model.predict(input_data)
        return float(yhat[0][0])
    except Exception as e:
        print(f"Error in forgery inference: {e}")
        return None


def perform_yolo_inference(image_path):
    try:
        results = yolo_model.predict(source=image_path, save=False)
        return results
    except Exception as e:
        print(f"Error in YOLO inference: {e}")
        return None


def overlay_bounding_boxes(image_path, yolo_results):
    try:
        img = cv2.imread(image_path)
        for result in yolo_results:
            boxes = result.boxes.xyxy
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        output_image_path = "output_image_with_boxes.jpg"
        cv2.imwrite(output_image_path, img)
        return output_image_path
    except Exception as e:
        print(f"Error in overlaying bounding boxes: {e}")
        return None


def detect_forgery(image_path):
    if forgery_model is None:
        print("Error: Forgery model not loaded.")
        return None

    ela_output_path = "ela_output.jpg"
    saved_ela_image_path = perform_ela(image_path, ela_output_path)

    if not saved_ela_image_path:
        print("ELA processing failed.")
        return None

    confidence_score = perform_forgery_inference(saved_ela_image_path)

    if confidence_score is None:
        print("Inference failed.")
        return None

    if confidence_score < 0.5:
        print("Forgery detected.")
        yolo_results = perform_yolo_inference(saved_ela_image_path)
        if yolo_results is None:
            print("YOLO inference failed.")
            return None
        output_image_path = overlay_bounding_boxes(image_path, yolo_results)
        print(f"Forgery bounding boxes saved at: {output_image_path}")
    else:
        print("No forgery detected.")
    os.remove(saved_ela_image_path)


if __name__ == "__main__":
    image_path = 'test_data\\forged\\2t.jpg'
    if not os.path.exists(image_path):
        print("Error: File not found.")
    else:
        detect_forgery(image_path)
