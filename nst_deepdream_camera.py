
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import os
from tkinter import Tk, filedialog

def list_styles_gui():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Bir stil görseli seçin",
        filetypes=[("Görsel dosyaları", "*.jpg *.png")]
    )
    return file_path

print("NST modeli yükleniyor...")
nst_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("NST modeli yüklendi.")

print("Deep Dream modeli yükleniyor...")
dream_base = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
dream_layers = ['mixed3', 'mixed5']
dream_model = tf.keras.Model(inputs=dream_base.input,
                             outputs=[dream_base.get_layer(name).output for name in dream_layers])
print("Deep Dream modeli yüklendi.")

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    return tf.reduce_sum([tf.reduce_mean(act) for act in layer_activations])

@tf.function
def deep_dream_step(img, model, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    grads = tape.gradient(loss, img)
    grads /= tf.math.reduce_std(grads) + 1e-8
    img += grads * step_size
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

def run_deep_dream(img, steps=30, step_size=0.01):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    for _ in range(steps):
        img = deep_dream_step(img, dream_model, step_size)
    return img.numpy()

def add_labels(triple_image, labels=("NST", "Cam-Dream", "Style-Dream")):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1
    width = triple_image.shape[1] // 3
    for i, label in enumerate(labels):
        x = i * width + 10
        cv2.putText(triple_image, label, (x, 25), font, font_scale, color, thickness, cv2.LINE_AA)
    return triple_image

def run_application():
    while True:
        style_path = list_styles_gui()
        style_img = cv2.imread(style_path)
        style_img = cv2.resize(style_img, (256, 256))
        style_img_rgb = style_img[..., ::-1] / 255.0
        style_tensor = tf.constant(style_img_rgb[np.newaxis, ...], dtype=tf.float32)

        print("Deep Dream uygulanıyor (stil görseline)...")
        deep_dream_result = run_deep_dream(style_img_rgb, steps=40, step_size=0.01)
        deep_dream_bgr = (deep_dream_result * 255).astype(np.uint8)[..., ::-1]

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Kamera açılamadı.")
            return

        print("Kamera aktif. Tuşlar: [s] Kaydet | [q] Çık | [r] Stil değiştir")

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        stylized_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        cam_dream_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        frame_count = 0
        update_interval = 5

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (256, 256))
            frame_rgb = frame_resized[..., ::-1] / 255.0
            content_tensor = tf.constant(frame_rgb[np.newaxis, ...], dtype=tf.float32)

            frame_count += 1
            if frame_count % update_interval == 0:
                stylized_tensor = nst_model(content_tensor, style_tensor)[0][0]
                stylized_rgb = stylized_tensor.numpy()
                stylized_bgr = (stylized_rgb * 255).astype(np.uint8)[..., ::-1]

                cam_dream_result = run_deep_dream(frame_rgb, steps=15, step_size=0.01)
                cam_dream_bgr = (cam_dream_result * 255).astype(np.uint8)[..., ::-1]

            combined = np.concatenate((stylized_bgr, cam_dream_bgr, deep_dream_bgr), axis=1)
            combined = add_labels(combined)

            cv2.imshow("🎨 NST + 📷 DeepDream (Cam & Stil)", combined)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key & 0xFF == ord('s'):
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"combo_output_{now}.jpg")
                cv2.imwrite(filename, combined)
                print(f"📸 Kaydedildi: {filename}")
            elif key & 0xFF == ord('r'):
                print("Stil değiştiriliyor...")
                cap.release()
                cv2.destroyAllWindows()
                break

run_application()
