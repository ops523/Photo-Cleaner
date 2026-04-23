import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import requests
import time

st.set_page_config(layout="wide")
st.title("🧱 Wall Image Cleaner (Hugging Face Only)")

# ---------------------------
# Load YOLO
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------
# Hugging Face Config
# ---------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

# ---------------------------
# Hugging Face Call (Retry Logic Added)
# ---------------------------
def call_huggingface(image_bytes, mask_bytes):

    for attempt in range(3):  # retry up to 3 times
        try:
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                files={
                    "image": ("image.png", image_bytes, "image/png"),
                    "mask": ("mask.png", mask_bytes, "image/png"),
                },
                data={
                    "prompt": "clean empty wall, natural background, no vehicles, no people, realistic texture",
                },
                timeout=90
            )

            # Model loading (common HF issue)
            if response.status_code == 503:
                time.sleep(5)
                continue

            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))

            else:
                st.warning(f"HF Error: {response.status_code}")
                return None

        except Exception as e:
            time.sleep(3)

    return None


# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    st.subheader("Select Flex Area (Protected Region)")

    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("x1", 0, w, 0)
        y1 = st.number_input("y1", 0, h, 0)
    with col2:
        x2 = st.number_input("x2", 0, w, w)
        y2 = st.number_input("y2", 0, h, h)

    # Preview
    preview = img_np.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0,255,0), 2)
    st.image(preview, caption="Flex Protected", width="stretch")

    if st.button("🚀 Process Image"):

        with st.spinner("Processing with Hugging Face..."):

            # ---------------------------
            # Object Detection
            # ---------------------------
            results = model(img_np)
            mask = np.zeros((h, w), dtype=np.uint8)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    if label in ["person", "car", "motorcycle", "truck", "bus"]:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                        pad = 20
                        x_min = max(0, x_min - pad)
                        y_min = max(0, y_min - pad)
                        x_max = min(w, x_max + pad)
                        y_max = min(h, y_max + pad)

                        if not (x_min > x1 and y_min > y1 and x_max < x2 and y_max < y2):
                            mask[y_min:y_max, x_min:x_max] = 255

            mask_pil = Image.fromarray(mask)

            # ---------------------------
            # Convert to Bytes
            # ---------------------------
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")

            mask_bytes = io.BytesIO()
            mask_pil.save(mask_bytes, format="PNG")

            img_bytes = img_bytes.getvalue()
            mask_bytes = mask_bytes.getvalue()

            # ---------------------------
            # Call Hugging Face
            # ---------------------------
            result_image = call_huggingface(img_bytes, mask_bytes)

            if result_image is None:
                st.error("❌ Hugging Face failed. Try again or adjust mask.")
                st.stop()

            result_np = np.array(result_image)

            # ---------------------------
            # Angle Variation
            # ---------------------------
            pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            shift = 15
            pts2 = np.float32([
                [shift,0],
                [w-shift,10],
                [0,h-10],
                [w,h]
            ])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(result_np, matrix, (w, h))

            # ---------------------------
            # Display
            # ---------------------------
            st.subheader("Results")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", width="stretch")
            with col2:
                st.image(warped, caption="Final Output", width="stretch")

            # ---------------------------
            # Download
            # ---------------------------
            buf = io.BytesIO()
            Image.fromarray(warped).save(buf, format="PNG")

            st.download_button(
                "⬇️ Download Image",
                data=buf.getvalue(),
                file_name="final_output.png",
                mime="image/png"
            )
