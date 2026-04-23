import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import torch
from diffusers import StableDiffusionInpaintPipeline

st.set_page_config(layout="wide")
st.title("🧱 Wall Image Cleaner V2 (AI Powered)")

# Load YOLO
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

# Load Inpainting Model
@st.cache_resource
def load_inpaint_model():
    return StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

yolo = load_yolo()
pipe = load_inpaint_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    st.subheader("Select Flex Area")

    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("x1", 0, w, 0)
        y1 = st.number_input("y1", 0, h, 0)
    with col2:
        x2 = st.number_input("x2", 0, w, w)
        y2 = st.number_input("y2", 0, h, h)

    preview = img_np.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0,255,0), 2)
    st.image(preview, caption="Flex Protected", width="stretch")

    if st.button("🚀 Process"):

        with st.spinner("AI Processing..."):

            results = yolo(img_np)
            mask = np.zeros((h, w), dtype=np.uint8)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = yolo.names[cls]

                    if label in ["person", "car", "motorcycle", "truck", "bus"]:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                        pad = 15
                        x_min = max(0, x_min - pad)
                        y_min = max(0, y_min - pad)
                        x_max = min(w, x_max + pad)
                        y_max = min(h, y_max + pad)

                        if not (x_min > x1 and y_min > y1 and x_max < x2 and y_max < y2):
                            mask[y_min:y_max, x_min:x_max] = 255

            # Convert to PIL
            mask_pil = Image.fromarray(mask).convert("RGB")

            # AI Inpainting
            result = pipe(
                prompt="clean wall, natural background, no vehicles, no people",
                image=image,
                mask_image=mask_pil
            ).images[0]

            result_np = np.array(result)

            # 🎯 Camera Angle Variation (Perspective Warp)
            pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            shift = 20
            pts2 = np.float32([
                [0+shift,0],
                [w-shift,0+10],
                [0,h-10],
                [w,h]
            ])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(result_np, matrix, (w, h))

            st.subheader("Results")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", width="stretch")
            with col2:
                st.image(warped, caption="AI Cleaned + Angle Variation", width="stretch")

            # Download
            buf = io.BytesIO()
            Image.fromarray(warped).save(buf, format="PNG")

            st.download_button(
                "⬇️ Download",
                data=buf.getvalue(),
                file_name="final.png",
                mime="image/png"
            )
