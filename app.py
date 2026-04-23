import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

st.set_page_config(layout="wide")
st.title("🧱 Wall Image Cleaner (MVP)")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload Wall Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("Step 1: Select Flex Area (Protected Region)")

    h, w, _ = img_np.shape

    col1, col2 = st.columns(2)

    with col1:
        x1 = st.number_input("x1", 0, w, 0)
        y1 = st.number_input("y1", 0, h, 0)

    with col2:
        x2 = st.number_input("x2", 0, w, w)
        y2 = st.number_input("y2", 0, h, h)

    # Draw selected area
    preview = img_np.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    st.image(preview, caption="Selected Flex Area (Protected)", width="stretch")

    if st.button("🚀 Process Image"):

        with st.spinner("Processing..."):

            results = model(img_np)

            mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

            st.subheader("🔍 Detected Objects")

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # Show detected labels
                    st.write(f"Detected: {label}")

                    # Objects to remove
                    if label in ["person", "car", "motorcycle", "truck", "bus"]:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                        # Add padding for better removal
                        pad = 10
                        x_min = max(0, x_min - pad)
                        y_min = max(0, y_min - pad)
                        x_max = min(w, x_max + pad)
                        y_max = min(h, y_max + pad)

                        # Skip if fully inside flex area
                        if not (x_min > x1 and y_min > y1 and x_max < x2 and y_max < y2):
                            mask[y_min:y_max, x_min:x_max] = 255

            # Show mask (debug)
            st.subheader("🧪 Removal Mask")
            st.image(mask, caption="White = Removed Areas", width="stretch")

            # Inpaint
            inpainted = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)

            st.subheader("✅ Output")

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_np, caption="Original", width="stretch")
            with col2:
                st.image(inpainted, caption="Cleaned", width="stretch")

            # Download (stable)
            img_pil = Image.fromarray(inpainted)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")

            st.download_button(
                label="⬇️ Download Image",
                data=buf.getvalue(),
                file_name="cleaned_image.png",
                mime="image/png"
            )
