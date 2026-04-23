import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

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

    st.subheader("Step 1: Draw Flex Area (Protected Region)")

    # Simple manual input (coordinates)
    h, w, _ = img_np.shape

    col1, col2 = st.columns(2)

    with col1:
        x1 = st.number_input("x1", 0, w, 0)
        y1 = st.number_input("y1", 0, h, 0)

    with col2:
        x2 = st.number_input("x2", 0, w, w)
        y2 = st.number_input("y2", 0, h, h)

    # Show selected area
    preview = img_np.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0,255,0), 2)
    st.image(preview, caption="Selected Flex Area (Protected)", use_column_width=True)

    if st.button("🚀 Process Image"):

        with st.spinner("Processing..."):

            results = model(img_np)

            mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # Only remove these
                    if label in ["person", "car", "motorbike", "truck", "bus"]:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

                        # Skip if inside flex area
                        if not (x_min > x1 and y_min > y1 and x_max < x2 and y_max < y2):
                            mask[y_min:y_max, x_min:x_max] = 255

            # Inpaint
            inpainted = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)

            st.subheader("✅ Output")

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_np, caption="Original", use_column_width=True)
            with col2:
                st.image(inpainted, caption="Cleaned", use_column_width=True)

            # Download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))

            with open(temp_file.name, "rb") as f:
                st.download_button(
                    label="⬇️ Download Image",
                    data=f,
                    file_name="cleaned_image.png",
                    mime="image/png"
                )
