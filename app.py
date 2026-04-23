import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import requests
import time

st.set_page_config(layout="wide")
st.title("🧱 Wall Image Cleaner (Stable MVP - No CV2)")

# ---------------------------
# Hugging Face Config
# ---------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-inpainting"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

# ---------------------------
# Hugging Face Call
# ---------------------------
time.sleep(2)
def call_huggingface(image, mask):

    image = image.resize((768, 512))
    mask = mask.resize((768, 512))

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")

    payload = {
        "inputs": {
            "image": img_bytes.getvalue(),
            "mask": mask_bytes.getvalue(),
            "prompt": "clean empty wall, realistic concrete texture, natural lighting, no people, no vehicles, seamless background"
        }
    }

    for attempt in range(5):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json=payload,
                timeout=120
            )

            if response.status_code == 503:
                st.warning("⏳ Model loading...")
                time.sleep(6)
                continue

            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))

            else:
                st.error(f"HF Error: {response.status_code}")
                return None

        except:
            time.sleep(4)

    return None
# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size

    st.subheader("Step 1: Select FLEX (Protected Area)")

    col1, col2 = st.columns(2)
    with col1:
        fx1 = st.number_input("Flex x1", 0, w, 0)
        fy1 = st.number_input("Flex y1", 0, h, 0)
    with col2:
        fx2 = st.number_input("Flex x2", 0, w, w)
        fy2 = st.number_input("Flex y2", 0, h, h)

    st.subheader("Step 2: Select REMOVE Area (Object Region)")

    col3, col4 = st.columns(2)
    with col3:
        rx1 = st.number_input("Remove x1", 0, w, 0)
        ry1 = st.number_input("Remove y1", 0, h, 0)
    with col4:
        rx2 = st.number_input("Remove x2", 0, w, w)
        ry2 = st.number_input("Remove y2", 0, h, h)

    # Preview
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    # Green = flex
    draw.rectangle([fx1, fy1, fx2, fy2], outline="green", width=3)

    # Red = removal
    draw.rectangle([rx1, ry1, rx2, ry2], outline="red", width=3)

    st.image(preview, caption="Green = Protected | Red = Remove", width="stretch")

    if st.button("🚀 Process Image"):

        with st.spinner("Processing..."):

            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)

            # Only remove region (exclude flex overlap)
            for y in range(ry1, ry2):
                for x in range(rx1, rx2):
                    if not (fx1 <= x <= fx2 and fy1 <= y <= fy2):
                        mask[y, x] = 255

            mask_pil = Image.fromarray(mask)

            # Convert to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")

            mask_bytes = io.BytesIO()
            mask_pil.save(mask_bytes, format="PNG")

            result_image = call_huggingface(image, mask_pil)

            if result_image is None:
                st.error("❌ Hugging Face failed. Try again.")
                st.stop()

            # Display
            st.subheader("Results")

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", width="stretch")
            with col2:
                st.image(result_image, caption="Final Output", width="stretch")

            # Download
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")

            st.download_button(
                "⬇️ Download Image",
                data=buf.getvalue(),
                file_name="final_output.png",
                mime="image/png"
            )
