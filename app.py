import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import random

st.set_page_config(layout="wide")
st.title("🧱 Wall Image Variations Generator (No AI)")

# ---------------------------
# Image Transform Functions
# ---------------------------

def random_crop(img):
    w, h = img.size
    crop_percent = random.uniform(0.85, 0.95)

    new_w = int(w * crop_percent)
    new_h = int(h * crop_percent)

    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)

    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h))


def color_adjust(img):
    brightness = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    contrast = ImageEnhance.Contrast(brightness).enhance(random.uniform(0.9, 1.1))
    color = ImageEnhance.Color(contrast).enhance(random.uniform(0.9, 1.1))
    return color


def add_noise(img):
    np_img = np.array(img).astype(np.int16)

    noise = np.random.normal(0, 5, np_img.shape)
    noisy = np_img + noise

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def slight_blur(img):
    if random.random() > 0.5:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return img


def perspective_shift(img):
    w, h = img.size

    shift = random.randint(5, 20)

    coeffs = (
        1, shift / w, 0,
        shift / h, 1, 0
    )

    return img.transform((w, h), Image.AFFINE, coeffs)


# ---------------------------
# Main Processing Pipeline
# ---------------------------

def generate_variation(image):
    img = image.copy()

    img = random_crop(img)
    img = perspective_shift(img)
    img = color_adjust(img)
    img = add_noise(img)
    img = slight_blur(img)

    return img


# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, width="stretch")

    num_variations = st.slider("Number of Variations", 1, 6, 3)

    if st.button("🚀 Generate Variations"):

        st.subheader("Generated Outputs")

        cols = st.columns(num_variations)

        outputs = []

        for i in range(num_variations):
            result = generate_variation(image)
            outputs.append(result)

            with cols[i]:
                st.image(result, caption=f"Variation {i+1}", width="stretch")

        # ---------------------------
        # Download ZIP
        # ---------------------------
        import zipfile

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for i, img in enumerate(outputs):
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                zip_file.writestr(f"variation_{i+1}.png", img_bytes.getvalue())

        st.download_button(
            "⬇️ Download All Variations (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="wall_variations.zip",
            mime="application/zip"
        )
