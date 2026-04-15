import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Pneumonia Detection", layout="wide")

MODEL_PATH = "backend/pneumonia_efficientnet.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

st.markdown(
    "<h1 style='text-align:center;'>🩺 Pneumonia Detection</h1>",
    unsafe_allow_html=True
)

# 🚨 If model missing → stop app cleanly
if model is None:
    st.error("❌ Model file not found!")
    st.markdown(
        "👉 Please download the model and place it inside `backend/` folder.\n\n"
        "Then restart the app."
    )
    st.stop()

left, right = st.columns([1, 1.2])

# -----------------------------
# LEFT
# -----------------------------
with left:
    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=420)

# -----------------------------
# RIGHT
# -----------------------------
with right:
    if uploaded_file:
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            prob = torch.sigmoid(output).item()

        pneumonia_conf = prob * 100
        normal_conf = (1 - prob) * 100

        if prob > 0.5:
            st.error(f"⚠️ Pneumonia ({pneumonia_conf:.1f}%)")
            st.markdown("• Lung opacity  \n• Patchy regions")
        else:
            st.success(f"✅ Normal ({normal_conf:.1f}%)")
            st.markdown("• Clear lungs  \n• No abnormalities")

        st.markdown("### Confidence")

        gcol1, gcol2, gcol3 = st.columns([1,2,1])

        with gcol2:
            fig, ax = plt.subplots(figsize=(3.5,2))
            ax.bar(["Normal", "Pneumonia"], [normal_conf, pneumonia_conf])
            ax.set_ylim(0, 100)
            ax.set_ylabel("%")

            st.pyplot(fig, use_container_width=False)
    else:
        st.info("Upload an image")