import streamlit as plt
import streamlit as st
import requests
from PIL import Image
import io
import base64

# Set up page configurations
st.set_page_config(page_title="Advanced Medical AI Platform", layout="wide")

st.title("Advanced Medical AI Intelligence Platform")
st.write("Upload a chest X-Ray image to generate deep learning predictions, Grad-CAM visualizations, and AI medical reports.")

# --- SIDEBAR HISTORY ---
st.sidebar.title("Patient Analysis History")
# Try fetching history from backend if available, otherwise show placeholder
try:
    # Adjust URL if your backend is hosted separately
    history_res = requests.get("http://127.0.0", timeout=2)
    if history_res.status_code == 200:
        records = history_res.json()
        for idx, rec in enumerate(records):
            st.sidebar.markdown(f"**Patient {idx+1}: {rec.get('filename')}**")
            st.sidebar.caption(f"Diagnosis: {rec.get('diagnosis')} ({rec.get('confidence')*100:.1f}%)")
            st.sidebar.divider()
    else:
        st.sidebar.info("No past records found in database.")
except Exception:
    st.sidebar.info("Backend database offline. Local history unavailable.")


# --- MAIN INTERFACE: FILE UPLOADER ---
uploaded_file = st.file_uploader("📂 Drop your Chest X-Ray image here or click to browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image immediately
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Chest X-Ray")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Analysis Controls")
        analyze_btn = st.button("🚀 Run Comprehensive Medical AI Diagnostics", type="primary")

    if analyze_btn:
        with st.spinner("Processing image through Deep Learning model, generating Grad-CAM, and querying LLM..."):
            try:
                # Convert image to bytes to send to API backend
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # Send request to FastAPI backend (Change URL if backend is deployed on Render/HuggingFace)
                files = {"file": (uploaded_file.name, img_bytes, f"image/{uploaded_file.type if hasattr(uploaded_file, 'type') else 'jpeg'}")}
                response = requests.post("http://127.0.0", files=files, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Update column 2 with Grad-CAM heatmap result
                    with col2:
                        st.success(f"Prediction Complete: **{data.get('diagnosis')}**")
                        st.metric(label="Model Confidence Score", value=f"{data.get('confidence') * 100:.2f}%")
                        
                        # Decode and display the Grad-CAM base64 string image
                        heatmap_encoded = data.get("heatmap_bytes")
                        if heatmap_encoded:
                            heatmap_bytes = base64.b64decode(heatmap_encoded)
                            st.image(heatmap_bytes, caption="Grad-CAM Pathology Localization Heatmap", use_column_width=True)
                    
                    # Display the AI Generated Report below columns
                    st.divider()
                    st.subheader("📝 AI-Assisted Clinical Radiology Report Draft")
                    st.info("⚠️ WARNING: This report is dynamically drafted by an AI Assistant. It must be verified and signed off by a licensed radiologist before clinical applications.")
                    st.text_area("Generated Draft Text (Editable)", value=data.get("report"), height=300)
                    
                else:
                    st.error(f"Backend API error returned code: {response.status_code}")
            except Exception as e:
                st.error(f"Could not reach backend API server. Details: {str(e)}")
