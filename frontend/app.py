import streamlit as st
import os
import sys

# 🔥 CRITICAL FIX: Direct absolute workspace path pinning
# This forces Streamlit to view the full repository folder before running any imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from PIL import Image
import io
import base64

# Safely resolve backend and core modules within the updated system path
try:
    from core.gradcam import generate_gradcam
    from backend.llm_service import generate_medical_report
    from backend.database import get_db, PredictionHistory, engine
    from sqlalchemy.orm import Session
except ImportError as e:
    st.error(f"📍 System Path Resolution Error: {str(e)}")
    st.stop()

# Initialize database tables on startup if they do not exist
try:
    from backend.database import Base
    Base.metadata.create_all(bind=engine)
except Exception:
    pass

# Streamlit UI Page configurations
st.set_page_config(page_title="Advanced Medical AI Platform", layout="wide")

st.title("Advanced Medical AI Intelligence Platform")
st.write("Upload a chest X-Ray image to generate deep learning predictions, Grad-CAM visualizations, and AI medical reports.")

# --- SIDEBAR PATIENT LOG HISTORY ---
st.sidebar.title("Patient Analysis History")

try:
    db: Session = next(get_db())
    records = db.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).all()
    
    if records:
        for idx, rec in enumerate(records):
            st.sidebar.markdown(f"**Patient {idx+1}: {rec.filename}**")
            st.sidebar.caption(f"Diagnosis: {rec.diagnosis} ({rec.confidence * 100:.1f}%)")
            st.sidebar.divider()
    else:
        st.sidebar.info("No past records found in database.")
except Exception:
    st.sidebar.info("Local history logs currently unavailable.")

# --- MAIN BLOCK INTERFACE: FILE UPLOADER ---
uploaded_file = st.file_uploader("📂 Drop your Chest X-Ray image here or click to browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Chest X-Ray")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Analysis Controls")
        analyze_btn = st.button("🚀 Run Comprehensive Medical AI Diagnostics", type="primary")

    if analyze_btn:
        with st.spinner("Processing image through Deep Learning model, generating Grad-CAM, and querying LLM..."):
            try:
                # 1. Evaluate image through core PyTorch models
                diagnosis, confidence, heatmap_bytes = generate_gradcam(file_bytes)
                
                # 2. Trigger the local clinical report text engine
                report_text = generate_medical_report(diagnosis, confidence)
                
                # 3. Log results to local SQLite database tracking schema
                try:
                    db: Session = next(get_db())
                    db_record = PredictionHistory(
                        filename=uploaded_file.name,
                        diagnosis=diagnosis,
                        confidence=float(confidence),
                        report=report_text
                    )
                    db.add(db_record)
                    db.commit()
                except Exception:
                    pass

                # 4. Display matching output panels side-by-side
                with col2:
                    st.success(f"Prediction Complete: **{diagnosis}**")
                    st.metric(label="Model Confidence Score", value=f"{confidence * 100:.2f}%")
                    
                    if heatmap_bytes:
                        st.image(heatmap_bytes, caption="Grad-CAM Pathology Localization Heatmap", use_container_width=True)
                
                # 5. Populate structured clinical draft report
                st.divider()
                st.subheader("📝 AI-Assisted Clinical Radiology Report Draft")
                st.info("⚠️ WARNING: This report is dynamically drafted by an AI Assistant. It must be verified and signed off by a licensed radiologist before clinical applications.")
                st.text_area("Generated Draft Text (Editable)", value=report_text, height=300)
                
                # Instantly refresh panel states to populate the dashboard metrics seamlessly
                st.rerun()
                
            except Exception as e:
                st.error(f"Internal processing failed. Details: {str(e)}")

# --- REGULATORY CLINICAL DISCLAIMER ---
st.divider()
st.caption("ℹ️ **Medical Disclaimer:** This software platform is an AI-powered diagnostic demonstration built for research evaluation. It does not provide definitive medical advice or replace professional human radiological interpretation.")