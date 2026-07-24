import streamlit as st
import os
import sys

# Force Python to recognize the absolute root directory first
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from PIL import Image
import io
import base64

# Now try importing the core layers safely
try:
    from core.gradcam import generate_gradcam
    from backend.llm_service import generate_medical_report
    from backend.database import get_db, PredictionHistory, engine
    from sqlalchemy.orm import Session
except ImportError as e:
    st.error(f"📍 Path resolution failure. Current Sys Path: {sys.path}. Error details: {str(e)}")

# Initialize database tables if they do not exist on the cloud instance
try:
    from backend.database import Base
    Base.metadata.create_all(bind=engine)
except Exception as db_init_err:
    pass

# Page configurations
st.set_page_config(page_title="Advanced Medical AI Platform", layout="wide")

st.title("Advanced Medical AI Intelligence Platform")
st.write("Upload a chest X-Ray image to generate deep learning predictions, Grad-CAM visualizations, and AI medical reports.")

# --- SIDEBAR HISTORY ---
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
except Exception as db_err:
    st.sidebar.info("Local history logs unavailable.")

# --- MAIN INTERFACE: FILE UPLOADER ---
uploaded_file = st.file_uploader("📂 Drop your Chest X-Ray image here or click to browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read raw image bytes for processing
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
                # 1. Run direct core logic for prediction and Grad-CAM
                diagnosis, confidence, heatmap_bytes = generate_gradcam(file_bytes)
                
                # 2. Run direct report generation logic
                report_text = generate_medical_report(diagnosis, confidence)
                
                # 3. Save logs to the database directly
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
                except Exception as db_save_err:
                    st.warning("Analysis completed, but prediction could not be saved to history log.")

                # 4. Render UI Visualizations side-by-side
                with col2:
                    st.success(f"Prediction Complete: **{diagnosis}**")
                    st.metric(label="Model Confidence Score", value=f"{confidence * 100:.2f}%")
                    
                    if heatmap_bytes:
                        st.image(heatmap_bytes, caption="Grad-CAM Pathology Localization Heatmap", use_container_width=True)
                
                # 5. Display the Generated Medical Report
                st.divider()
                st.subheader("📝 AI-Assisted Clinical Radiology Report Draft")
                st.info("⚠️ WARNING: This report is dynamically drafted by an AI Assistant. It must be verified and signed off by a licensed radiologist before clinical applications.")
                st.text_area("Generated Draft Text (Editable)", value=report_text, height=300)
                
                # Rerun to update the sidebar history record list smoothly
                st.rerun()
                
            except Exception as e:
                st.error(f"Internal processing failed. Details: {str(e)}")

# --- TECHNICAL HEALTH CLINICAL DISCLAIMER ---
st.divider()
st.caption("ℹ️ **Medical Disclaimer:** This software platform is an AI-powered diagnostic demonstration built for research evaluation. It does not provide definitive medical advice or replace professional human radiological interpretation.")