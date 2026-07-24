import io
import os
import sys

import streamlit as st
from PIL import Image

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from backend.database import PredictionHistory, engine, get_db
    from backend.llm_service import generate_medical_report
    from core.gradcam import generate_gradcam
    from sqlalchemy.orm import Session
except ImportError as exc:
    st.set_page_config(page_title="Advanced Medical AI Platform", layout="wide")
    st.error(f"Workspace setup failed: {exc}")
    st.stop()

try:
    from backend.database import Base

    Base.metadata.create_all(bind=engine)
except Exception:
    pass


def _load_history() -> list[PredictionHistory]:
    try:
        db: Session = next(get_db())
        return db.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).all()
    except Exception:
        return []


def main() -> None:
    st.set_page_config(page_title="Advanced Medical AI Platform", layout="wide")

    st.title("Advanced Medical AI Intelligence Platform")
    st.write(
        "Upload a chest X-Ray image to generate deep learning predictions, Grad-CAM visualizations, and AI medical reports."
    )

    st.sidebar.title("Patient Analysis History")
    records = _load_history()
    if records:
        for idx, rec in enumerate(records):
            st.sidebar.markdown(f"**Patient {idx + 1}: {rec.filename}**")
            st.sidebar.caption(f"Diagnosis: {rec.diagnosis} ({rec.confidence * 100:.1f}%)")
            st.sidebar.divider()
    else:
        st.sidebar.info("No past records found in database.")

    uploaded_file = st.file_uploader(
        "📂 Drop your Chest X-Ray image here or click to browse",
        type=["jpg", "jpeg", "png"],
    )

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
            with st.spinner("Processing image through the analysis pipeline..."):
                try:
                    diagnosis, confidence, heatmap_bytes = generate_gradcam(file_bytes)
                    report_text = generate_medical_report(diagnosis, confidence)

                    try:
                        db: Session = next(get_db())
                        db_record = PredictionHistory(
                            filename=uploaded_file.name,
                            diagnosis=diagnosis,
                            confidence=float(confidence),
                            report=report_text,
                        )
                        db.add(db_record)
                        db.commit()
                    except Exception:
                        st.warning("Analysis completed, but prediction could not be saved to history log.")

                    with col2:
                        st.success(f"Prediction Complete: **{diagnosis}**")
                        st.metric(label="Model Confidence Score", value=f"{confidence * 100:.2f}%")
                        if heatmap_bytes:
                            st.image(heatmap_bytes, caption="Grad-CAM Pathology Localization Heatmap", use_container_width=True)

                    st.divider()
                    st.subheader("📝 AI-Assisted Clinical Radiology Report Draft")
                    st.info(
                        "⚠️ WARNING: This report is dynamically drafted by an AI Assistant. It must be verified and signed off by a licensed radiologist before clinical applications."
                    )
                    st.text_area("Generated Draft Text (Editable)", value=report_text, height=300)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Internal processing failed. Details: {exc}")

    st.divider()
    st.caption(
        "ℹ️ **Medical Disclaimer:** This software platform is an AI-powered diagnostic demonstration built for research evaluation."
    )


if __name__ == "__main__":
    main()
