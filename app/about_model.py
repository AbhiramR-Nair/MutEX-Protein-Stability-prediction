import streamlit as st
import pandas as pd
from pathlib import Path

def render_about_model():
    """Renders the 'About the Model' tab."""

    # --------------------------------------------------
    # HEADER & INTRO
    # --------------------------------------------------
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🧬 MutEX - Protein Stability (ΔΔG) Predictor")
        st.markdown("""
        **Protein Stability (ΔΔG) Prediction** using Stage-2 Siamese Networks.
        
        This model predicts whether a single amino acid mutation makes a protein **more stable (Negative ΔΔG)** or **less stable (Positive ΔΔG)**.
        """)
    
    with col2:
        st.success("##### Key Features")
        st.markdown("""
        - **Structure-Agnostic:** Uses ESM-2 Embeddings.
        - **Antisymmetric:** Checks Forward & Reverse consistency.
        - **Lightweight:** Runs on standard CPU.
        """)

    st.divider()

    # --------------------------------------------------
    # PERFORMANCE METRICS (New Section)
    # --------------------------------------------------
    st.header("📈 Model Performance")
    
    # 1. Prepare Data
    # Predictive Metrics
    perf_data = [
        {"Metric": "Pearson Correlation (r)", "Value": 0.714, "Description": "Linear correlation (Higher is better)"},
        {"Metric": "Spearman Rho (ρ)", "Value": 0.679, "Description": "Rank correlation (Higher is better)"},
        {"Metric": "RMSE (kcal/mol)", "Value": 0.713, "Description": "Root Mean Square Error (Lower is better)"},
        {"Metric": "MAE (kcal/mol)", "Value": 0.509, "Description": "Mean Absolute Error (Lower is better)"},
        {"Metric": "Binary Accuracy", "Value": 0.798, "Description": "Correct classification of Stable/Unstable"},
        {"Metric": "Test Loss", "Value": 0.221, "Description": "Model convergence loss"},
    ]
    df_perf = pd.DataFrame(perf_data)

    # Antisymmetry Metrics
    anti_data = [
        {"Metric": "Consistency Correlation", "Value": 0.858, "Description": "Correlation between Forward & Reverse"},
        {"Metric": "Mean Bias", "Value": -0.222, "Description": "Average drift from symmetry"},
        {"Metric": "Absolute Bias", "Value": 0.313, "Description": "Magnitude of symmetry error"},
    ]
    df_anti = pd.DataFrame(anti_data)

    # 2. Display Tables side-by-side
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🎯 Predictive Accuracy")
        st.dataframe(
            df_perf, 
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.NumberColumn("Value", format="%.3f"),
                "Description": st.column_config.TextColumn("Meaning", width="large"),
            },
            hide_index=True,
            use_container_width=True
        )

    with c2:
        st.subheader("⚖️ Antisymmetry Checks")
        st.markdown("*Measures if ΔΔG(A→B) equals -ΔΔG(B→A)*")
        st.dataframe(
            df_anti,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.NumberColumn("Value", format="%.3f"),
                "Description": st.column_config.TextColumn("Meaning", width="large"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------
    # ARCHITECTURE & EXPLANATION
    # --------------------------------------------------
    with st.expander("📌 Problem Statement", expanded=False):
        st.markdown("""
        Predicting **protein stability change (ΔΔG)** is critical for:
        - Protein Engineering
        - Drug Resistance Analysis
        
        Experimental methods are slow. This ML model provides a fast approximation using **Protein Language Models**.
        """)
        
    # Safe Image Loading
    current_dir = Path(__file__).parent
        
    with st.expander("🧠 Model Architecture", expanded=False):
        st.markdown("""
        **Stage-2 Architecture:**
        1. Phase 1: Wild-type and mutant sequences are encoded by a frozen ESM2-650M model 
            (layers 30–33 averaged) to obtain position-specific embeddings, from which Δ, |Δ|,
            cosine similarity, and L2 distance are deterministically derived and normalized.

        2. Phase 2: WT, Mut, Δ, and |Δ| embeddings are processed through a shared Light Attention 
            block with Siamese weights to emphasize mutation-relevant signals without introducing 
            architectural bias.

        3. The attended embeddings are concatenated with cosine similarity and L2 distance and 
            passed through a deep MLP (512 → 256 → 128 → 1) to predict ΔΔG.

        4. Training enforces thermodynamic consistency by combining Huber loss with an antisymmetry 
            regularization term that penalizes ΔΔG(WT→Mut) + ΔΔG(Mut→WT).
        """)
        model_arch_image_path = current_dir / "Model_architecture.png"
        if model_arch_image_path.exists():
            st.image(str(model_arch_image_path), caption="Model_architecture", use_container_width=True)
        else:
            st.info("Model architecture image not found in `/images` folder.")
            

    with st.expander("📊 Training Curves", expanded=False):
        st.markdown("**Training Loss vs Validation Loss:**")
        
        
        training_curve_path = current_dir / "training_curves_stage2 (1).png"
        
        if training_curve_path.exists():
            st.image(str(training_curve_path), caption="Training Curves", use_container_width=True)
        else:
            st.info("Training curves image not found in `/images` folder.")
            
        st.markdown("**Anti-symmetry Evaluation Plots:**")
        
        
        eveluation_plots_path = current_dir / "evaluation_plots_stage2 (1).png"
        
        if eveluation_plots_path.exists():
            st.image(str(eveluation_plots_path), caption="Evaluation Plots", use_container_width=True)
        else:
            st.info("Training curves image not found in `/images` folder.")

    st.divider()
    st.caption("Developed for Research Purposes.")