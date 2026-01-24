import streamlit as st
import pandas as pd

# Import modules
from model_utils import load_model_from_file
from processing_utils import (
    get_remote_embeddings, 
    run_inference_single, 
    format_results_table, 
    apply_table_styling
)
from about_model import render_about_model

# ==========================================
# PAGE CONFIG (Must be first)
# ==========================================
st.set_page_config(
    page_title="Abyssal ΔΔG Predictor", 
    layout="wide", 
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.title("Settings")
    
    st.markdown("### 🔌 Server Connection")
    # 🔴 UPDATE NGROK URL HERE
    API_URL = st.text_input(
        "GPU Server URL (Ngrok)", 
        value="https://your-ngrok-url.ngrok-free.app/extract",
        help="Paste the URL from your Colab notebook here."
    )
    
    st.divider()
    
    st.markdown("### 🤖 Model Status")
    MODEL_FILENAME = "best_model_final.pt"
    
    # Load Model with spinner in sidebar
    with st.spinner("Loading Model..."):
        model = load_model_from_file(MODEL_FILENAME)
        
    if model:
        st.success(f"Loaded: `{MODEL_FILENAME}`")
    else:
        st.error("Model not found!")
        
    st.info("Developed for ABYSSAL-Style Prediction task.")

# ==========================================
# MAIN UI HEADER
# ==========================================
st.title("🧬 MutEX - Protein Stability (ΔΔG) Predictor")
st.markdown("""
    <style>
    .big-font { font-size:18px !important; color: #555; }
    </style>
    <p class="big-font">
    Predict protein stability changes (ΔΔG) using Siamese Neural Networks and ESM-2 Embeddings.
    <br><b>Negative ΔΔG</b> = Stabilizing | <b>Positive ΔΔG</b> = Destabilizing
    </p>
""", unsafe_allow_html=True)

st.divider()

# Session State
if 'results_data' not in st.session_state:
    st.session_state.results_data = []

# Tabs with Icons
tab_input, tab_results, tab_about = st.tabs(["📝 New Prediction", "📊 Results & Export", "ℹ️ About Model"])

# Standard Amino Acids
AMINO_ACIDS = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

# ---------------------------------------------------------------------
# TAB 1: INPUT
# ---------------------------------------------------------------------
with tab_input:
    # Input Method Selection using stylized segments
    col_mode, _ = st.columns([2, 1])
    with col_mode:
        input_method = st.radio(
            "Select Input Method:", 
            ["Standard Code", "Manual Details", "Batch CSV"], 
            horizontal=True,
            label_visibility="collapsed"
        )
        st.caption(f"Selected Mode: **{input_method}**")

    # --- CONTAINER FOR INPUTS ---
    with st.container(border=True):
        
        # --- A: STANDARD CODE ---
        if input_method == "Standard Code":
            c1, c2 = st.columns([3, 1])
            seq_in = c1.text_area("Sequence (WT)", height=150, placeholder="MVLSPADKTN...", key="seq_a")
            mut_in = c2.text_input("Mutation Code", placeholder="A23V", key="mut_a", help="Format: WT-Pos-Mut (e.g., A23V)")
            
            st.markdown("---")
            if st.button("Run Prediction ⚡", type="primary", use_container_width=True, disabled=(model is None)):
                if seq_in and mut_in:
                    with st.spinner("🚀 Extracting Embeddings & Predicting..."):
                        emb = get_remote_embeddings(seq_in.strip(), mut_in.strip(), API_URL)
                        if emb:
                            fwd, rev = run_inference_single(model, emb)
                            score = (fwd - rev) / 2
                            err = abs(fwd + rev)
                            
                            st.session_state.results_data.append({
                                "sequence": seq_in, "mutation": mut_in,
                                "predicted_ddg": score, "antisymmetry_error": err,
                                "cosine_similarity": emb['cosine_similarity'],
                                "l2_distance": emb['l2_distance'],
                                "delta_embedding": emb['delta_embedding']
                            })
                            st.toast(f"✅ Prediction Complete: {score:.3f} kcal/mol")

        # --- B: MANUAL DETAILS ---
        elif input_method == "Manual Details":
            seq_in = st.text_area("Sequence (WT)", height=100, placeholder="MVLSPADKTN...", key="seq_b")
            
            c1, c2, c3 = st.columns(3)
            pos_in = c1.number_input("Position", min_value=1, value=1, help="1-based index")
            mut_res_in = c2.selectbox("Mutant AA", AMINO_ACIDS)
            
            # Dynamic WT display
            wt_res_display = "❓"
            if seq_in and len(seq_in) >= pos_in:
                wt_res_display = seq_in[pos_in - 1].upper()
            c3.metric("Wildtype AA", wt_res_display)

            st.markdown("---")
            if st.button("Run Prediction ⚡", type="primary", use_container_width=True, disabled=(model is None)):
                if not seq_in:
                    st.error("Please enter a sequence.")
                elif len(seq_in) < pos_in:
                    st.error("Position out of bounds.")
                else:
                    code = f"{wt_res_display}{pos_in}{mut_res_in}"
                    with st.spinner(f"🚀 Predicting for {code}..."):
                        emb = get_remote_embeddings(seq_in.strip(), code, API_URL)
                        if emb:
                            fwd, rev = run_inference_single(model, emb)
                            score = (fwd - rev) / 2
                            err = abs(fwd + rev)
                            st.session_state.results_data.append({
                                "sequence": seq_in, "mutation": code,
                                "predicted_ddg": score, "antisymmetry_error": err,
                                "cosine_similarity": emb['cosine_similarity'],
                                "l2_distance": emb['l2_distance'],
                                "delta_embedding": emb['delta_embedding']
                            })
                            st.toast(f"✅ Prediction Complete: {score:.3f}")

        # --- C: BATCH CSV ---
        elif input_method == "Batch CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            st.info("CSV must have columns: `sequence`, `mutation`")
            
            if uploaded_file and st.button("Run Batch Processing ⚡", type="primary", use_container_width=True):
                df = pd.read_csv(uploaded_file)
                if 'sequence' in df.columns and 'mutation' in df.columns:
                    progress_bar = st.progress(0)
                    for i, row in df.iterrows():
                        s, m = str(row['sequence']).strip(), str(row['mutation']).strip()
                        emb = get_remote_embeddings(s, m, API_URL)
                        if emb:
                            fwd, rev = run_inference_single(model, emb)
                            score = (fwd - rev) / 2
                            err = abs(fwd + rev)
                            st.session_state.results_data.append({
                                "sequence": s, "mutation": m,
                                "predicted_ddg": score, "antisymmetry_error": err,
                                "cosine_similarity": emb['cosine_similarity'],
                                "l2_distance": emb['l2_distance'],
                                "delta_embedding": emb['delta_embedding']
                            })
                        progress_bar.progress((i + 1) / len(df))
                    st.success("Batch Processing Complete!")
                else:
                    st.error("Invalid CSV Format.")

# ---------------------------------------------------------------------
# TAB 2: RESULTS
# ---------------------------------------------------------------------
with tab_results:
    if st.session_state.results_data:
        # Latest Prediction Highlight
        latest = st.session_state.results_data[-1]
        
        st.subheader("Latest Prediction")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mutation", latest['mutation'])
        m2.metric("ΔΔG (kcal/mol)", f"{latest['predicted_ddg']:.3f}", 
                  delta="-Stabilizing" if latest['predicted_ddg'] < 0 else "+Destabilizing",
                  delta_color="inverse")
        m3.metric("Confidence (Sim)", f"{latest['cosine_similarity']:.3f}")
        m4.metric("Error Check", f"{latest['antisymmetry_error']:.4f}")
        
        st.divider()
        
        # Full Table
        st.subheader("History Log")
        df_display = format_results_table(st.session_state.results_data)
        
        # Apply Styling (Colors!)
        styled_df = apply_table_styling(df_display)
        
        # Display Styled Table (Exclude full embedding)
        view_cols = [c for c in df_display.columns if c != "_full_delta_embedding"]
        st.dataframe(styled_df, column_config={"_full_delta_embedding": None}, use_container_width=True, hide_index=True)
        
        # Download
        df_csv = df_display.rename(columns={"_full_delta_embedding": "Delta Embedding Vector"})
        st.download_button(
            "📥 Download Report (CSV)",
            df_csv.to_csv(index=False).encode('utf-8'),
            "ddg_results.csv",
            "text/csv",
            type="primary"
        )
        
        if st.button("Clear History", type="secondary"):
            st.session_state.results_data = []
            st.rerun()
    else:
        st.container(border=True).info("👋 No predictions yet. Go to the 'New Prediction' tab to start!")

# ---------------------------------------------------------------------
# TAB 3: ABOUT
# ---------------------------------------------------------------------
with tab_about:
    render_about_model()