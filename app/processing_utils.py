import requests
import torch
import pandas as pd
import re
import streamlit as st

def parse_mutation_code(mutation_code):
    """Parses 'A23V' into ('A', '23', 'V'). Returns None if invalid."""
    match = re.match(r"([A-Z])(\d+)([A-Z])", mutation_code.upper())
    if match:
        return match.groups() # (WT, Pos, Mut)
    return None, None, None

def get_remote_embeddings(seq, mut, api_url):
    """Calls the GPU Server."""
    try:
        resp = requests.post(api_url, json={"sequence": seq, "mutation_code": mut}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"⚠️ GPU Server Error: {e}")
        return None

def run_inference_single(model, data):
    """Runs a single prediction."""
    wt = torch.tensor([data['wt_embedding']])
    mut = torch.tensor([data['mut_embedding']])
    delta = torch.tensor([data['delta_embedding']])
    abs_delta = torch.tensor([data['abs_delta_embedding']])
    cos = torch.tensor([[data['cosine_similarity']]])
    l2 = torch.tensor([[data['l2_distance']]])

    with torch.no_grad():
        ddg_fwd = model(wt, mut, delta, abs_delta, cos, l2).item()
        ddg_rev = model(mut, wt, -delta, abs_delta, cos, l2).item()
        
    return ddg_fwd, ddg_rev

def format_results_table(results_list):
    """Formats the raw results into the specific columns requested."""
    data = []
    for r in results_list:
        wt_res, pos, mut_res = parse_mutation_code(r['mutation'])
        stability = "Stabilizing" if r['predicted_ddg'] < 0 else "Destabilizing"
        
        row = {
            "Sequence (WT)": r['sequence'],
            "Mutation": r['mutation'],
            "WT": wt_res,
            "Mut": mut_res,
            "Pos": pos,
            "ΔΔG (kcal/mol)": r['predicted_ddg'], # Renamed for clarity
            "Class": stability,
            "Antisym Error": r['antisymmetry_error'],
            "Cosine Sim": r['cosine_similarity'],
            "L2 Dist": r['l2_distance'],
            "_full_delta_embedding": str(r['delta_embedding']) 
        }
        data.append(row)
        
    return pd.DataFrame(data)

def apply_table_styling(df):
    """
    Color codes the ΔΔG column:
    - Negative (Stabilizing) -> Green text
    - Positive (Destabilizing) -> Red text
    """
    def color_ddg(val):
        color = 'green' if val < 0 else 'red'
        return f'color: {color}; font-weight: bold'

    # Apply style only if column exists
    if "ΔΔG (kcal/mol)" in df.columns:
        return df.style.map(color_ddg, subset=["ΔΔG (kcal/mol)"]) \
                       .format({"ΔΔG (kcal/mol)": "{:.3f}", "Antisym Error": "{:.4f}", "Cosine Sim": "{:.3f}"})
    return df