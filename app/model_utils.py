import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
from pathlib import Path
from dataclasses import dataclass
import streamlit as st

# ==========================================
# 1. MODEL CONFIGURATION CLASS
# ==========================================
@dataclass
class TrainingConfigStage2:
    # Dummy config to satisfy pickle loading
    batch_size: int = 32
    save_dir: Path = Path(".")
    device: str = "cpu"

# ==========================================
# 2. NEURAL NETWORK LAYERS
# ==========================================
class LightAttention(nn.Module):
    def __init__(self, input_dim, kernel_size=9):
        super().__init__()
        # Matches checkpoint: [1, 1, 9] -> 1 Channel, Length=Embedding Dim
        self.feature_conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2)
        self.attention_conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        features = self.feature_conv(x)
        attn_weights = torch.sigmoid(self.attention_conv(x))
        return features * attn_weights

class SiameseDDGPredictorStage2(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1280,
        attention_kernel: int = 9,
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.attention = LightAttention(embedding_dim, kernel_size=attention_kernel)
        
        input_dim = (4 * embedding_dim) + 2
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
        
        layers.append(nn.Linear(curr_dim, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, wt_embedding, mut_embedding, delta_embedding, delta_abs_embedding, cosine_sim, l2_dist):
        wt_input = wt_embedding.unsqueeze(1)
        mut_input = mut_embedding.unsqueeze(1)
        delta_input = delta_embedding.unsqueeze(1)
        delta_abs_input = delta_abs_embedding.unsqueeze(1)

        wt_att = self.attention(wt_input)
        mut_att = self.attention(mut_input)
        delta_att = self.attention(delta_input)
        delta_abs_att = self.attention(delta_abs_input)

        combined = torch.cat([
            wt_att.squeeze(1),
            mut_att.squeeze(1),
            delta_att.squeeze(1),
            delta_abs_att.squeeze(1),
            cosine_sim,
            l2_dist
        ], dim=1)

        return self.regressor(combined)

# ==========================================
# 3. MODEL LOADING LOGIC
# ==========================================
@st.cache_resource
def load_model_from_file(filename, esm_dim=1280):
    current_dir = Path(__file__).parent
    model_path = current_dir / filename
    
    if not model_path.exists():
        st.error(f"❌ Critical Error: Could not find '{filename}' in {current_dir}")
        return None

    try:
        # --- HACK: Inject classes & Fix Paths ---
        import __main__
        __main__.TrainingConfigStage2 = TrainingConfigStage2
        __main__.SiameseDDGPredictorStage2 = SiameseDDGPredictorStage2
        __main__.LightAttention = LightAttention

        # Windows Path Hack
        temp_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        finally:
            pathlib.PosixPath = temp_path # Cleanup
        
        # Initialize Architecture
        model = SiameseDDGPredictorStage2(embedding_dim=esm_dim)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint 
            
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None