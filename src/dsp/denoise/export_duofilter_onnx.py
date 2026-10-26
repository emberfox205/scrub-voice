import torch

import sys
from pathlib import Path
# Automatically resolve the project root (3 folders up: denoise -> dsp -> src -> scrub-voice)
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dsp.denoise.model.DuoFilter.DuoFilter import DuoFilter

# 1. Instantiate and load checkpoint
model = DuoFilter(use_ema_norm=True)
checkpoint = torch.load("src/dsp/denoise/checkpoint/duofilter_best_model.ckpt", map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model.eval()

# 2. Dummy STFT input: [Batch=1, Channels=2 (Real, Imag), Frames=T, FreqBins=257]
# 1 second of audio at 16kHz with hop=256 gives ~63 frames
dummy_stft = torch.randn(1, 2, 63, 257, dtype=torch.float32)

# 3. Export to ONNX with dynamic time axis
torch.onnx.export(
    model,
    dummy_stft,
    "duofilter.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["noisy_stft"],
    output_names=["enhanced_stft"],
    dynamic_axes={
        "noisy_stft": {0: "batch_size", 2: "num_frames"},
        "enhanced_stft": {0: "batch_size", 2: "num_frames"}
    }
)
print("Saved duofilter.onnx successfully!")