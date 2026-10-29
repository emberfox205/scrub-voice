import torch
import sys
from pathlib import Path

# Automatically resolve the project root (3 folders up: denoise -> dsp -> src -> scrub-voice)
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dsp.denoise.model.DuoFilter.DuoFilter import DuoFilter

# 1. Instantiate and load checkpoint
model = DuoFilter(use_ema_norm=False).eval()
checkpoint_path = project_root / "src" / "dsp" / "denoise" / "checkpoint" / "duofilter_best_model.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model.eval()

# 2. Script the model using TorchScript so CausalEmaNorm creates a dynamic ONNX Loop
# instead of statically unrolling 32 frames. This allows arbitrary audio lengths without breaking EMA norm!
scripted_model = torch.jit.script(model)

# 3. Dummy STFT input: [Batch=1, Channels=2 (Real, Imag), Frames=T=32, FreqBins=257]
dummy_stft = torch.randn(1, 2, 32, 257, dtype=torch.float32)

output_onnx = project_root / "src" / "dsp" / "denoise" / "onnx" / "duofilter.onnx"
output_onnx.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    scripted_model,
    dummy_stft,
    str(output_onnx),
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["noisy_stft"],
    output_names=["enhanced_stft"],
    dynamic_axes={
        "noisy_stft": {0: "batch_size", 2: "time_frames"},
        "enhanced_stft": {0: "batch_size", 2: "time_frames"},
    },
    dynamo=False,
)
print(f"Saved scripted duofilter.onnx successfully to {output_onnx}!")
