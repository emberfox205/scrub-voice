import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort

CURRENT_DIR = Path(__file__).resolve().parent
DENOISE_DIR = CURRENT_DIR.parent
REPO_ROOT = CURRENT_DIR.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dsp.denoise.model.DuoFilter.DuoFilter import DuoFilter
from src.dsp.denoise.stateful_onnx.DuoFilterStreaming import DuoFilterStreaming


def export_stateful_duofilter(
    checkpoint_path: Path,
    output_path: Path,
    chunk_frames: int = 32,  # 32 frames = 0.5s @ 16 kHz with hop=256
    opset_version: int = 14,
    verify: bool = True,
):
    print(f"[Export] Loading DuoFilter base model from: {checkpoint_path}")
    base_model = DuoFilter(use_ema_norm=False).eval()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()

    print(f"[Export] Wrapping with DuoFilterStreaming for chunk size T={chunk_frames} (0.5s)...")
    streaming_model = DuoFilterStreaming(base_model).eval()

    # Define chunk dummy inputs (Batch=1, Time=chunk_frames=32)
    dummy_inputs = (
        torch.randn(1, 2, chunk_frames, 257, dtype=torch.float32),   # noisy_frame: [B, 2, T, 257]
        torch.zeros(1, 9, 2, 128, dtype=torch.float32),              # enc0_cache
        torch.zeros(1, 64, 2, 128, dtype=torch.float32),             # enc1_cache
        torch.zeros(1, 64, 2, 64, dtype=torch.float32),              # enc2_cache
        torch.zeros(1, 64, 2, 32, dtype=torch.float32),              # enc3_cache
        torch.zeros(1, 64, 2, 16, dtype=torch.float32),              # glu_cache
        torch.zeros(1, 16, 64, dtype=torch.float32),                 # gru_state: [1, 16, 64]
    )

    input_names = [
        "noisy_frame",
        "enc0_cache",
        "enc1_cache",
        "enc2_cache",
        "enc3_cache",
        "glu_cache",
        "gru_state",
    ]

    output_names = [
        "enh_frame",
        "enc0_cache_out",
        "enc1_cache_out",
        "enc2_cache_out",
        "enc3_cache_out",
        "glu_cache_out",
        "gru_state_out",
    ]

    dynamic_axes = {
        "noisy_frame": {2: "time_frames"},
        "enh_frame": {2: "time_frames"},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Export] Exporting stateful chunk ONNX model to: {output_path}...")

    torch.onnx.export(
        streaming_model,
        dummy_inputs,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    print(f"[Export] Successfully exported to {output_path}!")

    if verify:
        print("[Verification] Running parity check between PyTorch and ONNX Runtime...")
        with torch.no_grad():
            pt_outputs = streaming_model(*dummy_inputs)

        sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_inputs = {name: tensor.numpy() for name, tensor in zip(input_names, dummy_inputs)}
        ort_outputs = sess.run(None, ort_inputs)

        max_diff = float(np.max(np.abs(pt_outputs[0].numpy() - ort_outputs[0])))
        print(f"[Verification] Max absolute difference: {max_diff:.2e}")
        assert max_diff < 1e-4, f"Parity check failed with difference {max_diff}"
        print("[Verification] Parity check PASSED successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Stateful Streaming DuoFilter to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DENOISE_DIR / "checkpoint" / "duofilter_epoch_045.ckpt"),
        help="Path to trained PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DENOISE_DIR / "onnx" / "duofilter_stateful.onnx"),
        help="Path to save exported ONNX model",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=32,
        help="Chunk size in STFT frames (default: 32 frames = 0.5s @ 16kHz, hop=256)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    args = parser.parse_args()

    export_stateful_duofilter(
        checkpoint_path=Path(args.checkpoint).resolve(),
        output_path=Path(args.output).resolve(),
        chunk_frames=args.chunk_frames,
        opset_version=args.opset,
    )
