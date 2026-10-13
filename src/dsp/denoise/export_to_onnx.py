import argparse
import torch
from model.DeepFilterNetDNN import DeepFilterNetDNN

def export_model(checkpoint_path, output_path="deepfilternet.onnx"):
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Initialize the model with default parameters matching training
    erb_bins = 32
    df_bins = 96
    
    model = DeepFilterNetDNN(erb_bins=erb_bins)
    
    # Load weights
    try:
        # Some checkpoints are saved as dict with 'model_state_dict', others just the state dict
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model.eval()

    # Create dummy inputs matching the forward pass
    # model(input_erb, complex_in)
    # input_erb: [B, 1, T, erb_bins]
    dummy_input_erb = torch.randn(1, 1, 100, erb_bins)
    
    # complex_in: [B, 2, T, df_bins] (real/imaginary channels from STFT)
    dummy_complex_in = torch.randn(1, 2, 100, df_bins)

    print(f"Exporting model to {output_path}...")
    
    torch.onnx.export(
        model, 
        (dummy_input_erb, dummy_complex_in), 
        output_path,
        export_params=True,
        opset_version=14,               # Opset 14 is generally stable for GRUs and complex operations
        do_constant_folding=True,
        input_names=['input_erb', 'complex_in'],
        output_names=['erb_gains', 'df_coefficients', 'alpha'],
        dynamic_axes={
            'input_erb': {2: 'time'}, 
            'complex_in': {2: 'time'},
            'erb_gains': {2: 'time'},
            'df_coefficients': {1: 'time'},
            'alpha': {1: 'time'}
        }
    )
    
    print("Export complete! You can now transfer the ONNX file to your Raspberry Pi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DeepFilterNet PyTorch model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to your .ckpt or .pt file")
    parser.add_argument("--output", type=str, default="deepfilternet.onnx", help="Output ONNX filename")
    
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)
