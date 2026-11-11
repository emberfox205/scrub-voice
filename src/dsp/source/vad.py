import numpy as np
import torch

class SileroVAD:
    # Gate 1: Voice Activity Detection
    # Evaluates whether an incoming multi-channel chunk contains human speech
    def __init__(self, sample_rate=16000, threshold=0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window_size_samples = 512

        # Load lightweight Silero VAD pre-trained model
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self.model.eval()

    def is_speech_active(self, audio_chunk: np.ndarray) -> bool:
        # Gate 1 decision
        # audio_chunk: (num_mics, num_samples), float32 normalized [-1.0, 1.0]
        # Returns True if speech is active and False otherwise

        if audio_chunk.ndim == 1:
            # Single channel
            ref_audio = audio_chunk
        else:
            # Multichannel array: (num_mics, num_samples)
            # Average the energy across all mics
            ref_audio = np.mean(audio_chunk, axis=0)

        # Convert to PyTorch tensor for Silero Model
        tensor_audio = torch.from_numpy(ref_audio).float()
        num_samples = len(tensor_audio)

        speech_probs = []
        for start in range(0, num_samples - self.window_size_samples + 1, self.window_size_samples):
            window = tensor_audio[start : start + self.window_size_samples]
            with torch.no_grad():
                speech_prob = self.model(window, self.sample_rate).item()
            speech_probs.append(speech_prob)

        if not speech_probs:
            return False

        # Consider chunk active if at least 3 windows (~96ms) exceed threshold,
        # or average speech probability across the chunk exceeds threshold.
        active_windows = sum(1 for p in speech_probs if p >= self.threshold)
        is_active = (active_windows >= 3) or (float(np.mean(speech_probs)) >= self.threshold)

        if not is_active:
            try:
                self.model.reset_states()
            except Exception:
                pass

        return is_active
