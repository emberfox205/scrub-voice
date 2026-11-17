from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi

# Dynamic path resolution
CURRENT_DIR = Path(__file__).resolve().parent
DSP_DIR = CURRENT_DIR.parent
DEFAULT_MODEL_PATH = DSP_DIR / "models" / "ecapa_tdnn" / "voxceleb_ECAPA512.onnx"

class SpeakerVerifier:
    """
    ECAPA-TDNN Speaker Verification Engine using ONNX Runtime.
    Extracts L2-normalized 192-D speaker embeddings from 16kHz mono audio
    and computes cosine similarity for voice biometrics (Target Speaker Extraction).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        similarity_threshold: float = 0.65,
    ):
        self.sample_rate = sample_rate
        self.threshold = similarity_threshold

        resolved_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not resolved_path.exists():
            raise FileNotFoundError(f"ECAPA-TDNN ONNX model not found at: {resolved_path}\n"
                f"Please ensure voxceleb_ECAPA512.onnx is located in src/dsp/models/ecapa_tdnn/")

        # Optimize ONNX Runtime for CPU Execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(resolved_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts an L2-normalized speaker embedding vector from raw mono audio.
        Args:
            audio: 1D numpy array of float32 in range [-1.0, 1.0] at 16kHz.
        Returns:
            1D numpy array of shape (192,), L2-normalized (||e||_2 = 1.0).
        """
        # Input sanity checks
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D mono audio array, got shape {audio.shape}")

        min_samples = int(self.sample_rate * 0.4)   # At least 400ms needed
        if len(audio) < min_samples:
            raise ValueError(f"Audio too short for voice biometrics ({len(audio)} samples). "
                f"Need at least {min_samples} samples (0.4s).")

        # Scale [-1.0, 1.0] to 16-bit PCM scale for standard Kaldi FBank
        tensor_wav = torch.from_numpy(audio * 32767.0).float().unsqueeze(0)

        # Extract 80-bin Log-Mel Filterbank features (WeSpeaker standard)
        feat = kaldi.fbank(
            tensor_wav,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            sample_frequency=self.sample_rate,
            window_type="hamming",
            dither=0.0
        )

        # Utterance-level Cepstral Mean Normalization (CMVN)
        feat = feat - torch.mean(feat, dim=0, keepdim=True)
        feat_np = feat.unsqueeze(0).numpy().astype(np.float32)  # Shape: (1, num_frames, 80)
        
        # Run ONNX Inference
        embedding = self.session.run([self.output_name], {self.input_name: feat_np})[0]
        embedding = embedding.flatten().astype(np.float32)

        # L2-Normalize vector to unit length (||e||_2 = 1.0)
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Computes cosine similarity between two L2-normalized embeddings.
        Returns:
            Cosine similarity score in range [-1.0, 1.0].
        """
        return float(np.dot(emb1, emb2))

    def verify(self, audio: np.ndarray, reference_embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Verifies if incoming audio matches the enrolled reference speaker.
        Returns:
            (is_match: bool, similarity_score: float)
        """
        cand_emb = self.extract_embedding(audio)
        score = self.compute_similarity(cand_emb, reference_embedding)

        return (score >= self.threshold), score 
