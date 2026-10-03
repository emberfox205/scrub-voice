# scrub-voice (Python 3.13.5)

An Offline, Noise-Robust, Edge Voice-to-Intent Assistant for Sterile Medical Environments.

## NLU Intent Classification (Raspberry Pi 4) 

A fine-tuned SetFit intent classification model (`setfit_bge-micro-v2_8k`) is included in `models/setfit_bge-micro-v2_8k/` for offline edge execution on Raspberry Pi 4 (ARM64).

### Quick Start on Raspberry Pi 4

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run benchmark and interactive tester:
   ```bash
   python test_rpi_intent.py
   ```

Detailed instructions and performance expectations are available in [docs/RPI4_DEPLOYMENT_GUIDE.md](docs/RPI4_DEPLOYMENT_GUIDE.md).

