"""Stand-in for the NLU stage: receives the ASR messages and prints them.

Use it to check that ASR output arrives, before NLU is connected - and as
the reference for NLU's receiving side (the few lines in main()). Run from
the repository root, then start run_asr.py in a second terminal:

    python src/asr/scripts/listen_asr.py
    python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav --send

Address: SCRUB_ASR_ADDR (default ipc:///tmp/scrub_asr.ipc; ipc:// works on
Linux / Raspberry Pi only - on Windows use e.g. --asr-addr tcp://127.0.0.1:5592
on both sides).
"""

import argparse
import json
import os
import time

DEFAULT_ASR_ADDR = os.getenv("SCRUB_ASR_ADDR", "ipc:///tmp/scrub_asr.ipc")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the messages ASR sends to NLU")
    parser.add_argument("--asr-addr", default=DEFAULT_ASR_ADDR)
    args = parser.parse_args()

    import zmq

    pull = zmq.Context.instance().socket(zmq.PULL)
    pull.bind(args.asr_addr)  # NLU binds, ASR connects (same pattern as src/orchestrator/)
    print(f"Waiting for ASR messages on {args.asr_addr} (Ctrl+C to stop)", flush=True)
    try:
        while True:
            msg = json.loads(pull.recv())
            received = time.time()
            # End of speech = start of speech + end of the last word.
            speech_end = msg["segment_start"] + (msg["words"][-1]["end"] if msg["words"] else 0.0)
            print(f"text={msg['text']!r}  normalized={msg['normalized']!r}  confidence={msg['confidence']}  "
                  f"end of speech -> here: {received - speech_end:.2f} s", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pull.close(linger=0)


if __name__ == "__main__":
    main()
