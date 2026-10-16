# Speech Recognition (ASR) Guide

How the speech recognition stage connects to the audio front end (DSP) and
to language understanding (NLU), and how to run it. Owner: Josias.
Design notes and measurements: `src/asr/CLAUDE.md`, `docs/ASR_FINDINGS.md`.

```
DSP (speech chunks) ──ZeroMQ──▶ ASR (src/asr/) ──ZeroMQ──▶ NLU ──▶ Orchestrator
```

## 1. What ASR does

* Takes the cleaned, beamformed 0.5 second chunks that the DSP passes on
  **only while someone speaks**.
* One run of speech chunks = one utterance. ASR turns it into text and sends
  **one JSON message per utterance** to NLU.
* It writes what was said, nothing more: no intent detection, no synonym
  mapping ("knife" stays "knife"). That is NLU's job.
* Engine: **Vosk** (`vosk-model-small-en-us-0.15`) with a grammar of our
  command words, plus a free-speech pass for the text after "log event" /
  "record milestone". Fully offline. About 0.07 s per command on a laptop,
  ~260 MB memory. Why this engine: `docs/ASR_FINDINGS.md`, "Decision".

## 2. Input from DSP

ASR **binds** a ZeroMQ `PULL` socket; DSP **connects** a `PUSH` socket.
Address: `SCRUB_AUDIO_ADDR` (default `ipc:///tmp/scrub_audio.ipc`).

| message | frames | meaning |
| :--- | :--- | :--- |
| speech chunk | 1. `{"type": "chunk", "capture_time": 1727800000.0}` (JSON)<br>2. raw audio bytes | `capture_time` = Unix time (seconds) of the chunk's **first sample**. Audio = **float32, mono, 16 kHz**, normally 8000 samples (0.5 s). |
| end of speech | 1. `{"type": "end"}` | Send this when your voice activity detector switches off. |

* Send only chunks with speech, in order, without overlap.
* **Please send `"end"`.** Without it ASR waits 0.8 s for the next chunk
  before it knows the command is over (measured: JSON after 0.94 s instead
  of 0.03 s).
* A gap in `capture_time` (a missing chunk) also ends the utterance.
* **Timing:** with 0.5 s chunks, the end of speech is only noticed when the
  next chunk is complete and silent. Measured end of speech → NLU:
  0.6–1.0 s, of which ASR itself takes 0.04 s. Faster: send `"end"` (or a
  shorter last chunk – any length is accepted) as soon as your detector
  (32 ms steps) hears the speech stop.
* Utterances longer than 15 s are cut (safety net if the detector sticks).

Python example (DSP side; the receiving side for NLU is in
`src/asr/scripts/listen_asr.py`):

```python
import orjson, zmq
sock = zmq.Context().socket(zmq.PUSH)
sock.connect("ipc:///tmp/scrub_audio.ipc")
sock.send_multipart([orjson.dumps({"type": "chunk", "capture_time": t}), beam.astype("float32").tobytes()])
sock.send_multipart([orjson.dumps({"type": "end"})])
```

If DSP and ASR run in one Python process, skip ZeroMQ and call
`asr.stream.UtteranceStream` directly (`feed(chunk, capture_time)`,
`end_utterance()`).

## 3. Output to NLU

ASR **connects** a ZeroMQ `PUSH` socket; NLU **binds** a `PULL` socket.
Address: `SCRUB_ASR_ADDR` (default `ipc:///tmp/scrub_asr.ipc`).
One message per utterance, UTF-8 JSON:

```json
{
  "text": "zoom in two x",
  "normalized": "zoom in 2 x",
  "confidence": 0.875,
  "language": "en",
  "words": [
    {"word": "zoom", "start": 0.0, "end": 0.48, "confidence": 1.0},
    {"word": "in", "start": 0.51, "end": 0.87, "confidence": 1.0},
    {"word": "two", "start": 0.96, "end": 1.23, "confidence": 0.5},
    {"word": "x", "start": 1.23, "end": 1.74, "confidence": 1.0}
  ],
  "segment_start": 1792150493.32
}
```

| field | meaning |
| :--- | :--- |
| `text` | What the engine produced (= `raw_transcript` in the command schema). Speech outside the command words appears as `[unk]`. |
| `normalized` | Cleaned text for classification: lowercase, no punctuation, no `[unk]`, numbers as digits, "please"/"thanks" removed, common mishearings fixed. **Empty** when nothing usable was heard. |
| `confidence` | 0–1, mean word confidence. **`null`** when only `[unk]` was heard. |
| `words` | Word timings in seconds from `segment_start`, with per-word confidence. |
| `segment_start` | Unix time (seconds) when the speech started. Use it for `latency_ms`. |

Written forms in `normalized` (please confirm, Khanh):

| said | `normalized` |
| :--- | :--- |
| "zoom in two x" | `zoom in 2 x` |
| "one point five x" | `1.5 x` |
| "go to slice one hundred twenty" | `go to slice 120` |
| "display X-ray" | `display xray` |
| "five centimeters" | `5 centimeters` |
| "fifteen degrees" | `15 degrees` |

Things NLU should know:
* **Everyday talk can turn into command words.** The grammar sometimes
  forces non-command speech into command words (measured: "we are running
  a bit late" → "blade"). Low `confidence` or `[unk]` in `text` are hints.
* Free text after "log event" comes from a general model and is often wrong
  for medical words ("hepatic" → "have pet take"). Store it verbatim.
* The first "log event" after start-up takes ~0.6 s longer (one-time setup).

## 4. Running it

```bash
# once, from the repository root
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # model downloads on first start, then offline

# service (pipeline)
python src/asr/scripts/run_asr.py

# simulation without the microphone array: plays wav files as a DSP stream, prints JSON
python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav

# test ASR -> NLU without the DSP: a stand-in NLU prints what arrives,
# the simulation plays the files in real time and sends the results
python src/asr/scripts/listen_asr.py                                  # terminal 1
python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav --send # terminal 2

# one file -> JSON
python src/asr/scripts/transcribe.py some_command.wav

# unit tests
python -m unittest discover -s src/asr/test -v
```

`ipc://` addresses work on Linux / Raspberry Pi. On Windows use TCP, e.g.
`--audio-addr tcp://127.0.0.1:5591 --asr-addr tcp://127.0.0.1:5592`.

## 5. Changing the command list

The grammar is built from the command words in `src/asr/vocab.py`
(`INTENT_TERMS`). A new command word that is not in that list cannot be
recognized. Add it there (and run the unit tests). Words missing from the
Vosk dictionary need a sound-alike spelling in `SPOKEN_FORMS` (example:
"hemostat" → "hemo stat").
