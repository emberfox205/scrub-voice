# ASR Module – speech recognition stage of scrub-voice

## Context
`scrub-voice` is a 5-person university project (VGU): an offline,
noise-robust voice assistant for sterile surgical environments. Surgeons
control a **UR10e robot arm** (tool handover, endoscope/light wand) and a
**medical image viewer** by voice.

Pipeline: ReSpeaker mic array + Raspberry Pi → DSP front end (`src/dsp/`;
beamforming, denoising; Minh, Quân) → streams cleaned audio in **0.5 s
chunks** → **ASR (`src/asr/`, this module): find the speech in the stream and
transcribe it** → NLU intent classifier (`src/nlu/`, Khanh) → UR10e controller
(`src/actuators/`) and viewer UI via ZeroMQ.

Owner of this module: Josias. Work happens on the branch **`asr_module`**.

I have never run an ML model locally before. Explain setup steps, commands and
errors in plain terms, and prefer simple, readable code over clever code.

## Repository rules (read `AGENTS.md` and `docs/` first)
- `AGENTS.md`: no new root-level folders; implementation code only inside
  `src/asr/`; models inside `src/asr/models/`; paths always relative to
  `Path(__file__)`; documentation in `docs/` with UPPERCASE names.
- `docs/GITHUB_WORKFLOW.md`: work on our branch, `git pull origin main` at the
  start of a session, open a pull request only when the code is stable,
  **never push to `main`**. Only edit files of our module (`src/asr/`,
  `data/asr/`, `docs/ASR_*.md`) to avoid merge conflicts.
- `docs/REPO_ORGANIZATION.md`: folder responsibilities.
- `schemas/command_schema.json` + `docs/COMMAND_SCHEMA_GUIDE.md`: the NLU
  command contract (owner Khanh) – source of the command list below.

## Status (2026-10-14) – read this first
- **Done:** Vosk and faster-whisper engines with a common interface and the
  JSON output below; a trustworthy benchmark (fair normalization, corpus WER
  with confidence intervals, command recall and precision, noise check, one
  process per config, repeated timing); three determinism bugs fixed; full
  engine comparison. All findings and numbers: `docs/ASR_FINDINGS.md` (team
  meeting notes at the top).
- **Decision (team meeting):** continue with exactly two setups:
  1. **Vosk + command grammar** (first choice: fastest, least memory,
     recall 0.98 / precision 0.98 on the old dev set)
  2. **faster-whisper tiny.en + prompt sentence** (recall 0.92 / precision
     1.00, better word-for-word text, but 0.87 s per clip on the laptop)
- **Dropped:** Moonshine (does not fit our streaming front end, no confidence
  score), Whisper base.en, beam 5, hot words (repetition loops, "zoom in" on
  pure noise).
- **The command set changed.** The old image-viewer commands (pan, window,
  axial, …) are replaced by the NLU contract.
  - **Phase A done (2026-10-14):** `vocab.py` and `normalize.py` rebuilt from
    the schema, `CORRECTIONS` emptied, `log_event` strategy implemented,
    28 unit tests in `src/asr/test/`. Details in `docs/ASR_FINDINGS.md`
    ("Phase A").
  - The first test set (46 clips, one speaker, old commands) is **not in this
    repo** (recordings of people are never committed). New phrase lists are
    ready in `data/asr/prompts/` → **Phase B (next)**.
- **The input changed.** DSP sends a continuous stream of 0.5 s chunks, not
  finished segments: ASR must detect speech start and end itself →
  **Phase C**.
- **History:** the original standalone project (old recordings, earlier run
  outputs, retired Moonshine code) stays in Josias's local folder
  `Desktop\projects\asr-module`; it is not needed for further work.

## Setup (once per machine)
```powershell
# from the repository root
python -m venv .venv            # .venv/ is already gitignored
.venv\Scripts\activate
pip install -r src/asr/requirements.txt
```
- Developed with Python 3.11 (the repo README mentions 3.13; check that
  `vosk` and `faster-whisper` install there before switching).
- Models: the engines download Whisper tiny.en and Vosk small into
  `src/asr/models/` (gitignored) on first use; after that everything runs
  offline. The Vosk engine adds `--dither=0` to the model config itself.
- `src/asr/requirements.txt` is this module's pinned list; the root
  `requirements.txt` (NLU, DSP) is shared – don't edit it without asking.

## Goal
Consume the stream of 0.5 s chunks of cleaned audio, detect each spoken
utterance (start and end), transcribe it, and emit **one JSON per utterance**
to the NLU stage. Fully offline, fast enough on a Raspberry Pi 4.

## Constraints
- **Offline at runtime.** Models live in `src/asr/models/`. No network calls at
  inference time.
- **Target hardware:** Raspberry Pi 4 (`docs/RPI4_DEPLOYMENT_GUIDE.md`). Laptop
  numbers are only for comparing setups.
- **Input:** 0.5 s chunks of cleaned, beamformed audio. Engines need 16 kHz
  mono float32 (`audio.py` resamples and downmixes). Chunk format,
  timestamps and transport still to confirm with DSP (open questions below;
  `src/dsp/source/` contains their audio streaming code – read it).
- **ASR owns voice activity detection and end-of-utterance detection.**
- **Latency target (tentative):** under 1 s from end of speech to JSON for a
  ~2 s command, on the Pi. The budget includes waiting for the chunk that
  contains the end of speech (up to 0.5 s), the silence needed to decide the
  utterance is over, and the transcription. Measure each part.
- **Determinism:** the same audio must give the same text.

## Scope
ASR finds speech in the stream, turns it into text, applies vocabulary
biasing (grammar / prompt sentence) and light text normalization (lowercase,
punctuation, fillers, number words → digits). It does **not** classify
intents, fill slots or canonicalize synonyms ("knife" → "scalpel" is NLU's
job, `docs/COMMAND_SCHEMA_GUIDE.md` §3A). ASR must transcribe faithfully,
including out-of-domain speech – never force speech into a command.

## Output schema (contract with NLU – do not change without telling Khanh)
One JSON object per detected utterance:
```json
{
  "text": "Zoom in two x, please.",
  "normalized": "zoom in 2 x",
  "confidence": 0.87,
  "language": "en",
  "words": [
    {"word": "Zoom", "start": 0.12, "end": 0.40, "confidence": 0.95},
    {"word": "in",   "start": 0.40, "end": 0.52, "confidence": 0.93}
  ],
  "segment_start": 1726658793.12
}
```
- `text`: raw transcript exactly as the engine produced it (= `raw_transcript`
  in the NLU schema).
- `normalized`: `normalize(text, corrections=True, drop_politeness=True)`.
- `confidence`: 0–1 for the whole transcript. faster-whisper →
  `exp(avg_logprob)`, null for empty output; Vosk → mean word confidence.
- `language`: fixed "en".
- `words`: word timestamps in seconds relative to `segment_start`, with
  per-word confidence.
- `segment_start`: absolute start time of the utterance = capture time of the
  chunk where speech starts + offset within that chunk.
- **Proposal to discuss with Khanh (not implemented):** add `segment_end`
  (absolute end of speech). NLU's `latency_ms` is measured from audio capture,
  and end of speech is the reference for our latency target.

Benchmark-only data (config name, processing time, RTF, RAM) goes to
`src/asr/results/`, never into this output.

## Commands to recognise (source: `schemas/command_schema.json`)
| intent (category) | spoken examples (guide) | words the ASR must get right |
|---|---|---|
| `pick_tool` (TOOL_HANDLING) | "Hand scalpel", "Pass the iris scissors", "Give me the clamp" | hand, pass, give me; scalpel, scissors, hemostat, forceps; synonyms (NLU maps them): knife, ten blade, blade, iris, metz, metzenbaum, clamp, snap, tweezers, pickups |
| `park_arm` (TOOL_HANDLING) | "Park arm", "Standby robot", "Retract arm" | park, standby, retract, arm, robot |
| `clear_emergency` (EMERGENCY, high priority) | "Emergency clear", "Clear area", "Back off" | emergency, clear, area, back off |
| `halt_emergency` (EMERGENCY, high priority) | "Emergency stop", "Halt robot", "Freeze" | emergency, stop, halt, freeze, robot |
| `open_scan` (MONITOR_DISPLAY) | "Open CT scan", "Show MRI", "Display X-ray" | open, show, display, scan; CT, MRI, X-ray (slot values `ct`, `mri`, `xray`) |
| `close_scan` | "Close scan", "Dismiss MRI", "Close viewer" | close, dismiss, scan, viewer (+ optional modality) |
| `zoom_display` | "Zoom", "Zoom in 2x", "Zoom out", "Magnify 3x" | zoom, magnify, in, out; factor 1.1–10 ("two x", "one point five x") |
| `navigate_slice` | "Next slice", "Previous slice", "Back 5 slices", "Go to slice 45" | next, previous, back, go to, slice(s); whole numbers ≥ 1 (up to the hundreds) |
| `log_event` | "Log event: first incision", "Record milestone: trocar insertion complete", "…, note patient vitals stable", "…, comment clean margins" | log, record, event, milestone, note, comment **+ free text** |
| `move_camera` (CAMERA_LIGHT) | "Camera left 5 cm", "Move up", "Camera forward" | camera, move; left, right, up, down, forward, backward; 1–25 centimeters |
| `tilt_camera` | "Tilt down 15 degrees", "Tilt right", "Tilt up" | tilt; up, down, left, right; 1–45 degrees |
| `reset_camera` | "Reset camera", "Center view", "Camera home" | reset, center, view, home |
| `unknown` | "It is raining outside", "Pass that thing", "Hello" | anything – transcribe faithfully |

What this means for the ASR design (decided in Phase A, measure in Phase B):
1. **`log_event` contains free text**, which a restricted Vosk grammar cannot
   produce (out-of-grammar words become `[unk]`). **Implemented:**
   `VoskEngine(freeform_log=True)` (config `vosk-grammar+freeform-log`) – when
   the grammar hears "log"/"record" among the first three words, a free-form
   recognizer (same model) decodes the same audio; the header ("log event")
   comes from the grammar, the words after it from the free-form result.
   Whisper needs nothing special. Measure both on the `log_event` clips.
2. **Numbers:** done. `words_to_digits` handles "one hundred (and) twenty" →
   120 and "one point five" → 1.5; Whisper's "1.5x" keeps its decimal point
   and is split to "1.5 x". **Written forms chosen (to confirm with Khanh):**
   digits; factor "2 x" (a separate "x" token); "xray"; "centimeters" (also
   for "cm"); "degrees" (also for "°").
3. **Emergency commands are safety-critical:** recognise them when repeated,
   in noise, and with the lowest latency. Idea: Vosk partial results can spot
   "emergency stop" / "halt" / "freeze" before the end-of-utterance wait.
4. **Synonyms:** recognise every synonym in the guide; NLU maps them.
5. **Correction list:** emptied (it was built for the old commands) →
   rebuild from errors on the new dev set only. First candidate seen on
   synthetic speech: the grammar heard "go **two** slice" for "go to slice".

## Engines (current setups – keep these settings)
**1. Vosk** – `vosk-model-small-en-us-0.15` (`engines/vosk_engine.py`)
- Command grammar built only from `vocab.py`, plus `[unk]`. It is a plain
  word list: a Vosk grammar accepts any sequence of its entries, so no
  command sentence has to be written into it. "hemostat" and "xray" are not
  in the model's dictionary → grammar uses "hemo stat" / "x ray",
  `normalize()` joins them back.
- Word times keep counting across `Reset()`; the engine shifts them back to
  0 per utterance (bug found and fixed in Phase A).
- `--dither=0` in the model's `mfcc.conf` (the engine adds it) – otherwise the
  same audio gives different words.
- One `KaldiRecognizer` per engine, `Reset()` after each utterance – a new
  recognizer per clip cost ~0.5 s.
- One `vosk.Model` per process – loading a second after freeing one
  segfaults.
- Streams natively: `AcceptWaveform(chunk)` returns True at end of utterance
  (endpoint rules in the model's `model.conf`), partial results available.

**2. faster-whisper tiny.en** – int8, CPU (`engines/faster_whisper_engine.py`)
- Beam 1, `initial_prompt` = one natural sentence with the key terms (must not
  contain test sentences), `language="en"`, `temperature=0`,
  `condition_on_previous_text=False`, `word_timestamps=True`.
- Not streaming: needs our voice activity detection to cut utterances.
- Says "you" on pure noise without the prompt; adds junk words when there is
  trailing silence → trim utterances tightly.
- The prompt costs latency (0.35 → 0.87 s per clip on the laptop) → a
  shorter prompt is ready as config `…+shortprompt` (`WHISPER_PROMPT_SHORT`).
  No hot words.

## Finding speech in the 0.5 s chunk stream (Phase C design notes)
- **Shared segmenter for both engines** (same boundaries → fair comparison):
  Silero VAD, bundled with faster-whisper (`faster_whisper.vad.SileroVADModel`,
  `faster_whisper/assets/silero_vad.onnx`, no extra download or dependency).
  Its `__call__(x, state, context, sr)` carries `state`/`context` between
  calls, so it runs window by window over our chunks (512-sample windows =
  32 ms at 16 kHz – check `get_speech_timestamps` in the installed source).
  Also look at `src/dsp/` – the DSP team already wraps a VAD there; reuse or
  align with it rather than duplicating.
- Parameters to tune on the dev set: speech threshold, silence needed to end
  an utterance (start with 300–500 ms), pre-roll (~200 ms kept before the
  detected start), maximum utterance length (`log_event` ~10 s).
- **Vosk variant:** its own endpointing (`AcceptWaveform` per chunk).
- Cases: utterance spanning several chunks, speech ending at a chunk
  boundary, two commands with a short pause, noise-only chunks, long
  `log_event`.
- **Test material:** `data/asr/sample_01–05.wav` (in the repo, 16 kHz mono,
  30–55 s each) are long passages read aloud (sample_03 is a story), not
  commands – useful as continuous non-command speech between commands and
  for long-utterance tests. `scripts/simulate_stream.py` (to write) joins recorded clips
  with pauses and noise, cuts the result into 0.5 s chunks and keeps the true
  start/end times. Metrics: utterances found / missed / split / merged, start
  and end error, end-of-speech → JSON latency.

## Module layout
```
src/asr/
  CLAUDE.md            # this file
  requirements.txt     # pinned dependencies of this module
  schema.py            # ASRResult / WordInfo + to_json() (output contract)
  audio.py             # load wav, downmix, resample to 16 kHz mono float32
  normalize.py         # one normalize() for scoring + the JSON `normalized` field; CORRECTIONS (empty)
  vocab.py             # terms per intent (schema), grammar words, prompt sentences, term metrics, leakage check
  metrics.py           # corpus WER, cluster bootstrap CI, paired differences
  bench_configs.py     # 6 configs: whisper tiny beam1 (none / prompt / short prompt),
                       #   vosk free-form / grammar / grammar + free-form for log_event
  engines/             # base.py, faster_whisper_engine.py, vosk_engine.py
  test/                # unit tests (python -m unittest discover -s src/asr/test -v)
  scripts/
    record_testset.py  # record a phrase list -> data/asr/<list>/ + manifest
    transcribe.py      # file(s) -> one JSON line each on stdout (--config <name>, default vosk-grammar)
    benchmark.py       # configs x clip sets, one process per config -> src/asr/results/
    bench_worker.py    # used by benchmark.py
    compare_configs.py # paired bootstrap: which differences are real
    make_noise_clips.py# generated noise-only clips -> data/asr/noise_only/
  models/              # gitignored, downloaded on first use
  results/             # gitignored benchmark outputs
data/asr/
  prompts/             # commands_dev.txt (60), commands_heldout.txt (34), non_command.txt (19)
  sample_01-05.wav     # team samples: long read-aloud passages, not commands
  commands_dev/ ...    # recordings + manifests (gitignored)
docs/ASR_FINDINGS.md   # findings log for the project report
```
To be written: `segmenter.py`, `stream.py` (chunks in → JSON out),
`scripts/simulate_stream.py`.

## Commands you'll use (from the repository root)
```powershell
.venv\Scripts\activate
python -m unittest discover -s src/asr/test -v                                # unit tests
python src/asr/scripts/transcribe.py data/asr/sample_01.wav --config vosk-grammar
python src/asr/scripts/record_testset.py --speaker <name> --prompts-file data/asr/prompts/commands_dev.txt
python src/asr/scripts/make_noise_clips.py
python src/asr/scripts/benchmark.py --configs vosk-grammar --runs 1           # quick check
python -u src/asr/scripts/benchmark.py > src/asr/results/run.log 2>&1         # full run
Get-Content src\asr\results\run.log -Wait                                     # follow it
python src/asr/scripts/benchmark.py --score-only                              # rescore, no inference
python src/asr/scripts/compare_configs.py                                     # which differences are real
```
The benchmark's dev set is `data/asr/commands_dev/manifest.csv` – it exists
only after recording (Phase B); until then the benchmark stops with a message.

## Measurement method (keep – details in docs/ASR_FINDINGS.md)
- One `normalize()` applied identically to reference and hypothesis. Corpus
  WER (total errors / total reference words) is the main number, mean
  per-clip WER secondary.
- 95% confidence interval by cluster bootstrap (resample command groups, not
  clips); compare configs pairwise on shared resamples. With the new data, the
  intent in the manifest `notes` column is the natural cluster.
- Term recall (said → output) and term precision (output → said): whole words,
  longest match first, per occurrence. Term list = words that carry intents
  and slot values.
- False commands: noise-only clips (any output = hallucination; a command term
  = false command) and non-command speech (`out_of_domain` section only;
  `near_miss` reported separately).
- One process per config; model load timed separately; 1 warm-up; each clip
  timed 3×, median kept; peak RAM; machine info; `runs_differ` must be 0.
- New for streaming: segmentation accuracy and end-of-speech → JSON latency.
- **Dev vs held-out:** tune only on `commands_dev`; run `commands_heldout` once
  at the end. Until then, label results "dev set".
- Grammar and prompt sentence must never contain test sentences
  (`check_leakage` asserts this).

## Next phases
A. ✅ **Vocabulary from the schema** (done 2026-10-14): `vocab.py`,
   `normalize.py`, empty `CORRECTIONS`, `log_event` free-form splice, unit
   tests.
B. **Record and re-measure:** `commands_dev` and `non_command` (held-out
   ideally by other speakers); run the 6 configs on whole clips. Check:
   full vs short prompt (speed and recall), grammar with vs without the
   free-form `log_event` pass, Whisper junk words at the end of clips with
   trailing silence (seen with the prompt on synthetic speech), first
   `CORRECTIONS` from dev errors only.
C. **Streaming:** `segmenter.py` (Silero VAD, window by window) +
   `scripts/simulate_stream.py`; Vosk native endpointing as a variant;
   measure segmentation and latency.
D. **Entry point:** `stream.py` – chunks (+ capture timestamps) in, one JSON per
   utterance out; transport (ZeroMQ?) once DSP confirms.
E. **Raspberry Pi 4:** run both setups end to end; decide.
F. **Held-out run** → final numbers in `docs/ASR_FINDINGS.md`; pull request
   to `main` once stable.
G. (Optional) Noise robustness: mix OR noise at 20/10/5/0 dB SNR into the dev
   clips.

## Open questions for the team
- **DSP (Minh, Quân):** chunk format (sample rate, channels, float or int)?
  Exactly 0.5 s, overlapping? Capture timestamp per chunk? Transport and
  message format? Do you still run a voice activity detector (a speech flag
  per chunk would help)? Is silence between commands sent continuously?
  Where do `data/asr/sample_01–05.wav` come from (recorded through the array
  and DSP chain?)
- **NLU (Khanh):** `segment_end` proposal; written forms we chose in
  `normalized` (digits, "zoom in 2 x", "xray", "5 centimeters",
  "15 degrees") OK?; verbatim free text for `log_event` OK?; handling of
  `confidence: null` and of an empty `normalized` (Vosk grammar on
  out-of-grammar speech gives text "[unk]", normalized "", confidence
  1.0 – the "[unk]" marker counts in the mean); who maintains the command
  list?
- **Everyone:** is the 1 s budget for the whole chain or ASR alone? Memory
  left on the Pi next to DSP and NLU? Fallback: ASR on the workstation?

## Working conventions
- Work in small steps. At the end of each phase, run the code and show me the
  output before moving on.
- Explain any new tool, library or concept briefly the first time it appears.
- Keep dependencies minimal and pinned in `src/asr/requirements.txt`.
- Use type hints, `pathlib`, and `argparse` for scripts; resolve paths from
  `Path(__file__)`.
- Verify library APIs against the installed package source, not memory.
- Never commit audio recordings of people or model files.
- Record findings (what was tried, numbers, conclusion) in
  `docs/ASR_FINDINGS.md`. Team-facing notes: short, plain words, no
  abbreviations, tables with key values.
- Long runs: `python -u … > src/asr/results/run.log 2>&1` and tell me how to
  follow it; the laptop must stay awake.
- Before trusting a number: `runs_differ` = 0 and dev/held-out kept apart.
- Commit small, meaningful steps on `asr_module`; never push to `main`.
