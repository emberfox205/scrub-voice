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

## Status (2026-10-16) – read this first
- **Done:** Vosk and faster-whisper engines with a common interface and the
  JSON output below; a trustworthy benchmark (fair normalization, corpus WER
  with confidence intervals, command recall and precision, noise check, one
  process per config, repeated timing); three determinism bugs fixed; full
  engine comparison. All findings and numbers: `docs/ASR_FINDINGS.md` (team
  meeting notes at the top).
- **FINAL DECISION (2026-10-16): Vosk, config `vosk-grammar+freeform-log`
  (`bench_configs.PRODUCTION_CONFIG`). Model testing is closed** – Josias
  wants to finish; don't propose more engine comparisons. 59/60 dev
  commands fully right, recall 0.986 / precision 0.986, 0.07 s per clip,
  261 MB. Weakness: forces everyday speech into command words ("a bit
  late" -> "blade"). Overview of all engines and the reasons:
  `docs/ASR_FINDINGS.md` "Decision". Whisper code stays as a fallback only.
- **Done (2026-10-16):** streaming entry point `stream.py` + service
  `scripts/run_asr.py` (ZeroMQ in from DSP, out to NLU), interface guide for
  the team: `docs/ASR_GUIDE.md`.
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
    repo** (recordings of people are never committed).
  - **Phase B done (2026-10-16):** dev (60) and non-command (19) sets
    recorded (Josias), grammar and correction list tuned, both setups
    measured. Details: `docs/ASR_FINDINGS.md` ("Phase B"). Optional, only
    if there is time: re-record 3 cut-off `log_event` clips
    (`--duration 7 --redo ...`); held-out set by other speakers.
- **The input:** DSP sends 0.5 s chunks, cleaned and beamformed, **only
  while its voice activity detector hears speech** (`src/dsp/source/`:
  per-chunk Silero gate, beamformer output = float32 mono 16 kHz, 8000
  samples). So ASR runs no speech detection of its own: a run of speech
  chunks = one utterance (`stream.py`).
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
Take the DSP's speech chunks, turn each utterance into text and emit **one
JSON per utterance** to the NLU stage – faithfully, without further
processing; NLU does the rest. Fully offline, fast enough on a Raspberry
Pi 4.

## Constraints
- **Offline at runtime.** Models live in `src/asr/models/`. No network calls at
  inference time.
- **Target hardware:** Raspberry Pi 4 (`docs/RPI4_DEPLOYMENT_GUIDE.md`). Laptop
  numbers are only for comparing setups.
- **Input:** 0.5 s speech chunks, float32 mono 16 kHz, with the capture
  time of the first sample; message format in `docs/ASR_GUIDE.md`.
- **End of utterance:** DSP "end" message (preferred), a gap in capture
  times, or 0.8 s without chunks (timeout); 15 s maximum.
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
  `exp(avg_logprob)`, null for empty output; Vosk → mean word confidence
  without "[unk]", null if only "[unk]" was heard.
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
   Whisper needs nothing special. Phase B: the free-form pass runs only
   after a full header ("log event", "record milestone"); otherwise the
   free-form text is used as it is ("Hello" had become "log"). Neither
   engine gets medical free text right ("hepatic" -> "have pet take" /
   "heptic"); NLU stores it verbatim.
2. **Numbers:** done. `words_to_digits` handles "one hundred (and) twenty" →
   120 and "one point five" → 1.5; Whisper's "1.5x" keeps its decimal point
   and is split to "1.5 x". **Written forms chosen (to confirm with Khanh):**
   digits; factor "2 x" (a separate "x" token); "xray"; "centimeters" (also
   for "cm"); "degrees" (also for "°").
3. **Emergency commands are safety-critical:** recognise them when repeated,
   in noise, and with the lowest latency. Idea: Vosk partial results can spot
   "emergency stop" / "halt" / "freeze" before the end-of-utterance wait.
4. **Synonyms:** recognise every synonym in the guide; NLU maps them.
5. **Correction list:** 19 rules from dev errors (Phase B), command words
   only. Never add phrases that can be meant literally ("close skin",
   "scope"); `test_no_test_phrase_is_changed` checks that no sentence in the
   phrase lists is altered.

## Engines (current setups – keep these settings)
**1. Vosk** – `vosk-model-small-en-us-0.15` (`engines/vosk_engine.py`)
- Command grammar built only from `vocab.py`, plus `[unk]`. A Vosk grammar
  accepts any sequence of its entries, but entries also teach it which
  words follow each other: multi-word terms ("go to", "back off", "zoom
  in") are entries, their words are not added alone (Phase B: "to" next to
  "two" gave "go two slice"); plus single-word terms, number words, small
  words and softeners ("little", "bit", "could", "you", "okay").
  "hemostat" and "xray" are not in the model's dictionary -> "hemo stat" /
  "x ray", `normalize()` joins them back.
- Per-word confidence does not separate invented words cleanly ("two" vs
  "to" both 0.5) -> no confidence filter; NLU gets the per-word values.
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
- Says "you" on pure noise without the prompt.
- **Cutting matters (Phase B):** a tight cut (0.3 s after the speech) adds
  junk words ("thank you", "good"): WER 0.08 -> 0.18. Appending 0.6-2 s of
  **digital silence** (zeros) after a tight cut fixes it (WER 0.077) at no
  waiting cost; appended room noise makes it far worse (WER 0.50). Keep
  >= 0.4 s before the speech (0.2 s lost "hand" in "hand scalpel").
- Short prompt (`WHISPER_PROMPT_SHORT`): 0.42 s per clip, same as no prompt;
  the long one took 0.78 s. No hot words.

## Utterances from the DSP stream (implemented, `stream.py`)
- `UtteranceStream(engine)`: `feed(chunk, capture_time)` collects speech
  chunks; the utterance is transcribed in one go when it ends (DSP "end",
  capture-time gap > 0.25 s, or 15 s). Same as the benchmark → measured
  accuracy holds. `segment_start` = first chunk's capture time + first
  word's start; word times relative to it.
- `scripts/run_asr.py`: ZeroMQ service (binds PULL `SCRUB_AUDIO_ADDR`,
  connects PUSH `SCRUB_ASR_ADDR`, `ipc:///tmp/...` defaults like
  `src/orchestrator/`) or `--wav` simulation (Silero from faster-whisper
  marks speech chunks, like the DSP).
- Measured (laptop, fake DSP over ZeroMQ): JSON 0.03 s after a DSP "end"
  message, 0.94 s when relying on the timeout → DSP should send "end".
- The first "log event" of a session takes ~0.6 s extra (one-time setup in
  Vosk's free-speech model; warming up with silence/noise did not help).
- Not done (ideas, only if needed): emergency fast path with Vosk partial
  results; Vosk's own endpointing inside long speech runs.

## Module layout
```
src/asr/
  CLAUDE.md            # this file
  requirements.txt     # pinned dependencies of this module
  schema.py            # ASRResult / WordInfo + to_json() (output contract)
  audio.py             # load wav, downmix, resample to 16 kHz mono float32
  normalize.py         # one normalize() for scoring + the JSON `normalized` field; CORRECTIONS
  vocab.py             # terms per intent (schema), grammar words, prompt sentences, term metrics, leakage check
  metrics.py           # corpus WER, cluster bootstrap CI, paired differences
  bench_configs.py     # 6 configs + PRODUCTION_CONFIG = "vosk-grammar+freeform-log"
  stream.py            # UtteranceStream: DSP speech chunks in, one ASRResult per utterance out
  engines/             # base.py, faster_whisper_engine.py, vosk_engine.py
  test/                # unit tests (python -m unittest discover -s src/asr/test -v)
  scripts/
    record_testset.py  # record a phrase list -> data/asr/<list>/ + manifest (--redo "<phrase>")
    run_asr.py         # the ASR service: ZeroMQ DSP -> ASR -> NLU, or --wav simulation
    transcribe.py      # file(s) -> one JSON line each on stdout (default: PRODUCTION_CONFIG)
    benchmark.py       # configs x clip sets, one process per config -> src/asr/results/
    bench_worker.py    # used by benchmark.py
    compare_configs.py # paired bootstrap: which differences are real
    per_intent.py      # recall and "fully right" clips per intent and config
    make_noise_clips.py# generated noise-only clips -> data/asr/noise_only/
  models/              # gitignored, downloaded on first use
  results/             # gitignored benchmark outputs
data/asr/
  prompts/             # commands_dev.txt (60), commands_heldout.txt (34), non_command.txt (19)
  sample_01-05.wav     # team samples: long read-aloud passages, not commands
  commands_dev/ ...    # recordings + manifests (gitignored)
docs/ASR_FINDINGS.md   # findings log for the project report
docs/ASR_GUIDE.md      # interface for DSP and NLU teammates, how to run
```

## Commands you'll use (from the repository root)
```powershell
.venv\Scripts\activate
python -m unittest discover -s src/asr/test -v                                # unit tests
python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav                # stream simulation
python src/asr/scripts/run_asr.py                                             # service (Pi; ipc:// is Linux-only)
python src/asr/scripts/transcribe.py data/asr/sample_01.wav
python src/asr/scripts/record_testset.py --speaker <name> --prompts-file data/asr/prompts/commands_dev.txt
python src/asr/scripts/make_noise_clips.py
python src/asr/scripts/benchmark.py --configs vosk-grammar --runs 1           # quick check
python -u src/asr/scripts/benchmark.py > src/asr/results/run.log 2>&1         # full run
Get-Content src\asr\results\run.log -Wait                                     # follow it
python src/asr/scripts/benchmark.py --score-only                              # rescore, no inference
python src/asr/scripts/compare_configs.py                                     # which differences are real
python src/asr/scripts/per_intent.py                                          # which commands fail
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
  The benchmark also warns when a config's text differs from its previous
  run (one unexplained case in Phase B, not reproduced in 4 reruns).
  Timing from a busy machine is misleading (Phase B: 0.79 s vs 0.42 s).
- **Dev vs held-out:** tune only on `commands_dev`; run `commands_heldout` once
  at the end. Until then, label results "dev set".
- Grammar and prompt sentence must never contain test sentences
  (`check_leakage` asserts this).

## Next phases
A. ✅ **Vocabulary from the schema** (done 2026-10-14): `vocab.py`,
   `normalize.py`, empty `CORRECTIONS`, `log_event` free-form splice, unit
   tests.
B. ✅ **Record and re-measure** (done 2026-10-16): setups fixed, grammar
   and correction list tuned, cutting rule for Whisper found. Left over:
   re-record 3 cut-off clips; held-out set by other speakers.
C. ✅ **Utterances from the stream** (done 2026-10-16): no own VAD needed –
   DSP gates speech; `stream.py`.
D. ✅ **Entry point** (done 2026-10-16): `scripts/run_asr.py`, ZeroMQ,
   `docs/ASR_GUIDE.md`.
E. **Raspberry Pi 4:** run the service with the real DSP chain; check speed
   (`transcribed in … s` in the log) and memory next to DSP and NLU.
F. **Pull request** to `main` once it runs on the Pi. (Held-out recordings
   by other speakers only if there is time – testing is closed otherwise.)

## Open questions for the team
- **DSP (Minh, Quân):** please send chunks in the format of
  `docs/ASR_GUIDE.md` (header with `capture_time` + float32 bytes) and an
  `"end"` message when speech stops. Which beam if several speakers are
  detected?
- **NLU (Khanh):** `segment_end` proposal; written forms we chose in
  `normalized` (digits, "zoom in 2 x", "xray", "5 centimeters",
  "15 degrees") OK?; verbatim free text for `log_event` OK?; handling of
  `confidence: null` and of an empty `normalized` (Vosk grammar on
  out-of-grammar speech: text "[unk]", normalized "", confidence null);
  NLU binds the PULL socket at `SCRUB_ASR_ADDR`; who maintains the command
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
