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

Owner of this module: Josias. Work happens on the branch
**`asr_refactorings`** (started from `main` on 2026-10-28; `asr_module` keeps
the history up to the squash merge, PR #14).

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
  `docs/ASR_FINDINGS.md` "Decision".
- **Vosk only (2026-10-28):** the benchmark phase is over. Whisper code,
  its prompt sentences and the correction list (Whisper's mishearings) were
  removed – **don't write or adapt code for other engines.** The benchmark
  stays as a regression check for the Vosk setups.
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
- **Rework from the team evaluation (2026-10-28,
  `evaluations/asr/README.md` on main):** in this order, one commit each
  (Josias commits; details and numbers: `docs/ASR_FINDINGS.md`, "Rework") –
  1. ✅ **Priority 3 – no normalization in ASR:** the `normalized` field is
     gone; `text` = the words as heard (`clean_words`: sound-alike spellings
     joined, `[unk]` dropped). `normalize.py` is for scoring only. Benchmark
     unchanged (recall 0.986 / precision 0.986).
  2. ✅ **Priority 1 – synonyms** in `vocab.py` `INTENT_TERMS` (+ context
     words in `GRAMMAR_CONTEXT_WORDS`, not terms): grammar 115 → 157
     entries. Dev recall 0.993 / precision 1.000, non-command false
     commands unchanged (0.08). Left out after measuring: "hold on" ("halt"
     → "hold"), "jump to" ("two" → "to"), "load" / "hide" / "closer"
     (everyday talk landed on them); "wait" (sounds like "eight"). Rule:
     short single words catch everyday speech – prefer phrases, and re-run
     the non-command set after every vocabulary change.
  3. ✅ **Priority 2 – spoken delimiter** "execute" / "over" / "now"
     (`vocab.DELIMITERS`, only after a command word; not "end" – sounds like
     "and"): `stream.py` feeds Vosk chunk by chunk (`accept_chunk`), checks
     the partial result after each chunk (`ends_with_delimiter`), sends at
     once; `finish()` drops the word. New output field `end_reason`. Vosk
     gets only whole 0.2 s steps – otherwise 0.5 s chunks changed words
     (3 of 99 clips); with it, chunked == one go on all clips. Smoke test:
     helps only when noise keeps the DSP detector on (sent 0.5 s after
     "now" instead of when the noise stops); in a quiet room ~0.5 s slower.
  4. ✅ **Vosk only:** Whisper engine, prompts, correction list,
     `engines/base.py` removed; `--wav` simulation uses the DSP's detector.
  5. ✅ **Emergency halt fast path** (`vocab.HALT_PHRASES`: "halt",
     "emergency stop", "hard stop"; `heard_halt`): first halt in the partial
     result → `stream.feed` also returns an early copy (`end_reason`
     "emergency_early", words empty, confidence null); the utterance goes
     on. `feed()` now returns a list. Not "freeze" ("next three slices" was
     read as "next freeze" for a moment), not "stop" alone. Never use
     `SetPartialWords(True)`: it delayed the partial result by ~2 s.
  Check after each step: production config, `--runs 1`, existing clips only
  (no new recordings). **The team-facing overview of the finished module is
  `docs/ASR_GUIDE.md` – keep it up to date.**

## Setup (once per machine)
```powershell
# from the repository root
python -m venv .venv            # .venv/ is already gitignored
.venv\Scripts\activate
pip install -r requirements.txt
```
- Developed with Python 3.11 (the repo README mentions 3.13; check that
  `vosk` installs there before switching).
- Model: the engine downloads Vosk small into `src/asr/models/`
  (gitignored) on first use; after that everything runs offline. The Vosk
  engine adds `--dither=0` to the model config itself.
- The `--wav` simulation uses the DSP's speech detector, which needs torch
  (in the root `requirements.txt`; CPU build:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
  Its Silero model downloads once from GitHub (torch.hub cache).
- Since the merge to main there is no `src/asr/requirements.txt`; the root
  `requirements.txt` is shared by all modules – don't edit it without asking.

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
- **End of utterance:** a spoken delimiter after a command ("next slice
  now"), DSP "end" message (preferred), a gap in capture times, or 0.8 s
  without chunks (timeout); 15 s maximum.
- **Latency target (tentative):** under 1 s from end of speech to JSON for a
  ~2 s command, on the Pi. The budget includes waiting for the chunk that
  contains the end of speech (up to 0.5 s), the silence needed to decide the
  utterance is over, and the transcription. Measure each part.
- **Determinism:** the same audio must give the same text.

## Scope
ASR finds speech in the stream, turns it into text and applies vocabulary
biasing (the Vosk grammar). It does **not** normalize (number
words → digits, fillers, politeness: NLU's normalizer, since 2026-10-28),
classify intents, fill slots or canonicalize synonyms ("knife" → "scalpel"
is NLU's job, `docs/COMMAND_SCHEMA_GUIDE.md` §3A). ASR must transcribe
faithfully, including out-of-domain speech – never force speech into a
command.

## Output schema (contract with NLU – do not change without telling Khanh)
One JSON object per detected utterance:
```json
{
  "text": "zoom in two x please",
  "confidence": 0.87,
  "language": "en",
  "words": [
    {"word": "zoom", "start": 0.12, "end": 0.40, "confidence": 0.95},
    {"word": "in",   "start": 0.40, "end": 0.52, "confidence": 0.93}
  ],
  "segment_start": 1726658793.12,
  "end_reason": "dsp_end"
}
```
- `text`: the words as heard (= `raw_transcript` in the NLU schema), not
  normalized. `clean_words` joins the grammar's sound-alike spellings back
  ("hemo stat" → "hemostat", "c t" → "ct", "x ray" → "x-ray") and drops
  `[unk]`; empty if only `[unk]` was heard.
- The `normalized` field was removed on 2026-10-28 (team evaluation,
  "Priority 3"); NLU already read `text`.
- `confidence`: 0–1, mean confidence of the output words; null if `text`
  is empty.
- `language`: fixed "en".
- `words`: the words of `text` with timestamps in seconds relative to
  `segment_start` and per-word confidence.
- `segment_start`: absolute start time of the utterance = capture time of the
  chunk where speech starts + offset within that chunk.
- `end_reason`: why the utterance ended – "keyword", "dsp_end", "gap",
  "timeout", "max_length"; null for a whole file. Telemetry, added
  2026-10-28 (additive – NLU ignores it). "emergency_early" marks the early
  copy of a halt (not an end; words empty, confidence null; the full
  utterance follows).
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

Since 2026-10-28 each intent also has synonyms from the team evaluation
("bring up", "pull up", "hard stop", "enlarge", "scroll", "go back", "jog",
"pitch", …): the full list is `vocab.py` `INTENT_TERMS`.

What this means for the ASR design (decided in Phase A, measure in Phase B):
1. **`log_event` contains free text**, which a restricted Vosk grammar cannot
   produce (out-of-grammar words become `[unk]`). **Implemented:**
   `VoskEngine(freeform_log=True)` (config `vosk-grammar+freeform-log`) – when
   the grammar hears "log"/"record" among the first three words, a free-form
   recognizer (same model) decodes the same audio; the header ("log event")
   comes from the grammar, the words after it from the free-form result.
   Phase B: the free-form pass runs only after a full header ("log event",
   "record milestone"); otherwise the free-form text is used as it is
   ("Hello" had become "log"). Medical free text is often wrong ("hepatic"
   -> "have pet take"); NLU stores it verbatim.
2. **Numbers:** ASR writes them as words; NLU turns them into digits
   (since 2026-10-28). `normalize.words_to_digits` ("one hundred (and)
   twenty" → 120, "one point five" → 1.5) is only used for scoring.
3. **Emergency commands are safety-critical:** recognise them when repeated,
   in noise, and with the lowest latency. Implemented: halts in Vosk's
   partial result are sent as an early copy (Status, item 5).
4. **Synonyms:** recognise every synonym in the guide; NLU maps them.
5. **No correction list** (removed 2026-10-28): its 19 rules fixed
   Whisper's mishearings and changed nothing on Vosk output.

## Engine: Vosk (keep these settings)
`vosk-model-small-en-us-0.15` (`engines/vosk_engine.py`)
- Command grammar built only from `vocab.py`, plus `[unk]`. A Vosk grammar
  accepts any sequence of its entries, but entries also teach it which
  words follow each other: multi-word terms ("go to", "back off", "zoom
  in") are entries, their words are not added alone (Phase B: "to" next to
  "two" gave "go two slice"); plus single-word terms, number words, small
  words and softeners ("little", "bit", "could", "you", "okay").
  "hemostat" and "xray" are not in the model's dictionary -> "hemo stat" /
  "x ray", `clean_words()` in the engine joins them back.
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
- Streams natively: fed chunk by chunk (`accept_chunk`, whole 0.2 s steps),
  partial results after each chunk. `AcceptWaveform` returns True when Vosk
  guesses the utterance is over (endpoint rules in `model.conf`) – ignored;
  the stream decides.

## Utterances from the DSP stream (implemented, `stream.py`)
- `UtteranceStream(engine)`: `feed(chunk, capture_time)` collects speech
  chunks; the utterance ends with a spoken delimiter after a command
  ("keyword"), DSP "end", a capture-time gap > 0.25 s, the service's
  timeout, or 15 s (`end_reason`). Vosk gets each chunk as it arrives
  (whole 0.2 s steps) and its result equals transcribing in one go, so the
  benchmark → measured accuracy holds. `segment_start` = first chunk's
  capture time + first word's start; word times relative to it.
- `scripts/run_asr.py`: ZeroMQ service (binds PULL `SCRUB_AUDIO_ADDR`,
  connects PUSH `SCRUB_ASR_ADDR`, `ipc:///tmp/...` defaults like
  `src/orchestrator/`) or `--wav` simulation (the DSP's own detector,
  `src/dsp/source/vad.py`, marks speech chunks with the same per-chunk rule
  as `run_dsp.py`; needs torch).
- Measured (laptop, fake DSP over ZeroMQ): JSON 0.03 s after a DSP "end"
  message, 0.94 s when relying on the timeout → DSP should send "end".
- End of speech → NLU measured 0.6–1.0 s with whole 0.5 s chunks (ASR's
  share 0.04 s): the chunk size dominates the latency budget. Ask DSP to
  send "end" / a shorter last chunk as soon as speech stops.
- The first "log event" of a session takes ~0.6 s extra (one-time setup in
  Vosk's free-speech model; warming up with silence/noise did not help).
- Emergency halt fast path: done (2026-10-28, see Status, item 5).
- Not done (idea, only if needed): Vosk's own endpointing inside long
  speech runs.

## Module layout
```
src/asr/
  CLAUDE.md            # this file
  schema.py            # ASRResult / WordInfo + to_json() (output contract)
  audio.py             # load wav, downmix, resample to 16 kHz mono float32
  normalize.py         # one normalize() for scoring only (benchmark)
  vocab.py             # terms per intent (schema), grammar words, delimiters, term metrics, leakage check
  metrics.py           # corpus WER, cluster bootstrap CI, paired differences
  bench_configs.py     # 3 Vosk setups + PRODUCTION_CONFIG = "vosk-grammar+freeform-log"
  stream.py            # UtteranceStream: DSP speech chunks in, one ASRResult per utterance out
  engines/             # vosk_engine.py (the only engine)
  test/                # unit tests (python -m unittest discover -s src/asr/test -v)
  scripts/
    record_testset.py  # record a phrase list -> data/asr/<list>/ + manifest (--redo "<phrase>")
    run_asr.py         # the ASR service: ZeroMQ DSP -> ASR -> NLU, or --wav simulation (--send: also to NLU)
    listen_asr.py      # stand-in NLU: prints what ASR sends (+ end of speech -> arrival time)
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
python src/asr/scripts/listen_asr.py                                          # stand-in NLU (terminal 1)
python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav --send         # ASR -> NLU test (terminal 2)
python src/asr/scripts/run_asr.py                                             # service (Pi; ipc:// is Linux-only)
python src/asr/scripts/transcribe.py data/asr/sample_01.wav
python src/asr/scripts/record_testset.py --speaker <name> --prompts-file data/asr/prompts/commands_dev.txt
python src/asr/scripts/make_noise_clips.py
python src/asr/scripts/benchmark.py --configs vosk-grammar+freeform-log --runs 1   # quick check after a change
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
- The grammar must never contain test sentences (`check_grammar_leakage`
  asserts this).

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
- **DSP (Minh, Quân):** **send one chunk of pre-roll** – `run_dsp.py`'s
  per-chunk gate drops the chunk where a word starts: 18 of 60 dev commands
  lost their first word ("Freeze" → nothing; recall 0.884 vs 0.993 with
  pre-roll, 2026-10-28, `docs/ASR_GUIDE.md` §7). `capture_time` is taken
  at the chunk's end, not its first sample (~0.5 s late). Which beam if
  several speakers are detected?
- **NLU (Khanh):** `segment_end` proposal; NLU's normalizer must now handle
  number words itself (today "one hundred and twenty" -> "100 and 20", see
  `docs/ASR_GUIDE.md` §3); the new synonyms (Priority 1) must be added to
  NLU too; verbatim free text for `log_event` OK?; who maintains the
  command list?
- **Everyone:** is the 1 s budget for the whole chain or ASR alone? Memory
  left on the Pi next to DSP and NLU? Fallback: ASR on the workstation?

## Working conventions
- Work in small steps. At the end of each phase, run the code and show me the
  output before moving on.
- Explain any new tool, library or concept briefly the first time it appears.
- Keep dependencies minimal (root `requirements.txt`, shared – ask first).
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
- Commit small, meaningful steps on `asr_refactorings`; never push to `main`.
