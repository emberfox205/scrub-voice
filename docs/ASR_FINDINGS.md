# ASR Module – Findings

> **All numbers in this file are "dev set" numbers.** The same 46 clips
> (one speaker) were used to find problems and tune settings, so results are
> optimistic. Final numbers will come from a held-out set recorded later.
>
> **All numbers so far were measured on the OLD image-viewer commands**
> (pan, window, axial, …). The command set has since changed (see below).
>
> This log was started in a standalone project before the module moved into
> this repository. References to `archive/`, `results/archive/…`, the first
> test set (`data/clean/`, `data/manifest.csv`) and `scripts/diagnose.py`
> point to that local project folder (owner: Josias) and are not part of the
> repo. The module itself lives in `src/asr/`.

## Decisions after the meeting and current state (2026-10-14)

### Decisions
| topic | decision | reason |
|---|---|---|
| Engines | Continue with **Vosk + command grammar** (first choice) and **Whisper Tiny + prompt sentence** (second choice) | Best command recall with almost no invented commands; both give a confidence score |
| Moonshine | **Dropped** (engine code and skill kept in `archive/moonshine/`) | Does not fit our streaming front end; no confidence score |
| Whisper Base, beam 5, hot words | Dropped | No real gain; hot words cause repetition and false commands |
| Input from the audio front end | Cleaned, beamformed audio arrives as a **continuous stream of 0.5 second pieces** | Decided with the audio front end team |
| Finding speech | **Our job now:** detect where each command starts and ends inside that stream | The audio front end no longer sends finished segments |
| Command list | Now defined by the language understanding contract, `schemas/command_schema.json` (12 commands in 4 groups: tool handling, emergency, monitor display, camera and light) | Agreed interface with Khanh |

### What still holds, and what must be redone
| still holds | must be redone for the new commands |
|---|---|
| Measurement method (fair scoring, error bars, commands found/invented, noise check, one process per engine, repeated timing) | Vocabulary, Vosk grammar and Whisper prompt sentence (`src/asr/vocab.py`) |
| Engine fixes (same input → same output, fast Vosk setup) | Correction list (built from errors on the old commands) |
| Lesson: teaching the vocabulary matters more than model size | Test recordings: new phrase lists are ready in `data/asr/prompts/` (60 development, 34 held-out, 19 non-command) |
| Whisper says "you" on noise without the prompt; trailing silence adds junk words | All accuracy and speed numbers, now measured on a simulated stream of 0.5 second pieces |

### New challenges from the new command list
- **"Log event" takes free text** ("record milestone, trocar insertion
  complete, note no bleeding"). A command grammar can't write arbitrary
  words, so Vosk needs a second, free-speech recognizer for this command (or
  Whisper takes over for it).
- **Numbers matter now:** zoom factors (1.1–10, e.g. "one point five"),
  slice numbers (up to the hundreds), distances (1–25 centimeters), angles
  (1–45 degrees).
- **Emergency commands** ("emergency stop", "halt robot", "freeze") must be
  recognised fastest and most reliably of all.

### Cleanup done (2026-10-14)
- Moonshine engine and its skill moved to `archive/moonshine/`; its model
  files (290 megabytes) deleted, package uninstalled.
- Benchmark configs reduced to four: Whisper Tiny with and without the
  prompt sentence, Vosk free speech and Vosk with grammar.
- Earlier results moved to `results/archive/` (v1, v2 stage 2, v3 final run).
- Phrase lists moved to `data/asr/prompts/`; the non-command list was fixed
  ("pass me the clamp" is a real command now).
- Command contract saved in `schemas/` (reference copies).
- The full plan for the next steps is in `src/asr/CLAUDE.md` (phases A–G).

## Meeting notes – speech recognition stage (2026-10-10)

**In one sentence:** all three speech recognition engines work offline and
deliver the agreed output format to the language understanding stage. After
fixing the measurement method and the engines, the best candidates for the
Raspberry Pi are **Vosk with a command grammar** and **Whisper Tiny with a
prompt sentence**. Next: a test on the Raspberry Pi and more recordings.

### How we got here
1. Built the speech recognition pipeline and recorded 46 test commands
   (one speaker so far).
2. The first measurement was **misleading**: it counted commas and "um" as
   errors, ignored invented commands, and mixed up the memory readings of
   different engines.
3. Rebuilt the measurement: fair scoring, error bars, "commands found"
   **and** "commands invented", a test with pure noise, and every engine
   measured on its own.
4. Fixed three engine bugs that made the same recording give a different
   text each time.
5. Tested six ways of teaching the engines our command vocabulary:
   - ✅ **Command grammar** (Vosk): only our command words are allowed,
     everything else is marked as "unknown".
     Commands found: 66% → 98%.
   - ✅ **Prompt sentence** (Whisper): one example sentence with our terms is
     given to the model before every command.
     Commands found: 60% → 92%.
   - ❌ **Hot words** (Whisper): a list of our terms the model should prefer.
     The model repeats itself ("zoom in, zoom in, zoom in, …") and says
     "zoom in" on pure noise.
   - ❌ **Prompt sentence and hot words together** (Whisper): the same
     problems as hot words alone.
   - ❌ **Key terms at three strengths** (Moonshine): no measurable
     improvement; the stronger the setting, the more often it outputs words
     on pure noise.
   - ✅ **Correction list** (all engines): fixes typical mishearings
     afterwards, for example "pen left" → "pan left" and
     "long window" → "lung window". Errors roughly halved – but the list was
     built from the same recordings, so the real gain is probably smaller.

### Engines at a glance (laptop, best setup of each)

| engine (setup) | commands found | invented commands | on pure noise | seconds per command | memory (megabytes) | confidence score | verdict |
|---|---|---|---|---|---|---|---|
| **Vosk + command grammar** | **98%** | 2% | silent | **0.07** | **216** | yes | ✅ first choice |
| **Whisper Tiny + prompt sentence** | **92%** | **0%** | silent | 0.87 | 325 | yes | ✅ second choice (check speed on the Raspberry Pi) |
| Whisper Tiny (no help) | 64% | 9% | says "you" (53%) | 0.31 | 323 | yes | works, but misses many commands |
| Whisper Base (no help) | 72% | 10% | says "you" (73%) | 0.90 | 386 | yes | ❌ slower, not better |
| Vosk (free speech) | 66% | 0% | says "huh" (33%) | 0.10 | 306 | yes | backup option |
| Moonshine Small | 66% | 6% | mostly silent | 1.85 | 398 | **no** | ❌ no confidence score, slow |
| Moonshine Tiny | 64% | 3% | silent | 0.76 | 240 | **no** | ❌ no confidence score |
| Whisper + hot words | – | – | **says "zoom in" (93%)** | – | – | yes | ❌ repeats itself, false commands |

- **Commands found:** share of spoken commands that came out correctly.
- **Invented commands:** share of output commands that were never said.
- **On pure noise:** what the engine outputs when nobody speaks.
- The error bars are wide (one speaker, 46 recordings): differences of less
  than about 15 percentage points are mostly not real – the first and second
  choice count as a tie.

### Progress along the way

| step | what changed | effect |
|---|---|---|
| First measurement | – | Moonshine Small looked best, Vosk grammar "98%" – both misleading |
| Fair scoring | error bars; invented commands counted | most differences turned out to be chance; Vosk grammar invented a command in 1 of 8 recordings |
| Engine fixes | same input now always gives the same output | unstable recordings 8 → 0; Vosk 8 times faster (0.80 → 0.10 seconds) |
| Vosk grammar with "unknown" word | grammar may now reject speech | invented commands 12% → 2% |
| Whisper prompt sentence | one sentence with our terms | Whisper Tiny: commands found 60% → 92% |
| Correction list | for example "pen left" → "pan left" | errors roughly halved for every engine (probably optimistic) |
| Dead ends | hot words, Moonshine key terms | repetition and false commands; no effect |

**What we learned:** teaching the engines our vocabulary (grammar, prompt
sentence, correction list) matters far more than choosing a bigger model.

### Questions and decisions for the team

**Audio front end – digital signal processing (Minh, Quân)**
- **Streaming?** Could you send small pieces of audio (about 100
  milliseconds) *while* someone is speaking, plus a signal when speech
  starts and ends, instead of the finished recording? Only needed if we
  choose Moonshine (answer time 1.2 seconds → 0.03 seconds); Vosk and
  Whisper work fine with finished recordings.
- In what format will the recordings arrive (sample rate, number of
  channels)? Will silence at the start and end be cut off? Extra silence
  made Whisper add junk words.
- How often will your voice activity detection pass on noise without any
  speech? (Whisper says "you" on noise.)

**Language understanding – natural language processing (Khanh)**
- What should happen when there is no confidence score, or with junk words
  like "you", "huh" or the "unknown" marker?
- Which is worse in the operating room: a missed command or an invented
  one?
- **Who owns the final list of commands?** The Vosk grammar must match it
  exactly, and every new command means updating the grammar. Do we need
  PACS (the hospital image archive) as a command? Vosk can't recognize it.

**Everyone – running on the Raspberry Pi**
- Is the limit of 1 second for speech recognition alone, or for the whole
  chain (audio processing + speech recognition + language understanding)?
- How much memory is left for speech recognition next to the audio
  processing on the Raspberry Pi 4? (Our candidates need about 220–330
  megabytes.)
- If the Raspberry Pi is too slow: could speech recognition run on the
  workstation instead?
- Flexible but slower (Whisper) or fast but rigid (Vosk grammar)? To be
  decided after the Raspberry Pi test.

### Next steps
1. Record normal operating room talk that is not a command (for example
   "pass me the clamp") – the biggest open risk for the Vosk grammar.
2. More speakers and a fresh set of recordings that we don't tune on, for
   the final numbers.
3. Run both candidates on the Raspberry Pi 4.

---

## Detailed summary (as of 2026-10-10)

### What has been done
- **Engines working end to end:** faster-whisper (tiny.en, base.en), Moonshine
  (Tiny, Small Streaming) and Vosk (free-form, restricted grammar), all
  offline, all producing the agreed JSON for NLP.
- **Test set:** 46 recorded commands (one speaker, 26 command groups) plus 15
  generated noise-only clips.
- **Benchmark rebuilt (v2)** so the numbers can be trusted: fair text
  normalization, corpus WER with 95% confidence intervals, term recall *and*
  precision, false-command checks on noise, each model measured in its own
  process, warm-up excluded, every clip timed 3×.
- **Three engine bugs found and fixed**, each making the same clip give
  different text on repeated runs: Moonshine's built-in VAD leaked state,
  Vosk added random dither noise, Whisper fell back to random sampling. All
  20 configs are now fully repeatable.
- **20 configs compared**, including every biasing method from CLAUDE.md
  (Whisper prompt and hotwords, Moonshine keyterms, Vosk grammar,
  post-correction map).

### Key results (details in stage 6)
1. **Shortlist for the Pi:** (1) **Vosk with the command grammar**,
   (2) **faster-whisper tiny.en, beam 1, with a vocabulary prompt.** Both find
   92–98% of the spoken commands and almost never invent one. Statistically
   they are tied on command recall; Vosk is ~13× faster and uses less RAM,
   Whisper gives the better word-for-word transcript.
2. **Moonshine is not recommended** despite the best raw WER (Small): it gives
   **no confidence score** (required by spec 2.2, needed by NLP to reject
   doubtful input), and Small is too slow (1.85 s per clip on a laptop).
3. **Whisper hotwords are unusable:** they cause repetition loops ("zoom in,
   zoom in, …" 70+ times) and made base.en output "zoom in" on 14 of 15
   noise-only clips.
4. **Moonshine keyterms had no measurable effect** on accuracy and made
   noise hallucinations worse as the boost went up.
5. **Most other differences are within noise.** Beam 1 vs 5 and tiny vs base
   are not distinguishable with 46 clips from one speaker.
6. **Every engine trips over the same words:** pan, zoom, sagittal, lung,
   bone, coronal, axial, annotate. A small post-correction map fixes many of
   them, but it was built from these same clips, so its gain is optimistic.

### Open questions for the meeting
- **More speakers:** who can record the test phrases? Error bars are wide
  (±0.08–0.12 WER), and we can't tell whether the "pan"/"zoom" errors are
  engine or pronunciation.
- **Khanh:** how should NLP treat `confidence: null`, and a lone "you" (what
  Whisper outputs on silence)?
- **Khanh:** which is worse in the OR, a missed command or an invented one?
  Both shortlisted configs are high-precision, but that choice decides
  thresholds.
- **DSP team (Minh, Quân):** could you stream audio chunks during speech
  instead of finished segments? That would cut Moonshine's response time to
  ~0.03 s, but it still lacks confidence (see stage 3).
- **Test set gaps:** no numbers and no acronyms (CT, MRI, DICOM, PACS) yet;
  no recordings of non-command OR speech yet (the most important missing
  test for Vosk grammar).
- **Next:** a held-out recording set for final numbers, then measure the
  shortlist on the Pi 4.

---

## Benchmark v2 – what changed in the method, and why

| Problem in v1 | Fix in v2 | Why it matters |
|---|---|---|
| Fillers ("um"), number words, acronym spellings, hyphens counted as errors | One `normalize()` (`src/asr/normalize.py`) applied identically to reference and transcript: lowercase, strip punctuation, split hyphens, number words → digits, canonical acronyms, drop fillers | Only real recognition errors are counted |
| Mean of per-clip WER | **Corpus WER** = total errors / total reference words (per-clip mean kept as a secondary column) | A 1-word clip no longer weighs as much as a 7-word sentence |
| No error bars | **95% confidence interval** by bootstrap over command groups (1000 resamples) | Shows which differences are real and which are luck |
| "Term accuracy" only checked if said commands came out | **Term recall** (said → output) and **term precision** (output → actually said), whole-word matching, counted per occurrence | Catches invented commands |
| All models in one process | Each config in its **own fresh process** | RAM numbers are no longer polluted by earlier models |
| First call (warm-up) included in timing, one run per clip | Model load timed separately, 1 warm-up thrown away, each clip timed **3×, median kept** | Latency reflects steady-state use |
| No hardware info | Machine info saved to `src/asr/results/machine.json` | Numbers can be compared with the Pi later |

**About the confidence interval.** Several clips are phrasings of the same
command ("Zoom in" / "Um, zoom in a little, please"), so they aren't
independent. The bootstrap therefore resamples whole command groups (26
groups), not single clips. Two configs are compared on identical resamples
("paired"): if the interval for their difference contains 0, the difference
is not distinguishable from noise.

### Test-set coverage of the vocabulary (`src/asr/vocab.py`)
- Covered (1–2 occurrences each): zoom in/out, pan left/right/up/down,
  rotate, flip, reset, contrast, brightness, window, level, lung window,
  bone window, soft tissue, axial, coronal, sagittal, next slice,
  previous slice, scroll, series, measure, annotate, undo.
- **Not covered at all: DICOM, PACS, CT, MRI.** There are also no numbers in
  any reference.

---

## Stage 1 – Where did the v1 WER come from?

Method (`scripts/diagnose.py` in the local project, on the v1 transcripts, nothing re-run): apply
each scoring fix one after another to both reference and transcript and
count how many errors each fix removes. What is left is genuine
misrecognition.

| engine | raw errors | punctuation / hyphens | numbers | acronyms | fillers | politeness | genuine |
|---|---|---|---|---|---|---|---|
| fw-tiny.en-b5 | 58 | 31 (53%) | 0 | 0 | 1 (2%) | 0 | 26 (45%) |
| fw-tiny.en-b1 | 66 | 37 (56%) | 0 | 0 | 1 (2%) | 0 | 28 (42%) |
| fw-base.en-b5 | 49 | 28 (57%) | 0 | 0 | 1 (2%) | 0 | 20 (41%) |
| fw-base.en-b1 | 54 | 32 (59%) | 0 | 0 | 1 (2%) | 0 | 21 (39%) |
| moonshine-tiny | 84 | 48 (57%) | 0 | 0 | 0 | 1 (1%) | 35 (42%) |
| moonshine-small | 67 | 49 (73%) | 0 | 0 | 0 | 0 | 18 (27%) |
| vosk-freeform | 46 | 11 (24%) | 0 | 0 | 2 (4%) | 0 | 33 (72%) |
| vosk-grammar | 73 | 7 (10%) | 0 | 0 | 2 (3%) | 6 (8%) | 58 (79%) |

How to read it:
- **Punctuation** looks large, but v1 already stripped punctuation, so it
  mostly explains why Whisper/Moonshine ("Zoom out.") and Vosk ("zoom out")
  looked so different in the raw output.
- **Numbers and acronyms: 0%** because the test set contains none – a gap
  in the test set, not proof the engines handle them.
- **Fillers: 1–2 errors per engine** (only 2 clips contain "um"/"uh").
  Whisper drops fillers on purpose.
- **Politeness words** only matter for Vosk grammar: its grammar can't
  produce most words around "please", so they become `[unk]` or get forced
  into a command.
- **Most remaining errors are real misrecognitions**, concentrated on a few
  words:

| said | heard as (by which engine) |
|---|---|
| pan | pen / penn (all), hand, and, "10" (Whisper tiny) |
| zoom | soon (most), room (Vosk), boom (Moonshine tiny), "so I'm" (Whisper tiny) |
| sagittal | "set your tail", "such a little", "said she to", "satchito", "such eternal" |
| lung | long (Whisper, Moonshine) |
| bone | phone (Whisper) |
| annotate | "and rotate" (Whisper base), "and notate" (Moonshine tiny), "and tate" (Vosk) |
| coronal | corona / corono |
| axial | exhale (Moonshine small), axiom (Vosk) |

Vosk with a grammar makes a different kind of error: it **forces speech into
commands** ("Reset the view, please" → "reset tissue slice", "Go to axial,
please" → "rotate axial series").

Two scoring bugs found and fixed for v2: Vosk's `[unk]` marker became the
word "unk" after punctuation stripping, and faster-whisper reported
confidence 1.0 for an empty transcript.

---

## Stage 2 – Same engines, new scoring and measurement

Machine: Windows 11, Intel 12th-gen laptop CPU (12 cores / 16 threads),
15.7 GB RAM, Python 3.11. Full details in
`results/archive/v3_final_2026-10-10/machine.json` (same laptop). Stage 2
files: `results/archive/v2_stage2_2026-10-07/`.
Engines were not changed yet (v1 settings), only the scoring and measurement.

| config | v1 WER | **v2 corpus WER [95% CI]** | v2 clip WER | v1 term acc | recall | precision | invented terms / clip | median / p95 latency (s) | RTF | peak RAM (MB) | conf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fw-tiny.en-b5 | 0.242 | **0.200** [0.117–0.291] | 0.240 | 0.656 | 0.638 | 0.909 | 0.07 | 0.39 / 1.36 | 0.123 | 383 | yes |
| fw-tiny.en-b1 | 0.251 | **0.215** [0.120–0.326] | 0.250 | 0.633 | 0.617 | 0.906 | 0.07 | 0.35 / 1.38 | 0.118 | 385 | yes |
| fw-base.en-b5 | 0.198 | **0.154** [0.083–0.238] | 0.193 | 0.744 | 0.723 | 0.895 | 0.09 | 0.71 / 0.85 | 0.193 | 516 | yes |
| fw-base.en-b1 | 0.192 | **0.162** [0.077–0.258] | 0.193 | 0.744 | 0.723 | 0.895 | 0.09 | 0.75 / 1.18 | 0.214 | 512 | yes |
| moonshine-tiny | 0.288 | **0.277** [0.170–0.390] | 0.286 | 0.600 | 0.617 | 0.906 | 0.07 | 0.62 / 0.99 | 0.161 | 241 | **no** |
| moonshine-small | 0.150 | **0.138** [0.081–0.202] | 0.151 | 0.711 | 0.723 | 0.971 | 0.02 | 1.50 / 1.92 | 0.372 | 391 | **no** |
| vosk-freeform | 0.269 | **0.262** [0.159–0.382] | 0.267 | 0.622 | 0.638 | 1.000 | 0.00 | 0.80 / 0.98 | 0.203 | 244 | yes |
| vosk-grammar | 0.397 | **0.469** [0.379–0.546] | 0.361 | 0.956 | 0.979 | 0.885 | 0.13 | 0.08 / 0.09 | 0.021 | 217 | yes |

Clips are 4 s long (fixed recording window), so RTF = latency / 4 s.
Latency target in CLAUDE.md: under 1 s for a ~2 s command **on the Pi**.

### Which differences are real? (paired bootstrap on corpus WER)

| comparison (A − B) | difference [95% CI] | verdict |
|---|---|---|
| tiny.en beam1 − beam5 | +0.015 [−0.015, +0.043] | not distinguishable |
| base.en beam1 − beam5 | +0.008 [−0.029, +0.051] | not distinguishable |
| base.en − tiny.en (beam5) | −0.046 [−0.123, +0.032] | not distinguishable |
| moonshine-small − fw-base.en-b5 | −0.015 [−0.090, +0.053] | not distinguishable |
| **moonshine-small − moonshine-tiny** | **−0.138 [−0.231, −0.052]** | **real** |
| vosk-freeform − fw-tiny.en-b5 | +0.062 [−0.030, +0.153] | not distinguishable |
| **vosk-grammar − vosk-freeform** | **+0.208 [+0.064, +0.326]** | **real** |

### What the numbers say
- **WER dropped for every engine except Vosk grammar** once fillers,
  hyphens etc. stopped counting. Vosk grammar went *up* (0.397 → 0.469)
  because its errors sit in the longer sentences, which per-clip averaging
  used to under-weight.
- **Term precision is the new important column.** Vosk grammar hears almost
  every command (recall 0.979) but 11.5% of the commands it outputs were never
  said. Free-form Vosk never invented one (precision 1.000); Moonshine Small
  rarely did (0.971).
- **RAM is now the whole process's peak** (Python + libraries + model,
  217–516 MB), i.e. what the process would really use on the Pi. The v1
  numbers were "extra RAM on top of earlier models" and can't be compared.
- **Latency no longer includes the warm-up call.** Moonshine Small's p95
  dropped from 2.57 s to 1.92 s for that reason; its median of 1.50 s for a
  4 s clip is still over the 1 s target even on a laptop.
- fw-tiny.en-b5's 4.3 s load time is a cold disk cache (it ran first), not
  the model itself.

### Which output fields does each engine actually fill? (CLAUDE.md schema)

| engine | `confidence` | `words` + timestamps | per-word `confidence` |
|---|---|---|---|
| faster-whisper | yes – exp(avg_logprob); **null for empty output** (was a false 1.0) | yes | yes, varies |
| Moonshine | **null** – no transcript-level score exists | yes | **null** – library returns a constant 1.0, which carries no information |
| Vosk | yes – mean word confidence | yes | yes, varies |

### Measurement caveats found in stage 2
- **Same clip, different transcript.** Across the 3 timed runs, the
  transcript changed on 8/46 clips for Moonshine Tiny, 4/46 for Vosk
  free-form, and 0–2 for the others. Only words the engine is unsure about
  flip. Being investigated as a bug in stage 3 (see below).
- **Laptop timings drift under sustained load:** repeating one Vosk clip went
  from 0.55 s to ~1.0 s within a few calls. Laptop latencies are only for
  comparing engines with each other; the Pi numbers are what count.
- **Vosk's latency is mostly setup, not recognition:** it built a new
  recognizer for every clip (~0.5 s). Reusing one with `Reset()` brings it to
  ~0.2 s. Fixed in stage 3.

---

## Stage 3 – Engine setups fixed

### Bug: the same clip gave different transcripts on repeated runs
In stage 2, repeating a clip 3× changed the transcript on 8/46 clips
(Moonshine Tiny), 4/46 (Vosk free-form), 2/46 (Moonshine Small) and 1/46
(faster-whisper beam 1). That makes every comparison noisier, so each case was
investigated as a bug.

| engine | root cause (how we found it) | fix | clips differing across 3 runs, before → after |
|---|---|---|---|
| Moonshine | Its **built-in voice-activity detector (VAD) keeps state between calls** – even into a brand-new `Transcriber`. Two fresh processes produced the *identical* drifting sequence, and silence clips between calls didn't reset it. Clips near the VAD threshold got split differently each time ("Corona, view," vs "Corona, \| View."). | `vad_threshold=0` (built-in VAD off). Our DSP stage already runs Silero-VAD, so segments arrive pre-cut. | Tiny 8 → **0**, Small 2 → **0** (tested 46 clips × 3 runs). A threshold of 0.1 only halved the problem. |
| Vosk | Kaldi's default **dither** adds a little random noise to the audio before feature extraction, drawn from a process-wide random generator. The model's `mfcc.conf` didn't set it. Uncertain words then flip ("room in" / "in" / "zoom in"). | `--dither=0` added to the model's `mfcc.conf` automatically by the engine. | 4 → **0** (free-form and grammar) |
| faster-whisper | When a greedy decode looks bad, the default retries with **random sampling** (temperature > 0). | `temperature=0` | 1 → **0** (clip 039, 6 repeats) |

Note: reusing a recognizer and calling `Reset()` (below) did **not** fix the
Vosk variation on its own – dither was the cause.

### Other engine fixes
- **Vosk:** one recognizer per engine, `Reset()` after each clip, instead of a
  new recognizer per clip. Median latency 0.80 s → **0.13 s** (free-form);
  building the recognizer cost more than recognizing.
- **Vosk grammar:** built only from `src/asr/vocab.py` (commands, digits
  zero–nine, "please", "by", "times") plus `[unk]`, so speech outside the
  grammar is rejected instead of forced into a command. Acronyms use spoken
  forms ("c t", "m r i", "die com"); PACS is left out (its spoken form
  "packs" is an everyday word). All 42 phrases exist in the model's
  vocabulary. An assert fails if a test sentence appears in any biasing list
  (grammar, prompt, hotwords, keyterms); it was checked to fire on a
  deliberately injected test sentence.
- **faster-whisper:** `language="en"`, `temperature=0`,
  `condition_on_previous_text=False`, `word_timestamps=True`.
- **All engines:** the JSON `normalized` field is now filled
  (normalize + correction map + politeness words dropped, per CLAUDE.md).

### Moonshine: why it's slower than its published numbers
Measured on all 46 clips (laptop, after ~1 h of full load, so absolute times
are ~2× slower than in stage 2):

| | Tiny Streaming | Small Streaming |
|---|---|---|
| audio → library format conversion | 5 ms | 7 ms |
| whole clip at once (`transcribe_without_streaming`), median / p95 | 1.17 / 1.86 s | 2.17 / 3.02 s |
| same, word timestamps off | 1.19 / 1.60 s | 2.25 / 3.48 s |
| **streaming**: compute spent *while* 4 s of audio arrives | 2.46 s (keeps up) | 4.58 s (**falls behind**) |
| **streaming: response time after the last audio chunk**, median / p95 | **0.03 / 0.09 s** | 0.19 / 3.03 s |

- Setup was not the problem: the model is loaded once and reused, clips use
  `transcribe_without_streaming` (no chunked feeding), the model files are
  the quantized ones (`quantized_26_08_21`), the architecture is correct, and
  there is no thread-count option (`num_threads` is rejected).
- **The difference is what gets measured.** The docs say "for streaming
  models most of the work is done as the audio is being added", and the
  README says "low latency by doing work while the user is still talking".
  Published latency is the response time *after* speech ends during
  streaming. We measured the time to process a finished clip, which is a
  different (and for Moonshine much less favourable) number.
- Moonshine Small can't stream in real time even on this laptop; on the Pi
  only Tiny is realistic for streaming.

### What streaming would mean for our pipeline (note only, not implemented)
Today DSP hands us a *finished* segment after Silero-VAD detects the end of
speech, and only then does ASR start. With Moonshine Tiny streaming:
- **Latency gain:** ASR would work while the surgeon is still talking. The
  wait after end of speech drops from ~1.2 s to ~0.03–0.09 s on this laptop,
  which would put us well under the 1 s target even with the Pi being slower.
- **Interface change with DSP (Minh, Quân):** instead of one array per
  segment, DSP would push small chunks (e.g. 100 ms) as they come out of the
  beamformer/denoiser, plus a "speech started" and "speech ended" signal from
  their VAD. That needs a streaming transport (a ZeroMQ PUSH/PULL socket or
  a queue), not a single function call.
- **Segment boundaries:** DSP's VAD would still decide when a command ends;
  Moonshine's own VAD stays off (it's the source of the determinism bug
  above). We'd call `stream.stop()` on DSP's end-of-speech signal.
- **JSON contract with Khanh:** unchanged. One JSON per command, emitted on
  end-of-speech. `segment_start` would be the timestamp of DSP's
  "speech started" signal instead of the start of a pre-cut array. Partial
  results (`on_text`) could optionally be sent earlier, but NLP must not act
  on them (they change).
- **Risk:** Moonshine still has no confidence score (see stage 2), so
  streaming would make it fast but not safer.

---

## Stage 4 – Vocabulary biasing setups

One shared term source, `src/asr/vocab.py`. Each biasing method is its own config
so its effect can be measured against its baseline. faster-whisper biasing
uses beam 1 only.

| method | configs | what is passed |
|---|---|---|
| Whisper `initial_prompt` | tiny.en / base.en beam1 `+prompt` | one natural sentence containing the commands (`WHISPER_PROMPT`) |
| Whisper `hotwords` | `+hot` | the 30 commands, comma-separated |
| both | `+prompt+hot` | both of the above |
| Moonshine `keyterms` | tiny / small `+kt1`, `+kt2`, `+kt3` (= `keyterm_boost` 1/2/3) | 30 commands **padded to 100 terms** with plausible viewer/radiology terms (`DEPLOY_PADDING`), because false insertions grow with list length. Docs: default boost 2.0, never above 4.0. |
| Vosk grammar | `vosk-grammar` vs `vosk-freeform` | see stage 3 |
| post-correction map | `<baseline>+corr` rows | same transcripts, rescored with `CORRECTIONS` – no extra inference |

Spot check: Moonshine keyterms *do* change `transcribe_without_streaming`
output (so they are applied), but in a quick test on Tiny they changed wrong
words into other wrong words ("Pen right" → "Penn Wright"). Full numbers in
stage 6.

### Post-correction map – where each entry comes from
Only errors seen in the stage-1 diagnosis, and only phrases that would almost
never be meant literally in an OR. Deliberately **not** included because they
could turn real speech into a false command: "and rotate this" (a real
rotate command), "room in", "hand down", "and down", "10 left".

| entry (normalized text) | → | seen from (stage 1) |
|---|---|---|
| pen / penn + left, right, up, down | pan … | fw-tiny, fw-base, Moonshine tiny/small ("pen"); Vosk free-form ("penn") |
| soon in / soon out | zoom in / out | fw-tiny, fw-base, Moonshine tiny/small, Vosk free-form |
| boom in | zoom in | Moonshine tiny |
| long window | lung window | fw-tiny, fw-base, Moonshine tiny/small |
| phone window | bone window | fw-tiny, fw-base |
| corona view, corono view | coronal view | fw-tiny, fw-base, Moonshine tiny, Vosk free-form ("corona"); Moonshine small ("corono") |
| paronal, plaronal, pluronal | coronal | Moonshine small, fw-base, fw-tiny |
| exhale view, axiom view | axial view | Moonshine tiny/small ("exhale"); Vosk free-form ("axiom") |
| saggitel, seggitel, segital, sekito, succito, satchito | sagittal | fw-base, Moonshine small, Moonshine tiny, fw-tiny, fw-base, Moonshine small |
| saggy tell | sagittal | Vosk free-form |
| set your tail / such a little / such eternal / said she to + view | sagittal view | fw-tiny / fw-base / Moonshine tiny / Vosk free-form |
| and notate, and tate | annotate | Moonshine tiny / Vosk free-form |

**Caution:** this map was built from the same 46 clips it is scored on, so
the `+corr` rows are the most optimistic numbers in this file. Its real value
can only be judged on the held-out set.

---

## Stage 5 – False-command check (setup)

- `data/asr/noise_only/` – 15 generated clips (`src/asr/scripts/make_noise_clips.py`,
  fixed seed): digital silence, white noise and pink noise, each at a low
  (−50 dBFS, quiet room) and medium (−30 dBFS, about our speech level) level,
  2/3/4 s, 16 kHz mono. Real room-noise recordings can be added to the same
  manifest and are picked up automatically.
- `data/asr/non_command/` – real speech that isn't a command. Empty for now. The
  phrase list (moved and updated on 2026-10-14, because "pass me the clamp"
  is a real command in the new schema) is `data/asr/prompts/non_command.txt`.
  Record with:
  `python src/asr/scripts/record_testset.py --speaker <name> --prompts-file data/asr/prompts/non_command.txt --outdir data/asr/non_command --manifest data/asr/non_command/manifest.csv`
- Metrics: **hallucination rate** = any output on noise-only clips (after
  normalization, `[unk]` removed); **false-command rate** = output contains
  any vocab term, reported separately for noise-only and non-command speech;
  `[unk]` rate per set.

---

## Stage 6 – Final results (dev set, 2026-10-10)

Laptop (Intel 12th gen, 12 cores, Windows 11), 46 command clips (4 s each)
+ 15 noise-only clips, each clip timed 3× (median kept). No non-command
speech recorded yet, so that column is still empty. Full data (archived on
2026-10-14) in `results/archive/v3_final_2026-10-10/`: `summary.csv`,
`per_clip.csv`, `errors_<config>.txt`, `raw/`, `machine.json`, `run.log`.

**Repeatability:** 0 clips differ across the 3 runs, for all 20 configs.

### All configs

| config | corpus WER [95% CI] | recall | precision | output on noise | false cmd on noise | median / p95 latency (s) | RTF | RAM MB | conf |
|---|---|---|---|---|---|---|---|---|---|
| fw-tiny.en-b5 | 0.200 [0.117–0.291] | 0.638 | 0.909 | 53% | 0% | 0.31 / 0.39 | 0.080 | 323 | yes |
| fw-tiny.en-b1 | 0.231 [0.122–0.351] | 0.596 | 0.903 | 0% | 0% | 0.35 / 0.38 | 0.088 | 324 | yes |
| **fw-tiny.en-b1 +prompt** | 0.138 [0.051–0.250] | 0.915 | 1.000 | 0% | 0% | 0.87 / 0.90 | 0.206 | 325 | yes |
| fw-tiny.en-b1 +hot | 12.25 [6.93–18.36] | 0.936 | 0.161 | 33% | 0% | 1.03 / 2.87 | 0.325 | 372 | yes |
| fw-tiny.en-b1 +prompt+hot | 10.54 [5.71–15.70] | 0.957 | 0.957 | 7% | 0% | 0.88 / 2.16 | 0.308 | 373 | yes |
| fw-base.en-b5 | 0.154 [0.083–0.238] | 0.723 | 0.895 | 73% | 0% | 0.90 / 1.14 | 0.224 | 386 | yes |
| fw-base.en-b1 | 0.185 [0.097–0.295] | 0.702 | 0.892 | 53% | 0% | 0.83 / 0.88 | 0.209 | 393 | yes |
| fw-base.en-b1 +prompt | 0.362 [0.277–0.441] | 1.000 | 0.979 | 0% | 0% | 1.64 / 1.72 | 0.397 | 395 | yes |
| fw-base.en-b1 +hot | 3.07 [0.67–7.07] | 0.979 | 0.220 | 93% | **93%** | 1.70 / 4.02 | 0.566 | 394 | yes |
| fw-base.en-b1 +prompt+hot | 6.19 [1.57–12.56] | 0.979 | 0.129 | 0% | 0% | 2.12 / 4.25 | 0.596 | 396 | yes |
| moonshine-tiny | 0.246 [0.136–0.367] | 0.638 | 0.968 | 0% | 0% | 0.76 / 1.02 | 0.197 | 240 | **no** |
| moonshine-tiny +kt1 / kt2 / kt3 | 0.254 / 0.292 / 0.262 | 0.66 / 0.64 / 0.64 | 1.000 | 0 / 7 / 13% | 0% | ~0.78 / ~1.1 | 0.20 | 242 | **no** |
| moonshine-small | 0.123 [0.067–0.193] | 0.660 | 0.939 | 7% | 0% | 1.85 / 2.29 | 0.468 | 398 | **no** |
| moonshine-small +kt1 / kt2 / kt3 | 0.115 / 0.146 / 0.138 | 0.68 / 0.66 / 0.70 | 0.94 | 7 / 13 / 20% | 0% | ~1.9 / ~2.3 | 0.48 | 398 | **no** |
| vosk-freeform | 0.254 [0.146–0.386] | 0.660 | 1.000 | 33% | 0% | 0.10 / 0.13 | 0.026 | 306 | yes |
| **vosk-grammar** | 0.362 [0.269–0.443] | 0.979 | 0.979 | 0% | 0% | 0.07 / 0.07 | 0.017 | 216 | yes |

"Output on noise" = hallucination rate (any words on a noise-only clip).
"False cmd" = output contained a vocabulary term. Latency is for 4 s clips
on a laptop; the target (< 1 s for a ~2 s command) applies to the Pi.

### With the post-correction map (same transcripts, rescored)

| config | corpus WER without → with map | recall without → with map |
|---|---|---|
| fw-tiny.en-b5 | 0.200 → 0.100 | 0.638 → 0.872 |
| fw-tiny.en-b1 | 0.231 → 0.146 | 0.596 → 0.787 |
| fw-base.en-b5 | 0.154 → 0.069 | 0.723 → 0.915 |
| fw-base.en-b1 | 0.185 → 0.108 | 0.702 → 0.872 |
| moonshine-tiny | 0.246 → 0.154 | 0.638 → 0.872 |
| moonshine-small | 0.123 → 0.038 | 0.660 → 0.894 |
| vosk-freeform | 0.254 → 0.146 | 0.660 → 0.894 |

All improvements are statistically real **on this data**, but the map was
written from these same clips, so this is the most optimistic table here.
Keep the map, and re-check it on the held-out set before trusting the gain.

### Which differences are real? (paired cluster bootstrap, `src/asr/scripts/compare_configs.py`)

| comparison (A − B) | corpus WER diff [95% CI] | term recall diff [95% CI] |
|---|---|---|
| **tiny.en-b1 +prompt − tiny.en-b1** | **−0.092 [−0.159, −0.025] real** | **+0.319 [+0.171, +0.478] real** |
| base.en-b1 +prompt − base.en-b1 | +0.177 [+0.071, +0.278] real (worse) | +0.298 [+0.160, +0.459] real |
| Moonshine +kt1/2/3 − baseline (tiny and small, 6 pairs) | all within noise | all within noise |
| **vosk-grammar − vosk-freeform** | +0.108 [−0.055, +0.243] noise | **+0.319 [+0.160, +0.500] real** |
| +corr − baseline (all 5 engines tested) | −0.085 to −0.108, all real | +0.19 to +0.23, all real |
| beam1 − beam5 (tiny and base) | noise | noise |
| base.en − tiny.en (beam 5) | noise | noise |
| moonshine-small − moonshine-tiny | −0.123 [−0.209, −0.046] real | noise |
| moonshine-small − fw-base.en-b5 | noise | noise |
| moonshine-tiny − fw-tiny.en-b5 | noise | noise |
| vosk-freeform − fw-tiny.en-b5 | noise | noise |
| tiny.en-b1 +prompt − base.en-b5 | noise | +0.191 [+0.044, +0.340] real |
| **vosk-grammar − tiny.en-b1 +prompt** | +0.223 [+0.060, +0.362] real (worse) | **+0.064 [−0.025, +0.184] noise** |
| vosk-grammar − moonshine-small | +0.238 real (worse) | +0.319 [+0.170, +0.490] real |
| vosk-grammar − fw-base.en-b5 | +0.208 real (worse) | +0.255 [+0.122, +0.404] real |

### What the results say
- **Vocabulary biasing is what moves command recall**, not model size or beam
  width: the Whisper prompt (+0.32), the Vosk grammar (+0.32) and the
  correction map (+0.19 to +0.23) are all real gains; model size, beam width
  and engine choice on their own are not distinguishable.
- **Whisper prompt:** clearly helps tiny.en on both WER and recall. On
  base.en it finds every command (recall 1.000) but appends junk words
  ("zoom in. Okay. Okay.") – probably decoded from the trailing silence in
  our fixed 4 s clips; real DSP segments are trimmed, so this needs
  re-checking on DSP output. Cost: the prompt is re-processed for every clip,
  so tiny.en latency goes from 0.35 s to 0.87 s.
- **Whisper hotwords: rejected.** Repetition loops (WER 3–12) and, for
  base.en, "zoom in" on 14 of 15 noise clips – the exact risk CLAUDE.md
  flagged. The loops happen because `temperature=0` (needed for repeatable
  output) removes Whisper's usual escape from loops; hotwords would need a
  loop guard (e.g. a repetition penalty) to be reconsidered.
- **Moonshine keyterms: no measurable benefit**; noise hallucinations rise
  with the boost (Small 7% → 13% → 20%).
- **Vosk grammar:** with `[unk]` it no longer forces speech into commands
  (precision 0.885 in stage 2 → 0.979 now) and ignores noise completely. Its
  word-for-word WER is poor by design: anything outside the grammar becomes
  `[unk]`, which NLP doesn't need anyway.
- **Noise:** Whisper says "You" (base.en also "All right.") on silence/noise –
  53–73% of noise clips for tiny.en beam 5 and base.en, none for tiny.en
  beam 1. Vosk free-form says "huh". Neither is a command, but
  NLP should treat a lone "you"/"huh" as no input. With the prompt, Whisper
  stays silent on noise.

### Recommendation – shortlist for the Pi

| rank | config | why | open risks |
|---|---|---|---|
| 1 | **vosk-grammar** | Recall 0.98 and precision 0.98 (tied with #2 on recall), 0% output on noise, real confidence score, 0.07 s per clip and the least RAM (216 MB) – the only option with a large margin under 1 s on a Pi 4. | Untested on **non-command speech** (biggest risk: a grammar can map ordinary talk onto commands). Can't produce PACS. Every new command needs a grammar update. |
| 2 | **faster-whisper tiny.en, beam 1, +prompt** | Recall 0.92, precision 1.00, best word-for-word WER of the shortlist, 0% output on noise, real confidence score. | 0.87 s per clip on the laptop – likely over 1 s on the Pi. Try a shorter prompt first. |
| – | Moonshine (all configs) | Best raw WER (Small, 0.123). | **No confidence score** – fails spec 2.2 and leaves NLP unable to reject doubtful input. The only workaround is running a second engine for confidence (e.g. agreement with Vosk-grammar), which costs Pi CPU/RAM. Small is also too slow offline; Tiny would only be fast with streaming (stage 3). |
| – | Whisper hotwords, beam 5, base.en without prompt | – | Loops / false commands (hotwords); no real gain over cheaper options (beam 5, base.en). |

**Plus, for whichever engine is chosen:** apply the post-correction map in
the `normalized` field (already done in code) and re-check it on the
held-out set.

### Before these numbers are final
1. Record **non-command OR speech** (`data/asr/prompts/non_command.txt`) and rerun –
   most important for Vosk grammar.
2. Record a **held-out set**, ideally several speakers, and rerun without any
   further tuning.
3. Run the shortlist **on the Pi 4** (`python src/asr/scripts/benchmark.py --configs vosk-grammar,faster-whisper-tiny.en-beam1+prompt`).

---

## History – earlier notes (phases 3–4)

> The phase 4 (v1) numbers below are **superseded** by benchmark v2 above.
> Kept for the record of what was tried.

### Phase 3 – core package smoke test (2026-10-07)
Built `schema.py`, `audio.py`, `engines/base.py`, `engines/faster_whisper_engine.py`,
`src/asr/scripts/transcribe.py`. Ran `transcribe.py` on real mic recordings from the
phase 2 test set (laptop mic, quiet room, peak levels 0.05–0.25).

- `tiny.en` (int8) consistently mis-transcribes "Zoom in" as "So I'm in" /
  "I'm so in". Not a volume issue (peak-normalizing to 0.9 didn't fix it) and
  not a pipeline bug (feeding the file straight to faster-whisper gives the
  identical wrong text).
- `base.en` (int8) correctly transcribes "Zoom in" and "Um, zoom in a little,
  please."
- Both `tiny.en` and `base.en` mis-hear "Lung window" as "long window".

### Phase 3 – Moonshine engine (2026-10-07)
Built `src/asr/engines/moonshine_engine.py` against the installed
`moonshine-voice==0.1.5` (API read from the package source, since the hosted
docs summary lacked exact signatures). Uses
`Transcriber.transcribe_without_streaming()` with models cached in `./models/`.

- Word-level `confidence` exists on `WordTiming` but was **1.0 for every
  word** on Tiny and Small Streaming, including clearly wrong words ("Boom
  in." for "Zoom in."). Reported as `null`.
- No line/transcript-level confidence field exists.
- Tiny Streaming also mis-hears "Zoom in" ("Boom in." / "Doom in.").
- `transcribe_without_streaming()` can return several lines for one short
  clip; the engine joins them, since one DSP segment = one command.

### Phase 3 – Vosk engine (2026-10-07)
Built `src/asr/engines/vosk_engine.py` against `vosk==0.3.45` with
`vosk-model-small-en-us-0.15` in `./models/`.

- Vosk's per-word `conf` is a real signal: in "room in" (for "Zoom in") the
  wrong word got 0.43, the right one 1.0.
- Vosk was the first engine to get "lung window" right.
- With a restricted grammar, "Zoom in" was transcribed correctly. (This
  early test used an ad-hoc 11-term list; the v2 grammar is built only from
  `src/asr/vocab.py`.)

### Phase 4 – benchmark v1 (2026-10-07) – superseded
Ran 8 configs × 46 clips with minimal normalization, mean per-clip WER, a
single process and no warm-up. Files archived in `results/archive/v1_2026-10-07/`.

**Bug found and fixed:** loading a second Vosk model after freeing the first
in the same process segfaults the Kaldi library. Fixed by sharing one cached
`vosk.Model` per process.

| engine | mean_wer | mean_term_acc | mean_rtf | p95_latency_s | ram_mb |
|---|---|---|---|---|---|
| fw-tiny.en-beam5 | 0.242 | 0.656 | 0.108 | 1.052 | 139 |
| fw-tiny.en-beam1 | 0.251 | 0.633 | 0.116 | 1.365 | 90 |
| fw-base.en-beam5 | 0.198 | 0.744 | 0.182 | 0.832 | 142 |
| fw-base.en-beam1 | 0.192 | 0.744 | 0.197 | 0.953 | 132 |
| moonshine-tiny-streaming | 0.288 | 0.600 | 0.215 | 1.115 | 88 |
| moonshine-small-streaming | 0.150 | 0.711 | 0.436 | 2.570 | 191 |
| vosk-freeform | 0.269 | 0.622 | 0.178 | 0.856 | 142 |
| vosk-grammar | 0.397 | 0.956 | 0.018 | 0.079 | ~1 |
