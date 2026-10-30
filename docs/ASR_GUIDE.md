# Speech Recognition (ASR) – Module Guide

The speech recognition stage of scrub-voice: what it does, what goes in and
out, what the other stages have to do, and how to run it.
Owner: Josias. State: **finished** (2026-10-28), Vosk only.
Details and all measurements: `docs/ASR_FINDINGS.md`. Notes for working on
the code: `src/asr/CLAUDE.md`.

```
microphones ─▶ DSP ──speech chunks──▶ ASR (src/asr/) ──text──▶ NLU ──command──▶ orchestrator ─▶ robot / monitor
              cleans the audio,        audio → words           words → intent + values
              sends only speech
```

---

## 1. In one minute

* **ASR turns speech into words – nothing more.** It does not decide what
  the command means, does not turn "two" into 2 and does not map "knife"
  to "scalpel". That is NLU's job.
* **Engine:** Vosk (`vosk-model-small-en-us-0.15`), fully offline, with a
  **list of our command words** (the "grammar"). Vosk only has to choose
  between our ~160 words and phrases instead of the whole English
  language. That makes it accurate and fast, but speech outside the list is
  left out.
* **In:** half-second pieces of cleaned audio from the DSP, only while
  someone speaks. **Out:** one JSON message per spoken command to NLU.
* **A command ends** when the DSP says the speaker stopped – or at once
  when the surgeon ends it with **"execute"**, **"over"** or **"now"**.
* **Emergency halts go out early:** as soon as Vosk hears "halt",
  "emergency stop" or "hard stop", a copy is sent to NLU right away,
  before the command is over.

| measured (laptop, our 94 test recordings, one speaker) | result |
| :--- | :--- |
| command words found / command words right | 99.3 % / 100 % |
| command words invented on non-command talk | 0.08 per sentence |
| command words invented on noise | none |
| time from the DSP's "end" to the message at NLU | ~0.03 s |
| memory | ~260 MB |

**Open problem on the DSP side (§7):** today the DSP drops the half-second
before a command starts, which cuts off the first word in about 1 of 3
commands ("halt robot" → `robot`). One extra chunk fixes it completely.

---

## 2. What happens to one command

1. **The DSP sends half-second pieces** ("chunks") while its speech
   detector hears speech. One run of chunks = one utterance.
2. **Vosk listens chunk by chunk** and keeps a running guess of the words
   heard so far.
3. **After every chunk ASR checks that guess:**
   * Does it end with an end word after a command ("next slice **now**")?
     → the command is finished, send it now.
   * Does it contain "halt", "emergency stop" or "hard stop" for the first
     time? → send an **early copy** now; keep listening.
4. **The utterance ends** with one of these (field `end_reason`):

   | `end_reason` | when |
   | :--- | :--- |
   | `keyword` | the surgeon said "execute", "over" or "now" after a command |
   | `dsp_end` | the DSP sent `"end"` (normal case) |
   | `gap` | the next chunk does not follow directly in time |
   | `timeout` | no chunk for 0.8 s and no `"end"` |
   | `max_length` | 15 s of speech (safety net if the DSP's detector gets stuck) |

5. **Vosk writes the final words.** ASR makes only two fixes (see §5):
   joins words the grammar has to spell as sounds ("hemo stat" →
   "hemostat") and leaves out speech that is not on the list. For "log
   event …" a second, free pass writes the note text.
6. **One JSON message goes to NLU.**

---

## 3. What the surgeon can say

### Command words
The grammar is built from `src/asr/vocab.py` (`INTENT_TERMS`). Synonyms
are recognized as said; NLU maps them to the command.

| command (intent) | words ASR recognizes |
| :--- | :--- |
| hand a tool (`pick_tool`) | hand, pass, give me, bring, need · scalpel, knife, blade, ten blade · scissors, iris, metz, metzenbaum, surgical scissors · hemostat, clamp, snap · forceps, tweezers, pickups, pickup, artery forceps, tissue forceps |
| park the arm (`park_arm`) | park, standby, retract, dock, arm, robot |
| emergency clear (`clear_emergency`) | emergency, clear, area, back off, evacuate |
| emergency halt (`halt_emergency`) | emergency, stop, halt, freeze, hard stop, robot |
| open a scan (`open_scan`) | open, show, display, bring up, pull up, scan · ct, mri, x-ray |
| close a scan (`close_scan`) | close, dismiss, viewer, scan · ct, mri, x-ray |
| zoom (`zoom_display`) | zoom, zoom in, zoom out, magnify, enlarge, bigger, smaller, shrink, make it bigger, make it smaller, increase, decrease, reduce, wider |
| slices (`navigate_slice`) | next, previous, back, go to, advance, rewind, scroll, go forward, go back, slice, slices |
| log an event (`log_event`) | log, record, event, milestone, note, comment **+ free text** |
| move the camera (`move_camera`) | camera, wand, move, jog · left, right, up, down, forward, backward, ahead, away · centimeter(s) |
| tilt the camera (`tilt_camera`) | tilt, pitch, yaw · up, down, left, right, upward, downward · degree(s) |
| reset the camera (`reset_camera`) | reset, center, view, home, camera, wand |
| numbers | zero … ninety-nine, hundred, "and", "point" (one point five), "x" / "times" |

Also on the list, so they are written as said (but they are not commands):
"the", "a", "me", "by", "please", "okay", "could you", "a little bit",
"patient", "active", "current", "immediately", "robotic", "position",
"surgical field", "safe height".

**Not on the list on purpose** – each made recognition worse when we
measured it: "hold on" ("halt robot" became "hold robot"), "wait" (sounds
like "eight"), "jump to" ("two" became "to"), "load", "hide", "closer"
(everyday talk landed on them), "end" as end word (sounds like "and").

### End words: "execute", "over", "now"
* "zoom in two x **execute**", "next slice **now**": the command is sent the
  moment the word is heard, and the word is left out of the text.
* Only counts **after a command word**: "now" alone or "now zoom in" does
  nothing.
* **When it helps:** in a noisy room, where noise keeps the DSP's speech
  detector on. Without the end word the command waits until the noise
  stops (up to 15 s); with it, it goes out about half a second after the
  word. **In a quiet room it is not faster** – the DSP's "end" comes just
  as quickly, and saying the word takes time. So it is optional.
* Leave a short pause after it before the next command.

### Emergency halt fast path
* When Vosk first hears **"halt"**, **"emergency stop"** or **"hard
  stop"**, ASR sends an early copy at once; the full command follows when
  the utterance ends. Example "Halt robot": early `halt`, then `halt robot`.
* Only halts, because a false halt only stops the robot. "Clear area" /
  "back off" make the robot move, so they wait for the end as usual.
* Not "stop" alone ("don't stop now") and not "freeze" ("next three
  slices" was heard as "next freeze" for a moment). Both still work as
  normal commands.
* **When it helps:** again mostly in noise or in long utterances ("emergency
  stop, emergency stop": 2 s sooner). In noise, measured: "halt robot" went
  out after 1.0 s instead of waiting for the noise to stop. For a short
  halt in a quiet room the normal message is just as fast.

### Speech that is not on the list
Left out of the text. "It is raining outside" and "Pass that thing" came
out as an empty text. If only part of a command is on the list, the rest is
missing – a command word can arrive without its value (a tool name that is
not on the list). Sometimes everyday talk is forced into command words:
"we are running a bit late" → `blade`, "where did I park my car" → `park
make arm`, "hello" → `little`.

---

## 4. Input from the DSP

ASR **binds** a ZeroMQ `PULL` socket; the DSP **connects** a `PUSH` socket.
Address: `SCRUB_AUDIO_ADDR` (default `ipc:///tmp/scrub_audio.ipc`).

| message | frames | meaning |
| :--- | :--- | :--- |
| speech chunk | 1. `{"type": "chunk", "capture_time": 1727800000.0}` (JSON)<br>2. raw audio bytes | `capture_time` = Unix time (seconds) of the chunk's **first sample**. Audio = **float32, mono, 16 kHz**, normally 8000 samples (0.5 s); any length is accepted. |
| end of speech | 1. `{"type": "end"}` | Send it when the speech detector switches off. |

```python
import orjson, zmq
sock = zmq.Context().socket(zmq.PUSH)
sock.connect("ipc:///tmp/scrub_audio.ipc")
sock.send_multipart([orjson.dumps({"type": "chunk", "capture_time": t}), beam.astype("float32").tobytes()])
sock.send_multipart([orjson.dumps({"type": "end"})])
```

If DSP and ASR run in one Python process, skip ZeroMQ and use
`asr.stream.UtteranceStream` directly: `feed(chunk, capture_time)` returns a
**list** of results to send (usually empty), `end_utterance()` returns the
finished one (or `None`).

---

## 5. Output to NLU

ASR **connects** a ZeroMQ `PUSH` socket; NLU **binds** a `PULL` socket.
Address: `SCRUB_ASR_ADDR` (default `ipc:///tmp/scrub_asr.ipc`). UTF-8 JSON,
one message per utterance – plus the early copy for halts.

```json
{
  "text": "zoom in two x",
  "confidence": 0.875,
  "language": "en",
  "words": [
    {"word": "zoom", "start": 0.0, "end": 0.48, "confidence": 1.0},
    {"word": "in", "start": 0.51, "end": 0.87, "confidence": 1.0},
    {"word": "two", "start": 0.96, "end": 1.23, "confidence": 0.5},
    {"word": "x", "start": 1.23, "end": 1.74, "confidence": 1.0}
  ],
  "segment_start": 1792150493.32,
  "end_reason": "dsp_end"
}
```

| field | meaning |
| :--- | :--- |
| `text` | The words as heard, lowercase, **not normalized**: numbers stay words, "please" stays, synonyms stay as said, end words are removed. **Empty** when nothing usable was heard. (= `raw_transcript` in the command schema) |
| `confidence` | 0–1, mean word confidence. `null` when `text` is empty, and in an early halt copy. |
| `words` | The words of `text`, times in seconds from `segment_start`, each with its confidence. Empty in an early halt copy. |
| `segment_start` | Unix time (seconds) when the speech started. Use it for `latency_ms`. |
| `end_reason` | Why the utterance ended (table in §2), or `"emergency_early"` for the early halt copy. `null` when a file is transcribed. |

The two fixes ASR makes in `text`:
* Words missing from Vosk's dictionary, which the grammar spells as
  sounds, are written as one word: `ct`, `mri`, `x-ray`, `hemostat`.
* Speech outside the list (Vosk's `[unk]`) is left out.

Typical messages:

| the surgeon says | NLU receives |
| :--- | :--- |
| "Pass the hemostat" | `text: "pass the hemostat"`, `end_reason: "dsp_end"` |
| "Go to slice one hundred and twenty" | `text: "go to slice one hundred and twenty"` |
| "Next slice now" | `text: "next slice"`, `end_reason: "keyword"` |
| "Halt robot" | 1. `text: "halt"`, `end_reason: "emergency_early"`, `words: []`, `confidence: null`<br>2. `text: "halt robot"`, `end_reason: "dsp_end"` |
| "It is raining outside" | `text: ""`, `confidence: null` |

---

## 6. What NLU has to do (checklist for Khanh)

**Must do:**
1. **Read `text`.** The `normalized` field is gone since 2026-10-28.
   (`src/nlu/run_nlu.py` already reads `text` first – nothing to change.)
2. **Turn number words into digits yourself.** ASR sends words only. Test
   with these – the first two fail in `src/nlu/normalizer.py` today:

   | `text` | needed | NLU today |
   | :--- | :--- | :--- |
   | "go to slice one hundred and twenty" | 120 | "100 and 20" |
   | "go to slice a hundred" | 100 | "a 100" |
   | "zoom in two x", "zoom in two times" | factor 2 | ok |
   | "zoom out one point five x" | factor 1.5 | ok |
   | "camera left five centimeters", "tilt down fifteen degrees" | 5 cm, 15 degrees | ok |

3. **Learn the new synonyms** – otherwise ASR hears them but NLU does not
   understand them. Not yet in NLU's training phrases or dictionaries:
   **pull up** (open scan), **scroll**, **go forward**, **go back** (slices),
   **enlarge**, **bigger**, **smaller**, **shrink**, **make it bigger**,
   **make it smaller** (zoom), **need** ("I need the scalpel").
4. **Treat "halt" alone as an emergency halt.** The early copy can be just
   `halt`; NLU's safety rule only matches "halt robot".
5. **Halts can arrive twice** (early copy + full message). Halting twice
   must be harmless. The copy is marked `end_reason: "emergency_early"`.

**Good to know:**
* **Empty `text`** = nothing usable was heard → skip it (`run_nlu.py`
  already does).
* **A command word without its value** can arrive (words not on the list
  are left out, e.g. a tool name we don't have). Ask or reject – don't
  guess a tool.
* **Everyday talk forced into command words** ("a bit late" → `blade`,
  "where did I park my car" → `park make arm`): be careful with short
  commands that have low `confidence`.
* **Words NLU knows that never arrive** (not on the grammar): load, jump,
  closer, reverse, prev, cm, hold on, wait, cat scan, computed tomography,
  magnetic resonance, mr, radiograph, chest / head / brain / spine /
  abdominal. Training phrases like "Load patient CT" or "Jump to slice one
  hundred" will not arrive as written – train on what ASR can say (§3).
* Spelling: `x-ray` with a hyphen; `ct`, `mri` lowercase; numbers as words.
* The note text after "log event" comes from a free pass and is often wrong
  for medical words ("hepatic" → "have pet take"). Store it as it is.
* End words never reach NLU after a command (ASR removes them); "now"
  elsewhere is already a filler in NLU.
* On Windows NLU listens on `tcp://127.0.0.1:5555` by default; start ASR
  with `--asr-addr tcp://127.0.0.1:5555`.

---

## 7. What the DSP has to do (checklist for Minh and Quân)

* **Most important – also send the chunk before the speech starts
  ("pre-roll").** The speech detector decides per half-second chunk. When a
  word starts near the end of a chunk, that chunk is judged "not speech"
  and dropped – and the start of the first word is lost. Measured on our
  60 commands, gated exactly like `run_dsp.py`:

  | | command words found / right | commands that came out different |
  | :--- | :--- | :--- |
  | today | 88.4 % / 97.6 % | 18 of 60 – e.g. "Halt robot" → `robot`, **"Freeze" → nothing**, "Back off" → `off`, "Park arm" → `jog arm`, "Clear area" → `view area` |
  | with one chunk of pre-roll | 99.3 % / 100 % | 0 of 60 (same as the whole recording) |

  Fix in `run_dsp.py`: keep the last chunk that was judged silent; when
  speech starts, process and send that chunk first (with its own
  `capture_time`), then the speech chunk. The cost is one extra chunk of
  processing at the start of each command.
* Send chunks in the format of §4 – `src/dsp/run_dsp.py` already does, and
  already sends `"end"` on the first silent chunk. ✓
* **`capture_time` should be the time of the chunk's first sample.**
  `run_dsp.py` takes `time.time()` right after reading the chunk, i.e. at
  its *end*, so `segment_start` comes out ~0.5 s late (and NLU's latency
  looks 0.5 s too small). Fix: subtract the chunk length (0.5 s).
* One beam only (one speaker direction).
* Faster still: send `"end"` (or a shorter last chunk) as soon as the
  detector hears the speech stop, instead of waiting for a full silent
  chunk – the half-second chunks are the biggest part of the delay.

---

## 8. Running it

```bash
# once, from the repository root
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # Vosk model downloads on first start, then offline

# service (pipeline)
python src/asr/scripts/run_asr.py

# simulation without the microphone array: plays wav files as a DSP stream, prints JSON.
# The DSP's own speech detector (src/dsp/source/vad.py) picks the speech chunks:
# needs torch; its Silero model downloads once from GitHub.
python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav

# test ASR -> NLU without the DSP: a stand-in NLU prints what arrives,
# the simulation plays the files in real time and sends the results
python src/asr/scripts/listen_asr.py                                  # terminal 1
python src/asr/scripts/run_asr.py --wav data/asr/sample_01.wav --send # terminal 2

# one file -> JSON
python src/asr/scripts/transcribe.py some_command.wav

# unit tests (51)
python -m unittest discover -s src/asr/test -v

# quick check after changing the word list (needs the recordings in data/asr/)
python src/asr/scripts/benchmark.py --configs vosk-grammar+freeform-log --runs 1
```

`ipc://` addresses work on Linux / Raspberry Pi. On Windows use TCP, e.g.
`--audio-addr tcp://127.0.0.1:5591 --asr-addr tcp://127.0.0.1:5555`.

---

## 9. Changing the word list

* Add words in `src/asr/vocab.py` (`INTENT_TERMS`; words that are not
  commands in `GRAMMAR_CONTEXT_WORDS`) and run the unit tests – they check
  that every word is in Vosk's dictionary and that no test sentence ended
  up in the grammar. A word missing from the dictionary needs a
  sound-alike spelling in `SPOKEN_FORMS` ("hemostat" → "hemo stat").
* **Every new word has a price:** everyday talk can be forced into it.
  Short single words are the risk; two-word phrases ("bring up") are safer.
  Run the quick check (§8) and compare the non-command results.
* A new word only helps if NLU knows it too.

---

## 10. What changed in the rework (2026-10-28)

After the team evaluation (`evaluations/asr/README.md`):

| | before | now | why |
| :--- | :--- | :--- | :--- |
| text tidying | ASR sent `text` + tidied `normalized` ("zoom in 2 x") | only `text`, words as heard | one place for tidying: NLU (evaluation "Priority 3") |
| `[unk]`, "hemo stat" in `text` | sent to NLU | left out / joined | NLU's word matching missed them |
| word list | 115 entries | 160 entries (synonyms, context words, end words) | surgeons use other words (evaluation "Priority 1") |
| end of a command | only when the DSP says so | also by end word "execute" / "over" / "now" | noise can keep the detector on (evaluation "Priority 2") |
| emergency halt | sent at the end of the utterance | early copy as soon as it is heard | lowest delay for the safety command |
| decoding | whole utterance at the end | chunk by chunk, same words as before | needed for end words and halts; checked on all recordings |
| `end_reason` field | – | new | shows why each message was sent |
| engines | Vosk + Whisper fallback | **Vosk only** | testing is closed; less code |

Accuracy stayed the same or got better at every step (§1 numbers; before:
98.6 % / 98.6 %).

---

## 11. Limits – please read

* **Until the DSP sends one chunk of pre-roll (§7), the first word of a
  command is often cut off** – 18 of 60 commands in our test, including
  "halt" and "freeze". The numbers in §1 assume the whole command reaches
  ASR. The `--wav` simulation uses the DSP's real rule, so it shows this
  too.
* Tested with **one speaker** (Josias) on recordings that were also used
  for tuning: real numbers will be lower.
* **Not re-measured on the Raspberry Pi after the rework.** The team
  evaluation on the Pi (before it) measured 0.12–0.14 s per command and
  185 MB.
* End words and the halt fast path are tested with unit tests, the
  existing recordings and spliced audio – **not yet with real recordings
  of commands that end in an end word.**
* An end word inside the "log event" note ("… bleeding is over the
  margin") ends the note early.
* Medical words in the "log event" note are often wrong.
