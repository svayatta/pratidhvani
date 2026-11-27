# Deep-Dive Report: `EXP-014` to `EXP-017`, Dual-LLM (`Opus`) Second-Opinion Audit, `Voice Atlas 2.0`, and the Two-Stage `Svara-TTS` + `Seed-VC` Pivot

This document provides the complete technical context for everything executed and discovered since `EXP-013`:
1. Why `EXP-014` and `EXP-016` regressed to `0%` listener resemblance despite high `ECAPA-TDNN` cosine similarity (`0.55–0.77`), while `EXP-015 Option 2` achieved the project's all-time high **`40%` listener resemblance** on the closing word *"ಬರ್ರಿ"*.
2. Why we paused blind training after `EXP-016`, upgraded hardware from a 16 GB Mac mini to an **Apple M5 Max (128 GB unified memory, `MPS`)**, and brought in an independent second LLM (**`Claude Opus`**) to audit all code, metadata, and audio outputs (`llm1-kannada-clone-output-exp16.zip`).
3. How `Opus`'s diagnostic ceiling tests (**`E003` `SNAC` codec round-trip** and **`E004` pitch shift**) and `Voice Atlas 1.0` were critically cross-audited and merged with full-archive `MPS` measurements into **`Voice Atlas 2.0`** across **11 archival Kannada & Marathi sermons (`99 min`)**, **5 reference anchors**, **16 `E003` pairs**, and **16 clone outputs**.
4. How the **five concrete root causes** uncovered by this dual-LLM audit directly shaped **`EXP-017A`** (Controlled `Svara-TTS` Plain LoRA on `0–4.2 kHz`-intact audio with Holdout Validation Loss and `c0`-only repetition penalty) and **`EXP-017B`** (Fine-Tuned `Seed-VC` Continuous Mel Conversion on `MPS`).

---

## 1. Chronological Progression from `EXP-014` to `EXP-016`

### 1.1 `EXP-014`: `IndicConformer-600M` `PBP` Hall Expansion (`0%` Resemblance)

- **What was tried:** Added 30 speech segments from public `PBP` hall recordings (`PBP 3`, `PBP 11`, `PBP 13`, `PBP 15`) transcribed with `ai4bharat/indic-conformer-600m-multilingual` (`CTC` + `RNN-T`) and filtered by `MMS CTC Forced Aligner` ($\text{pf} \ge -0.89$, $\text{sim} \ge 0.56$, metadata $F_0 \in [126, 186]\text{ Hz}$) onto the 102 close-mic `clips_exp12` rows (`132 clips, 13.48 min`). Trained Rank 32 LoRA for 3 epochs (`94 steps`) and synthesized in 2 breath clauses.
- **Telemetry & Verdict:** `ECAPA = 0.545`, duration `10.23s`, Praat median $F_0 = 141.2\text{ Hz}$. **Listener verdict: `0%` resemblance** (*"sounds nothing like the speaker"*).
- **Root-Cause Audit of `clips_exp12` & `clips_exp14`:**
  1. **Reverberant Hall Dilution:** Mixing 30 reverberant `PBP` hall clips with close-mic conversational clips (`DOC-0004`, `K13`, `K41`) averaged out the close-mic vocal tract formants (repeating the failure mode of `EXP-004` and `EXP-005`).
  2. **"Split-Brain" Transcript Poisoning:** In `clips_exp14`, only the 30 new `PBP` hall clips received verbatim `IndicConformer-600M` Kannada transcripts. The 102 close-mic `clips_exp12` rows (`lines 1–102`) were copied over unchanged with their earlier Whisper transcripts, which contained:
     - **Turkish Latin script** on `exp12_clip_0093.wav` (`"Ramlali aramadın. Ramlali. Oraya ne?"`)
     - **Hindi / Devanagari script** on `exp12_clip_0079.wav`, `0096.wav`, and `0098.wav`
     - **Bengali characters** on `exp12_clip_0013.wav`
     - **Garbled Whisper `REF_TEXT` on `exp12_clip_0000.wav`** prepended directly before Clause 1, which caused the opening cross-attention slur in `EXP-012`.
     - As a result, the model learned an unintended correlation: *clean Kannada text tokens = reverberant `PBP` hall voice*, whereas *close-mic speaker voice = Whisper gibberish*.
  3. **Under-Training (`3` vs `6` Epochs):** Stopping at 3 epochs (`94 steps` vs `144 steps` in `EXP-012`) left the adapter weights near the base model's generic Kannada centroid (the same regression seen with `checkpoint-00080` in `EXP-013`).

---

### 1.2 `EXP-015`: Three-Architecture Bake-Off (`Seed-VC` Zero-Shot vs. Patched `Svara-TTS` vs. `OmniVoice 0.6B`)

To test whether the bottleneck was our data/inference pipeline or the underlying model architecture, `EXP-015` evaluated three architectures side-by-side on the target evaluation sentence:

| Option | Architecture & Method | Duration | `ECAPA` Sim | Praat $F_0$ | `CPP` (dB) | Listener Verdict |
|---|---|---:|---:|---:|---:|---|
| **`EXP-015 Opt1`** | **Zero-Shot `Seed-VC`** (`DiT` flow-matching + `Whisper-small` content + `CAMPPlus` speaker conditioning + `BigVGAN` vocoder) converting the clean `EXP-013` Kannada take | `8.70s` | `0.768` | `132.2 Hz` | `14.37 dB` | **Rejected (`0%`)** — high `ECAPA` cosine, but zero-shot `CAMPPlus` embedding without weight adaptation missed his individual vocal tract timbre |
| **`EXP-015 Opt2`** | **Patched `Svara-TTS` 3B (Rank 32 LoRA, 6 Epochs)** with 4 targeted fixes (below) | `8.27s` | `0.576` | `138.2 Hz` | `17.30 dB` | **All-Time High `40%` Resemblance on *"ಬರ್ರಿ"*** (listener: the word *"ಬರ್ರಿ"* sounded about 40% like the speaker; the other two options less so) |
| **`EXP-015 Opt3`** | **Zero-Shot `OmniVoice 0.6B`** (`k2-fsa/OmniVoice` discrete flow-matching multilingual TTS) conditioned on clean reference | `7.60s` | `0.760` | `123.6 Hz` | `16.40 dB` | **Rejected (`0%`)** — generic voice; zero-shot conditioning failed to hold North Karnataka prosody or elderly vocal tract resonance |

#### The 4 Fixes Introduced in `EXP-015 Option 2` (`exp15_opt2_svara_fixed.mp3`)
1. **100% Verbatim `IndicConformer-600M` Re-Transcription of Close-Mic Clips:** Purged all Whisper transcripts from the 102 close-mic `clips_exp12` files (`9 ref + 81 DOC-0004 + 1 K13 + 11 K41`, zero `PBP` hall clips) and re-transcribed every clip with `IndicConformer-600M` + `MMS CTC Aligner` (`98` verified clips kept in `clips_exp15_clean`).
2. **Discovery of the HuggingFace `RepetitionPenaltyLogitsProcessor` Prompt-Penalization Bug:**
   - Standard `model.generate(..., repetition_penalty=1.15)` in HuggingFace `transformers` gathers logits for **all token IDs present in `input_ids`**—including the entire prompt!
   - In `Svara-TTS`, the prompt contains ~550 `SNAC` acoustic tokens of the reference speaker's voice. Standard `repetition_penalty` actively divides the logits of every `SNAC` acoustic code that appeared in the reference speaker's audio prompt, forcing the autoregressive decoder *away* from the speaker's own vocal timbre codes!
   - Replacing it with `PromptExcludedRepetitionPenalty` (penalizing only newly generated tokens after `prompt_len`) stopped the decoder from suppressing the reference speaker's acoustic tokens.
3. **Full 6-Epoch Rank 32 LoRA Training (`loss = 4.197`) + 2-Clause Breath Phrasing:** Restored full 6-epoch weight adaptation on pure close-mic audio while keeping 2-clause articulation.
4. **High-Frequency Bird-Chirp Filter (`strip_birds_preserve_fry`):** Applied spectral denoising above `2,600 Hz` on the reference clip to suppress background bird chirps (later found in `Voice Atlas 2.0` to be overly aggressive between `2.6–4.2 kHz`; see Section 3.3).

---

### 1.3 `EXP-016`: Aggressive Metadata Pitch/Sim Filtering (`65` Clips) & Rank 32 `RSLoRA` (`0%` Resemblance)

- **What was tried:** Aiming to extend the `40%` *"ಬರ್ರಿ"* resemblance of `EXP-015 Opt2` across the full utterance, `EXP-016` made four simultaneous changes:
  1. Filtered out 57 `DOC-0004` clips that had `sim < 0.52` or stored metadata `f0 > 192 Hz`, and added 47 `IndicConformer-600M`-verified clips from `K13` and `K41` (`65 clips, 7.08 min` in `clips_exp16_pure`).
  2. Restricted the training reference-swap pool strictly to clips with `sim >= 0.62`.
  3. Switched from standard LoRA (`scale = alpha / r = 64 / 32 = 2.0`) to **Rank-Stabilized LoRA (`use_rslora=True`, `scale = alpha / sqrt(r) = 64 / 5.66 = 11.31` — a $5.66\times$ larger adapter multiplier)** and trained for 7 epochs (`103 steps`), driving training loss down from `4.197` to **`3.152`** (without a holdout validation loss check).
  4. Changed the custom repetition penalty during `synthesize_exp16` back to penalizing all generated tokens (`c0, c1, c2`) at `1.12`.
- **Telemetry & Verdict:** `ECAPA` cosine rose to **`0.765`** (`8.88s`), full 2-clause articulation. However, the listener rejected `exp16_clone_voice.mp3` (`0%` resemblance):
  > The listener judged the real clip `S01__A` clearer than its codec round-trip `S01__B`, and said `exp16_clone_voice.mp3` did not sound like `S01__A` at all.

---

## 2. The Strategic Pivot: M5 Max (128 GB) Upgrade & Independent Second-Opinion Review (`Claude Opus`)

### 2.1 Why We Paused Blind Training After `EXP-016`
By `EXP-016`, two structural problems were clear:
1. **Metric-Perception Decoupling:** `ECAPA-TDNN` speaker cosine repeatedly rewarded outputs the listener rejected as `0%` (`EXP-015 Opt1` at `0.768`, `EXP-015 Opt3` at `0.760`, `EXP-016` at `0.765`), while scoring our best human-verified output (`EXP-015 Opt2`, `40%` on *"ಬರ್ರಿ"*) lower at `0.576`. Training blind to optimize training loss or `ECAPA` cosine was counterproductive.
2. **New Hardware Capabilities (16 GB Mac mini $\to$ 128 GB Apple M5 Max MacBook Pro):** Moving to an M5 Max with 128 GB unified memory (`MPS`) removed the 16 GB memory ceiling, making local fine-tuning of continuous flow-matching voice conversion (`Seed-VC` v0.4.3 on `MPS`) and full-archive acoustic signal processing fast and practical.

To get an independent second opinion, all code, logs, metadata, and audio deliverables through `EXP-016` were packaged into `llm1-kannada-clone-output-exp16.zip` and reviewed by **`Claude Opus`**.

---

### 2.2 `Opus` Round 1: Diagnostic Ceiling Experiments (`E003` & `E004`)

Before running any new training, `Opus` authored `E003_ceilings.py` to test two fundamental ceilings without training:
1. **`E003` (`SNAC 24 kHz` Codec Ceiling Test):**
   - Took 8 of the speaker's real close-mic clips (`S01__A.wav` through `S08__A.wav`, drawn from `clips_exp16_pure`) and passed each one through `Svara-TTS`'s neural audio codec (`SNAC 24 kHz` encode $\to$ decode, producing `S01__B.wav` through `S08__B.wav`).
   - **Purpose:** Since `Svara-TTS` can only speak through `SNAC 24 kHz` discrete tokens, `S01__B..S08__B` represent the theoretical upper bound of how close *any* `Svara-TTS` LoRA can ever sound to `S01__A..S08__A`.
   - **Listener Verdict:** Listening to `S01__A.wav` (real) vs `S01__B.wav` (`SNAC` round-trip), the listener immediately identified `S01__A.wav` as clearer and truer to the speaker's voice.
2. **`E004` (Praat PSOLA Pitch-Shift Test — And How It Exposed the `1.378x` Pitch Bug):**
   - `Opus` initially noticed that `VaniHarness`'s `metadata.jsonl` listed the speaker's clips at `172–184 Hz` while the clones (`EXP-008` to `EXP-016`) measured `125–142 Hz` (an apparent 4–7 semitone gap), and designed `E004` to shift the clones upward via Praat's `Change gender` (PSOLA pitch shift keeping formants fixed).
   - However, when `E003_ceilings.py` ran Praat (`parselmouth`) on both the real clips and the clones (`pitch_audit.tsv`), **both LLMs discovered that the speaker's real clips actually sit at `125.8–136.0 Hz`**—the exact same pitch register as the clones (`125–142 Hz`)! `E004` was immediately dropped, and this discovery exposed Root Cause #1 below.

---

### 2.3 Stepping Back to Acoustic First Principles: `Opus` Round 2 & `Voice Atlas 2.0`

Following `E003`, the listener asked a foundational question:
> The listener asked to step back and first understand what makes up the speaker's voice: which acoustic variables (timbre, pitch, range, speed) define it, how they vary, and how to compare a clone with it visually and numerically.

Both LLMs independently built acoustic voice-signature dashboards (`Voice Atlas 1.0` from `Opus`, and our 6-Pillar Acoustic Fingerprint Dashboard), and we then performed a **critical cross-audit** comparing the code, literature, and raw measurements of both LLMs to build **`Voice Atlas 2.0` (`voice_atlas_v2.py`)**.

#### Critical Audit: Where `Opus` Was 100% Right vs. What `Voice Atlas 2.0` Corrected

| Dimension | Where `Opus` Was Right (Adopted in `Voice Atlas 2.0`) | What `Opus` Missed / Got Wrong (Fixed in `Voice Atlas 2.0`) |
|---|---|---|
| **1. Connected-Speech Glottal Quality Metrics** | Cycle-to-cycle `Jitter` and `Shimmer` are clinically valid only on sustained isolated vowels (`/a:/`). On connected Kannada speech, rapid retroflex/dental stops and prosodic slopes fool pulse-boundary pickers. Replaced with **Cepstral Peak Prominence (`CPP`, dB)** (Hillenbrand 1994), **`H1-H2` (dB)**, **Alpha Ratio (`0–1 kHz` vs `1–5 kHz`)**, and **Hammarberg Index (`0–2 kHz` vs `2–5 kHz`)**. | `Opus`'s earlier HTML summary hardcoded *"SNAC consistently adds 45–85% jitter"*, which contradicted its own `e003_forensics.json` (where `SNAC` lowered jitter on `S05` from `2.43%` to `1.88%` and raised `HNR` on 4 of 8 pairs). `Voice Atlas 2.0` computes `CPP` and `ECAPA` dynamically across all 8 `E003` pairs, proving the real `SNAC` artifact is **artificial `CPP` over-periodicity (`16.3–18.3 dB` vs natural `10.9–14.7 dB`)** and a **universal `-0.07` to `-0.25` drop in `ECAPA` similarity across 8/8 pairs**. |
| **2. Bird-Filter High-Frequency Roll-off (`>3.5 kHz`)** | Spotted that `S01__A.wav` (`exp16_clip_0000.wav`) drops to `-77 dB` at `5 kHz` because `strip_birds_preserve_fry` attenuated `>2.6 kHz` and ran full-band `noisereduce`, corrupting Praat's 5-formant tracker (`max_formant=5000 Hz`). | `Opus` ran in a cloud sandbox without access to the raw archival sermons or `reference.wav`. By running locally on both **raw unfiltered `reference.wav`** (`Bandwidth = 5,892 Hz`, `F3/F4 = 2404 / 3352 Hz`) and **`S01__A.wav`** (`Bandwidth = 4,340 Hz`, `F3/F4 = 1797 / 2537 Hz`), `Voice Atlas 2.0` proved *quantitatively* that the old bird filter pulled `F2/F3/F4` down by `500–800 Hz`, and built the surgical `>4,200 Hz`-only notch (`exp17_ref_7s.wav`) that restores `F2/F3/F4` (`1159 / 2104 / 2921 Hz`) and raises `ECAPA` to `0.810`. |
| **3. Corpus Scale & `MPS` `ECAPA` Execution** | Designed clean 2-pass Praat pitch tracking (`75–500 Hz` pass 1 $\to$ $[0.72 \times q_{25}, \min(420, 1.65 \times q_{75})]$ pass 2), `1/3`-octave LTAS spectra, and pause-aware prosody segmentation. | `Opus`'s cloud HTML contained **0 archival sermons** (only 4 short clips, two of them duplicates) and an empty Voice Map (`PCA = null`). Furthermore, running `Opus`'s `voice_atlas.py` directly on the local folders caused a **plot explosion** (`65` separate enrollment clip boxplots + `51` output files including temp `.wav` chunks and `edge_tts` baselines). `Voice Atlas 2.0` pooled the 65 enrollment clips into `POOLED_65_Clean_Clips (7.09 min)` and ran batched `MPS` `ECAPA-TDNN` across **11 full Kannada & Marathi sermons (`99 min`)**, **5 reference anchors**, **16 `E003` clips**, and **16 clone outputs**. |
| **4. Sub-Second Trajectory of *"ಬರ್ರಿ"* (`EXP-015 Opt2` vs `EXP-016`)** | Noted that whole-clip averages on zero-shot `Seed-VC` and `OmniVoice` looked close to `S01__A.wav` even though the listener rejected both. | Added **Tab 2 (`Forensics: E003, Bird Filter & 'ಬರ್ರಿ'`)** with `0.25s`-windowed time-aligned trajectories of `F0(t)`, `CPP(t)`, and `F3(t)` across the final 28% of the utterance (`"ಎಲ್ಲಾರೂ ಬರ್ರಿ"`), proving physically why `EXP-015 Opt2` hit `40%` on *"ಬರ್ರಿ"* while `EXP-016` collapsed to `0%`. |

---

## 3. The Five Forensic Discoveries from `Voice Atlas 2.0`

### Discovery 1: The `1.378x` `librosa.pyin` Sample-Rate Bug (Why `EXP-016` Threw Away 95% of `DOC-0004`)
- **The Bug:** Earlier dataset curation scripts loaded audio at `16,000 Hz` (`load16`) but called `librosa.pyin(y, fmin=80, fmax=300)` **without passing `sr=16000`**. Because `librosa.pyin` defaults to `sr=22050`, every pitch value stored in `metadata.jsonl` was multiplied by:
  $$\frac{22050}{16000} = 1.378125\times \quad (+5.55\text{ semitones})$$
- **Measured Proof:**
  - `S01__A.wav` (`exp16_clip_0000.wav`) was listed in `metadata.jsonl` at **`172.6 Hz`**, but its true 2-pass Praat median pitch is **`125.8 Hz`** ($125.8 \times 1.378 = 173.4\text{ Hz}$).
  - Across all 65 clips in `clips_exp16_pure`, the true pooled median pitch is **`133.6 Hz`** (`P10–P90 = 115.6–164.3 Hz`), whereas `metadata.jsonl` listed `184.1 Hz`.
- **How This Sabotaged `EXP-016`:**
  - `EXP-016` enforced a strict pitch ceiling of `metadata f0 <= 192 Hz` intending to remove voices above `192 Hz`.
  - In true physical pitch, $192 / 1.378125 = \mathbf{139.3\text{ Hz}}$!
  - Because the speaker's natural animated conversational pitch in `DOC-0004` sits around `138–165 Hz`, that filter **discarded 77 out of 81 close-mic `DOC-0004` clips**, starving `EXP-016` of the very clips that drove `EXP-007`, `EXP-008`, `EXP-012`, and `EXP-015 Opt2`!

### Discovery 2: Why the Old Bird Filter (`strip_birds_preserve_fry`) Collapsed `F2/F3/F4` Formants
- Comparing the three versions of the primary reference anchor in `Voice Atlas 2.0`:

| Reference Anchor | Filtering Applied | Effective Bandwidth (`99%` Roll-Off) | `ECAPA` Sim to Centroid | `CPP` (dB) | Praat Formants `F1 / F2 / F3 / F4` (Hz) |
|---|---|---:|---:|---:|---|
| **`reference.wav`** | **Raw Unfiltered Close-Mic (`82s`)** | **`5,892 Hz`** | `0.785` | `14.70 dB` | **`466 / 1363 / 2404 / 3352 Hz`** |
| **`S01__A.wav` (`exp16_clip_0000`)** | **Old `strip_birds_preserve_fry`** (`>2.6 kHz` split + full-band `noisereduce` `prop=0.55`) | `4,340 Hz` (`-77 dB` at `5 kHz`) | `0.786` | `16.24 dB` | `460 / 880 / 1797 / 2537 Hz` *(collapsed!)* |
| **`exp17_ref_7s.wav` (`EXP-017`)** | **New `strip_birds_above_4200_only`** (`0–4.2 kHz` 100% untouched; notch `>4.2 kHz` only) | **`5,057 Hz`** | **`0.810`** | `15.92 dB` | **`474 / 1159 / 2104 / 2921 Hz`** *(restored!)* |

- **Physical Explanation:** Full-band `noisereduce` in `EXP-015`/`016` stripped 30 dB of upper-formant energy between `2.6 kHz` and `4.2 kHz`, artificially smoothing glottal micro-rasp (`CPP` jumped from `14.70 dB` to `16.24 dB`) and causing Praat's 5-formant LPC tracker to fit two formants into the low-frequency band (`F3` dropped from `2404 Hz` to `1797 Hz`). Keeping `0–4,200 Hz` 100% untouched in `EXP-017` restored `F2/F3/F4` and boosted `ECAPA` cosine to `0.810`.

### Discovery 3: The `SNAC 24 kHz` Codec Ceiling (`E003` Quantitative Audit Across All 8 Pairs)
- Measured across all 8 `E003` real clips (`S01__A..S08__A`) vs their `SNAC 24 kHz` round-trip reconstructions (`S01__B..S08__B`):

| Pair | Source Clip | `F0` Real $\to$ `SNAC` (Hz) | `CPP` Real $\to$ `SNAC` (dB) | `HNR` Real $\to$ `SNAC` (dB) | `F3` Real $\to$ `SNAC` (Hz) | `ECAPA` Cosine Real $\to$ `SNAC` ($\Delta$) |
|---|---|---|---|---|---|---|
| **`S01`** | `exp16_clip_0000` (`ref`) | `125.8` $\to$ `126.1` | `16.24` $\to$ `16.06` (`-0.18`) | `11.7` $\to$ `11.1` | `1797` $\to$ `1804` | **`0.786` $\to$ `0.583` (`-0.203`)** |
| **`S02`** | `exp16_clip_0001` (`ref`) | `127.6` $\to$ `128.4` | `14.43` $\to$ `15.05` (`+0.62`) | `11.8` $\to$ `11.7` | `2049` $\to$ `2009` | **`0.766` $\to$ `0.631` (`-0.135`)** |
| **`S03`** | `exp16_clip_0004` (`ref`) | `125.0` $\to$ `124.2` | `16.36` $\to$ `16.95` (`+0.59`) | `11.9` $\to$ `12.6` | `1992` $\to$ `2084` | **`0.831` $\to$ `0.678` (`-0.153`)** |
| **`S04`** | `exp16_clip_0009` (`DOC-0004`) | `135.4` $\to$ `134.8` | `14.92` $\to$ `16.22` (`+1.30`) | `11.0` $\to$ `11.4` | `1919` $\to$ `1983` | **`0.786` $\to$ `0.667` (`-0.119`)** |
| **`S05`** | `exp16_clip_0010` (`DOC-0004`) | `135.1` $\to$ `136.6` | `13.94` $\to$ `14.84` (`+0.90`) | `10.1` $\to$ `10.7` | `2305` $\to$ `2280` | **`0.715` $\to$ `0.562` (`-0.153`)** |
| **`S06`** | `exp16_clip_0012` (`DOC-0004`) | `129.6` $\to$ `128.8` | `13.68` $\to$ `14.70` (`+1.02`) | `10.2` $\to$ `9.9` | `1890` $\to$ `1918` | **`0.685` $\to$ `0.615` (`-0.070`)** |
| **`S07`** | `exp16_clip_0045` (`K41`) | `125.4` $\to$ `125.7` | `13.38` $\to$ `14.12` (`+0.74`) | `12.5` $\to$ `11.5` | `2343` $\to$ `2267` | **`0.737` $\to$ `0.483` (`-0.254`)** |
| **`S08`** | `exp16_clip_0052` (`K41`) | `123.5` $\to$ `124.2` | `13.48` $\to$ `14.86` (`+1.38`) | `12.5` $\to$ `13.4` | `2301` $\to$ `2247` | **`0.822` $\to$ `0.742` (`-0.080`)** |

- **Takeaway:** Passing real speech through `SNAC 24 kHz` alone—with zero TTS modeling error—**reduces `ECAPA` similarity on 8 out of 8 clips (`-0.070` to `-0.254`, median `-0.127`)** and **inflates `CPP` periodicity on every natural/raspy clip (`S02–S08`) by `+0.59 to +1.38 dB`**, smoothing out the elderly glottal micro-tremor and sub-harmonics into a cleaner, more synthetic waveform. This directly explains why a second-stage continuous vocoder/converter (`Seed-VC` with continuous `BigVGAN` mel synthesis in `EXP-017B`) is needed to break through `SNAC`'s acoustic ceiling.

### Discovery 4: Why `EXP-015 Opt2` Achieved 40% Resemblance on *"ಬರ್ರಿ"* While `EXP-016` Dropped to 0%
- Sub-second (`0.25s` step, `0.45s` window) tracking across the final 28% of the utterance (`"ಎಲ್ಲಾರೂ ಬರ್ರಿ"`) revealed:

| Audio File | Tail Segment (`"ಎಲ್ಲಾರೂ ಬರ್ರಿ"`) Median $F_0$ | Tail `CPP` (Periodicity) | Tail `F3` (Pharyngeal Formant) | Physical Alignment to Unfiltered Speaker |
|---|---:|---:|---:|---|
| **`reference_10s.wav` (`10s` Raw Anchor)** | **`133.2 Hz`** | **`15.90 dB`** | **`2443 Hz`** | **Ground-Truth Baseline** |
| **`exp15_opt2_svara_fixed.mp3` (`EXP-015 Opt2`, 40% on *"ಬರ್ರಿ"*)** | **`132.5 Hz`** (`-0.1 st`) | **`16.29 dB`** (`+0.39 dB`) | **`2431 Hz`** (`-12 Hz`) | **Locked onto true $F_0$, `CPP`, and `F3` during *"ಬರ್ರಿ"*!** |
| **`exp16_clone_voice.mp3` (`EXP-016 RSLoRA`, 0%)** | `142.8 Hz` (`+1.2 st`) | `17.92 dB` (`+2.02 dB` buzzy) | `1849 Hz` (`-594 Hz` muffled) | Overfitted to bird-filtered formants & `RSLoRA` `11.3x` scale |

- In addition to `RSLoRA` overfitting (`scale = 11.31` vs `2.0` without holdout validation), `EXP-016` applied repetition penalty across all 3 `SNAC` codebooks (`c0, c1, c2`), perturbing the fine acoustic timbre codebooks (`c1, c2`), whereas penalizing **only coarse `c0` framing tokens** (`C0OnlyGeneratedRepetitionPenalty` in `EXP-017A`) prevents loop stalls without distorting vocal tract timbre.

### Discovery 5: Complete Multi-Sermon Acoustic Baseline (`Voice Atlas 2.0` Across 11 Archival Sermons, `99 Min`)

| Recording / Anchor | Category | Analyzed Min | `ECAPA` Purity (`>=0.50`) | True Median $F_0$ (Hz) | Pitch Span (`P10–P90`, st) | `CPP` (dB) | `H1-H2` (dB) | `F1 / F2 / F3 / F4` (Hz) | Syllables/s |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| **`POOLED_65_Clean_Clips`** | Pooled Close-Mic | `7.09` | `100%` (`0.785`) | `133.6` | `6.1 st` | `14.66` | `-6.31` | `460 / 1138 / 2142 / 2948` | `5.04` |
| **`reference.wav`** | Raw Unfiltered Anchor | `1.37` | `100%` (`0.785`) | `134.8` | `6.1 st` | `14.70` | `-6.54` | `466 / 1363 / 2404 / 3352` | `4.95` |
| **`exp17_ref_7s.wav`** | `0–4.2 kHz` Intact Ref | `0.12` | `100%` (`0.810`) | `125.8` | `5.3 st` | `15.92` | `-1.82` | `474 / 1159 / 2104 / 2921` | `5.66` |
| **`DOC-0004`** | Close-Mic Phone | `9.00` | `83%` (`0.619`) | `150.7` | `7.1 st` | `14.59` | `-5.31` | `444 / 1198 / 2258 / 3050` | `4.79` |
| **`K41`** | Intimate Informal | `9.00` | `94%` (`0.684`) | `122.4` | `5.0 st` | `11.88` | `-6.04` | `454 / 1299 / 2387 / 3241` | `4.77` |
| **`K13`** | Intimate Informal | `9.00` | `74%` (`0.569`) | `126.9` | `11.0 st` | `10.90` | `-7.42` | `495 / 1368 / 2472 / 3355` | `4.68` |
| **`kan_40`** | Kannada Sermon | `9.00` | `96%` (`0.625`) | `157.5` | `8.5 st` | `13.48` | `-5.91` | `423 / 1185 / 2219 / 2989` | `4.57` |
| **`K15`** | Kannada Sermon | `9.00` | `73%` (`0.549`) | `156.9` | `5.4 st` | `12.56` | `-6.37` | `520 / 1387 / 2393 / 3347` | `4.74` |
| **`K20`** | Kannada Sermon | `9.00` | `88%` (`0.628`) | `151.3` | `7.1 st` | `11.57` | `-8.50` | `508 / 1475 / 2520 / 3383` | `4.77` |
| **`K32`** | Kannada Sermon | `9.00` | `70%` (`0.550`) | `149.4` | `6.7 st` | `12.69` | `-6.73` | `510 / 1383 / 2421 / 3354` | `4.81` |
| **`K47`** | Kannada Sermon | `9.00` | `95%` (`0.674`) | `151.2` | `5.6 st` | `12.39` | `-7.73` | `510 / 1397 / 2424 / 3394` | `4.76` |
| **`PBP 1`** | Kannada Sermon | `9.00` | `86%` (`0.594`) | `146.9` | `6.4 st` | `13.21` | `-7.56` | `454 / 1280 / 2378 / 3260` | `4.75` |
| **`M01` (Marathi)** | Marathi Discourse | `9.00` | `89%` (`0.601`) | `142.3` | `8.0 st` | `12.66` | `-5.82` | `498 / 1372 / 2472 / 3371` | `4.81` |
| **`M02` (Marathi)** | Marathi Discourse | `9.00` | `92%` (`0.595`) | `149.8` | `6.6 st` | `12.57` | `-6.97` | `518 / 1386 / 2394 / 3352` | `4.79` |

---

## 4. Execution of `EXP-017` (`EXP-017A` & `EXP-017B`)

### 4.1 Step 1: Rebuilding the Close-Mic Dataset with `0–4,200 Hz` 100% Untouched (`clips_exp17_unfiltered`)
1. **Surgical `>4,200 Hz`-Only Bird Filter (`strip_birds_above_4200_only`):**
   - Decomposed audio via a steep 6th-order Butterworth crossover at **`4,200 Hz`**.
   - Kept **`0–4,200 Hz` (`F0` and `F1–F4` vocal tract resonances) 100% bit-for-bit untouched** (`zero` `noisereduce` below `4,200 Hz`), applying gentle spectral gating only to the `>4,200 Hz` band where outdoor bird chirps (`4.5–8.0 kHz`) reside.
2. **True 2-Pass Praat Pitch Gate & Strict Text Hygiene:**
   - Re-scored all candidate segments from the 4 intimate close-mic recordings (`reference.wav`, `DOC-0004`, `K13`, `K41`) using **true 2-pass Praat $F_0 \in [105, 175]\text{ Hz}$**, `ECAPA sim >= 0.50`, `RMS >= 0.035`, `duration 2.2–10.5s`, syllable rate `2.5–7.5 syl/s`, and `100%` Kannada Unicode grapheme verification (`IndicConformer-600M` + `MMS CTC`).
   - Admitted **`87` high-purity close-mic clips (`9.43 min`, true Praat median $F_0 = 135.9\text{ Hz}`, mean `ECAPA sim = 0.601`)** into `clips_exp17_unfiltered`, plus **`31` top clips (`sim >= 0.58`)** into `seedvc_ft_clips`.

### 4.2 Step 2 (`EXP-017A`): Controlled `Svara-TTS` Plain LoRA + Holdout Validation Loss + `c0`-Only Repetition Penalty
1. **Plain LoRA (`use_rslora=False`, Rank 32, $\alpha=64$, scale `2.0`):** Reverted from `RSLoRA` (`scale = 11.31`) back to standard Plain LoRA (`scale = 2.0`) on `81` training clips + `6` stratified holdout clips (`seed=42`, `lr=1.2e-4`, 6 epochs).
2. **Per-Epoch Holdout Validation Loss Selection:**
   - Evaluated cross-entropy loss on the 6 unseen holdout clips at the end of every epoch:
     - Epoch 1 (`step 20`): `train_loss = 5.197`, **`val_loss = 5.2845`**
     - Epoch 2 (`step 40`): `train_loss = 4.654`, **`val_loss = 5.0989`**
     - Epoch 3 (`step 60`): `train_loss = 4.404`, **`val_loss = 4.9963`**
     - Epoch 4 (`step 80`): `train_loss = 4.249`, **`val_loss = 4.9562`**
     - Epoch 5 (`step 100`): `train_loss = 4.132`, **`val_loss = 4.9403`**
     - Epoch 6 (`step 120`, selected best): `train_loss = 4.058`, **`val_loss = 4.9347`**
   - Unlike `RSLoRA` (which drove training loss to `3.152` by memorizing training targets), Plain LoRA decreased unseen holdout validation loss monotonically through Epoch 6 (`4.9347`).
3. **`C0OnlyGeneratedRepetitionPenalty` & 2-Clause Synthesis:**
   - Penalized only newly generated coarse `c0` framing tokens (`128266..132361`) at `1.12`, leaving fine timbre codebooks (`c1, c2`) completely unpenalized, conditioned on `exp17_ref_7s.wav` (`0–4.2 kHz` untouched).

### 4.3 Step 3 (`EXP-017B`): Fine-Tuning `Seed-VC` on `MPS` Over Unfiltered Close-Mic Clips
1. **Fine-Tuning `Seed-VC` (`DiT` Continuous Flow-Matching):**
   - Trained `Seed-VC` (`config_dit_mel_seed_uvit_whisper_small_wavenet.yml`, `22.05 kHz` continuous mel + `BigVGAN` vocoder) on the M5 Max `MPS` GPU for **`300 steps` (`batch_size=2`, `max_epochs=150`)** on the 31 unfiltered close-mic clips (`seedvc_ft_clips`). Diffusion flow-matching loss converged cleanly from `0.79` to **`0.26`** (`ft_model.pth`).
2. **Two-Stage Conversion (`Svara-TTS` Dialect & Pacing $\to$ Fine-Tuned `Seed-VC` Vocal Tract & Glottal Quality):**
   - Converted both **`EXP-017A` (`exp17a_svara_controlled.mp3`)** and **`EXP-015 Option 2` (`exp15_opt2_svara_fixed.mp3`, the 40% *"ಬರ್ರಿ"* winner)** through the fine-tuned `Seed-VC` checkpoint (`diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7`) using a `20s` unfiltered reference clip (`speaker_unfiltered_20s.wav`).

### 4.4 Acoustic Telemetry Comparison Across Key Deliverables (`Voice Atlas 2.0`)

| Deliverable | Architecture | Duration | `ECAPA` Cosine (vs Pooled Profile) | True Median $F_0$ (Hz) | `CPP` (dB) *(Target: `10.9–14.7`)* | `H1-H2` (dB) *(Target: `-7.5 to -5.3`)* | `F1 / F2 / F3 / F4` (Hz) |
|---|---|---:|---:|---:|---:|---:|---|
| **`POOLED_65_Clean_Clips` (Speaker Ground Truth)** | Real Speech (`7.09 min`) | — | `0.785` | `133.6 Hz` | **`14.66 dB`** | **`-6.31 dB`** | `460 / 1138 / 2142 / 2948` |
| **`reference.wav` (Raw Unfiltered Anchor)** | Real Speech (`1.37 min`) | — | `0.785` | `134.8 Hz` | **`14.70 dB`** | **`-6.54 dB`** | `466 / 1363 / 2404 / 3352` |
| `exp12_clone_voice.mp3` (`EXP-012`) | `Svara-TTS` LoRA (`r=32`) | `8.28s` | `0.680` | `136.4 Hz` | `17.50 dB` *(+2.8 dB)* | `-4.72 dB` | `461 / 1219 / 2167 / 2806` |
| `exp15_opt2_svara_fixed.mp3` (`EXP-015 Opt2`, 40% *"ಬರ್ರಿ"*) | `Svara-TTS` LoRA (`r=32`, Prompt-Excl Penalty) | `8.27s` | `0.587` | `138.2 Hz` | `17.30 dB` *(+2.6 dB)* | `-2.78 dB` | `508 / 1207 / 2305 / 2925` |
| `exp16_clone_voice.mp3` (`EXP-016`, `0%`) | `Svara-TTS` `RSLoRA` (`r=32`, Bird-Filtered) | `8.88s` | `0.720` | `133.8 Hz` | `17.42 dB` *(+2.7 dB)* | `-6.37 dB` | `520 / 1106 / 1945 / 2678` |
| **`exp17a_svara_controlled.mp3` (`EXP-017A`)** | `Svara-TTS` Plain LoRA + Holdout Val + `0–4.2k` Intact + `c0`-Only Penalty | `8.69s` | `0.679` (`0.726` on `seed77`) | **`125.9 Hz`** | `16.40 dB` *(lowest `Svara` `CPP`)* | `-5.32 dB` | `472 / 1126 / 2167 / 2852` |
| **`exp17b_svara17a_seedvc_ft.mp3` (`EXP-017B-1`)** | **`EXP-017A` $\to$ Fine-Tuned `Seed-VC` (`300 steps, MPS`)** | `8.69s` | **`0.739`** | **`126.3 Hz`** | **`14.03 dB`** *(inside natural range!)* | **`-7.90 dB`** | `428 / 1222 / 1906 / 2826` |
| **`exp17b_svara15_seedvc_ft.mp3` (`EXP-017B-2`)** | **`EXP-015 Opt2` (40% *"ಬರ್ರಿ"*) $\to$ Fine-Tuned `Seed-VC` (`300 steps, MPS`)** | `8.27s` | **`0.726`** | **`122.7 Hz`** | **`13.62 dB`** *(inside natural range!)* | **`-9.18 dB`** | `427 / 1226 / 1984 / 2780` |
