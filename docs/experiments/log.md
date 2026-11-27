# Experiment log

One row per experiment. New experiments are added by their pull request,
with the quality-gate report linked. Rows before EXP-009 predate the gate.

For the full technical narrative, acoustic tables across 11 archival sermons,
the independent second-opinion audit (`Claude Opus`), `E003`/`E004` ceiling
tests, `Voice Atlas 2.0` root-cause discoveries, and `EXP-017A`/`EXP-017B`
telemetry, see
[**`exp-014-to-017-forensics-and-pivot.md`**](exp-014-to-017-forensics-and-pivot.md).

## Current Status & Architectural Pivot (`EXP-017A` & `EXP-017B` — 8 Nov 2026)

- **Why We Paused Blind Training After `EXP-016` & Consulted a Second LLM (`Claude Opus`):**
  1. **Metric-Perception Decoupling:** Automatic `ECAPA-TDNN` cosine similarity repeatedly scored rejected `0%` outputs higher (`0.768` on `EXP-015 Opt1`, `0.760` on `EXP-015 Opt3`, `0.765` on `EXP-016`) than `EXP-015 Opt2` (`0.576`), which achieved our all-time high **`40%` listener resemblance** on *"ಬರ್ರಿ"*.
  2. **Hardware Upgrade (16 GB Mac mini $\to$ 128 GB Apple M5 Max `MPS`):** Unlocked local fine-tuning of continuous flow-matching voice conversion (`Seed-VC` on `MPS`) and full-archive acoustic forensics across 11 Kannada & Marathi sermons (`99 min` analyzed).
  3. **Five Forensic Root Causes Uncovered by the Dual-LLM (`Voice Atlas 2.0`) Audit:**
     - **The `1.378x` `librosa.pyin` Sample-Rate Pitch Bug:** Earlier scripts called `librosa.pyin` without `sr=16000` (defaulting to `sr=22050`), inflating stored `f0` values in `metadata.jsonl` by $22050/16000 = 1.378\times$. The speaker's true 2-pass Praat median pitch is **`126–136 Hz`** (not `173–184 Hz`), refuting the `E004` pitch-gap premise. Because `EXP-016` filtered on stored `f0 <= 192 Hz`, it inadvertently rejected clips with true pitch $>139.3\text{ Hz}$—throwing away **77 of the 81** expressive `DOC-0004` segments.
     - **The Bird-Filter Bandwidth Trap (`>3.5 kHz` Roll-Off & Formant Collapse):** Full-band `noisereduce` in `strip_birds_preserve_fry` (`EXP-015`/`016`) stripped 30 dB above `3.5 kHz` (`-77 dB` at `5 kHz` on `S01__A.wav`), pulling Praat's 5-formant tracker down by `500–800 Hz` (`F2/F3/F4` collapsed from `1363 / 2404 / 3352 Hz` on raw `reference.wav` to `880 / 1797 / 2537 Hz`). Replacing it with a surgical `>4,200 Hz`-only notch (`0–4,200 Hz` 100% untouched in `exp17_ref_7s.wav`) restored bandwidth to **`5,057 Hz`**, `F2/F3/F4` to **`1159 / 2104 / 2921 Hz`**, and `ECAPA` similarity to **`0.810`**.
     - **The `SNAC 24 kHz` Over-Periodicity Ceiling (`E003` Codec Audit):** Across 11 archival sermons, the speaker's natural connected-speech **Cepstral Peak Prominence (`CPP`)** is **`10.9 to 14.7 dB`** (natural elderly glottal breathiness/rasp). Every `Svara-TTS` (`SNAC 24 kHz`) take sits at **`16.3 to 18.3 dB` (`+2.5 to +5.5 dB` too periodic/buzzy)**. Passing 8 real speaker clips (`S01..S08`) through a lossless `SNAC 24 kHz` round-trip (`E003`) reduced `ECAPA` similarity on **8 out of 8 clips (`-0.070` to `-0.254`, median `-0.127`)** and inflated `CPP` on noisy/raspy clips (`S02–S08`) by `+0.59 to +1.38 dB`.
     - **Sub-Second (`0.25s`) Forensic Proof of Why *"ಬರ್ರಿ"* Reached `40%` in `EXP-015 Opt2`:** During the final 28% of the utterance (`"ಎಲ್ಲಾರೂ ಬರ್ರಿ"`), `EXP-015 Opt2` settled to **`F0 = 132.5 Hz`, `CPP = 16.29 dB`, `F3 = 2431 Hz`**, matching unfiltered `reference_10s.wav` (`F0 = 133.2 Hz`, `CPP = 15.90 dB`, `F3 = 2443 Hz`), whereas `EXP-016` (`RSLoRA` scale `11.31` vs Plain LoRA `2.0`, plus penalizing `c1/c2` timbre codebooks) jumped to `F0 = 142.8 Hz`, spiked `CPP` to `17.92 dB`, and muffled `F3` to `1849 Hz`.
     - **Prompt & Codebook Repetition Penalty:** Standard `RepetitionPenaltyLogitsProcessor` penalizes prompt `SNAC` tokens (`EXP-001`–`014`); even after excluding the prompt (`EXP-015`/`016`), penalizing `c1/c2` codebooks perturbs vocal tract timbre. `C0OnlyGeneratedRepetitionPenalty` (`EXP-017A`) penalizes only coarse `c0` framing tokens (`128266..132361`).

## EXP-009 to EXP-017B (3-8 Nov 2026)

| Run | Change | Similarity | Listener | What it suggested |
|---|---|---|---|---|
| **017B** | **Two-stage pipeline (`Svara-TTS` prosody $\to$ Fine-Tuned `Seed-VC` continuous mel on `MPS`):** Fine-tuned `Seed-VC` (`DiT` + `BigVGAN`, `300 steps, batch_size=2, MPS`, loss `0.79` $\to$ `0.26`) on 31 unfiltered `0–4.2 kHz` close-mic clips (`seedvc_ft_clips`); converted both `EXP-017A` (`exp17b_svara17a_seedvc_ft.mp3`) and `EXP-015 Opt2` (`exp17b_svara15_seedvc_ft.mp3`) using `20s` unfiltered reference | **0.74** (`17B-1`) / **0.73** (`17B-2`) | Pending review | Bypassing `SNAC 24 kHz`'s discrete vocoder with fine-tuned `Seed-VC` continuous mel synthesis dropped connected-speech `CPP` from `16.40 dB` down to **`14.03 dB` (`17B-1`, $F_0 = 126.3\text{ Hz}$) and `13.62 dB` (`17B-2`, $F_0 = 122.7\text{ Hz}$)**—squarely inside the speaker's natural **`10.9–14.7 dB`** sermon range. |
| **017A** | **Controlled `Svara-TTS` Plain LoRA (`use_rslora=False`, $r=32, \alpha=64$, scale `2.0`, 6 epochs) + Holdout Validation Loss + `0–4.2 kHz` Intact Audio + `C0OnlyGeneratedRepetitionPenalty`:** Re-curated 87 close-mic clips (`9.43 min`, true Praat median $F_0 = 135.9\text{ Hz}$) with surgical `>4.2 kHz`-only bird notch (`0–4.2 kHz` 100% untouched); holdout `val_loss` improved monotonically (`5.2845` $\to$ `4.9347` at epoch 6) | **0.68** (`seed42`) / **0.73** (`seed77`) | Pending review | Restored `0–4.2 kHz` formant bandwidth (`5,057 Hz`, `F2/F3/F4 = 1126 / 2167 / 2852 Hz`), locked pitch to **`125.9 Hz`**, and achieved the lowest `CPP` (`16.40 dB`) of any direct `Svara-TTS` output while articulating 100% of both clauses cleanly. |
| **E003 / E004** | **Diagnostic ceiling checks (`Opus` second-opinion audit + `Voice Atlas 2.0`):** `E003` passed 8 real speaker clips (`S01..S08`) through `SNAC 24 kHz` encode $\to$ decode without TTS; `E004` audited true 2-pass Praat pitch across all clips and clones | `0.786` $\to$ `0.583` on `S01` (`-0.127` median drop across 8/8 pairs) | `S01__A` (real) chosen as clearer & truer than `S01__B` (`SNAC`) | `E003` proved `SNAC 24 kHz` drops `ECAPA` similarity on 8/8 real clips and inflates `CPP` periodicity (`+0.6 to +1.4 dB`). `E004` exposed the `1.378x` `librosa.pyin` sample-rate bug in old metadata (`sr=22050` default on `16 kHz` audio), confirming real median pitch is `126–136 Hz` (`E004` pitch-shift dropped). |
| **016** | **High-purity close-mic filter (`65 clips, 7.08 min`, stored `f0 <= 192 Hz`, `sim >= 0.53`) + Rank 32 `RSLoRA` (`use_rslora=True`, scale `11.31`, 7 epochs, train loss `3.152`)** | 0.77 (`0.72` vs pooled) | 0% (rejected vs `S01__A.wav`) | Five compounding bugs identified by `Voice Atlas 2.0`: (1) `1.378x` metadata pitch bug meant `f0 <= 192 Hz` threw away clips with true $F_0 > 139.3\text{ Hz}$ (discarding 77/81 `DOC-0004` clips); (2) full-band `noisereduce` bird filter stripped `>3.5 kHz` by `-30 dB` and collapsed `F3` to `1849 Hz`; (3) `RSLoRA` `5.7x` larger scale overfit without holdout validation; (4) penalized `c1/c2` timbre tokens; (5) `SNAC` `CPP` spiked to `17.92 dB` on tail. |
| **015** | **Three-architecture bake-off:** **Opt1:** Zero-shot `Seed-VC` (`DiT` + `BigVGAN`); **Opt2:** Patched `Svara-TTS` (Rank 32 LoRA, 6 epochs, 100% `IndicConformer-600M` text on 98 close-mic `clips_exp12` clips, `PromptExcludedRepetitionPenalty`); **Opt3:** Zero-shot `OmniVoice 0.6B` | Opt1: 0.77, **Opt2: 0.58**, Opt3: 0.76 | **Opt2: 40% on *"ಬರ್ರಿ"* (all-time high)**; Opt1 & Opt3: 0% | Fixing HuggingFace's `repetition_penalty` (which previously penalized reference prompt `SNAC` tokens) + 100% `IndicConformer-600M` transcripts on `clips_exp12` allowed `EXP-015 Opt2` to lock onto the speaker's exact tail trajectory (`F0 = 132.5 Hz`, `CPP = 16.29 dB`, `F3 = 2431 Hz`) on *"ಬರ್ರಿ"*. Zero-shot `Seed-VC` and `OmniVoice` failed without weight adaptation. |
| 014 | Expanded corpus with 30 `IndicConformer-600M` + `MMS CTC` verified `PBP` hall clips (`clips_exp14`, 132 clips), exact ref text on `reference.wav`, LoRA Rank 32 (3 epochs / 94 steps), 2-clause synthesis | 0.55 | 0% (rejected: "sounds nothing like the speaker") | Mixing 30 reverberant `PBP` hall clips diluted close-mic vocal tract resonance (same trap as EXP-004/005), copying the 102 `clips_exp12` rows unchanged left Whisper Turkish/Hindi/gibberish text on close-mic clips ("split-brain" dataset), and cutting training from 6 to 3 epochs (`94 steps`) left timbre near the base model. |
| 013 | Phrased conversational synthesis (Greeting + Invitation, 80ms dither transition), checkpoint-00080 (step 80/144) | 0.59 | Timbre regression (clean articulation & accent, 0% speaker resemblance) | Cured 'drunk person' slurring and 8.0s cutoff: 100% of sentence (8.70s, 36 syllables through "ಎಲ್ಲಾರೂ ಬರ್ರಿ") articulated crisply with correct accent. However, checkpoint-00080 lost speaker's distinct vocal tract rasp/gravel, smoothing into a generic clear voice. |
| 012 | Sanitized conversational corpus with automated regex repetition filter (`(.)\1{2,}`) and RMS floor (>=0.035), LoRA Rank 32 | 0.72 (all-time record similarity) | ~30-35% (+5% gain) | Voice quality improved 5% more, accent right, pacing authentic. Sentence 1 slurred due to archaic chant reference conditioning; tail ("ಎಲ್ಲಾರೂ ಬರ್ರಿ") cut off by 8.0s budget. |
| 011 | Expanded multi-source corpus with informal talks (K13 & K41, 109 clips, 11.4 min), LoRA Rank 32 (6 epochs, lr 1.2e-4) | 0.39 | Mixed (+5% on phrase, stall on tail) | Conversational informal talks captured authentic timbre (+5% resemblance on "ಆರಾಮ ಇದ್ದೀರಿ ಏನು"), but whispered hallucination clips in K41 (`ದೆದೆದೆ...`) caused autoregressive stalling on the second sentence. |
| 010 | Return to clean VAD phone recording corpus (`clips_exp7`), LoRA Rank 32 (step 105/126, lr 1.2e-4, max-ref-sec 6.5s) | 0.67 (record similarity) | ~25% (plateaued) | Restored clean articulation (all clauses spoken, duration 8.19s) and accent improved over EXP-009, but vocal timbre resemblance plateaued at baseline (~25%). |
| 009 | Human verbatim forced-aligned transcript (`DOC-0004.rtf`), MMS CTC alignment, pitch window 122.8-208.9 Hz, LoRA Rank 32 (5 epochs, lr 1.5e-4) | 0.51 (non-leaky, against ref) | 0% (rejected) | Mumbled and slurred like a drunk person; approximate RTF timestamps (15s, 20s, 45s) aligned text to non-speech harmonium intro drone. PR closed. |

## EXP-001 to EXP-008 (31 Oct - 2 Nov 2026, before the gate)

These runs were scored on one sentence, one seed and a different
conditioning clip each time. EXP-003, 006, 007 and 008 trained on chunks of
the same reference recording they were scored against, so their similarity is
inflated. The listener's resemblance estimates are one person's impression
and were not blind. Read the rows as leads to test on the benchmark, not as
results.

| Run | Change | Similarity* | Listener | What it suggested |
|---|---|---|---|---|
| 001 | Zero-shot, 4.8 s vault clip as reference | 0.12 | 0% | Zero-shot conditioning keeps the dialect but not the voice. |
| 002 | Zero-shot, 10 s reference with its text | 0.19 | 0% | A longer reference does not fix it: the weights must adapt (LoRA). |
| 003 | First LoRA (rank 16, 3 epochs, ~11 min of a close-mic phone recording, DOC-0004) | 0.70 (leaky) | 10% | LoRA is the step that matters. The transcripts came from Whisper. |
| 004 | Added two hall recordings (K47, K20), 105 clips | 0.66 | 0% | Other voices, chanting and instruments passed the voiceprint filter; their pitch ran 230-560 Hz against the speaker's ~177 Hz. |
| 005 | Fixed 135-195 Hz pitch gate | 0.38 | 0% | 86% of clips still came from reverberant halls; output slurred, and loudness normalisation amplified the hiss. |
| 006 | Informal-talk recording (K41) plus the reference only | 0.03 | 0% | Whisper looped on whispered speech; mismatched text made generation stall. |
| 007 | Back to DOC-0004 plus the reference, ~90 clips, 5 epochs, reference text in the prompt | 0.64 (leaky) | 20% | Clean close-mic audio helps. The lowest training loss (4.31) on ~9 min is not evidence on its own. |
| 008 | Rank 32, alpha 64, 6 epochs | 0.63 (leaky) | 25% | More capacity on ~9 min of data; "3x RMS power" measured loudness after a gain step, not voice quality. |

\* ECAPA cosine of one take against the reference recording. For scale, the
reference adapter (`lora_v4`, trained on verbatim IndicConformer text) scored
about 0.70 with best-of-3 selection, against about 0.75 for the speaker's
own clips.

### What carries forward

- Train only on the speaker alone, from clean close-mic recordings. Reverberant
  halls and mixed voices hurt more than having less data.
- Transcripts must be verbatim; Whisper's paraphrases and loops broke two runs.
  This is now a ground rule.
- Never apply full-band spectral gating (`noisereduce`) below `4,200 Hz` on
  close-mic speech: it strips upper formant energy (`F3/F4`) and inflates
  artificial periodicity. Keep `0–4,200 Hz` untouched.
- Measure pitch with explicit sample rate (`sr=16000`) or 2-pass Praat
  autocorrelation (`105–175 Hz`), and measure connected-speech voice quality
  with **Cepstral Peak Prominence (`CPP`)** and **`H1-H2`** rather than
  sustained-vowel jitter/shimmer.
- Because `SNAC 24 kHz` imposes an over-periodic `CPP` ceiling (`16.3–18.3 dB`
  vs natural `10.9–14.7 dB`) and drops `ECAPA` similarity by `-0.13` median on
  round-trip encoding (`E003`), use the two-stage architecture (`Svara-TTS`
  Plain LoRA for North Karnataka dialect/prosody $\to$ fine-tuned `Seed-VC`
  continuous mel conversion for authentic vocal tract & glottal texture).
- See [experiment-plan.md](../experiment-plan.md) and
  [exp-014-to-017-forensics-and-pivot.md](exp-014-to-017-forensics-and-pivot.md).
