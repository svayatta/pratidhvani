# Experiment plan

What to try next, in order, and how to run each one so the result counts.
The rules are in [CONTRIBUTING.md](../CONTRIBUTING.md), what was tried
so far is in [experiments/log.md](experiments/log.md), and the full technical
report on `EXP-014` through `EXP-017B`, the `Claude Opus` second-opinion audit,
and `Voice Atlas 2.0` is in
[experiments/exp-014-to-017-forensics-and-pivot.md](experiments/exp-014-to-017-forensics-and-pivot.md).

## Current Plan: Two-Stage Architecture (`Svara-TTS` Plain LoRA + Fine-Tuned `Seed-VC` on M5 Max `MPS` — Updated 8 Nov 2026)

### Why the Plan Reshaped After `EXP-016` (`Opus` Audit + `Voice Atlas 2.0` + 128 GB M5 Max)
1. **Hardware Shift (16 GB Mac mini $\to$ 128 GB Apple M5 Max `MPS`):** Memory is no longer the bottleneck. `Seed-VC` (v0.4.3, GPLv3) automatically runs and fine-tunes on Apple Silicon `MPS` from small close-mic datasets, making a **two-stage pipeline** (*"`Svara-TTS` Plain LoRA for North Karnataka dialect, phrasing, and cadence $\to$ Fine-Tuned `Seed-VC` continuous flow-matching for authentic vocal tract formants and glottal rasp"*) practical and 100% local.
2. **What `E003` (`SNAC 24 kHz` Codec Ceiling) Proved:** Passing 8 real close-mic speaker clips (`S01..S08`) through `SNAC 24 kHz` encode $\to$ decode without any TTS generation dropped `ECAPA` similarity on **8 out of 8 clips (`-0.070` to `-0.254`, median `-0.127`)** and artificially inflated connected-speech **Cepstral Peak Prominence (`CPP`)** from the speaker's natural `10.9–14.7 dB` range up to `14.8–17.0 dB` (`16.3–18.3 dB` on `Svara-TTS` outputs). Converting `Svara-TTS` takes through fine-tuned `Seed-VC` (`EXP-017B`) brought `CPP` right back into the speaker's natural range (**`13.62–14.03 dB`**).
3. **What `E004` & `Voice Atlas 2.0` Fixed in Data Preparation:**
   - **Fixed the `1.378x` `librosa.pyin` Sample-Rate Bug:** Old metadata scripts ran `librosa.pyin` at default `sr=22050` on `16 kHz` audio, inflating stored `f0` by $1.378\times$ and causing `EXP-016`'s `f0 <= 192 Hz` rule to discard clips with true pitch $>139.3\text{ Hz}$ (throwing away 77 of 81 `DOC-0004` clips). All pitch gating now uses 2-pass Praat autocorrelation (`105–175 Hz`, true median $F_0 = 135.9\text{ Hz}$).
   - **Replaced Full-Band `noisereduce` with a Surgical `>4,200 Hz`-Only Bird Notch:** Keeping **`0–4,200 Hz` (`F0` and `F1–F4`) 100% untouched** restored effective reference bandwidth from `4,340 Hz` to **`5,057 Hz`**, restored `F2/F3/F4` from `880 / 1797 / 2537 Hz` to **`1159 / 2104 / 2921 Hz`**, and raised reference `ECAPA` similarity to **`0.810`**.
   - **Plain LoRA + Holdout Validation Loss + `C0OnlyGeneratedRepetitionPenalty`:** Reverted from `RSLoRA` (`scale = 11.31`, which overfit in `EXP-016`) to Plain LoRA (`use_rslora=False`, `scale = 2.0`) selected by per-epoch holdout validation loss (`5.2845` $\to$ `4.9347` at epoch 6), and restricted repetition penalty strictly to coarse `c0` framing tokens (`128266..132361`) after `prompt_len`.

### Prioritized Active Backlog (`EXP-017` Listening Evaluation & Next Steps)

| Priority | Experiment | The Change | Why / Expected Outcome |
|---|---|---|---|
| **P0 (`EXP-017A` & `017B`)** | **Human Listening Evaluation of `EXP-017A` & `EXP-017B` Deliverables** | Compare `exp17a_svara_controlled.mp3` (`Svara` Plain LoRA + `0–4.2 kHz` intact), `exp17b_svara17a_seedvc_ft.mp3` (`EXP-017A` + Fine-Tuned `Seed-VC`), and `exp17b_svara15_seedvc_ft.mp3` (`EXP-015 Opt2` 40% *"ಬರ್ರಿ"* take + Fine-Tuned `Seed-VC`) against `S01__A.wav` and `exp17_ref_7s.wav` | Determines whether `EXP-017A`'s restored `0–4.2 kHz` formants alone improve `Svara-TTS` or whether `EXP-017B`'s fine-tuned `Seed-VC` (`CPP = 13.6–14.0 dB`) breaks through the `SNAC` codec ceiling |
| **P1 (`EXP-018A`)** | **`44.1 kHz` `f0`-Conditioned `Seed-VC` Fine-Tuning (`1000 steps` on `MPS`)** | Fine-tune `Seed-VC`'s `44.1 kHz` pitch-conditioned model (`config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`) for `1000 steps` on the `87` unfiltered `0–4.2 kHz` close-mic clips (`clips_exp17_unfiltered`) | Explicitly locks the `F0` contour from `Svara-TTS` while doubling output sample rate (`22.05 kHz` $\to$ `44.1 kHz` BigVGAN) and restoring upper-formant `F3` (`2400 Hz`) |
| **P2 (`EXP-018B`)** | **Expand `0–4.2 kHz`-Untouched Close-Mic Corpus (`DOC-0004` + `K41` + `K13`)** | Harvest additional verified segments from `DOC-0004` (`83%` speaker purity across 32 min) and `K41` (`94%` purity across 45 min) using the `0–4.2 kHz`-untouched pipeline and true 2-pass Praat pitch gate (`105–175 Hz`) | Expands training coverage from `9.4 min` (`87` clips) to `20+ min` of pure close-mic conversational speech |
| **P3 (`Goal 2: Marathi`)** | **Marathi Voice Cloning from Archival Marathi discourses (`M01`–`M15`)** | Apply the `0–4.2 kHz`-untouched curation pipeline + `IndicConformer-600M` Marathi (`mr`) ASR + `Svara-TTS` / `Seed-VC` fine-tuning across the 13 archival Marathi recordings (`M01` and `M02` already verified at `89–92%` speaker purity in `Voice Atlas 2.0`) | Unlocks Goal 2 (Marathi spiritual discourse synthesis in the speaker's authentic voice) |

---

## How an experiment runs

1. **Pick the next item below** and open a draft pull request with its
   experiment card filled in: the hypothesis and the one change.
2. **Make the change** in `ci/recipe.json`, in the code, or both. Explore
   locally as much as you like, but only the gate's run counts.
3. **Check locally:** `pytest` and `python ci/check_hygiene.py` must pass. Mark
   the pull request ready and add the `experiment` label.
4. **The owner reviews and adds `benchmark`.** The gate trains your recipe on
   the shared data, benchmarks it against main's run and posts the verdict.
5. **On a pass** (better on at least one metric, worse on none), the owner
   merges, and your recipe becomes main's. **On a fail**, add one line to the
   log on what you learnt and close the pull request.

The shared data (recordings, transcribed chunks, the benchmark's held-out
recordings) lives on the gate runner and is managed by the owner. Ask them
for a copy to explore with. Don't upload the recordings anywhere else: they
are a family archive.

## Backlog

Each item names its change, what should move on the gate, and its rough
cost on the runner.

| # | Experiment | The one change | Expect | Cost |
|---|---|---|---|---|
| 0 | **Baseline** (owner) | Score main's recipe once | Sets the numbers every other run is compared with | 1 run |
| 1 | **Clean close-mic data, verbatim** | The owner prepares DOC-0004 (`prepare`, IndicConformer text). Add its chunk folder to the recipe's `build-clips` | Similarity up; CER no worse. EXP-003/007 saw the gain with Whisper text, and this tests it with verbatim text | 1 run |
| 2 | **Pitch gate from his own clips** | Add `--f0-from-enroll` to `build-clips` | Similarity up when hall recordings are in the set; no change without them | 1 run |
| 3 | **More of his speech from the hall recordings** | New `diarize` step (pyannote, MIT) keeping only his turns, then `build-clips` | More training minutes at the same purity; similarity up | Code, then 1 run |
| 4 | **Dereverberation** | Same recipe, chunks enhanced with MossFormer2 (`prepare --enhance-python`) versus not | Answers whether reverb broke EXP-005. CER should fall if it helps | 1 run |
| 5 | **Pick the checkpoint by held-out loss** | `train` computes validation loss on held-out recordings; the recipe uses the best checkpoint | Avoids the overfitting that low training loss hides | Code, then 1 run |
| 6 | **Capacity** | Rank 32 (and later epochs and learning rate), one at a time | Only worth it once 1-4 give 20+ minutes of clean speech | 1 run each |
| 7 | **Best of 3 in the product** | Studio previews generate 3 takes and keep the best by similarity and CER (`best-of`) | Better previews without retraining; checked by listening, not the gate | Code only |

Items 1, 2 and 4 are recipe changes and can go in order. Items 3 and 5 are
code first: one pull request for the new step, with tests, which the gate
checks for "no worse". Then an experiment pull request uses it.

## Claims about the accent

The gate measures voice similarity and intelligibility, not accent. For an
accent claim, generate the A/B page from the gate's two runs
(`voiceclone listen-page`). Have at least three native speakers of the dialect
answer it, and attach `voiceclone compare --answers ...` to the pull request.

## Not worth repeating

- Zero-shot conditioning alone (EXP-001/002, EXP-015 Opt1/Opt3): it can't hold this voice without weight adaptation.
- Choosing the conditioning clip or the reference by its score against the
  evaluation target: that is the leak the benchmark refuses.
- Judging by training loss (`EXP-016` `RSLoRA` hit `3.152` train loss while overfitting), by `ECAPA` cosine alone without listening, or by audio after full-band `noisereduce` below `4,200 Hz`.
