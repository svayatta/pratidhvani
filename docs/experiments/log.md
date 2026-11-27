# Experiment log

One row per experiment. New experiments are added by their pull request,
with the quality-gate report linked. Rows before EXP-009 predate the gate.

## EXP-009 to EXP-013 (3-4 Nov 2026)

| Run | Change | Similarity | Listener | What it suggested |
|---|---|---|---|---|
| 009 | Human verbatim forced-aligned transcript (`DOC-0004.rtf`), MMS CTC alignment, pitch window 122.8-208.9 Hz, LoRA Rank 32 (5 epochs, lr 1.5e-4) | 0.51 (non-leaky, against ref) | 0% (rejected) | Mumbled and slurred like a drunk person; approximate RTF timestamps (15s, 20s, 45s) aligned text to non-speech harmonium intro drone. PR closed. |
| 010 | Return to clean VAD phone recording corpus (`clips_exp7`), LoRA Rank 32 (step 105/126, lr 1.2e-4, max-ref-sec 6.5s) | 0.67 (record similarity) | ~25% (plateaued) | Restored clean articulation (all clauses spoken, duration 8.19s) and accent improved over EXP-009, but vocal timbre resemblance plateaued at baseline (~25%). |
| 011 | Expanded multi-source corpus with informal talks (K13 & K41, 109 clips, 11.4 min), LoRA Rank 32 (6 epochs, lr 1.2e-4) | 0.39 | Mixed (+5% on phrase, stall on tail) | Conversational informal talks captured authentic timbre (+5% resemblance on "ಆರಾಮ ಇದ್ದೀರಿ ಏನು"), but whispered hallucination clips in K41 (`ದೆದೆದೆ...`) caused autoregressive stalling on the second sentence. |
| 012 | Sanitized conversational corpus with automated regex repetition filter (`(.)\1{2,}`) and RMS floor (>=0.035), LoRA Rank 32 | 0.72 (all-time record similarity) | ~30-35% (+5% gain) | Voice quality improved 5% more, accent right, pacing authentic. Sentence 1 slurred due to archaic chant reference conditioning; tail ("ಎಲ್ಲಾರೂ ಬರ್ರಿ") cut off by 8.0s budget. |
| 013 | Phrased conversational synthesis (Greeting + Invitation, 80ms dither transition), checkpoint-00080 (step 80/144) | 0.59 | Timbre regression (clean articulation & accent, 0% speaker resemblance) | Cured 'drunk person' slurring and 8.0s cutoff: 100% of sentence (8.70s, 36 syllables through "ಎಲ್ಲಾರೂ ಬರ್ರಿ") articulated crisply with correct accent. However, checkpoint-00080 lost speaker's distinct vocal tract rasp/gravel, smoothing into a generic clear voice. |

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
- The pitch gate is now `build-clips --min-f0/--max-f0`, or `--f0-from-enroll`,
  which takes the window from the speaker's own clips instead of a fixed
  135-195 Hz.
- Open questions for the benchmark: does dereverberation rescue the hall
  recordings (EXP-005)? Does rank 32 help once there is more data (EXP-008)?
  See [experiment-plan.md](../experiment-plan.md).
