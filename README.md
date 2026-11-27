<img src="src/voiceclone/studio/static/logo.svg" alt="pratidhvani logo" width="72">

# pratidhvani

प्रतिध्वनि, *pratidhvani*: Sanskrit for "echo", a returning sound. Formerly
`indic-voice-clone`.

Open, self-hosted tools to keep the voices of India's speakers, accent and
dialect included. The code is open source under Apache 2.0; no audio is
licensed (see [Licence](#licence)).

## Why

Much of India's spoken heritage survives only in recordings: lectures,
discourses, oral histories, radio archives and family tapes. Commercial
text-to-speech offers a few studio voices per language and flattens regional
accents, so it cannot bring back a particular person, or the way people speak
in a particular place. This project's aim: a family, archive, school or
community should be able to take imperfect recordings of someone and produce
a voice that sounds like that person and speaks like where they are from, on
their own hardware, without handing the recordings to anyone.

What it stands for:

- **Faithful accent and dialect.** Verbatim transcripts, light-touch
  restoration, and native-speaker listening tests, not just numeric scores.
- **Built for archival audio.** Noisy, reverberant, decades-old recordings,
  not studio data.
- **Sovereign.** It runs locally, so the recordings and the trained voice
  stay with their owners.
- **Consent and provenance.** Only voices with consent. Every generated clip
  carries a [watermark](#watermark) that `voiceclone detect` can find.
- **Measured, not claimed.** Frozen benchmarks, guards against testing on
  training audio, and a quality gate on every change.
- **Open.** Apache-2.0 code on openly licensed models, growing from one
  dialect towards all of India's languages.

Uses it is meant for: hearing a teacher or an elder read new text in their own
voice; course material in a teacher's voice and dialect; voice banking for
speakers of Indic languages who are losing their speech; restoring and
re-voicing archival media with consent; and preserving dialects that
commercial voice products ignore.

The first test case is a northern Kannada speaker's lecture tapes, cloned on
a 16 GB Apple-silicon laptop.

## How it works

The recordings are cleaned lightly and cut into short chunks.
Each chunk is transcribed verbatim in the dialect, and only clean chunks of
the target speaker are kept. Those chunks LoRA-fine-tune the
[Svara-TTS voice-clone model](https://huggingface.co/kenpath/svara-tts-voiceclone-beta)
in the same "reference-swap" prompt format the model uses at inference.

## Getting started

Set up the repository and run the tests. This needs no GPU, no models and no
recordings, and takes a few minutes. To use the app without setting up the
source, run a published image instead: see [Docker](#docker).

You need Python 3.10 or newer (3.12 is tested), [uv](https://docs.astral.sh/uv/)
and [ffmpeg](https://ffmpeg.org/) on your `PATH`. On a Mac:

```bash
brew install uv ffmpeg
```

On Debian or Ubuntu, `apt install ffmpeg`, and uv from its website.

```bash
git clone https://github.com/svayatta/pratidhvani.git
cd pratidhvani
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]" -c constraints.txt
```

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/python ci/check_hygiene.py
```

The tests cover transliteration, transcript parsing, the Svara token and
prompt layout, pause segmentation, take selection, the benchmark's leakage
guard, paired comparison and the listening page, the closeness report (each
dimension blames only the change that was made), the quality gate's rules,
recipe checks and the studio's API. `tests/test_watermark.py` needs torch and
AudioSeal and is skipped without them. The hygiene check enforces the
repository rules in [CONTRIBUTING.md](CONTRIBUTING.md): no audio, weights or
home-folder paths in git.

To clone voices, install every extra into the same environment. This pulls
torch and the model libraries, about 1 GB:

```bash
uv pip install --python .venv/bin/python -e ".[all]" -c constraints.txt
source .venv/bin/activate
voiceclone --help
```

Transcription uses AI4Bharat IndicConformer, which is gated: open
[its model page](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual),
accept the terms, then run `hf auth login` (or set `HF_TOKEN`). Building the
aligner needs a C++ compiler once (`xcode-select --install` on a Mac,
`build-essential` on Debian or Ubuntu). Then try the
[voice studio](#voice-studio-web-app) with `voiceclone serve`, or follow
[Clone a voice from a recording](#clone-a-voice-from-a-recording).

`constraints.txt` pins the exact versions behind the reference results
(Python 3.12, macOS arm64). Leave it out to take the newest releases, at the
risk of behaviour changes. The extras are `tts` for synthesis and training,
`align` for forced alignment and character error, `speaker` for voiceprints,
`closeness` for the closeness report's identity and naturalness models,
`asr` for IndicConformer, `web` for the studio, `dev` for the tests, and `all`
for everything except `dev`.

## Requirements for cloning a voice

**Hardware.** An Apple-silicon Mac with 16 GB of memory, or a Linux machine
with an NVIDIA GPU with 16 GB or more of VRAM. Only the Mac has been tested;
the code picks CUDA automatically when it is available. Generating speech
with the 3B model on a CPU takes minutes per sentence, and training needs a
GPU. Allow about 15 GB of disk: the Svara model alone is 6.2 GB and the
alignment model 1.2 GB. Models download on first use: Svara and IndicConformer
go to the Hugging Face cache, and the alignment model to `~/ctc_forced_aligner/`.

**Recordings.** At least 10 to 20 minutes of the target speaker after
cleaning; the reference results used about 23 minutes from one 32-minute
lecture. Any format ffmpeg reads; MP3 at 128 kbps is fine. You also need the
right to use the voice: only clone someone who has consented, or whose estate
has.

**Languages.** Text in Kannada, Telugu, Tamil, Malayalam, Devanagari (Hindi,
Marathi, Sanskrit, Nepali), Bengali/Assamese, Gujarati, Gurmukhi and Odia is
transliterated for alignment, length limits and scoring. Only Kannada has
been tested end to end. Text in other scripts, such as Urdu, can still be
synthesised, but only with the 20-second length cap, and the alignment-based
steps cannot use it.

**Optional tools.**
- Audacity 3.x, for removing hiss, mains hum and clicks from tape. Audacity
  4 has no scripting. Enable `mod-script-pipe` under Preferences, Modules,
  then restart Audacity. The `audacity-denoise` step currently assumes
  macOS and `~/Applications/Audacity 3.7.app`; pass `--app` for another path.
  It quits and relaunches Audacity for every file, so it refuses to start
  while Audacity is open unless you pass `--force-restart`.
- ClearerVoice neural denoising, in a second environment because
  `clearvoice` needs `numpy<2`:

  ```bash
  uv venv --python 3.12 .venv-enh
  uv pip install --python .venv-enh/bin/python clearvoice "numpy<2" -e .
  ```

### Which install to use

| Your machine | Use | Runs on |
|---|---|---|
| Mac with Apple silicon | [Getting started](#getting-started) (from source) | Apple GPU |
| Linux with an NVIDIA GPU | [`:cuda` Docker image](#docker), or from source | CUDA |
| Anything else, or a quick try | [`:latest` Docker image](#docker) | CPU |

The package is not on PyPI yet. Each release will publish it (see
[CONTRIBUTING.md](CONTRIBUTING.md#releasing)), after which
`uv tool install "pratidhvani[all]"` installs the `voiceclone` command
without a checkout.

## Clone a voice from a recording

This walk-through turns one recording, `talk.mp3`, into a fine-tuned voice
and six test sentences. Run it from the repository root. Everything is
written under `data/`, `models/` and `outputs/`, which git ignores.

**1. Prepare the recording.** This decodes the file, restores it lightly,
cuts it into 3 to 10 second chunks at natural pauses, and transcribes each
chunk verbatim.

```bash
voiceclone prepare talk.mp3 talk --lang kn
```

The chunks and `transcripts.jsonl` land in `data/vad_talk/`. Add
`--enhance-python .venv-enh/bin/python` to denoise the chunks with
ClearerVoice first; the output then lands in `data/vad_talk_se/`.
Transcription takes a while: each chunk is a separate model call.

**2. Pick enrollment clips.** The filter in the next step keeps only chunks
that sound like the target speaker, so it needs a voiceprint to compare
against. Listen to a few chunks and copy 10 or more that are certainly the
target speaker, then index them:

```bash
mkdir -p data/enroll
cp data/vad_talk/talk_0012.wav data/vad_talk/talk_0015.wav ... data/enroll/
python -c "import json,pathlib; d=pathlib.Path('data/enroll'); \
open(d/'metadata.jsonl','w').writelines(json.dumps({'file':p.name})+'\n' for p in sorted(d.glob('*.wav')))"
```

**3. Build the training set.** This drops chunks that are another speaker,
too few words, an implausible speaking rate, or a transcript that does not
fit the audio. It prints what was dropped and why.

```bash
voiceclone build-clips --vad data/vad_talk --enroll data/enroll --n-enroll 10 --out data/clips
```

Repeat `--vad` to combine several prepared recordings.

**4. Fine-tune.** Six clips are held out for evaluation and listed in
`models/lora/holdout.json`. The final adapter is saved in `models/lora/`,
and every 50 optimizer steps a copy is kept in `models/lora/checkpoint-NNNNN/`
(about 100 MB each), so you can go back to an earlier one if it sounds
better. Change the interval with `--save-every`.

```bash
voiceclone train --clips data/clips --out models/lora --epochs 2
```

On an Apple M5 with 16 GB, 23 minutes of clips over 2 epochs took about 70
minutes, at 31 to 42 seconds per optimizer step.

**5. Synthesize.** Choose a clean 4 to 8 second clip from `data/clips` as the
reference, and put its text from `data/clips/metadata.jsonl` in a file.

```bash
voiceclone synth --lora models/lora \
    --ref data/clips/talk_0073.wav --ref-text-file ref.txt \
    --text-file examples/test_sentences_kn.txt --min-sps 2.5 \
    --out outputs/lora
```

Each sentence takes about 30 to 75 seconds on the same laptop. The output is
`outputs/lora/sample_XX.wav` plus `log.json`. Use `--text "..."` for a
single sentence.

**6. Tidy and measure.**

```bash
voiceclone trim --samples outputs/lora
voiceclone eval-similarity --clips data/clips --samples outputs/lora_trimmed
voiceclone eval-intelligibility --clips data/clips --samples outputs/lora_trimmed \
    --holdout models/lora/holdout.json
```

Both reports include the speaker's real clips as a baseline, so compare the
clone against those numbers rather than against perfect scores. In the
reference run, the speaker's real clips scored about 0.75 to 0.78 voice
similarity and the clone scored 0.61 to 0.72. Numbers do not judge accent, though: have native
speakers of the dialect listen.

**7. Improve (optional).** Generate several takes with different `--seed`
values and keep the best one per sentence with `best-of`. It pairs takes by
the sentence recorded in each folder's `log.json`, so folders made from
differently ordered sentence files still compare correctly. Add more
recordings with `prepare`, then retrain or continue training with
`train --init-adapter models/lora --lr 1e-4`.

## Voice studio (web app)

A local web page for the whole flow: name a voice, drop a 5-45 s sample or
record one in the browser, and the server cleans it (by default the same
ffmpeg restoration as `prepare`), picks the clearest 3-10 s stretch as the
reference, transcribes it verbatim and makes an instant clone you can type
text for. You can hear the upload and
the cleaned audio side by side, and correct the reference words. With longer
recordings of the same person (8 minutes or more of their speech alone), it
also trains a LoRA clone. The sample decides which voice to keep, other
speakers are filtered out, and previews then use the trained adapter.

```bash
voiceclone serve --root studio            # http://127.0.0.1:8000
```

The page shows a voice as four steps in order: the cleaned sample, the check
of the words, the text for the voice to say, and training. While a text
generates, a panel shows the step and the time. Versions of the same text
stay together, and the earlier versions are in a history row.

Every step has options, under **Advanced options**. Each group has a
**Reset** button that sets its defaults again:

| Panel | Options |
|---|---|
| Noise cleaning (new voice, or "Clean again") | declip; high-pass and its cut-off; spectral denoise, its strength and noise floor; MossFormer2 enhancement, its mix and super-resolution; peak normalisation |
| Clip to copy the voice from | the clearest 3-10 s stretch, or the whole sample (first 15 s) |
| How the voice speaks | the trained voice or the voice from the sample; number of versions and the seed; temperature, top-p, repetition penalty, longest sentence |
| Training | enhancement of the recordings; speaker-match threshold; pitch gate and its range; epochs, learning rate, LoRA rank, held-out clips, seed |

The defaults are the settings behind the reference results. MossFormer2
options are offered only when the server is started with
`--enhance-python` pointing at an environment with `clearvoice`. Each voice
records the options it was cleaned and trained with, and each preview records
its generation settings.

Voices are kept under `studio/voices/<id>/`: the upload, `clean.wav`,
`reference.wav`, previews and training output. Jobs run one at a time,
because the 3B model and training do not fit in memory together. The server
listens on this computer only unless you pass `--host`.

## Docker

Each merge to `main` publishes two images to `ghcr.io/svayatta/pratidhvani`:
`latest` (CPU, linux/amd64 and linux/arm64) and `cuda` (CUDA 12.6,
linux/amd64), plus `sha-…` and `sha-…-cuda` tags per commit. Both contain the
`voiceclone` command and every extra, and run the voice studio by default.

### Before you start

- **A Docker engine.** Docker Desktop, OrbStack or Podman on macOS or
  Windows; Docker Engine on Linux.
- **Memory for the engine's VM.** Give it at least 16 GB if you'll generate
  speech: on CPU the 3B model loads in 32-bit floats, about 12 GB. Docker
  Desktop's default is often lower (Settings → Resources). Cleaning and
  transcription alone need about 4 GB.
- **Disk.** About 2 GB for the image and 10 GB for the models it downloads on
  first use.
- **A Hugging Face token** (read-only is enough) for the gated
  IndicConformer model; see [Getting started](#getting-started).

### Run the studio

`:latest` runs on any machine with CPU-only torch. On a Linux host with an
NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/),
use `:cuda` with `--gpus all` instead (linux/amd64, about 6 GB, CUDA 12.6;
any CUDA 12 driver, 525 or newer):

```bash
docker run -d --name voiceclone --gpus all -p 127.0.0.1:8000:8000 -v voiceclone:/data -e HF_TOKEN=hf_... ghcr.io/svayatta/pratidhvani:cuda
```

On a CPU-only machine:

```bash
docker pull ghcr.io/svayatta/pratidhvani:latest
```

```bash
docker run -d --name voiceclone -p 127.0.0.1:8000:8000 -v voiceclone:/data -e HF_TOKEN=hf_... ghcr.io/svayatta/pratidhvani:latest
```

Open http://127.0.0.1:8000. The first sample takes a few minutes while the
models download into the `voiceclone` volume; later runs reuse them.
`127.0.0.1:` keeps the studio reachable from this computer only.

| To | Run |
|---|---|
| Follow the log | `docker logs -f voiceclone` |
| Stop it | `docker stop voiceclone` |
| Start it again (same settings and data) | `docker start voiceclone` |
| Update to the newest image | `docker pull ghcr.io/svayatta/pratidhvani:latest`, then `docker rm -f voiceclone` and the `docker run` above |
| Remove the container | `docker rm -f voiceclone` (voices and models stay in the volume) |
| Delete all voices and downloaded models | `docker volume rm voiceclone` |

To keep the data in a visible folder instead of a volume, use
`-v "$HOME/voiceclone-data:/data"`. On Linux, that folder must be writable by
user id 1000, the container's user.

### Other commands

Any `voiceclone` subcommand runs the same way. Mount the folder with your
recordings at `/work`:

```bash
docker run --rm -v voiceclone:/data -v "$PWD:/work" ghcr.io/svayatta/pratidhvani:latest restore /work/talk.mp3 --decode-first --out /work/talk_clean.wav
```

To check a recording for the [watermark](#watermark):

```bash
docker run --rm -v voiceclone:/data -v "$PWD:/work" ghcr.io/svayatta/pratidhvani:latest detect /work/suspect.mp3
```

### Build the image yourself

From a checkout of this repository:

```bash
docker build -t pratidhvani .
```

```bash
docker run -d --name voiceclone -p 127.0.0.1:8000:8000 -v voiceclone:/data -e HF_TOKEN=hf_... pratidhvani
```

To build for the other architecture (for example amd64 on an Apple-silicon
Mac), use `docker buildx build --platform linux/amd64 -t pratidhvani --load .`.
It builds under emulation, so it takes longer.

### What it can't do

Docker on a Mac can't use the Apple GPU, so there the image always runs on
CPU: cleaning, transcription and the studio are fine, but synthesis takes
minutes per sentence and training is impractical. Use a
[source install](#getting-started) on a Mac.

## Watermark

Every take the software generates carries an inaudible watermark: studio
previews, `synth`, `long-form` and benchmark runs alike. It is
[AudioSeal](https://github.com/facebookresearch/audioseal) (Meta, MIT) with a
fixed 16-bit tag that identifies this software. There is no option to turn it
off. To check a recording:

```bash
voiceclone detect suspect.mp3 another.wav
```

```
suspect.mp3: made with pratidhvani (detector 1.00, 16/16 tag bits)
another.wav: no watermark found (detector 0.00, 8/16 tag bits)
no watermark doesn't prove a recording is genuine: the code can be changed, and heavy noise can hide it
```

The command exits 0 only when every file carries the pratidhvani tag, and
exits 1 if any file has another AudioSeal mark or no mark. Pass `--json` for
a machine-readable list with `file`, `verdict`, `probability`, `bit_errors`
and `ours` fields.

The detector score (`probability`) is the marked share of the most-marked two
seconds of the file, not of the whole file. A file counts as made with
pratidhvani when it holds cloned speech anywhere: a clone cut into a longer
genuine recording, `long-form`'s `cloned.wav` with long silences between the
takes, and its `ab.wav`, where each original line is followed by its clone.

It sits about 26 dB below the voice, and genuine recordings of the speaker
scored 0. Every build checks that it survives the ways clips get shared
(`tests/test_watermark.py`):

| Shared as | Detector score (found at 0.5) |
| --- | --- |
| MP3 or AAC at 32 kbps | 1.00 |
| Telephone-band audio (8 kHz) | 1.00 |
| A messaging-app voice note (Opus, 16 kbps) | about 0.8 |
| A cropped, quieter excerpt | found |
| Three cloned seconds inside a minute of other speech | found |
| `long-form` output with 3 s takes 20 s apart | found |

Scores are from a synthetic voice-like test signal, so real recordings will
vary.

What it can't do:

- It marks what *this software* makes. Anyone can edit the code to remove the
  mark, or run a trained adapter with other software. It is a label that
  honest use keeps, not a lock.
- Heavy added noise weakens it, and very low bitrates strip it (Opus at
  8 kbps scores about 0.02). Not finding it doesn't prove a recording is
  genuine.
- "Another source" means an AudioSeal mark with a different tag, from other
  software that uses AudioSeal.

## Comparing runs

`eval-similarity` and `eval-intelligibility` say how one folder of samples
scores. To decide whether one adapter (or training recipe) is better than
another, the scores must come from the same test: the same held-out speech,
the same reference clip, the same sentences and seeds. `benchmark` fixes all
of that in a spec file and refuses adapters that trained on the test audio.

**1. Hold out whole recordings.** Similarity differs between clips of the
same session and clips of another session, and an adapter that trained on a
recording has heard its room, microphone and words. So hold out complete
recordings, not clips. Prepare one or
two recordings that no adapter will train on and build their clips into a
folder of their own:

```bash
voiceclone prepare talk9.mp3 TALK9 --lang kn
voiceclone build-clips --vad data/vad_TALK9 --enroll data/enroll --out data/clips_holdout
```

**2. Write the spec.** Copy `examples/bench_spec.example.json` (outside
git, for example to `data/bench.json`) and fill in the held-out clip folder,
the recording names in `holdout.sources`, and a conditioning clip with its
exact text. A recording's name is a clip's `source` field (kept by
`import-vault`), or else its file name without the chunk number: `prepare`
names chunks `TAG_NNNN.wav`, so the recording is `TAG`. The sentences are
stored in the spec itself, and the spec's hash is stored with every result,
so editing a spec starts a new benchmark rather than silently changing an
old one. `seeds` (default 0, 1, 2) sets how many takes each sentence gets.

**3. Run each adapter.**

```bash
voiceclone benchmark --spec data/bench.json --lora models/lora --out outputs/bench/lora
voiceclone benchmark --spec data/bench.json --lora models/lora_v2/checkpoint-00150 --out outputs/bench/v2_150
voiceclone benchmark --spec data/bench.json --out outputs/bench/zeroshot
```

Each run writes `seed_S/sample_XX.wav` (with `log.json`, so `trim` and
`best-of` work on them), `spec.json` and `scores.json`, and appends one row
to `outputs/bench/results.jsonl`: the adapter, its training parameters and
the recordings it trained on, and the scores. A run that is interrupted
resumes where it stopped. Six sentences and three seeds are 18 takes, about
10 to 25 minutes on an M5.

Scores per take, each also given for the speaker's real held-out clips:

| Score | Meaning | Better |
|---|---|---|
| `sim` | ECAPA cosine to the held-out voiceprint; each real clip is scored against the other held-out clips | higher |
| `cer` | character error of the MMS recogniser's decode | lower |
| `len_ratio` | length against 3.2 syllables/s; `len_err` is how far it is from the speaker's own ratio | closer to real |
| `stopped` | generation ended by itself, not at the length cap | higher |

The relative scores divide by the real clips' value: `sim` 90% means the
clone is 90% as close to the voiceprint as his own recordings are. A take
with no audio scores as no voice and nothing heard instead of being dropped.

Before synthesizing anything, `benchmark` reads the adapter's
`train_log.json` (a `checkpoint-NNNNN/` uses its run folder's), the clip
folder it names and its `metadata.jsonl`, and follows `--init-adapter`
chains. If a held-out recording or the conditioning clip's recording is
among the training clips, or the training data cannot be found, it refuses.
`--train-clips DIR` names the training clips when the log's relative path
does not resolve from here; `--allow-leak` scores anyway and flags the row.

**4. Compare two runs.**

```bash
voiceclone compare outputs/bench/lora outputs/bench/v2_150
```

Takes are paired by sentence and seed, and the mean difference (B minus A)
gets a 95% bootstrap interval that resamples whole sentences. A metric says
`B better` or `A better` only when its interval excludes zero; otherwise it
says `no clear difference`, which with six sentences is a common and honest
answer. Add sentences to the spec, as a new benchmark, to resolve smaller
differences.

**5. Listen blind.** Numbers do not judge accent, so ask native speakers:

```bash
voiceclone listen-page outputs/bench/lora outputs/bench/v2_150 --out outputs/bench/listen.html
```

This writes one self-contained HTML file (the audio is embedded, roughly 10
to 20 MB for 18 takes per run) and `listen.key.json`. Send only the HTML. For each
sentence and seed it plays the two runs' takes as 1 and 2 in random order and
asks which sounds more like him and which sounds like natural northern
Kannada; about one trial in six is a hidden check that pairs a real held-out
recording with a clone. The listener downloads their answers as JSON. Keep
the key file to yourself, and score the answers with:

```bash
voiceclone compare outputs/bench/lora outputs/bench/v2_150 \
    --key outputs/bench/listen.key.json --answers answers_X.json --answers answers_Y.json
```

This adds B's win rate per question with a 95% interval, and how often each
listener picked the real recording in the checks. A listener who cannot pick
out the real recording is guessing, so weigh their answers accordingly.

To find out *how* close each run is, not only which is preferred, add `--rate`:

```bash
voiceclone listen-page outputs/bench/lora outputs/bench/v2_150 --rate --out outputs/bench/rate.html
```

Each item plays a real recording of the speaker, then one take, and the
listener rates the take 1 to 5 on voice, accent, naturalness and pacing, with
an optional note on what sounds wrong. Takes from both runs are mixed in random
order. Some items are hidden anchors: a real recording rated against another.
They show what "sounds like the speaker" scores when it is the speaker. A
listener counts only after rating every anchor, averaging 4 or more on voice.
The interval resamples whole listeners, so one person's many sentences count
as one voter. The listener downloads
a file, which includes their notes, or copies an answer code, which holds the
ratings only. `compare --key outputs/bench/rate.key.json --answers ...` takes
either and reports each run's mean rating, the anchors' mean and B - A per
dimension with a 95% interval.

**6. See where a clone falls short.** `compare` says whether a run is better;
`closeness` says in what way one clone differs from him, so you know what to
fix next. It reports six dimensions, each banded close, noticeably different
or far: identity (two independent speaker models), timbre, pitch (level,
range, intonation), rhythm (pace, pauses), energy (loudness spread, stress)
and naturalness (predicted MOS). There is no overall score on purpose.

```bash
# best: a long-form folder, which pairs every clone chunk with his own words
voiceclone closeness --long-form outputs/long --baseline data/K14.wav --out outputs/closeness
# or any two files; add --same-words only if they say the same text
voiceclone closeness --clone take.wav --original K06.wav --baseline K14.wav --out outputs/closeness
```

`--baseline` is a second real recording of him (not the same take). Every
number is then shown next to what his own recordings reach, and a difference
no larger than theirs counts as close, because on tape even two recordings of
him disagree. Identity becomes a fraction of his own similarity, with the raw
voiceprint cosines on a row of their own. Timbre, the two contours, pace and
pauses need the same words, so they are not measured for different texts.
Timbre is compared only where the clone's pitch is within a semitone of his;
when the pitch is off everywhere it is not measured, so fix pitch first.
Pass `--lora` to refuse an original the adapter trained on; `--original-source`
names its recording tag (such as K06) when the file name doesn't, as for a
long-form folder. It writes `closeness.html` (audio, table and a pitch
contour per pair) and `closeness.json`. The timbre thresholds are provisional.

Identity asks two speaker models with different architectures, SpeechBrain's
ECAPA-TDNN and ResNet-TDNN, and takes the worse band, so a clone that fools
one model but not the other is not reported close. The ResNet's cosines have
not been calibrated on his recordings, so without `--baseline` they are shown
but not banded. Naturalness runs UTMOS (a predictor of the 1-5 score
listeners give) on both the clone and his recording and bands the gap, not
the score: archival tape scores low on its own, and a clone is judged by how
far below his recording it falls. Its thresholds are provisional too. Both
need the `closeness` extra; without it those rows read "not measured" and
the rest of the report is unchanged. `--no-identity` and `--no-naturalness`
skip them. The UTMOS weights (411 MB) are downloaded on first use into
`--utmos-dir` and checked against a pinned SHA-256.

## Command reference

All steps are subcommands of `voiceclone`; run `voiceclone <step> --help`
for every option.

| Step | Command | Output |
|---|---|---|
| Restore audio | `restore talk.mp3 --decode-first` | mono 24 kHz WAV, mild denoise |
| Cut at pauses | `vad --audio clean.wav --out data/vad` | 3-10 s chunks + `segments.jsonl` |
| Transcribe | `transcribe --vad data/vad` | `transcripts.jsonl`, verbatim, resumable |
| Whole preparation for one file | `prepare talk.mp3 TAG` | `data/vad_TAG[_se]/` |
| Import Vāṇī-Harness vault (optional) | `import-vault metadata.jsonl --vault-dir VAULT --out data/vault_kn` | vault clips as chunks; then `transcribe` |
| Analogue noise (optional) | `audacity-denoise talk.mp3 TAG --reuse-chunks data/vad_TAG_se` | hiss, hum, clicks removed |
| Rank recordings | `rank DIR --enroll data/enroll` | which talks are the speaker, how clean |
| Build training set | `build-clips --vad data/vad --enroll data/enroll --out data/clips` | speaker- and alignment-filtered clips |
| Fine-tune | `train --clips data/clips --out models/lora` | LoRA adapter + `holdout.json` |
| Synthesize | `synth --lora models/lora --ref clip.wav --ref-text-file ref.txt --text "..."` | `sample_XX.wav` + `log.json` |
| Trim run-ons | `trim --samples outputs/synth` | cut after the last word |
| Best take | `best-of --takes A --takes B --clips data/clips --out best` | best take per sentence |
| Voice similarity | `eval-similarity --clips data/clips --samples outputs/synth` | ECAPA cosine vs real clips |
| Intelligibility | `eval-intelligibility --clips data/clips --samples outputs/synth` | character error vs real clips |
| Same-words test | `long-form --lora models/lora --chunks data/vad_X --source X.wav --ref clip.wav --ref-text-file ref.txt --out outputs/long` | clone vs original, A/B file |
| Closeness report | `closeness --long-form outputs/long --baseline real.wav --out outputs/closeness` | identity, timbre, pitch, rhythm, energy, naturalness, banded |
| Benchmark a run | `benchmark --spec data/bench.json --lora models/lora --out outputs/bench/lora` | scores + `results.jsonl` row |
| Compare two runs | `compare outputs/bench/A outputs/bench/B` | paired differences, 95% interval |
| Blind listening | `listen-page outputs/bench/A outputs/bench/B --out listen.html` | A/B page + answer key |
| Listener ratings | `listen-page outputs/bench/A outputs/bench/B --rate --out rate.html` | 1-5 rating page with hidden anchors + key |
| Listening status | `listening-status --key rate.key.json --comments c.jsonl --listeners a,b --out status.json` | scores `/listening` codes for a pull request (CI) |
| Run a recipe | `run-recipe --recipe ci/recipe.json --data DATA --work RUN --spec DATA/bench.json` | adapter + benchmark run |
| Quality gate | `gate MAIN_RUN PR_RUN [--require-improvement]` | pass/fail + Markdown report |
| Web app | `serve --root studio` | the voice studio on port 8000 |
| Check for the watermark | `detect FILE...` | whether each file was made with this software |

If you have a timed human transcript (`[SS:MM:00] [Speaker] : text`) instead
of verbatim ASR, use `parse-transcript`, `align` and `segment --speaker NAME`
in place of `vad`, `transcribe` and `build-clips`. Its clips can also serve
as the enrollment set. Verbatim ASR gave better results, because a cleaned-up
transcript teaches the model words the speaker never said.

## Using clips from Vāṇī-Harness

Vāṇī-Harness is a companion restoration pipeline. It
separates the voice from music and crowd noise, removes room reverberation,
enhances, and writes admitted 3 to 15 second clips to a "cloning vault" with a
JSON-lines index. Those clips can replace or add to step 1 of Getting started:

```bash
voiceclone import-vault path/to/vault/metadata.jsonl --vault-dir path/to/vault --out data/vault_talk
voiceclone transcribe --vad data/vault_talk
voiceclone build-clips --vad data/vault_talk --vad data/vad_talk --enroll data/enroll --out data/clips
```

- **Transcripts are redone.** The vault's Whisper transcripts are not imported;
  on this archival speech they were unreliable (wrong language, repeated phrases).
  `transcribe` produces verbatim IndicConformer text instead.
- **Paths are matched by file name**, so the vault can be copied anywhere.
  `--min-dnsmos` and `--min-spk` add the vault's own quality and speaker
  scores as extra gates.
- **Check the accent before switching.** Dereverberation and enhancement change
  the sound the model learns. Train one adapter on vault clips and one on
  lightly restored clips, then compare them with `eval-similarity`,
  `eval-intelligibility` and native listeners.
- **Clip edges.** The vault splits long speech at fixed lengths, so clips can
  start or end mid-word. `build-clips` does not detect partial edge words yet.

## Troubleshooting

- **Out of memory while training.** Lower `--max-tgt-sec` (default 10.5) and
  `--max-ref-sec` (default 4.5), or `--rank`. Close other large apps.
- **The clone rambles past the sentence.** Run `trim`, or raise `--min-sps`
  to tighten the length cap.
- **The clone stops after a syllable.** Lower `--max-sps` to raise the
  minimum length.
- **Few clips survive `build-clips`.** Check the drop reasons it prints.
  `other_speaker` means the enrollment clips do not match, and `score` means
  transcripts do not fit the audio, often because of noise.
- **IndicConformer fails to download.** Accept its terms on Hugging Face
  and log in; see [Getting started](#getting-started).
- **Two runs of the same command give slightly different audio.** That is
  expected: the SNAC decoder adds a little random noise, and sampling is
  random unless `--seed` is fixed.

## Notes on keeping the accent

- Transcripts stay verbatim in the dialect. Pronunciation is learned from
  the audio, but the text must match what was said.
- Restoration is mild. Aggressive denoising shifts formants, which is where
  an accent lives. ClearerVoice (`prepare --enhance-python`) runs speech
  enhancement only: its super-resolution (`--steps se,sr` when running
  `python -m voiceclone.enhance.clearvoice` directly) shifted the median
  formants of archival lecture audio by hundreds of hertz. `--enhance-wet 0.9`
  mixes 10% of the original back in.
- Training uses the same reference-swap prompt as inference, so the adapter
  learns this speaker and dialect rather than a new speaker-ID token.
- The LoRA stays unmerged at inference: merging fp32 deltas into bf16
  weights rounds most of them away.
- Generation passes `use_cache=True`. The base model's saved config disables
  the cache, which makes generation about 10x slower.

## Package layout

```
src/voiceclone/
  text.py         Kannada transliteration (keeps the inherent vowel), syllables, CER
  audio.py        loading, normalising, ffmpeg decode and mild restoration
  transcript.py   timed-transcript parser
  segment.py      pause-based and alignment-based chunking
  asr.py          IndicConformer verbatim transcription
  align.py        MMS CTC forced alignment and greedy decoding
  speaker.py      ECAPA voiceprints and centroids
  dataset.py      training-clip filter
  rank.py         recording ranking
  svara/tokens.py prompt layout, SNAC token mapping, generation length budget (pure Python)
  svara/codec.py  SNAC encode/decode
  svara/synth.py  Synthesizer (zero-shot or LoRA); watermarks every take
  watermark.py    AudioSeal embed and detect
  train.py        reference-swap LoRA fine-tuning
  evaluate.py     similarity, intelligibility, best-of selection
  benchmark.py    frozen benchmark spec, leakage guard, scoring, leaderboard
  compare.py      paired bootstrap comparison and listening win rates
  listen.py       blind A/B listening page
  rating.py       listener rating page (1-5, hidden anchors), answer codes, listening status
  closeness.py    how close one clone is to the original, dimension by dimension
  utmos.py        UTMOS22 naturalness predictor (vendored, MIT)
  postprocess.py  trim to text
  longform.py     same-words re-voicing
  enhance/        ClearerVoice and Audacity 3.x denoisers
  recipe.py       ci/recipe.json: the pipeline the quality gate runs
  gate.py         pass/fail rules for a candidate against main
  studio/         the web app: upload, cleaning, previews, training jobs
  cli.py          the `voiceclone` command
ci/               hygiene check, gate script and runner notes
```

## Models and licences

| Component | Licence |
|---|---|
| `kenpath/svara-tts-voiceclone-beta` (3B, 19 Indic languages) | Apache 2.0 |
| `hubertsiuzdak/snac_24khz` codec | MIT |
| AudioSeal watermark (code and weights) | MIT |
| AI4Bharat IndicConformer 600M | MIT |
| SpeechBrain ECAPA-TDNN | Apache 2.0 |
| SpeechBrain ResNet-TDNN (`spkrec-resnet-voxceleb`, second speaker model in `closeness`) | Apache 2.0 |
| UTMOS22 strong learner (code vendored from SpeechMOS v1.2.0, weights converted from the official checkpoint) | MIT (code and weights) |
| ClearerVoice MossFormer2 | Apache 2.0 |
| `ctc-forced-aligner` code / MMS weights | MIT / CC-BY-NC: use for data prep and evaluation only |
| Audacity 3.x | GPL, used as an external tool |

## Contributing and the quality gate

Run the tests and the hygiene check as in [Getting started](#getting-started)
before opening a pull request. Every pull request runs `ci` (the same tests
and repository rules) on GitHub. Changes
that can affect the voice also go through `quality-gate`, which trains and
benchmarks the pull request's recipe with the real model on a self-hosted
runner and fails if any metric is worse than main. Experiments must also show
an improvement. See [CONTRIBUTING.md](CONTRIBUTING.md) for the ground rules and
[ci/README.md](ci/README.md) for the runner.

## Licence

Copyright 2026 Svayatta Forge. The code, documentation and Docker images are
released under the [Apache License 2.0](LICENSE), commercial use included.
See [NOTICE](NOTICE).

**No audio is licensed.** The recordings used to develop and test this
project, their transcripts, and any weights or adapters trained on them are
not part of the release, and no licence is granted for them: neither Svayatta
Forge nor the contributors hold the right to distribute them. You need the
right to use any voice and recording you process yourself.

The models the code downloads have their own licences (see
[Models and licences](#models-and-licences)). In particular, the MMS
alignment weights are CC-BY-NC, so commercial use of the full pipeline needs
a different aligner.
