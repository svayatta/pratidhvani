"""The models behind the studio, loaded on first use.

Only one job runs at a time (the 3B model and training do not fit in memory
together), so the engine frees the synthesizer before training and the
other way round.
"""
import gc
import shutil
from pathlib import Path

from voiceclone.studio.options import ENHANCE
from voiceclone.svara.synth import DEFAULT_MODEL, GenParams

MIN_TRAIN_MINUTES = 8.0   # the reference adapter trained on 23 min; 10-minute runs were data-starved


class NotEnoughSpeech(Exception):
    pass


class Engine:
    def __init__(self, model: str = DEFAULT_MODEL, ecapa_dir: str = "models/ecapa",
                 enhance_python: str | None = None, min_train_minutes: float = MIN_TRAIN_MINUTES):
        self.model, self.ecapa_dir = model, ecapa_dir
        self.enhance_python = enhance_python
        self.min_train_minutes = min_train_minutes
        # what this server can run beyond the basics (see options.py `requires`)
        self.capabilities = {ENHANCE} if enhance_python else set()
        self._asr = None
        self._synth = None   # (lora, Synthesizer)

    # ---------- cleaning ----------
    def clean(self, src, dst, opts: dict | None = None) -> Path:
        from voiceclone.studio.clean import clean_audio

        return clean_audio(src, dst, opts, self.enhance_python)

    # ---------- transcription ----------
    def asr(self):
        if self._asr is None:
            from voiceclone.asr import IndicConformer

            self._asr = IndicConformer()
        return self._asr

    def transcribe(self, wav, lang: str) -> str:
        import soundfile as sf

        x, sr = sf.read(str(wav), dtype="float32")
        return self.asr().transcribe(x, sr, lang)

    # ---------- synthesis ----------
    def synth_ready(self, lora: str | None = None) -> bool:
        """False when the next `synthesize` must load the model first, which takes minutes."""
        return self._synth is not None and self._synth[0] == lora

    def load_synth(self, lora: str | None = None) -> None:
        """Load the model for this adapter if it is not loaded. `synthesize` calls this too."""
        if not self.synth_ready(lora):
            self.release()
            from voiceclone.svara.synth import Synthesizer

            self._synth = (lora, Synthesizer(self.model, lora))

    def synthesize(self, text: str, ref_wav, ref_text: str, lora: str | None = None, seed: int = 0,
                   params: dict | None = None):
        """Returns (waveform at 24 kHz, number of generated tokens). `params`: the
        generation options (temperature, top_p, repetition_penalty, max_sec)."""
        self.load_synth(lora)
        synth = self._synth[1]
        p = GenParams(**{k: params[k] for k in ("temperature", "top_p", "repetition_penalty", "max_sec")}) \
            if params else GenParams()
        return synth.generate(text, synth.reference(ref_wav), ref_text, seed=seed, p=p)

    def watermark(self, wav):
        """Mark audio that the studio joins from takes of the loaded synthesizer."""
        from voiceclone.audio import TTS_SR

        return self._synth[1].watermark.embed(wav, TTS_SR)

    def release(self) -> None:
        self._synth = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---------- training ----------
    def train(self, sample_wav, recordings: list[Path], work, lang: str, log=print,
              clean_opts: dict | None = None, opts: dict | None = None) -> Path:
        """Recordings -> cleaned chunks -> verbatim transcripts -> speaker/alignment filter -> LoRA.

        The sample defines the speaker: chunks whose voiceprint is far from it
        (other people in the recording) are dropped by `build_clips`. The
        recordings are cleaned like the sample (`clean_opts`); `opts` are the
        training options (options.py), which may add enhancement.
        """
        from voiceclone.studio.options import defaults
        from voiceclone.align import Aligner
        from voiceclone.asr import transcribe_dir
        from voiceclone.audio import TTS_SR, load_wav
        from voiceclone.dataset import ClipFilter, build_clips
        from voiceclone.jsonl import read_jsonl
        from voiceclone.segment import vad_segment
        from voiceclone.speaker import SpeakerEncoder, centroid
        from voiceclone.train import TrainParams, finetune

        opts = opts or defaults("train")
        rec_clean = {**(clean_opts or defaults("clean")), "enhance": opts["enhance"]}
        self.release()
        work = Path(work)
        for stale in ("chunks", "clean", "clips"):   # old transcripts and SNAC tokens must not meet new audio
            shutil.rmtree(work / stale, ignore_errors=True)
        chunk_dirs = []
        for i, rec in enumerate(recordings):
            tag = f"R{i:02d}"
            log(f"cleaning {Path(rec).name}")
            clean = self.clean(rec, work / "clean" / f"{tag}.wav", rec_clean)
            d = work / "chunks" / tag
            log(vad_segment(load_wav(clean, TTS_SR), TTS_SR, d, prefix=tag))
            log(f"transcribing {Path(rec).name}")
            transcribe_dir(d, lang=lang, log=lambda *_: None, asr=self.asr())
            chunk_dirs.append(d)
        self._asr = None
        gc.collect()

        log("keeping chunks that are this speaker and whose words fit the audio")
        enc = SpeakerEncoder(self.ecapa_dir)
        gate = opts["pitch_gate"]
        f = ClipFilter(min_sim=opts["min_sim"], min_f0=opts["min_f0"] if gate else None,
                       max_f0=opts["max_f0"] if gate else None)
        summary = build_clips(chunk_dirs, work / "clips", enc, Aligner(), centroid([enc.embed_file(sample_wav)]), f, log=log)
        minutes = sum(c["duration"] for c in read_jsonl(work / "clips" / "metadata.jsonl")) / 60
        log(f"usable speech: {minutes:.1f} min ({summary})")
        if minutes < self.min_train_minutes:
            raise NotEnoughSpeech(f"only {minutes:.1f} min of usable speech; training needs at least "
                                  f"{self.min_train_minutes:.0f} min of this speaker alone")
        del enc
        gc.collect()

        log("training the LoRA adapter")
        p = TrainParams(epochs=opts["epochs"], lr=opts["lr"], rank=int(opts["rank"]), holdout=opts["holdout"],
                        seed=opts["seed"], save_every=0)
        finetune(work / "clips", work / "adapter", self.model, p, log=log)
        return work / "adapter"
