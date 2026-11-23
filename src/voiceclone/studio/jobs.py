"""The studio's work queue: cleaning, previews and training run one at a time
on a single background thread, and every step is recorded in voice.json.
"""
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from voiceclone.audio import TTS_SR, write_wav
from voiceclone.studio.clean import SAMPLE_MAX_SEC, SAMPLE_MIN_SEC, duration, pick_reference
from voiceclone.studio.options import defaults
from voiceclone.studio.store import VoiceDeleted, VoiceStore
from voiceclone.text import split_sentences

LOG_LINES = 40
SENTENCE_GAP_SEC = 0.25   # the same pause as longform.py


class Studio:
    def __init__(self, root, engine, start: bool = True):
        self.store = VoiceStore(root)
        self.engine = engine
        self.q: queue.Queue = queue.Queue()
        self.store.fail_interrupted()
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    # ---------- requests (return at once; the work is queued) ----------
    def add_sample(self, vid: str, original: Path, options: dict | None = None) -> None:
        """`options`: validated {"clean": ..., "reference": ...} (options.py); defaults when left out."""
        def mark(v):
            v["status"], v["error"], v["original"] = "queued", None, original.name
            v["options"] = options or {"clean": defaults("clean"), "reference": defaults("reference")}
        self.store.update(vid, mark)
        self.q.put((self._clean, vid))

    def reclean(self, vid: str, options: dict) -> None:
        """Clean the sample again with other options. The reference clip may change, so
        its words are transcribed again."""
        v = self.store.get(vid)
        if v["status"] in {"queued", "cleaning", "transcribing"}:
            raise ValueError("the sample is already being cleaned")
        if v["train"]["status"] in {"queued", "preparing", "training"}:
            raise ValueError("wait for training to finish")

        def mark(v):
            v.update(status="queued", error=None, options=options, ref_text=None, ref_text_by=None)
        self.store.update(vid, mark)
        self.q.put((self._clean, vid))

    def preview(self, vid: str, text: str, params: dict | None = None) -> list[dict]:
        """One preview per take, seeds params["seed"], +1, ... `params`: generation options."""
        params = params or defaults("generate")
        v = self.store.get(vid)
        if v["status"] != "ready":
            raise ValueError("the voice is not ready yet")
        trained = params["trained"] and v["train"]["status"] == "trained"
        gen = {k: params[k] for k in ("temperature", "top_p", "repetition_penalty", "max_sec")}
        ps = [{"id": uuid.uuid4().hex[:8], "text": text, "seed": params["seed"] + i, "status": "queued",
               "trained": trained, "params": gen, "created": time.time()} for i in range(params["takes"])]
        self.store.update(vid, lambda v: v.update(previews=ps + v["previews"]))   # newest first, takes in order
        for p in ps:
            self.q.put((self._preview, vid, p["id"]))
        return ps

    def add_recording(self, vid: str, path: Path) -> None:
        self.store.update(vid, lambda v: v["recordings"].append({"file": path.name, "sec": duration_or_none(path)}))

    def train(self, vid: str, options: dict | None = None) -> None:
        v = self.store.get(vid)
        if v["status"] != "ready":
            raise ValueError("clean the voice sample first")
        if not v["recordings"]:
            raise ValueError("add the speaker's longer recordings first")
        if v["train"]["status"] in {"queued", "preparing", "training"}:
            raise ValueError("training is already running")

        def mark(v):
            v["train"].update(status="queued", error=None, log=[], options=options or defaults("train"))
        self.store.update(vid, mark)
        self.q.put((self._train, vid))

    # ---------- the worker ----------
    def _run(self):
        while True:
            fn, *args = self.q.get()
            self.run_job(fn, *args)

    def run_job(self, fn, *args):
        try:
            fn(*args)
        except VoiceDeleted:
            pass   # the voice was deleted while queued
        except Exception as e:   # noqa: BLE001 - every failure is shown to the user in voice.json
            traceback.print_exc()
            self._fail(fn, args, e)

    def _fail(self, fn, args, e):
        msg = str(e) or type(e).__name__
        vid = args[0]

        def mark(v):
            if fn == self._clean:
                v["status"], v["error"] = "failed", msg
            elif fn == self._train:
                v["train"]["status"], v["train"]["error"] = "failed", msg
            elif fn == self._preview:
                for p in v["previews"]:
                    if p["id"] == args[1]:
                        p["status"], p["error"] = "failed", msg
        try:
            self.store.update(vid, mark)
        except KeyError:
            pass

    def _set(self, vid, **kw):
        self.store.update(vid, lambda v: v.update(kw))

    def _clean(self, vid):
        d = self.store.dir(vid)
        v = self.store.get(vid)
        self._set(vid, status="cleaning")
        opts = v.get("options") or {"clean": defaults("clean"), "reference": defaults("reference")}
        clean = self.engine.clean(d / v["original"], d / "clean.wav", opts["clean"])
        sec = duration(clean)
        self._set(vid, sample_sec=round(sec, 1))
        if not SAMPLE_MIN_SEC <= sec <= SAMPLE_MAX_SEC:
            raise ValueError(f"the sample is {sec:.0f} s long; use {SAMPLE_MIN_SEC:.0f} to {SAMPLE_MAX_SEC:.0f} s "
                             "(longer recordings go under training)")
        pick_reference(clean, d / "reference.wav", d / "work" / "ref", opts["reference"]["mode"])
        self._set(vid, status="transcribing")   # the words of the reference clip, not of the whole sample
        text = self.engine.transcribe(d / "reference.wav", v["lang"])
        self._set(vid, ref_text=text, ref_text_by="asr")
        self._set(vid, status="ready")

    def _preview(self, vid, pid):
        d = self.store.dir(vid)
        v = self.store.get(vid)
        p = next(p for p in v["previews"] if p["id"] == pid)
        lora = v["train"]["adapter"] if p["trained"] else None
        sentences = split_sentences(p["text"]) or [p["text"]]   # ponytail: one sentence longer than max_sec is still cut
        # Progress for the page: `stage` is loading, speaking or joining; `sentence` counts the finished sentences.
        # The model loads inside the first `synthesize` call, so "loading" covers the first sentence too.
        loaded = getattr(self.engine, "synth_ready", lambda lora: True)(lora)
        self._set_preview(vid, pid, status="running", stage="speaking" if loaded else "loading",
                          sentence=0, sentences=len(sentences))
        t0 = time.time()
        gap = np.zeros(int(SENTENCE_GAP_SEC * TTS_SR), np.float32)
        parts = []
        for i, sentence in enumerate(sentences):
            if i:
                self._set_preview(vid, pid, stage="speaking", sentence=i)
            wav, _ = self.engine.synthesize(sentence, d / "reference.wav", v["ref_text"], lora, p["seed"], p.get("params"))
            parts += [gap, wav]
        wav = np.concatenate(parts[1:])
        if len(parts) > 2:
            self._set_preview(vid, pid, stage="joining", sentence=len(sentences))
            # The pauses carry no mark, and the detector scores the marked share of the file.
            # A join of short takes can thus fall below the threshold. One mark on the join prevents this.
            wav = self.engine.watermark(wav)
        (d / "previews").mkdir(exist_ok=True)
        write_wav(d / "previews" / f"{pid}.wav", wav, TTS_SR)
        self._set_preview(vid, pid, status="done", seconds=round(len(wav) / TTS_SR, 1), gen_time=round(time.time() - t0, 1))

    def _set_preview(self, vid, pid, **kw):
        def mark(v):
            for p in v["previews"]:
                if p["id"] == pid:
                    p.update(kw)
        self.store.update(vid, mark)

    def _train(self, vid):
        d = self.store.dir(vid)
        v = self.store.get(vid)

        def log(*parts):
            line = " ".join(str(x) for x in parts)

            def add(v):
                v["train"]["log"] = (v["train"]["log"] + [line])[-LOG_LINES:]
                if line.startswith("training the LoRA"):
                    v["train"]["status"] = "training"
                if line.startswith("usable speech:"):
                    v["train"]["speech_min"] = float(line.split()[2])
            self.store.update(vid, add)

        self.store.update(vid, lambda v: v["train"].update(status="preparing"))
        recs = [d / "recordings" / r["file"] for r in v["recordings"]]
        clean_opts = (v.get("options") or {}).get("clean")
        adapter = self.engine.train(d / "clean.wav", recs, d / "train", v["lang"], log=log,
                                    clean_opts=clean_opts, opts=v["train"].get("options"))
        self.store.update(vid, lambda v: v["train"].update(status="trained", adapter=str(adapter)))


def duration_or_none(path):
    try:
        return round(duration(path), 1)
    except Exception:   # noqa: BLE001 - compressed formats soundfile cannot read; ffmpeg handles them later
        return None
