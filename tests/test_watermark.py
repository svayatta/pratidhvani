"""Every take the software generates must carry the watermark, and the check
must not accuse real recordings. These need torch and AudioSeal; CI runs them
in their own job (`watermark` in .github/workflows/ci.yml)."""
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("audioseal")
import torch  # noqa: E402

from voiceclone.watermark import Watermark  # noqa: E402

SR = 24000


def voice_like(seconds=4.0, seed=0) -> np.ndarray:
    """A gliding harmonic tone with syllable-rate loudness, close enough to
    speech for the detector, without putting a recording in the repository."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 25)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    x = 0.3 * x / np.abs(x).max() + 0.003 * np.random.default_rng(seed).normal(size=len(t))
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def wm():
    return Watermark()


def test_unmarked_audio_is_not_claimed(wm):
    # a false "made with pratidhvani" would accuse a genuine recording
    assert not wm.detect(voice_like(), SR).marked


def test_marked_audio_is_recognised_as_ours(wm):
    d = wm.detect(wm.embed(voice_like(), SR), SR)
    assert d.ours, d


def test_mark_is_quiet(wm):
    # the mark must not change how the clone sounds; AudioSeal sits near 26 dB below the voice
    x = voice_like()
    y = wm.embed(x, SR)
    assert 10 * np.log10((x ** 2).sum() / ((y - x) ** 2).sum()) > 20


def test_mark_survives_a_cropped_quieter_excerpt(wm):
    y = wm.embed(voice_like(6.0), SR)
    assert wm.detect(0.4 * y[SR: 4 * SR], SR).ours


def test_a_long_unmarked_recording_is_not_claimed(wm):
    # the detector looks for the mark in every stretch of the file, so a long
    # genuine recording gives it many chances to be wrong: none may count
    speech = [x for i in range(12) for x in (voice_like(4.0, seed=i), np.zeros(SR, np.float32))]
    assert wm.detect(np.concatenate(speech), SR).probability < 0.1


def test_a_clone_cut_into_a_genuine_recording_is_found(wm):
    # three cloned seconds inside a minute of real speech are 5% of the file:
    # judged by the share of the whole file, the splice would pass as genuine
    spliced = np.concatenate([voice_like(30.0, seed=1), wm.embed(voice_like(3.0), SR), voice_like(27.0, seed=2)])
    d = wm.detect(spliced, SR)
    assert d.ours, d


@pytest.mark.parametrize("take_sec, slot_sec", [(3.0, 3.5), (2.0, 10.0), (3.0, 20.0), (0.5, 10.0)],
                         ids=["in-step", "sparse", "very-sparse", "short-and-sparse"])
def test_long_form_files_are_marked(wm, tmp_path, take_sec, slot_sec):
    """`long-form` places each marked take at its chunk's original start in
    a buffer of silence (cloned.wav), and interleaves each unmarked original
    line with its clone (ab.wav). A speaker who talks for a few seconds now
    and then leaves cloned.wav mostly silent, and ab.wav is half genuine
    recording by design. Both hold cloned speech, so both must be recognised
    as ours; original.wav is the real recording and must not be."""
    import soundfile as sf

    from voiceclone.jsonl import write_jsonl
    from voiceclone.longform import long_form

    n = 4
    d = tmp_path / "chunks"
    d.mkdir()
    write_jsonl(d / "segments.jsonl", [{"file": f"c{i}.wav", "start": i * slot_sec,
                                        "end": i * slot_sec + min(take_sec, slot_sec)} for i in range(n)])
    write_jsonl(d / "transcripts.jsonl", [{"file": f"c{i}.wav", "text": f"ವಾಕ್ಯ {i}"} for i in range(n)])
    sf.write(tmp_path / "source.wav", voice_like(n * slot_sec + 1), SR)

    class Synth:   # stands in for Synthesizer: generate returns marked takes
        def reference(self, ref_wav):
            return None

        def generate(self, text, ref_tokens, ref_text, seed=0, p=None):
            return wm.embed(voice_like(take_sec, seed=seed), SR), {}

    out = long_form(Synth(), d, tmp_path / "source.wav", None, "", tmp_path / "out", log=lambda *a: None)
    for name in ("cloned.wav", "ab.wav"):
        wav, sr = sf.read(out / name, dtype="float32")   # the 16-bit files
        found = wm.detect(wav, sr)
        assert found.ours, (name, found)
    assert not wm.detect(sf.read(out / "original.wav", dtype="float32")[0], SR).marked


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("name,args", [
    # clones get shared in these forms; a mark that one of them strips is no use
    ("mp3", ["-b:a", "32k"]),
    ("m4a", ["-c:a", "aac", "-b:a", "32k"]),
    # a messaging-app voice note: mono 16 kHz Opus at 16 kbps, tuned for speech
    ("ogg", ["-c:a", "libopus", "-b:a", "16k", "-ac", "1", "-ar", "16000", "-application", "voip"]),
    # a phone call: narrowband 8 kHz audio
    ("wav", ["-ar", "8000"]),
], ids=["mp3-32k", "aac-32k", "voice-note-opus-16k", "phone-8khz"])
def test_mark_survives_how_clips_get_shared(wm, tmp_path, name, args):
    import soundfile as sf

    from voiceclone.audio import load16
    from voiceclone.watermark import SR as WM_SR

    sf.write(tmp_path / "a.wav", wm.embed(voice_like(), SR), SR, subtype="PCM_16")
    out = tmp_path / f"shared.{name}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp_path / "a.wav", *args, out], check=True)
    d = wm.detect(load16(out), WM_SR)
    assert d.ours, d


def test_every_generated_take_is_marked(wm, monkeypatch):
    """Synthesizer.generate is the one way audio comes out of the model: the
    studio, `synth`, `long-form` and the benchmark all call it. Whatever the
    codec decodes must leave it marked."""
    from voiceclone.svara.synth import Synthesizer
    from voiceclone.svara.tokens import EOS

    class Model:
        def generate(self, ids, **k):
            return torch.cat([ids, torch.tensor([[11] * 70 + [EOS]])], dim=1)

    class Codec:
        def decode(self, tokens):
            return voice_like()

    s = Synthesizer.__new__(Synthesizer)
    s.device, s.tok, s.model, s.codec, s.watermark, s._torch = "cpu", None, Model(), Codec(), wm, torch
    monkeypatch.setattr("voiceclone.svara.synth.build_prompt", lambda *a, **k: [1, 2, 3])
    wav, _ = s.generate("ನಮಸ್ಕಾರ", None, None)
    assert wm.detect(wav, SR).ours


@pytest.mark.parametrize("n, sentence_sec", [(4, 3.0), (6, 0.5), (10, 0.5), (8, 0.3)])
def test_studio_preview_of_several_sentences_is_marked(wm, tmp_path, n, sentence_sec):
    """The file the user plays is the join of the takes with pauses between
    them. The detector reports the marked share of the most-marked two
    seconds. With only the marks of the takes, eight 0.3 s sentences score
    0.63 against a threshold of 0.5, so the studio marks the join. The join
    must stay ours and the two marks together must stay quiet."""
    import soundfile as sf

    from voiceclone.studio.jobs import SENTENCE_GAP_SEC, Studio

    takes = [voice_like(sentence_sec, seed=i) for i in range(n)]

    class Engine:
        def __init__(self):
            self.todo = list(takes)

        def synthesize(self, text, ref_wav, ref_text, lora=None, seed=0, params=None):
            return wm.embed(self.todo.pop(0), SR), 0   # as Synthesizer.generate returns it

        def watermark(self, wav):
            return wm.embed(wav, SR)                   # as studio.engine.Engine.watermark does

    studio = Studio(tmp_path, Engine(), start=False)
    vid = studio.store.create("v", "kn")["id"]
    studio.store.update(vid, lambda v: v.update(status="ready", ref_text="x"))
    pid = studio.preview(vid, " ".join(f"ವಾಕ್ಯ {i}." for i in range(n)))[0]["id"]
    while not studio.q.empty():
        fn, *args = studio.q.get()
        fn(*args)
    wav, sr = sf.read(studio.store.dir(vid) / "previews" / f"{pid}.wav", dtype="float32")   # the 16-bit file
    d = wm.detect(wav, sr)
    assert d.ours, d
    gap = np.zeros(int(SENTENCE_GAP_SEC * SR), np.float32)
    clean = np.concatenate([x for t in takes for x in (gap, t)][1:])
    assert len(wav) == len(clean)
    assert 10 * np.log10((clean ** 2).sum() / ((wav - clean) ** 2).sum()) > 20


# take / slot seconds for four takes. `long_form` places each take at its
# chunk's start and fills the rest with silence, and the detector scores the
# share of the whole file that is marked, so the sparse layouts are the hard ones.
LONG_FORM_LAYOUTS = {
    "in-step": (3.0, 3.5),
    "short-gaps": (1.0, 1.5),
    "takes-run-long": (4.0, 2.0),
    "sparse": (2.0, 10.0),        # a few seconds of speech every ten, as in an interview
    "very-sparse": (3.0, 20.0),
}
# Known to come back as not ours today, though the tag decodes intact: see
# https://github.com/svayatta/pratidhvani/issues/63. Strict, so each one fails
# loudly once the detector or long_form is fixed and the mark can be removed.
# ab.wav interleaves unmarked original lines with the clones; whether it should
# count as ours is still open in #63, and these assume it should.
NOT_YET_OURS = {
    ("cloned.wav", "sparse"), ("cloned.wav", "very-sparse"),
    ("ab.wav", "in-step"), ("ab.wav", "short-gaps"), ("ab.wav", "sparse"), ("ab.wav", "very-sparse"),
}


def _long_form_cases():
    for name in ("cloned.wav", "ab.wav"):
        for layout in LONG_FORM_LAYOUTS:
            marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason="#63: share-of-file threshold")] \
                if (name, layout) in NOT_YET_OURS else []
            yield pytest.param(name, layout, marks=marks, id=f"{name}-{layout}")


@pytest.fixture(scope="module")
def long_form_out(wm, tmp_path_factory):
    """Runs `long-form` once per layout with AudioSeal-marked takes and an
    unmarked source, and returns its output folder."""
    import soundfile as sf

    from voiceclone.jsonl import write_jsonl
    from voiceclone.longform import long_form

    class Synth:   # stands in for Synthesizer: generate returns marked takes
        def __init__(self, take_sec):
            self.take_sec = take_sec

        def reference(self, ref_wav):
            return None

        def generate(self, text, ref_tokens, ref_text, seed=0, p=None):
            return wm.embed(voice_like(self.take_sec, seed=seed), SR), {}

    done = {}

    def get(layout):
        if layout not in done:
            take_sec, slot_sec = LONG_FORM_LAYOUTS[layout]
            n = 4
            tmp = tmp_path_factory.mktemp(layout)
            d = tmp / "chunks"
            d.mkdir()
            write_jsonl(d / "segments.jsonl", [{"file": f"c{i}.wav", "start": i * slot_sec,
                                                "end": i * slot_sec + min(take_sec, slot_sec)} for i in range(n)])
            write_jsonl(d / "transcripts.jsonl", [{"file": f"c{i}.wav", "text": f"ವಾಕ್ಯ {i}"} for i in range(n)])
            sf.write(tmp / "source.wav", voice_like(n * slot_sec + 1), SR)   # unmarked, like a real recording
            done[layout] = long_form(Synth(take_sec), d, tmp / "source.wav", None, "", tmp / "out", log=lambda *a: None)
        return done[layout]

    return get


@pytest.mark.parametrize("name, layout", list(_long_form_cases()))
def test_long_form_output_is_marked(wm, long_form_out, name, layout):
    """`long-form` writes cloned.wav by copying each marked take into a
    buffer of silence at its chunk's start, and ab.wav by interleaving each
    original line with its clone. Both contain cloned speech, so both must be
    recognised as ours, however much silence or original audio surrounds the
    takes."""
    import soundfile as sf

    wav, sr = sf.read(long_form_out(layout) / name, dtype="float32")   # the 16-bit file
    d = wm.detect(wav, sr)
    assert d.ours, d


@pytest.mark.parametrize("layout", LONG_FORM_LAYOUTS)
def test_long_form_original_is_not_claimed(wm, long_form_out, layout):
    # the original next to the clone is a real recording and must not be claimed
    import soundfile as sf

    wav, sr = sf.read(long_form_out(layout) / "original.wav", dtype="float32")
    assert not wm.detect(wav, sr).marked
