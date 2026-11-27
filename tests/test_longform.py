"""`long-form` re-voices a stretch of a recording so it can be heard next to
the original. Each take is placed at its chunk's original start, or right
after the previous take if that one ran long, and generated takes are cached
so an interrupted run resumes. A fake synthesizer stands in for Svara, so
these need no model."""
import numpy as np
import pytest
import soundfile as sf

from voiceclone.audio import TTS_SR
from voiceclone.jsonl import write_jsonl
from voiceclone.longform import LONGFORM_PARAMS, long_form
from voiceclone.svara.synth import GenParams

GAP = 0.25


class FakeSynth:
    """Returns a constant-level take of a chosen length per text, and records
    every call with all its arguments, so placement can be read back from
    cloned.wav and what reached the model can be checked."""

    REF_TOKENS = object()   # what `reference` returns; long_form must pass it on unchanged

    def __init__(self, takes: dict[str, tuple[float, float]]):
        self.takes = takes          # text -> (seconds, level)
        self.calls = []             # (text, ref_tokens, ref_text, seed, p) per generate call
        self.references = []        # ref_wav per reference call

    @property
    def texts(self):
        return [c[0] for c in self.calls]

    def reference(self, ref_wav):
        self.references.append(ref_wav)
        return self.REF_TOKENS

    def generate(self, text, ref_tokens, ref_text, seed=0, p=None):
        self.calls.append((text, ref_tokens, ref_text, seed, p))
        seconds, level = self.takes[text]
        return np.full(int(seconds * TTS_SR), level, dtype=np.float32), {}


def chunks(tmp_path, rows, level=0.0):
    """segments.jsonl and transcripts.jsonl as `prepare` and `transcribe`
    write them, plus a source recording as long as the last chunk that is
    `level` inside each chunk and silent between them."""
    d = tmp_path / "chunks"
    d.mkdir()
    write_jsonl(d / "segments.jsonl",
                [{"file": f"c{i}.wav", "start": s, "end": e} for i, (s, e, _) in enumerate(rows)])
    write_jsonl(d / "transcripts.jsonl", [{"file": f"c{i}.wav", "text": t} for i, (_, _, t) in enumerate(rows)])
    src = tmp_path / "source.wav"
    wav = np.zeros(int(rows[-1][1] * TTS_SR) + TTS_SR, np.float32)
    for s, e, _ in rows:
        wav[int(s * TTS_SR): int(e * TTS_SR)] = level
    sf.write(src, wav, TTS_SR)
    return d, src


def run(tmp_path, synth, chunks_dir, src, out="out", **kw):
    return long_form(synth, chunks_dir, src, tmp_path / "ref.wav", "ref", tmp_path / out, gap=GAP,
                     log=lambda *a: None, **kw)


def level_at(wav, t):
    return float(wav[int(t * TTS_SR)])


def test_a_take_that_runs_long_pushes_the_next_one_back(tmp_path):
    # "one" has a 1 s slot but its take is 2 s; "two" starts at 1.5 s in the original
    d, src = chunks(tmp_path, [(0.0, 1.0, "one"), (1.5, 2.5, "two"), (6.0, 7.0, "three")])
    synth = FakeSynth({"one": (2.0, 0.25), "two": (1.0, 0.5), "three": (0.5, 0.75)})
    out = run(tmp_path, synth, d, src)
    clone, sr = sf.read(out / "cloned.wav", dtype="float32")
    assert sr == TTS_SR
    tol = 1e-3   # cloned.wav is 16-bit
    # "one" is whole: "two" was not written over its tail
    assert level_at(clone, 1.9) == pytest.approx(0.25, abs=tol)
    # "two" starts at the cursor (end of "one" plus the gap), not at its original 1.5 s
    assert level_at(clone, 2.1) == pytest.approx(0.0, abs=tol)
    assert level_at(clone, 2.0 + GAP + 0.01) == pytest.approx(0.5, abs=tol)
    assert level_at(clone, 2.0 + GAP + 0.99) == pytest.approx(0.5, abs=tol)
    # once back in step, a take returns to its original start rather than the cursor
    assert level_at(clone, 5.9) == pytest.approx(0.0, abs=tol)
    assert level_at(clone, 6.01) == pytest.approx(0.75, abs=tol)


def test_takes_never_overlap(tmp_path):
    # every take runs long; each must still be whole in cloned.wav
    rows = [(i * 1.0, i * 1.0 + 0.8, f"line {i}") for i in range(5)]
    d, src = chunks(tmp_path, rows)
    synth = FakeSynth({t: (1.5, 0.1 * (i + 1)) for i, (_, _, t) in enumerate(rows)})
    clone = sf.read(run(tmp_path, synth, d, src) / "cloned.wav", dtype="float32")[0]
    start = 0.0
    for i in range(5):
        take = clone[int(start * TTS_SR) + 1: int((start + 1.5) * TTS_SR) - 1]
        assert np.allclose(take, 0.1 * (i + 1), atol=1e-3), i
        start += 1.5 + GAP


def test_a_cached_take_is_not_generated_again(tmp_path):
    d, src = chunks(tmp_path, [(0.0, 1.0, "one"), (2.0, 3.0, "two")])
    synth = FakeSynth({"one": (0.5, 0.25), "two": (0.5, 0.5)})
    out = run(tmp_path, synth, d, src)
    assert synth.texts == ["one", "two"]
    first = sf.read(out / "cloned.wav", dtype="float32")[0]

    synth.calls.clear()
    run(tmp_path, synth, d, src)
    assert synth.texts == []
    assert np.array_equal(sf.read(out / "cloned.wav", dtype="float32")[0], first)

    # an interrupted run resumes: only the missing take is generated
    (out / "chunks" / "001.wav").unlink()
    run(tmp_path, synth, d, src)
    assert synth.texts == ["two"]


def test_an_empty_transcript_is_skipped(tmp_path):
    d, src = chunks(tmp_path, [(0.0, 1.0, "one"), (1.5, 2.0, " "), (3.0, 4.0, "three")])
    synth = FakeSynth({"one": (0.5, 0.25), "three": (0.5, 0.5)})
    out = run(tmp_path, synth, d, src)
    assert synth.texts == ["one", "three"]
    assert (out / "script.tsv").read_text(encoding="utf-8").count("\n") == 3


def test_every_take_gets_the_reference_voice_its_own_seed_and_the_long_form_params(tmp_path):
    # a run that ignored the reference would produce the wrong voice with every placement test green
    d, src = chunks(tmp_path, [(0.0, 1.0, "one"), (2.0, 3.0, "two"), (4.0, 5.0, "three")])
    synth = FakeSynth({"one": (0.5, 0.25), "two": (0.5, 0.5), "three": (0.5, 0.75)})
    run(tmp_path, synth, d, src, seed=7)
    assert synth.references == [tmp_path / "ref.wav"]
    assert synth.calls == [("one", FakeSynth.REF_TOKENS, "ref", 7, LONGFORM_PARAMS),
                           ("two", FakeSynth.REF_TOKENS, "ref", 8, LONGFORM_PARAMS),
                           ("three", FakeSynth.REF_TOKENS, "ref", 9, LONGFORM_PARAMS)]
    assert LONGFORM_PARAMS == GenParams(min_sps=2.5)

    # params passed in reach the model in place of the default
    mine = GenParams(min_sps=1.0)
    synth.calls.clear()
    run(tmp_path, synth, d, src, out="out2", p=mine)
    assert [c[4] for c in synth.calls] == [mine] * 3


def test_only_chunks_that_end_within_minutes_are_revoiced(tmp_path):
    # minutes=0.05 is 3 s: "three" ends at 4.5 s and is left out of every file
    d, src = chunks(tmp_path, [(0.0, 1.0, "one"), (2.0, 2.9, "two"), (3.5, 4.5, "three")], level=0.5)
    synth = FakeSynth({"one": (0.5, 0.25), "two": (0.5, 0.5), "three": (0.5, 0.75)})
    out = run(tmp_path, synth, d, src, minutes=0.05)
    assert synth.texts == ["one", "two"]
    assert [line.split("\t")[2] for line in (out / "script.tsv").read_text(encoding="utf-8").splitlines()] == ["one", "two"]
    orig, sr = sf.read(out / "original.wav", dtype="float32")
    assert len(orig) == pytest.approx(2.9 * sr, abs=1)


def test_ab_alternates_each_original_line_with_its_clone(tmp_path):
    # ab.wav is: original line, 0.4 s silence, its clone, 1.0 s silence, for each line with a transcript
    rows = [(0.0, 1.0, "one"), (1.5, 2.0, " "), (3.0, 3.6, "three")]
    d, src = chunks(tmp_path, rows, level=0.9)
    synth = FakeSynth({"one": (0.5, 0.25), "three": (0.8, 0.5)})
    ab, sr = sf.read(run(tmp_path, synth, d, src) / "ab.wav", dtype="float32")
    expected = []
    for s, e, text in rows:
        if text.strip():
            seconds, level = synth.takes[text]
            expected += [np.full(int(e * sr) - int(s * sr), 0.9), np.zeros(int(0.4 * sr)),
                         np.full(int(seconds * sr), level), np.zeros(int(1.0 * sr))]
    expected = np.concatenate(expected).astype(np.float32)
    assert len(ab) == len(expected)
    assert np.allclose(ab, expected, atol=1e-3)
