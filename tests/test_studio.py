"""The studio's API and job flow, with a stand-in engine (no models, no ffmpeg).

What matters to a user: a sample is cleaned before anything is cloned from
it; samples outside 5-45 s are refused with a reason; the clone is
conditioned on words that match its reference clip; training only starts
with recordings; once trained, previews use the trained adapter; a server
restart never leaves a job looking busy forever.
"""
import io
import json
import shutil

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from voiceclone.studio.app import create_app  # noqa: E402
from voiceclone.studio.jobs import Studio  # noqa: E402

SR = 24000


def speech_like(sec, sr=SR, seed=0):
    """Bursts of tone separated by pauses, so VAD finds 3-10 s chunks."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * t / 7.0) > -0.6)
    return (x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def wav_bytes(sec):
    buf = io.BytesIO()
    sf.write(buf, speech_like(sec), SR, format="WAV")
    return buf.getvalue()


class FakeEngine:
    min_train_minutes = 8.0

    capabilities = set()

    def __init__(self):
        self.cleaned, self.transcribed, self.synth_calls, self.trained = [], [], [], []
        self.marked = []   # lengths of the joins that got a mark

    def clean(self, src, dst, opts=None):
        self.cleaned.append((src, opts))
        shutil.copy(src, dst)
        return dst

    def transcribe(self, wav, lang):
        self.transcribed.append((wav, lang))
        return "ನಮಸ್ಕಾರ ರೀ"

    def synthesize(self, text, ref_wav, ref_text, lora=None, seed=0, params=None):
        self.synth_calls.append({"text": text, "ref_text": ref_text, "lora": lora, "seed": seed, "params": params})
        return np.zeros(SR, np.float32), 84

    def watermark(self, wav):
        self.marked.append(len(wav))
        return wav

    def train(self, sample_wav, recordings, work, lang, log=print, clean_opts=None, opts=None):
        log("usable speech: 12.5 min ({})")
        log("training the LoRA adapter")
        self.trained.append({"recordings": list(recordings), "clean_opts": clean_opts, "opts": opts})
        (work / "adapter").mkdir(parents=True)
        return work / "adapter"


@pytest.fixture
def env(tmp_path):
    engine = FakeEngine()
    studio = Studio(tmp_path, engine, start=False)
    client = TestClient(create_app(studio))

    def drain():
        while not studio.q.empty():
            fn, *args = studio.q.get()
            studio.run_job(fn, *args)
    return client, studio, engine, drain


def create(client, sec=12, name="Narrator", **form):
    return client.post("/api/voices", data={"name": name, "lang": "kn", **form},
                       files={"file": ("sample.wav", wav_bytes(sec), "audio/wav")})


def test_sample_is_cleaned_then_transcribed_then_ready(env):
    client, _, engine, drain = env
    r = create(client)
    assert r.status_code == 201
    vid = r.json()["id"]
    assert r.json()["status"] == "queued"
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert v["status"] == "ready", v["error"]
    assert len(engine.cleaned) == 1
    # the reference clip (not the whole sample) is what gets transcribed and conditioned on
    assert engine.transcribed[0][0].name == "reference.wav"
    assert v["ref_text"] == "ನಮಸ್ಕಾರ ರೀ" and v["ref_text_by"] == "asr"
    for kind in ("original", "clean", "reference"):
        assert client.get(f"/api/voices/{vid}/audio/{kind}").status_code == 200


def test_transcript_sent_at_creation_does_not_replace_the_asr_of_the_reference_clip(env):
    client, _, engine, drain = env
    vid = create(client, ref_text="ಇವತ್ತು ಸಂಜೆ").json()["id"]   # an old client may still send it
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert v["ref_text"] == "ನಮಸ್ಕಾರ ರೀ" and v["ref_text_by"] == "asr"
    assert len(engine.transcribed) == 1


@pytest.mark.parametrize("sec", [2, 60])
def test_sample_length_outside_5_to_45_s_fails_with_reason(env, sec):
    client, _, _, drain = env
    vid = create(client, sec=sec).json()["id"]
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert v["status"] == "failed"
    assert "5 to 45 s" in v["error"]


def test_upload_checks(env):
    client, studio, _, _ = env
    assert create(client, name="  ").status_code == 400
    r = client.post("/api/voices", data={"name": "x"}, files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/voices", data={"name": "x", "lang": "xx"}, files={"file": ("a.wav", wav_bytes(6), "audio/wav")})
    assert r.status_code == 400
    assert studio.store.list() == []     # refused uploads leave nothing behind


def test_oversize_upload_is_refused_and_removed(env, monkeypatch):
    client, studio, _, _ = env
    monkeypatch.setattr("voiceclone.studio.app.SAMPLE_MAX_MB", 0)
    assert create(client).status_code == 413
    assert studio.store.list() == []


def test_preview_uses_reference_words_and_is_served(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    assert client.post(f"/api/voices/{vid}/previews", json={"text": "ನಮಸ್ಕಾರ"}).status_code == 409   # not ready yet
    drain()
    p = client.post(f"/api/voices/{vid}/previews", json={"text": "ಮಕ್ಕಳಿಗೆ ಚಲೋ ವಿದ್ಯೆ ಕೊಡ್ರಿ", "options": {"seed": 2}}).json()[0]
    drain()
    call = engine.synth_calls[-1]
    assert {k: call[k] for k in ("text", "ref_text", "lora", "seed")} == \
        {"text": "ಮಕ್ಕಳಿಗೆ ಚಲೋ ವಿದ್ಯೆ ಕೊಡ್ರಿ", "ref_text": "ನಮಸ್ಕಾರ ರೀ", "lora": None, "seed": 2}
    v = client.get(f"/api/voices/{vid}").json()
    assert v["previews"][0]["status"] == "done"
    assert client.get(f"/api/voices/{vid}/audio/preview", params={"id": p["id"]}).status_code == 200


def test_corrected_reference_words_are_used(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    client.patch(f"/api/voices/{vid}", json={"ref_text": "ನಮಸ್ಕಾರ ರೀ ಎಲ್ಲಾರೂ"})
    client.post(f"/api/voices/{vid}/previews", json={"text": "ನಮಸ್ಕಾರ"})
    drain()
    assert engine.synth_calls[-1]["ref_text"] == "ನಮಸ್ಕಾರ ರೀ ಎಲ್ಲಾರೂ"


def test_preview_of_several_sentences_is_one_take_per_sentence_joined_by_pauses(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    pid = client.post(f"/api/voices/{vid}/previews", json={"text": "ಒಂದು. ಎರಡು? Pi is 3.5! ಮೂರು।"}).json()[0]["id"]
    drain()
    calls = engine.synth_calls
    assert [c["text"] for c in calls] == ["ಒಂದು.", "ಎರಡು?", "Pi is 3.5!", "ಮೂರು।"]
    assert len({c["seed"] for c in calls}) == 1   # the take keeps its seed
    wav, _ = sf.read(io.BytesIO(client.get(f"/api/voices/{vid}/audio/preview", params={"id": pid}).content))
    assert len(wav) == 4 * SR + 3 * int(0.25 * SR)
    assert engine.marked == [len(wav)]   # the join gets one mark of its own (see test_watermark.py)


def test_preview_of_one_sentence_has_no_pause(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    client.post(f"/api/voices/{vid}/previews", json={"text": "ನಮಸ್ಕಾರ ರೀ ಎಲ್ಲಾರೂ"})
    drain()
    assert [c["text"] for c in engine.synth_calls] == ["ನಮಸ್ಕಾರ ರೀ ಎಲ್ಲಾರೂ"]
    assert client.get(f"/api/voices/{vid}").json()["previews"][0]["seconds"] == 1.0
    assert engine.marked == []   # one take already carries its mark


def test_training_needs_recordings_then_previews_use_the_adapter(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    assert client.post(f"/api/voices/{vid}/train").status_code == 409
    r = client.post(f"/api/voices/{vid}/recordings",
                    files=[("files", ("talk one.wav", wav_bytes(20), "audio/wav")), ("files", ("b.mp3", b"ID3fake", "audio/mpeg"))])
    assert [x["file"] for x in r.json()["recordings"]] == ["000_talk_one.wav", "001_b.mp3"]
    assert client.post(f"/api/voices/{vid}/train").status_code == 202
    assert client.post(f"/api/voices/{vid}/train").status_code == 409   # already queued
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert v["train"]["status"] == "trained" and v["train"]["speech_min"] == 12.5
    assert [p.name for p in engine.trained[0]["recordings"]] == ["000_talk_one.wav", "001_b.mp3"]
    client.post(f"/api/voices/{vid}/previews", json={"text": "ನಮಸ್ಕಾರ"})
    drain()
    assert engine.synth_calls[-1]["lora"] == v["train"]["adapter"]


def test_engine_failure_is_shown_not_swallowed(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()

    def boom(*a, **k):
        raise RuntimeError("only 3.1 min of usable speech")
    engine.train = boom
    client.post(f"/api/voices/{vid}/recordings", files=[("files", ("a.wav", wav_bytes(6), "audio/wav"))])
    client.post(f"/api/voices/{vid}/train")
    drain()
    t = client.get(f"/api/voices/{vid}").json()["train"]
    assert t["status"] == "failed" and "3.1 min" in t["error"]


def test_restart_marks_interrupted_jobs_failed(tmp_path):
    engine = FakeEngine()
    s1 = Studio(tmp_path, engine, start=False)
    v = s1.store.create("x", "kn")
    s1.store.update(v["id"], lambda v: v.update(status="cleaning"))
    s2 = Studio(tmp_path, engine, start=False)
    assert s2.store.get(v["id"])["status"] == "failed"


def test_audio_route_does_not_escape_the_voice_folder(env):
    client, _, _, drain = env
    vid = create(client).json()["id"]
    drain()
    assert client.get(f"/api/voices/{vid}/audio/preview", params={"id": "../voice"}).status_code == 404
    assert client.get(f"/api/voices/{vid}/audio/voice.json").status_code == 404
    assert client.get("/api/voices/..%2F..%2Fetc/audio/clean").status_code == 404


def test_delete_and_page(env):
    client, studio, _, _ = env
    vid = create(client).json()["id"]
    assert client.get("/").status_code == 200 and "Voice studio" in client.get("/").text
    assert client.delete(f"/api/voices/{vid}").status_code == 204
    assert client.get(f"/api/voices/{vid}").status_code == 404


# ---------- options ----------
def test_options_default_and_are_recorded(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert v["options"]["clean"]["denoise"] is True and v["options"]["reference"]["mode"] == "auto"
    assert engine.cleaned[0][1] == v["options"]["clean"]


def test_chosen_cleaning_options_reach_the_cleaner(env):
    client, _, engine, drain = env
    o = {"clean": {"denoise": False, "highpass_hz": 80}, "reference": {"mode": "whole"}}
    vid = create(client, options=json.dumps(o)).json()["id"]
    drain()
    used = engine.cleaned[0][1]
    assert used["denoise"] is False and used["highpass_hz"] == 80 and used["declip"] is True
    assert client.get(f"/api/voices/{vid}").json()["status"] == "ready"


@pytest.mark.parametrize("o", [{"clean": {"highpass_hz": 5}},          # below the allowed range
                               {"clean": {"denoise": "yes"}},          # not a bool
                               {"clean": {"loudness": True}},          # unknown option
                               {"reference": {"mode": "middle"}}])
def test_bad_options_are_refused_and_leave_nothing(env, o):
    client, studio, _, _ = env
    r = create(client, options=json.dumps(o))
    assert r.status_code == 400
    assert studio.store.list() == []


def test_enhancement_is_off_when_the_server_cannot_run_it(env):
    # the page greys it out, and a direct request cannot switch it on either
    client, _, engine, drain = env
    create(client, options=json.dumps({"clean": {"enhance": True}}))
    drain()
    assert engine.cleaned[0][1]["enhance"] is False
    cfg = client.get("/api/config").json()["options"]["clean"]["fields"]
    assert next(f for f in cfg if f["key"] == "enhance")["available"] is False


def test_takes_seeds_and_generation_settings(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    ps = client.post(f"/api/voices/{vid}/previews",
                     json={"text": "ನಮಸ್ಕಾರ", "options": {"takes": 3, "seed": 10, "temperature": 0.8}}).json()
    assert [p["seed"] for p in ps] == [10, 11, 12]
    drain()
    assert [c["seed"] for c in engine.synth_calls] == [10, 11, 12]
    assert all(c["params"]["temperature"] == 0.8 and c["params"]["top_p"] == 0.95 for c in engine.synth_calls)
    assert client.post(f"/api/voices/{vid}/previews", json={"text": "x", "options": {"takes": 9}}).status_code == 400


def test_trained_clone_can_be_switched_off_for_comparison(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    client.post(f"/api/voices/{vid}/recordings", files=[("files", ("a.wav", wav_bytes(6), "audio/wav"))])
    client.post(f"/api/voices/{vid}/train")
    drain()
    client.post(f"/api/voices/{vid}/previews", json={"text": "ನಮಸ್ಕಾರ", "options": {"trained": False}})
    drain()
    assert engine.synth_calls[-1]["lora"] is None


def test_training_options_and_the_voices_cleaning_reach_training(env):
    client, _, engine, drain = env
    vid = create(client, options=json.dumps({"clean": {"denoise": False}})).json()["id"]
    drain()
    client.post(f"/api/voices/{vid}/recordings", files=[("files", ("a.wav", wav_bytes(6), "audio/wav"))])
    bad = {"options": {"pitch_gate": True, "min_f0": 200, "max_f0": 150}}
    assert client.post(f"/api/voices/{vid}/train", json=bad).status_code == 400
    r = client.post(f"/api/voices/{vid}/train", json={"options": {"epochs": 5, "rank": "32", "pitch_gate": True}})
    assert r.status_code == 202
    drain()
    t = engine.trained[0]
    assert t["opts"]["epochs"] == 5 and t["opts"]["rank"] == "32" and t["opts"]["pitch_gate"] is True
    assert t["clean_opts"]["denoise"] is False      # recordings are cleaned like the sample


def test_reclean_uses_new_options_and_retranscribes(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    client.patch(f"/api/voices/{vid}", json={"ref_text": "ಇವತ್ತು ಸಂಜೆ"})
    assert len(engine.transcribed) == 1
    r = client.post(f"/api/voices/{vid}/clean", json={"clean": {"declip": False}, "reference": {"mode": "whole"}})
    assert r.status_code == 202
    assert client.post(f"/api/voices/{vid}/clean", json={}).status_code == 409    # already queued
    drain()
    v = client.get(f"/api/voices/{vid}").json()
    assert engine.cleaned[-1][1]["declip"] is False and v["options"]["reference"]["mode"] == "whole"
    assert len(engine.transcribed) == 2 and v["ref_text_by"] == "asr"   # the reference may have changed


@pytest.mark.parametrize("text, sentences", [
    ("ಒಂದು।ಎರಡು।ಮೂರು", ["ಒಂದು।", "ಎರಡು।", "ಮೂರು"]),       # a danda with no space after it still ends a sentence
    ("ಒಂದು.ಎರಡು", ["ಒಂದು.", "ಎರಡು"]),
    ("೩.೫ and 3.5", ["೩.೫ and 3.5"]),                      # a number stays whole, in each script
    ('ನಿಲ್ಲು... "ಯಾಕೆ?" ಅಂದ.', ["ನಿಲ್ಲು...", '"ಯಾಕೆ?" ಅಂದ.']),   # a row of marks and a closing quote do not split
])
def test_preview_text_splits_at_sentence_ends_with_or_without_a_space(env, text, sentences):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    client.post(f"/api/voices/{vid}/previews", json={"text": text})
    drain()
    assert [c["text"] for c in engine.synth_calls] == sentences


def test_a_take_records_its_progress_for_the_page(env):
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    seen = []
    real = engine.synthesize

    def spy(*a, **kw):   # what the page reads while this sentence is generated
        p = client.get(f"/api/voices/{vid}").json()["previews"][0]
        seen.append((p["status"], p["stage"], p["sentence"], p["sentences"]))
        return real(*a, **kw)
    engine.synthesize = spy
    client.post(f"/api/voices/{vid}/previews", json={"text": "ಒಂದು. ಎರಡು."})
    drain()
    assert seen == [("running", "speaking", 0, 2), ("running", "speaking", 1, 2)]


def test_a_take_shows_the_model_load_and_then_the_first_sentence(env):
    # The load takes minutes. If "loading" stays during sentence 1, the page never shows "Speak sentence 1".
    client, _, engine, drain = env
    vid = create(client).json()["id"]
    drain()
    seen = []
    stage = lambda: client.get(f"/api/voices/{vid}").json()["previews"][0]["stage"]
    real = engine.synthesize
    engine.synth_ready = lambda lora: False
    engine.load_synth = lambda lora: seen.append(("load", stage()))
    engine.synthesize = lambda *a, **kw: seen.append(("speak", stage())) or real(*a, **kw)
    client.post(f"/api/voices/{vid}/previews", json={"text": "ಒಂದು."})
    drain()
    assert seen == [("load", "loading"), ("speak", "speaking")]


def test_page_and_script_are_revalidated_on_each_load(env):
    client = env[0]
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/config").headers
