"""HTTP API and page for the studio (FastAPI). `voiceclone serve` runs it."""
import json
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from voiceclone.studio.jobs import Studio
from voiceclone.studio.options import OptionError, parse, schema

STATIC = Path(__file__).parent / "static"
SAMPLE_MAX_MB = 30
RECORDING_MAX_MB = 500
LANGUAGES = {"kn": "Kannada", "mr": "Marathi", "hi": "Hindi", "te": "Telugu",
             "ta": "Tamil", "ml": "Malayalam", "gu": "Gujarati", "bn": "Bengali"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4"}


class PreviewIn(BaseModel):
    text: str = Field(min_length=1, max_length=400)
    options: dict | None = None     # generation options (options.py)


class CleanIn(BaseModel):
    clean: dict | None = None
    reference: dict | None = None


class TrainIn(BaseModel):
    options: dict | None = None


class VoicePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    ref_text: str | None = Field(default=None, min_length=1, max_length=1000)


def _safe_name(filename: str | None, default: str) -> str:
    name = Path(filename or "").name
    stem = re.sub(r"[^\w.-]+", "_", Path(name).stem)[:60] or default
    ext = Path(name).suffix.lower()
    if ext not in AUDIO_EXT:
        raise HTTPException(415, f"unsupported file type {ext or '(none)'}; use one of {', '.join(sorted(AUDIO_EXT))}")
    return stem + ext


async def _save(upload: UploadFile, dst: Path, max_mb: int) -> Path:
    """Stream to disk, refusing files over the limit without reading them whole."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with open(dst, "wb") as f:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > max_mb << 20:
                f.close(); dst.unlink()
                raise HTTPException(413, f"{upload.filename} is over {max_mb} MB")
            f.write(chunk)
    if size == 0:
        dst.unlink()
        raise HTTPException(400, f"{upload.filename} is empty")
    return dst


def create_app(studio: Studio) -> FastAPI:
    app = FastAPI(title="Voice studio", docs_url=None, redoc_url=None)
    store = studio.store
    caps = set(getattr(studio.engine, "capabilities", set()))

    def opts(group, data):
        try:
            return parse(group, data, caps)
        except OptionError as e:
            raise HTTPException(400, str(e)) from None

    def voice(vid):
        try:
            return store.get(vid)
        except KeyError:
            raise HTTPException(404, "no such voice") from None

    @app.middleware("http")
    async def revalidate_page(request, call_next):
        """The page and its script change together. `no-cache` makes the browser ask if its copy is current
        (a 304 when it is), so a reload never mixes an old page with a new script."""
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config():
        return {"languages": LANGUAGES, "sample_max_mb": SAMPLE_MAX_MB, "recording_max_mb": RECORDING_MAX_MB,
                "min_train_minutes": getattr(studio.engine, "min_train_minutes", None), "options": schema(caps)}

    @app.get("/api/voices")
    def list_voices():
        return store.list()

    @app.post("/api/voices", status_code=201)
    async def create_voice(name: str = Form(...), lang: str = Form("kn"),
                           options: str = Form("{}"), file: UploadFile = File(...)):
        name = name.strip()
        if not name or len(name) > 80:
            raise HTTPException(400, "give the voice a name (up to 80 characters)")
        if lang not in LANGUAGES:
            raise HTTPException(400, f"language must be one of {sorted(LANGUAGES)}")
        try:
            o = json.loads(options or "{}")
            assert isinstance(o, dict)
        except (ValueError, AssertionError):
            raise HTTPException(400, "options must be a JSON object") from None
        chosen = {"clean": opts("clean", o.get("clean")), "reference": opts("reference", o.get("reference"))}
        fname = "original_" + _safe_name(file.filename, "sample")
        v = store.create(name, lang)
        try:
            await _save(file, store.dir(v["id"]) / fname, SAMPLE_MAX_MB)
        except HTTPException:
            store.delete(v["id"])
            raise
        studio.add_sample(v["id"], Path(fname), chosen)
        return store.get(v["id"])

    @app.get("/api/voices/{vid}")
    def get_voice(vid: str):
        return voice(vid)

    @app.patch("/api/voices/{vid}")
    def patch_voice(vid: str, body: VoicePatch):
        voice(vid)

        def edit(v):
            if body.name:
                v["name"] = body.name.strip()
            if body.ref_text:
                v["ref_text"], v["ref_text_by"] = body.ref_text.strip(), "user"
        return store.update(vid, edit)

    @app.delete("/api/voices/{vid}", status_code=204)
    def delete_voice(vid: str):
        voice(vid)
        store.delete(vid)

    @app.post("/api/voices/{vid}/clean", status_code=202)
    def reclean(vid: str, body: CleanIn):
        voice(vid)
        chosen = {"clean": opts("clean", body.clean), "reference": opts("reference", body.reference)}
        try:
            studio.reclean(vid, chosen)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return store.get(vid)

    @app.post("/api/voices/{vid}/previews", status_code=202)
    def add_preview(vid: str, body: PreviewIn):
        voice(vid)
        params = opts("generate", body.options)
        try:
            return studio.preview(vid, body.text.strip(), params)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/voices/{vid}/recordings", status_code=201)
    async def add_recordings(vid: str, files: list[UploadFile] = File(...)):
        voice(vid)
        d = store.dir(vid) / "recordings"
        n = len(list(d.glob("*"))) if d.exists() else 0
        for i, f in enumerate(files):
            path = await _save(f, d / f"{n + i:03d}_{_safe_name(f.filename, 'recording')}", RECORDING_MAX_MB)
            studio.add_recording(vid, path)
        return store.get(vid)

    @app.post("/api/voices/{vid}/train", status_code=202)
    def train(vid: str, body: TrainIn | None = None):
        voice(vid)
        chosen = opts("train", body.options if body else None)
        try:
            studio.train(vid, chosen)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return store.get(vid)

    @app.get("/api/voices/{vid}/audio/{kind}")
    def audio(vid: str, kind: str, id: str | None = None):
        v = voice(vid)
        d = store.dir(vid)
        if kind == "original":
            path = d / v.get("original", "")
        elif kind in ("clean", "reference"):
            path = d / f"{kind}.wav"
        elif kind == "preview" and id and re.fullmatch(r"[0-9a-f]{8}", id):
            path = d / "previews" / f"{id}.wav"
        else:
            raise HTTPException(404, "no such audio")
        if not path.is_file():
            raise HTTPException(404, "not ready")
        return FileResponse(path)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
