"""The choices the studio offers for cleaning, the reference clip, generation
and training. This one schema drives both the page (served from /api/config)
and the server's validation, so the two cannot drift apart.

Each field: key, label, type (bool | number | int | choice), default, and for
numbers min/max/step; `needs` names another bool field of the same group that
must be on for this one to matter; `requires` names a server capability.
"""

ENHANCE = "enhance"   # capability: an environment with clearvoice (MossFormer2) was configured

SCHEMA = {
    "clean": {
        "label": "Noise cleaning",
        "fields": [
            {"key": "declip", "type": "bool", "default": True, "label": "Repair clipped peaks",
             "help": "For a recording that was too loud and distorts."},
            {"key": "highpass", "type": "bool", "default": True, "label": "Cut rumble and hum",
             "help": "Removes low noise, such as handling sounds and mains hum."},
            {"key": "highpass_hz", "type": "number", "default": 60, "min": 20, "max": 200, "step": 5,
             "label": "Cut-off (Hz)", "needs": "highpass"},
            {"key": "denoise", "type": "bool", "default": True, "label": "Reduce steady hiss",
             "help": "Gentle by default. A strong setting dulls the voice."},
            {"key": "denoise_db", "type": "number", "default": 8, "min": 1, "max": 30, "step": 1,
             "label": "Reduction (dB)", "needs": "denoise"},
            {"key": "noise_floor_db", "type": "number", "default": -42, "min": -80, "max": -20, "step": 1,
             "label": "Noise floor (dB)", "needs": "denoise"},
            {"key": "enhance", "type": "bool", "default": False, "requires": ENHANCE,
             "label": "AI speech enhancement",
             "help": "Removes noise and some echo with the MossFormer2 model. Slower."},
            {"key": "enhance_wet", "type": "number", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
             "requires": ENHANCE, "label": "Enhanced share", "needs": "enhance",
             "help": "1 = fully enhanced; 0.9 keeps a little of the original warmth."},
            {"key": "super_res", "type": "bool", "default": False, "requires": ENHANCE, "needs": "enhance",
             "label": "Restore high frequencies (super-resolution)",
             "help": "Off by default: on old lecture tapes it moved the formants that carry the accent."},
            {"key": "normalize", "type": "bool", "default": True, "label": "Even out the level",
             "help": "Sets the volume to a standard level."},
        ],
    },
    "reference": {
        "label": "Clip to copy the voice from",
        "fields": [
            {"key": "mode", "type": "choice", "default": "auto", "label": "Copy the voice from",
             "choices": [["auto", "the clearest 3-10 s stretch of the sample"],
                         ["whole", "the whole sample (first 15 s)"]]},
        ],
    },
    "generate": {
        "label": "How the voice speaks",
        "fields": [
            {"key": "trained", "type": "bool", "default": True, "label": "Use the trained voice",
             "help": "Available after training in step 4. Off: the voice from the short sample."},
            {"key": "takes", "type": "int", "default": 1, "min": 1, "max": 5, "label": "Versions for each text",
             "help": "Each version sounds a little different. Keep the one that sounds most like the speaker."},
            {"key": "seed", "type": "int", "default": 0, "min": 0, "max": 9999, "label": "Variation number (seed)",
             "help": "A different number gives a different version of the same text."},
            {"key": "temperature", "type": "number", "default": 0.6, "min": 0.1, "max": 1.5, "step": 0.05,
             "label": "Temperature", "help": "Lower is steadier, higher is livelier and less predictable."},
            {"key": "top_p", "type": "number", "default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01, "label": "Top-p"},
            {"key": "repetition_penalty", "type": "number", "default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05,
             "label": "Repetition penalty"},
            {"key": "max_sec", "type": "number", "default": 20, "min": 5, "max": 60, "step": 1,
             "label": "Longest sentence (s)"},
        ],
    },
    "train": {
        "label": "Training",
        "fields": [
            {"key": "enhance", "type": "bool", "default": False, "requires": ENHANCE,
             "label": "AI speech enhancement on the recordings",
             "help": "On top of the voice's cleaning options, which also apply to the recordings."},
            {"key": "min_sim", "type": "number", "default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05,
             "label": "Speaker match threshold",
             "help": "Chunks less like the sample's voice than this are dropped (other speakers score near 0)."},
            {"key": "pitch_gate", "type": "bool", "default": False, "label": "Pitch gate",
             "help": "Also drop chunks whose median pitch is outside the range: chanting, singing, other voices."},
            {"key": "min_f0", "type": "number", "default": 80, "min": 50, "max": 400, "step": 5,
             "label": "Lowest pitch (Hz)", "needs": "pitch_gate"},
            {"key": "max_f0", "type": "number", "default": 250, "min": 60, "max": 500, "step": 5,
             "label": "Highest pitch (Hz)", "needs": "pitch_gate"},
            {"key": "epochs", "type": "int", "default": 3, "min": 1, "max": 10, "label": "Epochs"},
            {"key": "lr", "type": "number", "default": 2e-4, "min": 1e-5, "max": 1e-3, "step": 1e-5,
             "label": "Learning rate"},
            {"key": "rank", "type": "choice", "default": "16", "label": "LoRA rank",
             "choices": [["8", "8"], ["16", "16"], ["32", "32"], ["64", "64"]],
             "help": "Higher ranks need more training data."},
            {"key": "holdout", "type": "int", "default": 6, "min": 0, "max": 50, "label": "Clips held out of training"},
            {"key": "seed", "type": "int", "default": 0, "min": 0, "max": 9999, "label": "Seed"},
        ],
    },
}


class OptionError(ValueError):
    pass


def schema(capabilities: set[str]) -> dict:
    """The schema with each field marked available or not on this server."""
    return {g: {**s, "fields": [{**f, "available": f.get("requires") in (None, *capabilities)} for f in s["fields"]]}
            for g, s in SCHEMA.items()}


def defaults(group: str) -> dict:
    return {f["key"]: f["default"] for f in SCHEMA[group]["fields"]}


def parse(group: str, data: dict | None, capabilities: set[str] = frozenset()) -> dict:
    """Validated options: defaults filled in, unknown keys and out-of-range values refused,
    options this server cannot run switched off."""
    data = dict(data or {})
    fields = {f["key"]: f for f in SCHEMA[group]["fields"]}
    unknown = set(data) - set(fields)
    if unknown:
        raise OptionError(f"unknown {group} options: {', '.join(sorted(unknown))}")
    out = {}
    for k, f in fields.items():
        v = data.get(k, f["default"])
        if f["type"] == "bool":
            if not isinstance(v, bool):
                raise OptionError(f"{f['label']}: on or off")
        elif f["type"] in ("number", "int"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (f["type"] == "int" and v != int(v)):
                raise OptionError(f"{f['label']}: a {'whole ' if f['type'] == 'int' else ''}number")
            if not f["min"] <= v <= f["max"]:
                raise OptionError(f"{f['label']}: between {f['min']} and {f['max']}")
            v = int(v) if f["type"] == "int" else float(v)
        elif f["type"] == "choice":
            if v not in [c[0] for c in f["choices"]]:
                raise OptionError(f"{f['label']}: one of {', '.join(c[0] for c in f['choices'])}")
        if f.get("requires") and f["requires"] not in capabilities:
            v = f["default"]
        out[k] = v
    if group == "train" and out["pitch_gate"] and out["min_f0"] >= out["max_f0"]:
        raise OptionError("Pitch gate: the lowest pitch must be below the highest")
    return out
