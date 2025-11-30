import os
import sys
import tempfile
from glob import glob
from structlog import get_logger
import bark

log = get_logger(__name__)

# -----------------------------------------------------------
#  AUTO-CONFIG WITHOUT config.toml
# -----------------------------------------------------------

BASE_DIR = os.path.abspath(os.getcwd())

def load_dynamic_config():
    """
    Dynamically configure everything without config.toml.
    - Automatically detects RVC models under ./models (or env var RVC_MODEL_DIR)
    - Automatically detects Bark voices from installed bark package
    """

    # RVC model root: env override or ./models
    rvc_model_dir = os.environ.get("RVC_MODEL_DIR", os.path.join(BASE_DIR, "models"))

    if not os.path.isdir(rvc_model_dir):
        log.warn(f"RVC model directory '{rvc_model_dir}' does not exist. RVC support disabled.")
        rvc_model_dir = None
    else:
        log.info(f"Using RVC model directory: {rvc_model_dir}")

    # Output folder: system temp (unused by you now, but kept for compatibility)
    output_dir = tempfile.gettempdir()
    log.info(f"TTS output directory: {output_dir}")

    return {
        "rvc": {
            "model_dir": rvc_model_dir,
        },
        "tts": {
            "output_dir": output_dir
        }
    }


# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

def relative_bark_paths(paths, start_path):
    return [os.path.relpath(os.path.splitext(p)[0], start_path) for p in paths]


# -----------------------------------------------------------
# LOAD SPEAKERS DYNAMICALLY
# -----------------------------------------------------------

def load_speakers(config):
    """Load all Bark & RVC voices dynamically."""
    # ---------------------------------------------------
    # Load Bark voices (required)
    # ---------------------------------------------------
    bark_voice_dir = os.path.join(bark.__path__[0], "assets/prompts")
    if not os.path.isdir(bark_voice_dir):
        log.error(f"Bark voice directory not found: {bark_voice_dir}")
        sys.exit(1)

    voices_full_path = glob(os.path.join(bark_voice_dir, "**", "*.npz"), recursive=True)
    bark_voices = relative_bark_paths(voices_full_path, bark_voice_dir)

    if not bark_voices:
        log.error(f"No Bark voices found under {bark_voice_dir}")
        sys.exit(1)

    log.info(f"Loaded {len(bark_voices)} Bark voices")

    # ---------------------------------------------------
    # Load RVC models dynamically
    # ---------------------------------------------------
    rvc_model_dir = config["rvc"]["model_dir"]
    rvc_speakers = {}

    if rvc_model_dir:
        rvc_full_path = glob(os.path.join(rvc_model_dir, "**", "*.pth"), recursive=True)

        for model_path in rvc_full_path:
            rel_path = os.path.relpath(model_path, rvc_model_dir)
            speaker_name = os.path.dirname(rel_path)
            model_dir = os.path.dirname(model_path)

            # Find index file in the same folder
            index_files = [f for f in os.listdir(model_dir) if f.endswith(".index")]

            if len(index_files) != 1:
                log.error(
                    f"RVC model '{speaker_name}' MUST contain exactly 1 .index file "
                    f"(found {len(index_files)} in {model_dir})"
                )
                sys.exit(1)

            index_file = os.path.join(model_dir, index_files[0])
            index_rel = os.path.relpath(index_file, rvc_model_dir)

            # Default Bark voice mapping
            bark_voice = "v2/en_speaker_6"

            rvc_speakers[speaker_name] = {
                "id": rel_path,
                "bark_voice": bark_voice,
                "index": index_rel,
            }

        if not rvc_speakers:
            log.warn(f"No RVC models found in {rvc_model_dir}")
        else:
            log.info(f"Loaded {len(rvc_speakers)} RVC speakers")

    return bark_voices, rvc_speakers


# -----------------------------------------------------------
# RUN ON IMPORT
# -----------------------------------------------------------

config = load_dynamic_config()
bark_voices, rvc_speakers = load_speakers(config)
