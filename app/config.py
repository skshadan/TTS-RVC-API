import tomli
import os
import sys
import tempfile
from fastapi import HTTPException
from structlog import get_logger
import bark
from glob import glob

log = get_logger(__name__)

def parse_config():
    """Parse and validate config on startup
    Raise errors for invalid config

    Set defaults for undefined config options
    """

    config = {
        "rvc": {},
        "tts": {},
    }
    config_file = None
    try:
        with open("config.toml", mode="rb") as fp:
            config_file = tomli.load(fp)
    except FileNotFoundError:
        log.error("FAILURE TO START. Configuration file \"config.toml\" was not found")
        sys.exit(1)

    try:
        if config_file["rvc"]["model_dir"]:
            rvc_model_dir = config_file["rvc"]["model_dir"]
            if not os.path.isdir(rvc_model_dir):
                log.error(f"FAILURE TO START. Config item \"rvc.model_dir\" was defined but path \"{rvc_model_dir}\" was not found")
                log.info(f"Remove config item \"rvc.model_dir\" from \"config.toml\" if you don't want to use RVC models")
                sys.exit(1)
            else:
                config["rvc"]["model_dir"] = config_file["rvc"]["model_dir"]
    except KeyError:
        config["rvc"]["model_dir"] = False
        log.warn(f"Config item \"rvc.model_dir\" is missing from config file. RVC features are disabled")

    try:
        if config_file["rvc"]["bark_voice_map"]:
            if not config["rvc"]["model_dir"]:
                log.error(f"FAILURE TO START. Config item \"rvc.bark_voice_map\" was defined but \"rvc.model_dir\" was not")
                log.info(f"Config item \"rvc.model_dir\" is required to use RVC models. Either set this value or remove \"rvc.bark_voice_map\"")
                sys.exit(1)
            config["rvc"]["bark_voice_map"] = config_file["rvc"]["bark_voice_map"]
    except KeyError:
        # Suno Favourite from voice Bark Speaker Library (v2)
        config["rvc"]["bark_voice_map"] = {"default": "v2/en_speaker_6"}
        log.warn("Config item \"rvc.bark_voice_map\" is undefined. Setting \"v2/en_speaker_6\" for all RVC models")

    try:
        temp = config_file["tts"]["output_dir"]
        if temp:
            if not os.path.isdir(temp):
                log.error(f"FAILURE TO START. Config item \"tts.output_dir\" was defined but path \"{temp}\" was not found")
                log.info(f"Either remove config item \"tts.output_dir\" from \"config.toml\" to use system default temp dir, or set the value as an existing directory path")
                sys.exit(1)
            else:
                config["tts"]["output_dir"] = config_file["tts"]["output_dir"]
    except KeyError:
        temp_dir = tempfile.gettempdir()
        config["tts"]["output_dir"] = temp_dir
        log.warn(f"Config item \"tts.output_dir\" is undefined. Using system default: {temp_dir}")

    log.info(f"STARTUP CONFIG: {config}")
    return config

# Return list of relative paths for input of list of paths and start path
def relative_bark_paths(paths, start_path):
    p = []
    for i in paths:
        p += [os.path.relpath(os.path.splitext(i)[0], start_path)]
    return p

def load_speakers(config):
    """Load all available speakers on system. Including Bark voices and RVC models
    """
    # Get bark voices from bark package files
    bark_voice_dir = os.path.join(bark.__path__[0], "assets/prompts")
    if not os.path.isdir(bark_voice_dir):
        log.error(f"FAILURE TO START. Bark voice directory was not found at {bark_voice_dir}")
        sys.exit(1)

    voices_full_path = glob(os.path.join(bark_voice_dir, "**", f"*.npz"), recursive=True)
    bark_voices = relative_bark_paths(voices_full_path, bark_voice_dir)
    if not bark_voices:
        log.error(f"FAILURE TO START. No Bark speaker npz files were found in a recursive search of existing directory {bark_voice_dir}. Bark speakers are required")
        sys.exit(1)

    # Get RVC models
    rvc_model_dir = config["rvc"]["model_dir"]
    rvc_speakers = {}
    if rvc_model_dir:
        rvc_full_path = glob(os.path.join(rvc_model_dir, f"**", "*.pth"), recursive=True)
        for s in rvc_full_path:
            id = os.path.relpath(s, rvc_model_dir)
            name = os.path.split(id)[0]
            dir = os.path.dirname(s)
            index = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".index")]
            if len(index) != 1:
                log.error(f"FAILURE TO START. RVC model {name} should have 1 index file. It has {len(index)}")
                sys.exit(1)
            index_relative = os.path.relpath(
                index[0],
                rvc_model_dir
            )
            try:
                bv = config["rvc"]["bark_voice_map"][name]
            except KeyError:
                bv = config["rvc"]["bark_voice_map"]["default"]
            rvc_speakers[name] = {"id": id, "bark_voice": bv, "index": index_relative}

        if not rvc_speakers:
            log.error(f"FAILURE TO START. No RVC model files were found in a recursive search of the defined and existing {rvc_model_dir}")
            log.info(f"You must supply any RVC models you wish to use. Either remove or fix the config item \"rvc.model_dir\"")
            sys.exit(1)
    
    return bark_voices, rvc_speakers

config = parse_config()
bark_voices, rvc_speakers = load_speakers(config)