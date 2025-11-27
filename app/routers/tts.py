from TTS.api import TTS
import huggingface_hub, numpy as np, os, time, io, glob
from nltk.tokenize import sent_tokenize
from scipy.io.wavfile import write
from typing import Optional
from fastapi import HTTPException
from ..config import config, bark_voices, rvc_speakers
from ..rvc.misc import (
    load_hubert,
    get_vc,
    vc_single
)
from structlog import get_logger

BASE_DIR = os.path.abspath(os.getcwd())
RVC_MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = os.path.join(BASE_DIR, "tmp")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(RVC_MODEL_DIR, exist_ok=True)


import torch
from structlog import get_logger

log = get_logger(__name__)

def detect_tts_device():
    """Return 'cuda' if a usable GPU is available, otherwise 'cpu'."""

    # is any GPU visible?
    if not torch.cuda.is_available():
        log.warn("No CUDA GPU detected → using CPU.")
        return "cpu"

    try:
        major, minor = torch.cuda.get_device_capability()
        cc = major * 10 + minor

        log.info(f"Found GPU compute capability: {major}.{minor} (cc={cc})")

        # PyTorch 2.1+ requires at least compute capability 7.5
        if cc < 75:
            log.warn(f"GPU compute capability {major}.{minor} is too old for PyTorch "
                     f"(requires >= 7.5). Falling back to CPU.")
            return "cpu"

        # Try a tiny CUDA op to verify
        try:
            test = torch.tensor([1.0]).cuda() * 2
            log.info("CUDA test operation succeeded → USING GPU.")
            return "cuda"
        except Exception as e:
            log.error(f"CUDA test operation failed → GPU unusable → CPU. Error: {e}")
            return "cpu"

    except Exception as e:
        log.error(f"Error checking GPU: {e}. Falling back to CPU.")
        return "cpu"


device = detect_tts_device()
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=(device=="cuda"))
log.info(f"TTS initialized on device: {device}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log = get_logger(__name__)

def get_output_filename(user_name: Optional[str] = None):
    """
    - If the user provides a name → use it (auto-add .wav if missing)
    - If empty or None → generate output.wav, output_1.wav, output_2.wav, ...
    """

    if user_name and user_name.strip():
        name = user_name.strip()

        # add .wav if user forgot it
        if not name.lower().endswith(".wav"):
            name += ".wav"

        return os.path.join(OUTPUT_DIR, name)

    files = glob.glob(os.path.join(OUTPUT_DIR, "*.wav"))

    if len(files) == 0:
        return os.path.join(OUTPUT_DIR, "output.wav")

    numbers = []
    for f in files:
        base = os.path.basename(f)
        name, _ = os.path.splitext(base)

        if name == "output":
            numbers.append(0)
        elif name.startswith("output_"):
            try:
                num = int(name.split("_")[1])
                numbers.append(num)
            except ValueError:
                pass

    if len(numbers) == 0:
        return os.path.join(OUTPUT_DIR, "output.wav")

    next_number = max(numbers) + 1
    return os.path.join(OUTPUT_DIR, f"output_{next_number}.wav")


def server(
        text: str,
        speaker_name: str,
        file_name: Optional[str] = None,
        emotion: Optional[str] = None,
        speed: Optional[float] = 1.0
    ):


    # TTS output file path
    tts_tmp_file = os.path.join(TMP_DIR, "audio_tmp.wav")

    # Is the speaker an RVC model?
    rvc_speaker_id = None
    if speaker_name in rvc_speakers:
        rvc_speaker_id = rvc_speakers[speaker_name]["id"]
    else:
        raise HTTPException(status_code=400, detail=f"speaker_name \"{speaker_name}\" was not found")

    # Prepare the text
    script = text.replace("\n", " ").strip()
    sentences = sent_tokenize(script)
    full_script = " ".join(sentences)

    # Generate audio using TTS
    tts.tts_to_file(text=full_script, file_path=tts_tmp_file, emotion="Surprise", speed=1.0)

    t0 = time.time()
    generation_duration_s = time.time() - t0
    log.info(f"took {generation_duration_s:.0f}s to generate audio")

    if rvc_speaker_id and RVC_MODEL_DIR:
        hubert_model = None
        hubert_path = huggingface_hub.hf_hub_download(
            repo_id="lj1995/VoiceConversionWebUI",
            filename="hubert_base.pt",
            revision="1c75048c96f23f99da4b12909b532b5983290d7d",
            local_dir="models/hubert/",
            local_dir_use_symlinks=True,
        )
        hubert_model = load_hubert(hubert_path)
        
        get_vc(rvc_speaker_id, RVC_MODEL_DIR, 0.33, 0.5)
        
        rvc_index = os.path.join(RVC_MODEL_DIR, rvc_speakers[speaker_name]["index"])
        wav_opt = vc_single(
            0, 
            tts_tmp_file,
            0, 
            None, 
            "pm", 
            rvc_index,
            '',
            0.88,
            3,
            0,
            1,
            0.33,
        )

        if wav_opt is None or wav_opt[1] is None:
            raise HTTPException(500, "RVC conversion failed: wav_opt returned None.")


        wav = io.BytesIO()
        write(wav, wav_opt[1][0], wav_opt[1][1])
        wav.seek(0)
        save_path = get_output_filename(file_name)  # re-use the same naming logic

        with open(save_path, "wb") as audio_file:
            audio_file.write(wav.getvalue())
        log.info(f"Saved final audio file: {save_path}")
        return rvc_speaker_id, wav
    else:
        return rvc_speaker_id, tts_tmp_file
