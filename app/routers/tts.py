from TTS.api import TTS
import huggingface_hub
from nltk.tokenize import sent_tokenize
import numpy as np
import os
from scipy.io.wavfile import write
import time
import io
from typing import Optional
from fastapi import HTTPException
from ..config import config, bark_voices, rvc_speakers
from ..rvc.misc import (
    load_hubert,
    get_vc,
    vc_single
)
from structlog import get_logger



# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log = get_logger(__name__)

def server(
        text: str,
        tts_output_dir: str,
        speaker_name: str,
        emotion: Optional[str] = None,
        speed: Optional[float] = 1.0
    ):
    rvc_model_dir = config["rvc"]["model_dir"]

    # TTS output file path
    tts_file = os.path.join(tts_output_dir, "bark_out.wav")

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
    tts.tts_to_file(text=full_script, file_path=tts_file, emotion="Surprise", speed=1.0)

    t0 = time.time()
    generation_duration_s = time.time() - t0
    log.info(f"took {generation_duration_s:.0f}s to generate audio")

    if rvc_speaker_id and rvc_model_dir:
        hubert_model = None
        hubert_path = huggingface_hub.hf_hub_download(
            repo_id="lj1995/VoiceConversionWebUI",
            filename="hubert_base.pt",
            revision="1c75048c96f23f99da4b12909b532b5983290d7d",
            local_dir="models/hubert/",
            local_dir_use_symlinks=True,
        )
        hubert_model = load_hubert(hubert_path)
        
        get_vc(rvc_speaker_id, rvc_model_dir, 0.33, 0.5)
        
        rvc_index = os.path.join(rvc_model_dir, rvc_speakers[speaker_name]["index"])
        wav_opt = vc_single(
            0, 
            tts_file,
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

        wav = io.BytesIO()
        write(wav, wav_opt[1][0], wav_opt[1][1])
        
        return rvc_speaker_id, wav
    else:
        return rvc_speaker_id, tts_file
