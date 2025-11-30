from fastapi import APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from .tts import server
from pydantic import BaseModel
from typing import Optional
import io

router = APIRouter(
    prefix="/generate",
    tags=["generate"],
    responses={404: {"description": "Not found"}},
)

class Generation(BaseModel):
    speaker_name: Optional[str] = None  # Change this line
    input_text: str
    emotion: Optional[str] = None  # Added this line
    speed: Optional[float] = 1.0

@router.post("/")
async def generate(gen: Generation):
    from fastapi.concurrency import run_in_threadpool
    rvc_speaker_id, audio_data = await run_in_threadpool(
        server,
        text=gen.input_text,
        speaker_name=gen.speaker_name,
        emotion=gen.emotion,
        speed=gen.speed 
    )
    if isinstance(audio_data, io.BytesIO):
        audio_data.seek(0)
        return StreamingResponse(audio_data, media_type="audio/wav")

    # TTS-only returns a filepath
    return FileResponse(audio_data, media_type="audio/wav")
