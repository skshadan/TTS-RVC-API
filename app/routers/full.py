from fastapi import APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from app.routers.tts import server
from pydantic import BaseModel
from app.config import config
from typing import Optional

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

    rvc_speaker_id, wav = server(
        text=gen.input_text,
        speaker_name=gen.speaker_name,
        emotion=gen.emotion,
        speed=gen.speed 
    )
    if rvc_speaker_id:
        return StreamingResponse(wav, media_type="audio/x-wav")
    else:
        return FileResponse(wav, media_type="audio/x-wav")
