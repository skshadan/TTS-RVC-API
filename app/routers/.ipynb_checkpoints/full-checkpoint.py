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

from fastapi import Response
import time

# class TimedStreamingResponse(Response):
#     media_type = "audio/x-wav"

#     def __init__(self, content: io.BytesIO, *args, **kwargs):
#         def content_generator():
#             yield content.getvalue()

#         super().__init__(content_generator(), *args, **kwargs)

#     async def __call__(self, scope, receive, send):
#         self.start_time = time()
#         await super().__call__(scope, receive, send)

#     def body_iterator(self):
#         for chunk in self.body:
#             if not hasattr(self, 'first_chunk_sent_time'):
#                 self.first_chunk_sent_time = time()
#                 print(f"First chunk sent after {self.first_chunk_sent_time - self.start_time} seconds")
#             yield chunk

class Generation(BaseModel):
    speaker_name: Optional[str] = None  # Change this line
    input_text: str
    emotion: Optional[str] = None  # Added this line
    speed: Optional[float] = 1.0

@router.post("/")
async def generate(gen: Generation):

    t1 = time.time()
    rvc_speaker_id, wav = server(
        text=gen.input_text,
        tts_output_dir=config["tts"]["output_dir"],
        speaker_name=gen.speaker_name,
        emotion=gen.emotion,
        speed=gen.speed 
    )
    ghanta_duration = time.time() - t1
    print(f"GHANTA time {ghanta_duration:.2f}s")
    
    if rvc_speaker_id:
        return FileResponse(wav, media_type="audio/x-wav")
    else:
        response = FileResponse(wav, media_type="audio/x-wav")
    tot_duration = time.time() - t1
    print(f"FINAL time {tot_duration:.2f}s")
    
    return response
    
