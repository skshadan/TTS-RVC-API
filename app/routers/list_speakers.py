import os
import bark
import re
from glob import glob
from fastapi import APIRouter, HTTPException
from structlog import get_logger
from ..config import config, bark_voices, rvc_speakers

log = get_logger(__name__)

router = APIRouter(
    prefix="/speakers",
    tags=["speakers"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def list_speakers(b: str = None, r: str = None):
    """ List all speakers

    Search with query parameters which accepts regex. `b` bark voices, `r` rvc models E.g.

    `/speakers?b=en_speaker`: All english bark voices (name containing "en_speaker")

    `/speakers?r=custom_model`: All RVC models in `rvc_model_dir` with basename matching string "custom_model"
    
    `/speakers?b=en_speaker&r=custom_model`: Combination of both
    """

    speakers = []

    # If no query parameters, return all speakers
    if not (b or r):
        b, r = ".*", ".*"

    # Filter from available bark voices in bark package path
    if b:
        filtered_bark = [ x for x in bark_voices if re.match(f".*{b}.*", x) ]
        if not filtered_bark:
            raise HTTPException(
                status_code=400,
                detail=f"No Bark speaker npz files which matched regex \".*{b}.*\" were found."
            )
        speakers += filtered_bark

    # Get RVC models
    rvc_model_dir = config["rvc"]["model_dir"]
    if r and rvc_model_dir:
        filtered_rvc = [ x for x in rvc_speakers if re.match(f".*{r}.*", x) ]
        if not filtered_rvc:
            raise HTTPException(
                status_code=400,
                detail=f"No RVC model files which matched regex \".*{r}.*\" were found in {rvc_model_dir}"
            )
        speakers += filtered_rvc

    return speakers

@router.get("/{speaker_id}")
def get_speaker(speaker_id):
    """Get details on speaker using speaker name

    Where the speaker name is the name of the directory that contains the RVC model pth and index files.
    The speaker name is returned by the `/speakers/` endpoint
    """

    speaker = None
    try:
        speaker = rvc_speakers[speaker_id]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"No RVC model file with speaker_id {speaker_id}"
        )        

    return speaker

# Return list of relative paths for input of list of paths and start path
def relative_paths(paths, start_path):
    p = []
    for i in paths:
        p += [os.path.relpath(i, start_path)]
    return p
