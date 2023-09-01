# TODO
- FastAPI
  - Generate:
    - other params? (temperature, rvc f0_method, min_eos_p)

- Bark TTS
  - Set speech sentiment in request body, then prefix each sentence e.g. `[Happy]<input_text>`
  - Should I aim to combine sentences which will fit in the largest clip length? (14s?)
    - More consistent tone etc in bark output?
  - Should map the rvc model to a chosen bark voice (incl. default)?
    - Or set via request body?
    - Currently hardcoded for bark voice `v2/en_speaker_9` (works well with all tested RVC models, regardless of gender etc)

- RVC
  - Should I re-use the Config details? (GPU info etc)
    - Set `CUDA_VISIBLE_DEVICES` for bark
  - Should I load hubert model for each request? Precious VRAM

- Smaller container image
  - Currently used to confirm app dependencies only, don't care that it's 6 GiB

# Issues
- Splits on sentences. So a single sentence which takes longer than ~14 seconds will be a mess
- Significantly slower bark generation when ran via API vs directly in python script
  - Generation time roughly equals audio length (tested on 3090)