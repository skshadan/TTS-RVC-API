
# TTS-RVC-API

Yes, we can use Coqui with RVC!

# Why combine the two frameworks?

Coqui is a text-to-speech framework (vocoder and encoder), but cloning your own voice takes decades and offers no guarantee of better results. That's why we use RVC (Retrieval-Based Voice Conversion), which works only for speech-to-speech. You can train the model with just 2-3 minutes of dataset as it uses Hubert (a pre-trained model to fine-tune quickly and provide better results).


## Installation

How to use Coqui + RVC api?

```python
https://github.com/skshadan/TTS-RVC-API.git
```
```python
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install TTS
python -m uvicorn app.main:app
```
Now update `config.toml` with relative paths
config `model_dir` path or set a `speaker_name` in the request body

Where the RVC v2 model is mounted on the container at:
```python
  /
└── models
      └── speaker1
          ├── speaker1.pth
          └── speaker1.index
```

Now Run this 
```python
  python -m uvicorn app.main:app
```
## POST REQUEST

```python
  http://localhost:8000/generate
```
```python
    emotions : happy,sad,angry,dull
    speed = 1.0 - 2.0
```
```python
  {
  "speaker_name": "speaker3",
  "input_text": "Hey there! Welcome to the world",
  "emotion": "Surprise",
  "speed": 1.0
}
```
   
# CODE SNIPPET

```python
import requests
import json
import time

url = "http://127.0.0.1:8000/generate"

payload = json.dumps({
  "speaker_name": "speaker3",
  "input_text": "Are you mad? The way you've betrayed me is beyond comprehension, a slap in the face that's left me boiling with an anger so intense it's as if you've thrown gasoline on a fire, utterly destroying any trust that was left.",
  "emotion": "Dull",
  "speed": 1.0
})
headers = {
  'Content-Type': 'application/json'
}

start_time = time.time()  # Start the timer

response = requests.request("POST", url, headers=headers, data=payload)

end_time = time.time()  # Stop the timer

if response.status_code == 200:
    audio_content = response.content
    
    # Save the audio to a file
    with open("generated_audio.wav", "wb") as audio_file:
        audio_file.write(audio_content)
        
    print("Audio saved successfully.")
    print("Time taken:", end_time - start_time, "seconds")
else:
    print("Error:", response.text)
```
## Feedback

If you have any feedback, issues please reach out to shadankhantech@gmail.com

