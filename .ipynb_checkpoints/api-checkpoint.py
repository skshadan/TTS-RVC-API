import requests
import json
import time

url = "http://127.0.0.1:8000/generate"

payload = json.dumps({
  "speaker_name": "speaker3",
    "input_text": "Hello, everyone. Thanks for being here on this livestream.",
  "emotion": "Dull",
  "speed": 1.0
})
headers = {
  'Content-Type': 'application/json'
}

start_time = time.time()  # Start the timer

response = requests.request("POST", url, headers=headers, data=payload, stream = True)

end_time = time.time()  # Stop the timer

if response.status_code == 200:
    audio_content = response.content
    
    # Save the audio to a file
    with open("generated_audio.wav", "wb") as audio_file:
        for chunk in response.iter_content(chunk_size=128):
            if chunk:  # filter out keep-alive new chunks
                audio_file.write(chunk)
        
    print("Audio saved successfully.")
    print("Time taken:", end_time - start_time, "seconds")
else:
    print("Error:", response.text)
