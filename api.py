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
