# TTS-RVC-API
## Yes, we can use Coqui with RVC!

Why combine the two frameworks? Coqui is a text-to-speech framework (vocoder and encoder), but cloning your own voice takes decades and offers no guarantee of better results. That's why we use RVC (Retrieval-Based Voice Conversion), which works only for speech-to-speech. You can train the model with just 2-3 minutes of dataset as it uses Hubert (a pre-trained model to fine-tune quickly and provide better results).
