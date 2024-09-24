import yaml
import assemblyai as aai


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

aai.settings.api_key = config['assemblyai']['api_key']

if __name__ == '__main__':
    file = open("files/audio/in-25050300-99060607-20240924-152232-1727180552.541352.wav", "rb")

    aai_config = aai.TranscriptionConfig(speaker_labels=True, language_detection=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file, config=aai_config)

    for utterance in transcript.utterances:
        print(f"Speaker {utterance.speaker}: {utterance.text}")
