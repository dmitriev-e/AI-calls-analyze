import json
import yaml
import openai
from pathlib import Path

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

client = openai.OpenAI(api_key=config['openai']['api_key'])

# Define the folder path
audio_folder = Path('files/audio/')
transcript_folder = 'files/transcriptions/openai_whisper/'

# Get a list of all *.wav files in folder
wav_files = [f for f in audio_folder.glob('*.wav') if f.is_file()]


def write_transcript_json(transcript, file_name):
    # write transcript to JOSN file
    with open(transcript_folder + file_name + '.json', 'w', encoding='utf-8') as f:
        f.write(transcript.model_dump_json(indent=4))


def write_transcript_txt(transcript_text, file_name):
    # write transcript to TXT file in transcript folder
    with open(transcript_folder + file_name + '.txt', 'w', encoding='utf-8') as f:
        f.write(transcript_text)


if __name__ == '__main__':
    for wav_file in wav_files:
        if wav_file.is_file():
            print(f"Processing file: {wav_file}")

            try:
                # Open the audio file in binary mode
                with open(wav_file, 'rb') as audio_file:
                    # Transcribe the audio file using OpenAI's Whisper API
                    transcription = client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                        temperature=0
                    )

                # Save the transcription to a text file
                write_transcript_txt(transcription.text, wav_file.name)

                # Save the transcription to a JSON file
                write_transcript_json(transcription, wav_file.name)
                print(f"Transcription saved")
            except Exception as e:
                print(f"Could not transcribe file: {wav_file}; Error: {e}")

    print("All files processed.")