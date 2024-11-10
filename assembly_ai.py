import json
import os
import re

import yaml
import assemblyai as aai


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

aai.settings.api_key = config['assemblyai']['api_key']
audio_folder = 'files/audio/'
transcript_folder = 'files/transcriptions/assembly_ai/'


def get_file_list(folder_path):
    file_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_list.append(file)
    return file_list


def read_transcript(file_name):
    with open(transcript_folder + file_name + '.json', 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    return transcript


def transcript_json_exist(file_name):
    transcript_path = os.path.join(transcript_folder, file_name + '.json')
    return os.path.exists(transcript_path)


def transcript_txt_exist(file_name):
    transcript_path = os.path.join(transcript_folder, file_name + '.txt')
    return os.path.exists(transcript_path)


def do_aai_transcript(file_name):
    file = open(audio_folder + file_name, "rb")

    aai_config = aai.TranscriptionConfig(speaker_labels=True, language_detection=True, speakers_expected=2)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file, config=aai_config)

    return transcript


def ask_gpt(transcript):
    text_with_speaker_labels = ""
    for utt in transcription.utterances:
        text_with_speaker_labels += f"Speaker {utt.speaker}: {utt.text}\n"

    unique_speakers = set(utterance.speaker for utterance in transcription.utterances)

    speakers_words = []
    for speaker in unique_speakers:
        speakers_words.append(f'\"Speaker {speaker}\"')

    result = aai.Lemur().task(
        f"This is a speaker-labeled transcript of a phone call between Customer and Manager of Customer Support Service.\n"
        f"Your task is to identify speaker names and change words {', '.join(speakers_words)} to their names.\n"
        f"If you know Customer's agreement number, add it in speaker name in parentheses.\n"
        f"Do not add and change any other words in conversation. In answer use same language as in original transcript.\n",
        input_text=text_with_speaker_labels,
        final_model=aai.LemurModel.claude3_5_sonnet,
    )

    return result


def write_transcript_json(transcript, file_name):
    # write transcript to JOSN file
    with open(transcript_folder + file_name + '.json', 'w', encoding='utf-8') as f:
        json.dump(transcript.json_response, f, indent=4, ensure_ascii=False)


def write_transcript_txt(transcript_text, file_name):
    # write transcript to TXT file in transcript folder
    with open(transcript_folder + file_name + '.txt', 'w', encoding='utf-8') as f:
        f.write(transcript_text)


if __name__ == '__main__':

    for file in get_file_list(audio_folder):
        transcript_id = None
        print(f'Processing {file}')
        # =========================
        if transcript_json_exist(file):
            print('JSON file already exists.')
            # read transcript from JSON file
            transcript = read_transcript(file)
            transcript_id = transcript['id']
            transcription = aai.Transcript.get_by_id(transcript_id)

        else:
            print('JSON file does not exist. Transcribing in AssemblyAI ...')
            transcription = do_aai_transcript(file)
            print('AssemblyAI transcription done. Writing to JSON file ...')
            write_transcript_json(transcription, file)

        # =========================
        if transcript_txt_exist(file):
            print('TXT file already exists.\n=====================')
        else:
            print('TXT file does not exist. Asking GPT ...')
            gpt_response = ask_gpt(transcription)
            print('GPT response received. Writing to TXT file ...')
            write_transcript_txt(gpt_response.response, file)

    print('All done.')
