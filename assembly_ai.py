import re

import yaml
import assemblyai as aai


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

aai.settings.api_key = config['assemblyai']['api_key']

if __name__ == '__main__':
    file = open("files/audio/in-25050300-99060607-20240924-152232-1727180552.541352.wav", "rb")

    aai_config = aai.TranscriptionConfig(speaker_labels=True, language_detection=True, speakers_expected=3)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file, config=aai_config)

    # assemblyai get transcript by ID
    # transcript = aai.Transcript.get_by_id(transcript_id='74f1040b-1516-415e-bc3c-38f29d6281a1')

    text_with_speaker_labels = ""
    for utt in transcript.utterances:
        text_with_speaker_labels += f"Speaker {utt.speaker}:\n{utt.text}\n"

    unique_speakers = set(utterance.speaker for utterance in transcript.utterances)

    questions = []
    for speaker in unique_speakers:
        questions.append(
            aai.LemurQuestion(
                question=f"Who is speaker {speaker}?",
                answer_format="<First Name> <Last Name (if applicable)>"
            )
        )

    result = aai.Lemur().question(
        questions,
        input_text=text_with_speaker_labels,
        context="Your task is to infer the speaker's name from the speaker-labelled transcript"
    )

    speaker_mapping = {}
    for qa_response in result.response:
        pattern = r"Who is speaker (\w)\?"
        match = re.search(pattern, qa_response.question)
        if match and match.group(1) not in speaker_mapping.keys():
            speaker_mapping.update({match.group(1): qa_response.answer})

    for utterance in transcript.utterances:
        speaker_name = speaker_mapping[utterance.speaker]
        print(f"{speaker_name}: {utterance.text}")
