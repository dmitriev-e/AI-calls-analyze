import json
import yaml
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        api_key=config['yandex']['speech_api_key']
    )
)

model = model_repository.recognition_model()

model.model = 'general'
model.language = 'ru-RU'
model.speaker_labeling = True
model.data_logging = True
model.audio_processing_type = AudioProcessingType.Full

file_name = 'in-s-79836338557-20241109-190208-1731150128.2132010.wav'
audio_folder = 'files/audio/'
transrip_folder = 'files/transcriptions/'

result = model.transcribe_file(audio_folder + file_name)
print(result)

# write result to JSON file in files/transcriptions
# with open(transrip_folder + file_name + '.json', 'w') as f:
#     f.write(result)

for c, res in enumerate(result):
    print(f'channel: {c}\n'
          f'raw_text: {res.raw_text}\n'
          f'norm_text: {res.normalized_text}\n')
