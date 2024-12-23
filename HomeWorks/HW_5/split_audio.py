from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

# Убедитесь, что путь к модели указан правильно
cache_dir = os.path.join(os.path.dirname(__file__), 'models', 'pyannote')
pipeline_name = "pyannote/speaker-diarization-3.1"
auth_token = "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"

# Проверка наличия модели в локальной директории
if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=auth_token, cache_dir=cache_dir)
    pipeline.model.save_pretrained(cache_dir)
else:
    pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=auth_token, cache_dir=cache_dir)

# Функция для разделения аудио на файлы по голосам
def split_audio_by_speakers(audio_file):
    print("Начало разделения аудио на файлы по голосам...")
    
    # Жестко заданные значения для диапазона количества спикеров
    min_speakers = 2
    max_speakers = 5
    
    # Применение пайплайна к аудиофайлу с указанием диапазона количества спикеров
    diarization = pipeline(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
    
    # Загрузка аудиофайла с помощью pydub
    audio = AudioSegment.from_wav(audio_file)
    
    # Создание директории для сохранения файлов, если она не существует
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Создание отдельных аудиофайлов для каждого спикера
    speaker_files = []
    speakers = set()
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = int(turn.start * 1000)  # Конвертация в миллисекунды
        end_time = int(turn.end * 1000)  # Конвертация в миллисекунды
        speaker_audio = audio[start_time:end_time]
        
        if speaker not in speaker_segments:
            speaker_segments[speaker] = speaker_audio
        else:
            speaker_segments[speaker] += speaker_audio
        
        speakers.add(speaker)
        
        # Вывод информации о спикере и временных метках
        print(f"Спикер: {speaker}, Время начала: {start_time} мс, Время окончания: {end_time} мс, Продолжительность: {end_time - start_time} мс")
    
    # Сохранение объединенных аудиофайлов для каждого спикера
    for speaker, audio_segment in speaker_segments.items():
        speaker_file = os.path.join(output_dir, f"speaker_{speaker}.wav")
        audio_segment.export(speaker_file, format="wav")
        speaker_files.append(speaker_file)
    
    print("Разделение аудио завершено.")
    return speaker_files, len(speakers)
