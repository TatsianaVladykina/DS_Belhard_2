from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import AudioFile
import os
import torch
from .split_audio import split_audio_by_speakers
from .emotion_recognition import recognize_emotions, GRUModel

# Словарь для преобразования меток эмоций
emotion_labels = {
    '01': '"нейтральность"',
    '02': '"спокойствие"',
    '03': '"счастье"',
    '04': '"грусть"',
    '05': '"злость"',
    '06': '"испуг"',
    '07': '"отвращение"',
    '08': '"удивление"'
}

def index(request):
    return render(request, 'audioapp/upload.html')

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        audio_file = request.FILES['file']
        audio_instance = AudioFile(file=audio_file)
        audio_instance.save()
        print("Файл загружен, ID:", audio_instance.id)
        return JsonResponse({'id': audio_instance.id})

@csrf_exempt
def analyze_file(request, file_id):
    if request.method == 'POST':
        audio_instance = AudioFile.objects.get(id=file_id)
        file_path = audio_instance.file.path
        print("Начало анализа файла:", file_path)

        # Разделение аудиофайла на сегменты по голосам
        speaker_files, num_speakers = split_audio_by_speakers(file_path)
        print("Анализ завершен. Количество спикеров:", num_speakers)

        # Путь к модели
        model_path = "D:/DS_Belhard_2/Homeworks/HW_5/myprogect_f/myproject/audioapp/models/emotion_recognition_model/emotion_recognition_weights.pth"
        
        # Создание экземпляра модели
        model = GRUModel(input_size=64, hidden_size=32, num_layers=3, num_classes=len(emotion_labels))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Распознавание эмоций для каждого спикера
        emotions = {}
        for i, speaker_file in enumerate(speaker_files):
            try:
                predicted_emotions = recognize_emotions(model, speaker_file)
                emotions[f"Спикер {i+1}"] = predicted_emotions[0]  # Предполагаем, что одна эмоция на файл
                print(f"Эмоции для спикера {i+1}: {predicted_emotions}")
            except Exception as e:
                print(f"Ошибка при распознавании эмоций для спикера {i+1}: {e}")
                emotions[f"Спикер {i+1}"] = "Ошибка распознавания"

        # Форматирование результата
        result = f"<p>Количество спикеров: {num_speakers}</p>"
        for speaker, emotion in emotions.items():
            result += f"<p>{speaker}: эмоция {emotion}</p>"
        
        return HttpResponse(result)
