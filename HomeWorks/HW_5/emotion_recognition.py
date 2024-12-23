import torch
import os
import torchaudio
import torch.nn as nn
from tqdm import tqdm

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

# Функция для выравнивания длины аудиофайлов с использованием данных из того же файла
def pad_audio_with_repetition(spectrogram, target_length):
    while spectrogram.size(0) < target_length:
        spectrogram = torch.cat((spectrogram, spectrogram), dim=0)
    return spectrogram[:target_length, :]

# Класс модели GRU
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Функция для распознавания эмоций
def recognize_emotions(model, audio_file):
    # Загрузка аудиофайла
    audio, sample_rate = torchaudio.load(audio_file)
    print(f"Форма звука после загрузки: {audio.shape}")  # [channels, samples]
    
    # Если аудио многоканальное, преобразовать в моно
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
        print(f"Форма звука после преобразования в моно: {audio.shape}")  # [1, samples]
    
    # Преобразование аудиоданных в спектрограмму
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_mels=64, n_fft=400)
    spectrogram = mel_spectrogram(audio)
    print(f"Форма спектрограммы: {spectrogram.shape}")  # [1, n_mels, time]
    
    # Преобразование спектрограммы в 2D тензор
    spectrogram_2d = spectrogram.squeeze(0).transpose(0, 1)
    print(f"Спектрограмма 2D форма: {spectrogram_2d.shape}")  # [time, n_mels]
    
    # Выравнивание длины аудиофайлов с использованием данных из того же файла
    spectrogram_2d = pad_audio_with_repetition(spectrogram_2d, 16000)
    print(f"Спектрограмма 2D после заполнения: {spectrogram_2d.shape}")  # [16000, 64]
    
    # Прогнозирование эмоций
    with torch.no_grad():
        input_tensor = spectrogram_2d.unsqueeze(0)  # [1, 16000, 64]
        print(f"Введите форму тензора в модель: {input_tensor.shape}")
        logits = model(input_tensor)
        print(f"Форма логита: {logits.shape}")  # [1, num_classes]
    
    # Получение меток эмоций
    predicted_ids = torch.argmax(logits, dim=-1).tolist()  # [1]
    print(f"Размерность ids: {predicted_ids}")
    predicted_emotions = [emotion_labels[str(i).zfill(2)] for i in predicted_ids]
    print(f"Размерность predicted_emotions: {predicted_emotions}")
    
    return predicted_emotions

# Пример использования
if __name__ == "__main__":
    model_path = "D:/DS_Belhard_2/Homeworks/HW_5/myprogect_f/myproject/audioapp/models/emotion_recognition_model/emotion_recognition_weights.pth"
    audio_dir = "D:/DS_Belhard_2/Homeworks/HW_5/myprogect_f/myproject/audioapp/output"
    
    # Создание экземпляра модели
    model = GRUModel(input_size=64, hidden_size=32, num_layers=3, num_classes=len(emotion_labels))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Проверка наличия аудиофайлов в директории output
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    if not audio_files:
        print("Нет аудиофайлов в директории output.")
    else:
        # Обработка всех аудиофайлов в директории output
        total_files = 0
        processed_files = 0
        for file_name in tqdm(audio_files, desc="Обработка файлов"):
            total_files += 1
            audio_file = os.path.join(audio_dir, file_name)
            try:
                print(f"Обработка файла {file_name}...")
                emotions = recognize_emotions(model, audio_file)
                processed_files += 1
                print(f"Распознанные эмоции для {file_name}: {emotions}")
            except Exception as e:
                print(f"Ошибка при обработке {file_name}: {e}")

            # Отображение прогресса обработки файла
            progress = (processed_files / total_files) * 100
            print(f"Прогресс обработки: {progress:.2f}%")

        if total_files > 0:
            processed_percentage = (processed_files / total_files) * 100
        else:
            processed_percentage = 0

        print(f"Всего файлов: {total_files}")
        print(f"Обработано файлов: {processed_files}")
        print(f"Процент обработанных файлов: {processed_percentage:.2f}%")
