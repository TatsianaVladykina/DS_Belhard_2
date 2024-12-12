import pandas as pd
import json

# Чтение данных из Excel-файла
df = pd.read_excel('tags.xlsx')

# Создание структуры для JSON
intents = {"intents": []}

# Группировка данных по тегам
grouped = df.groupby('tag')

for tag, group in grouped:
    patterns = group['patterns'].dropna().tolist()
    responses = group['responses'].dropna().tolist()
    intent = {
        "tag": tag,
        "patterns": patterns,
        "responses": responses
    }
    intents["intents"].append(intent)

# Запись данных в JSON-файл
with open('intents.json', 'w', encoding='utf-8') as json_file:
    json.dump(intents, json_file, ensure_ascii=False, indent=4)

print("Данные успешно преобразованы и сохранены в intents.json")