import random
import json
import torch
from datetime import datetime

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def log_interaction(question, answer, log_file='interaction_log.json'):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer
    }
    try:
        with open(log_file, 'r+', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump([log_entry], f, ensure_ascii=False, indent=4)

print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    questions = sentence.split('?')
    responses = []
    understand_flag = True

    for question in questions:
        if question.strip() == "":
            continue

        question_tokens = tokenize(question)
        X = bag_of_words(question_tokens, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    responses.append(response)
                    log_interaction(question, response)
        else:
            understand_flag = False
            log_interaction(question, "I do not understand...")

    if understand_flag or len(responses) == 0:
        final_response = ". ".join(responses) if responses else "I do not understand..."
        print(f"{bot_name}: {final_response}")
    else:
        final_response = ". ".join(responses)
        print(f"{bot_name}: {final_response}")