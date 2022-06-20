import json
import random

import torch

from model import NerualNet
from nltk_utils import bag_of_words, stem, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as f:
    intents = json.load(f)

FILE = "model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NerualNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    # print(X)
    # print(X.shape)
    X = X.reshape(1, X.shape[0])
    # print(X)
    # print(X.shape)
    X = torch.from_numpy(X).float().to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted[0]]

    tag = tags[predicted.item()]
    print(f"{bot_name}: {tag} - prob: {prob}")
