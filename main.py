import random

import spacy
from spacy.training import Example
from spacy.util import minibatch

train_data = [
    (
        "What is the price of 10 bananas?",
        {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]},
    ),
    (
        "How much do 5 apples cost?",
        {"entities": [(13, 14, "QUANTITY"), (15, 20, "PRODUCT")]},
    ),
    (
        "What is the price of 2 kilograms of rice?",
        {"entities": [(21, 22, "QUANTITY"), (34, 38, "PRODUCT")]},
    ),
    (
        "I want to buy 3 oranges.",
        {"entities": [(14, 15, "QUANTITY"), (16, 23, "PRODUCT")]},
    ),
    (
        "How much is 1 watermelon?",
        {"entities": [(12, 13, "QUANTITY"), (14, 24, "PRODUCT")]},
    ),
    (
        "Give me the price of 12 mangoes.",
        {"entities": [(23, 25, "QUANTITY"), (26, 33, "PRODUCT")]},
    ),
    (
        "What does 6 potatoes cost?",
        {"entities": [(10, 11, "QUANTITY"), (12, 20, "PRODUCT")]},
    ),
    (
        "How much for 4 tomatoes?",
        {"entities": [(13, 14, "QUANTITY"), (15, 23, "PRODUCT")]},
    ),
    (
        "What is the cost of 8 onions?",
        {"entities": [(20, 21, "QUANTITY"), (22, 28, "PRODUCT")]},
    ),
    (
        "Tell me the price of 7 carrots.",
        {"entities": [(21, 22, "QUANTITY"), (23, 30, "PRODUCT")]},
    ),
    (
        "How much do 3 pineapples cost?",
        {"entities": [(13, 14, "QUANTITY"), (15, 25, "PRODUCT")]},
    ),
    (
        "What is the price of 5 lemons?",
        {"entities": [(21, 22, "QUANTITY"), (23, 29, "PRODUCT")]},
    ),
    ("I need 9 cucumbers.", {"entities": [(7, 8, "QUANTITY"), (9, 18, "PRODUCT")]}),
    (
        "How much are 2 coconuts?",
        {"entities": [(13, 14, "QUANTITY"), (15, 23, "PRODUCT")]},
    ),
    (
        "What is the price of 4 papayas?",
        {"entities": [(21, 22, "QUANTITY"), (23, 30, "PRODUCT")]},
    ),
    (
        "Please tell me the cost of 6 pears.",
        {"entities": [(27, 28, "QUANTITY"), (29, 34, "PRODUCT")]},
    ),
    (
        "How much do 8 strawberries cost?",
        {"entities": [(13, 14, "QUANTITY"), (15, 28, "PRODUCT")]},
    ),
    (
        "What is the price of 3 grapes?",
        {"entities": [(21, 22, "QUANTITY"), (23, 29, "PRODUCT")]},
    ),
    (
        "I want to buy 5 kilograms of potatoes.",
        {"entities": [(14, 15, "QUANTITY"), (32, 40, "PRODUCT")]},
    ),
    (
        "How much does 2 kilograms of sugar cost?",
        {"entities": [(14, 15, "QUANTITY"), (29, 34, "PRODUCT")]},
    ),
    (
        "What is the price of 10 kilograms of flour?",
        {"entities": [(21, 23, "QUANTITY"), (37, 42, "PRODUCT")]},
    ),
    (
        "Give me 3 liters of milk.",
        {"entities": [(8, 9, "QUANTITY"), (19, 23, "PRODUCT")]},
    ),
    (
        "How much for 2 bottles of water?",
        {"entities": [(13, 14, "QUANTITY"), (27, 32, "PRODUCT")]},
    ),
    (
        "What is the cost of 5 eggs?",
        {"entities": [(20, 21, "QUANTITY"), (22, 26, "PRODUCT")]},
    ),
    ("I need 12 eggs.", {"entities": [(7, 9, "QUANTITY"), (10, 14, "PRODUCT")]}),
    (
        "How much do 4 loaves of bread cost?",
        {"entities": [(13, 14, "QUANTITY"), (28, 33, "PRODUCT")]},
    ),
    (
        "What is the price of 2 packets of biscuits?",
        {"entities": [(21, 22, "QUANTITY"), (32, 40, "PRODUCT")]},
    ),
    (
        "Tell me how much 7 bars of chocolate cost.",
        {"entities": [(17, 18, "QUANTITY"), (27, 36, "PRODUCT")]},
    ),
    (
        "What is the price of 3 bags of rice?",
        {"entities": [(21, 22, "QUANTITY"), (31, 35, "PRODUCT")]},
    ),
    (
        "How much do 6 bottles of juice cost?",
        {"entities": [(13, 14, "QUANTITY"), (27, 32, "PRODUCT")]},
    ),
]


nlp = spacy.load("en_core_web_lg")

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

for _, annotations in train_data:
    for ent in annotations:
        if ent[2] not in ner.labels:  # type: ignore
            ner.add_label(ent[2])  # type: ignore


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    epochs = 50

    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}  # type: ignore
        batches = minibatch(train_data, size=2)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Epoch {epoch + 1}, Losses: {losses}")

nlp.to_disk("custom_ner_model")
trained_nlp = spacy.load("custom_ner_model")

test_texts = [
    "How much for 3 oranges?",
    "I want 15 chairs for the conference.",
    "Can you give me the price for 6 desks.",
]

for text in test_texts:
    doc = trained_nlp(text)
    print(f"Text: {text}")
    print("Entities: ", [(ent.text, ent.label_) for ent in doc.ents])
    print()
