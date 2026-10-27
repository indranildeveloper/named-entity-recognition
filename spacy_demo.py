import spacy

# texts = [
#     "John goes for a walk in Berlin.",
#     "Mike is going to the store.",
#     "Elon Musk is the CEO at Twitter.",
#     "Bob Smith is the guy behind XYZ-Soft Inc.",
#     "Indranil Halder is the guy behind Phlama.",
# ]

texts = [
    "What is the price of 4 bananas?",
    "How much are 16 chairs?",
    "Give me the value of 5 laptops",
    "Give me the value of five laptops",
]

nlp = spacy.load("en_core_web_lg")

# ner_labels = nlp.get_pipe("ner").labels  # type: ignore
# print(ner_labels)

# categories = ["ORG", "PERSON", "LOC"]

docs = [nlp(text) for text in texts]

for doc in docs:
    entities = []
    for ent in doc.ents:
        # if ent.label_ in categories:
        entities.append((ent.text, ent.label_))
    print(entities)
