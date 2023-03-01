# This code extracts common nouns and verbs from a given text and redacts all personal names and locations

# Import necessary libraries
from collections import Counter
import spacy
import pandas as pd

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Extract common nouns from the text
nouns = [token.text for token in doc if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN']
word_freq = Counter(nouns)
common_nouns = word_freq.most_common(10)
print(common_nouns)

# Extract common verbs from the text
verbs = [token.text for token in doc if token.is_punct !=True and token.pos_ == 'VERB']
print(Counter(verbs).most_common(10))

# Function to redact all personal names from the text
def sanitize_names(text):
    doc = nlp(text)
    redacted_sentences = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ == 'PERSON':
            redacted_sentences.append("[REDACTED]")
        else:
            redacted_sentences.append(token.string)
    return "".join(redacted_sentences)

# Function to redact all locations from the text
def sanitize_locations(text):
    doc = nlp(text)
    redacted_sentences = []
    for ent in doc.ents:
        ent.merge()
    for token in doc:
        if token.ent_type_ == 'GPE':
            redacted_sentences.append("[REDACTED]")
        else:
            redacted_sentences.append(token.string)
    return "".join(redacted_sentences)

# Function to tokenize a sentence using Spacy
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# Function to extract selected topics from a given model
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
              for i in topic.argsort()[:-top_n - 1:-1]])

# Extract all entities (personal names and locations) from a given text
entities = []
for text in nlp.pipe(iter(bestdf['Description']), batch_size=1000, n_threads=-1):
    entities.append([(X.text, X.label_) for X in text.ents])
    print([(X.text, X.label_) for X in text.ents])

# Create a DataFrame to store all extracted entities
index = range(0, 13911)
full_df = pd.DataFrame(index=index, columns=['DOC NR','CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'])

# Add each extracted entity to the DataFrame
for index, doc in enumerate(entities):
    for entity, entity_type in doc:
        if pd.isna(full_df.at[index, entity_type]):
            full_df.at[index, entity_type] = entity
        else:
            full_df.at[index, entity_type] = full_df.at[index, entity_type] + ", {}".format(entity)