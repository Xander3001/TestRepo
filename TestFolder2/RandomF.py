"""
This script includes multiple functions that perform various NLP tasks using spaCy library.
"""

from collections import Counter
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

def get_common_nouns(doc):
    """
    Returns a list of 10 most common nouns in the given doc object.
    """
    nouns = [token.text for token in doc if not token.is_stop and not token.is_punct and token.pos_ == 'NOUN']
    word_freq = Counter(nouns)
    common_nouns = word_freq.most_common(10)
    return common_nouns

def get_common_verbs(doc):
    """
    Returns a list of 10 most common verbs in the given doc object.
    """
    verbs = [token.text for token in doc if not token.is_punct and token.pos_ == 'VERB']
    return Counter(verbs).most_common(10)

def sanitize_names(text):
    """
    Replaces all PERSON entities in the given text with [REDACTED] tag.
    """
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

def sanitize_locations(text):
    """
    Replaces all GPE entities in the given text with [REDACTED] tag.
    """
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

def spacy_tokenizer(sentence):
    """
    Returns a cleaned version of the given sentence using spaCy tokenizer.
    Removes stopwords and punctuations.
    """
    stopwords = nlp.Defaults.stop_words
    punctuations = nlp.Defaults.punctuations
    parser = nlp(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in parser]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    return " ".join([i for i in mytokens])

def selected_topics(model, vectorizer, top_n=10):
    """
    Returns top n topics from the given LDA model.
    """
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

def extract_entities(texts):
    """
    Extracts various entities from the given list of texts using spaCy EntityRecognizer.
    Returns a pandas dataframe with entitiy categories as columns and extracted entities as values.
    """
    index = range(len(texts))
    full_df = pd.DataFrame(index=index, columns=['DOC NR','CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT',
                                                  'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT',
                                                  'PRODUCT', 'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'])
    entities = []
    for text in nlp.pipe(iter(texts), batch_size=1000, n_threads=-1):
        entities.append([(X.text, X.label_) for X in text.ents])
    for i, doc in enumerate(entities):
        for entity, entity_type in doc:
            if pd.isna(full_df.at[i, entity_type]):
                full_df.at[i, entity_type] = entity
            else:
                full_df.at[i, entity_type] = full_df.at[i, entity_type] + ", " + entity
    return full_df