"""
This code does various NLP tasks using SpaCy as the NLP library. 

nouns function extracts the most common 10 nouns of the analyzed text.
verbs function extracts the most 10 common verbs of the analyzed text.
sanitize_names masks any PERSON entity with the string "[REDACTED]".
sanitize_locations masks any GPE entity with the string "[REDACTED]".
spacy_tokenizer tokenizes and lemmatizes a sentence after removing stopwords and punctuation.
selected_topics prints the top 10 topics for a given LDA model and vectorizer.
entities extracts all named entities and their types from a column in a pandas dataframe.
full_df extracts all named entities by type and stores them in a pandas dataframe.

"""


nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == 'NOUN']
word_freq = Counter(nouns)
common_nouns = word_freq.most_common(10)
print(common_nouns)


verbs = [token.text for token in doc if token.is_punct != True and token.pos_ == 'VERB']
print(Counter(verbs).most_common(10))


def sanitize_names(text):
    """
    This function takes a text as input and replaces all PERSON entities with the string '[REDACTED]'. 
    
    Parameters:
    text (str): The string to sanitize.
    
    Returns:
    str: The sanitized string.
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
    This function takes a text as input and replaces all GPE entities with the string '[REDACTED]'. 
    
    Parameters:
    text (str): The string to sanitize.
    
    Returns:
    str: The sanitized string.
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
    This function tokenizes and lemmatizes a sentence after removing stopwords and punctuation.
    
    Parameters:
    sentence (str): The sentence to tokenize and lemmatize.
    
    Returns:
    str: The tokenized and lemmatized sentence.
    """
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


def selected_topics(model, vectorizer, top_n=10):
    """
    This function prints the top n topics for a given LDA model and vectorizer.
    
    Parameters:
    model (LDA model): The LDA model to use.
    vectorizer (vectorizer object): The vectorizer object to use.
    top_n (int, optional): The number of top topics to print. Defaults to 10.
    """
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])


entities = []
for text in nlp.pipe(iter(bestdf['Description']), batch_size=1000, n_threads=-1):
    entities.append([(X.text, X.label_) for X in text.ents])
    print([(X.text, X.label_) for X in text.ents])


index = range(0, 13911)
full_df = pd.DataFrame(index=index, columns=['DOC NR', 'CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT',
                                              'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT', 'PRODUCT',
                                              'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'])


for index, doc in enumerate(entities):
    for entity, entity_type in doc:
        if pd.isna(full_df.at[index, entity_type]):
            full_df.at[index, entity_type] = entity
        else:
            full_df.at[index, entity_type] = full_df.at[index, entity_type] + ", {}".format(entity)