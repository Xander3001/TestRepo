# This code performs various NLP tasks using the Spacy library

# Counter for most common nouns
nouns = [ token.text for token in doc if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN']
word_freq = Counter(nouns)
common_nouns = word_freq.most_common(10)
print(common_nouns)

# Counter for most common verbs
verbs = [ token.text for token in doc if token.is_punct !=True and token.pos_ == 'VERB']
print(Counter(verbs).most_common(10))

# Function to redact names in text
def sanitize_names(text):
    """
    Given a piece of text, this function uses Spacy to redact names and return the modified text.
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

# Function to redact locations in text
def sanitize_locations(text):
    """
    Given a piece of text, this function uses Spacy to redact locations and return the modified text.
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

# Function to tokenize and preprocess text for modeling
def spacy_tokenizer(sentence):
    """
    Given a sentence, this function preprocesses the text by outputting lowercase lemmatized words that aren't stopwords or punctuation marks.
    """
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# Function to print the top topics from a fitted LSA or NMF model
def selected_topics(model, vectorizer, top_n=10):
    """
    Given a fitted LSA or NMF model and vectorizer, this function prints the top topics by feature weights.
    """
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

# Code to extract named entities from a dataframe column and add them to a new dataframe
entities = []
for text in nlp.pipe(iter(bestdf['Description']), batch_size = 1000, n_threads=-1):  
  entities.append([(X.text, X.label_) for X in text.ents])
  print([(X.text, X.label_) for X in text.ents])

# Code to populate a new dataframe with named entities
index=range(0,13911)
full_df = pd.DataFrame(index = index, columns=['DOC NR','CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'])

for index, doc in enumerate(entities):
    for entity, entity_type in doc:
        if pd.isna(full_df.at[index, entity_type]):
            full_df.at[index, entity_type] = entity
        else:
            full_df.at[index, entity_type] = full_df.at[index, entity_type] + ", {}".format(entity)