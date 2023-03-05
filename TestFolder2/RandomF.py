# Extract only non-stop and non-punctuation nouns from a spacy document and create a list
nouns = [ token.text for token in doc if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN']

# Count the frequency of each noun and retrieve the 10 most common
word_freq = Counter(nouns)
common_nouns = word_freq.most_common(10)

# Print the most common nouns
print(common_nouns)

# Extract only non-punctuation verbs from a spacy document and create a list
verbs = [ token.text for token in doc if token.is_punct !=True and token.pos_ == 'VERB']

# Count the frequency of each verb and retrieve the 10 most common
print(Counter(verbs).most_common(10))

# Define a function to redact names from a spacy-processed text
def sanitize_names(text):
    doc = nlp(text)
    redacted_sentences = []
    for ent in doc.ents:
        ent.merge()
    # Replace PERSON entities with "[REDACTED]"
    for token in doc:
        if token.ent_type_ == 'PERSON':
            redacted_sentences.append("[REDACTED]")
        else:
            redacted_sentences.append(token.string)
    # Join the edited sentences into a single string and return it
    return "".join(redacted_sentences)

# Define a function to redact locations from a spacy-processed text
def sanitize_locations(text):
    doc = nlp(text)
    redacted_sentences = []
    for ent in doc.ents:
        ent.merge()
    # Replace GPE (location) entities with "[REDACTED]"
    for token in doc:
        if token.ent_type_ == 'GPE':
            redacted_sentences.append("[REDACTED]")
        else:
            redacted_sentences.append(token.string)
    # Join the edited sentences into a single string and return it
    return "".join(redacted_sentences)

# Define a function to tokenize a sentence using spacy, lemmatize words, remove stopwords and punctuation, and return cleaned tokens as a string
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    # Lemmatize words and convert pronouns to lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    # Remove stopwords and punctuation
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    # Join the cleaned tokens back into a single string and return it
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# Define a function to retrieve the top N topics from an LDA model, printing the top words for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        # Print the top N words and their weights for the current topic
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

# Extract named entities (entities with recognized types such as PERSON, ORG, GPE) from each text in bestdf['Description'] using spacy, and store them as a list of tuples in the 'entities' list
entities = []
for text in nlp.pipe(iter(bestdf['Description']), batch_size = 1000, n_threads=-1):  
  entities.append([(X.text, X.label_) for X in text.ents])
  # Print the named entities found in the current text, along with their types
  print([(X.text, X.label_) for X in text.ents])

# Create an empty dataframe with columns for each named entity type we're interested in
index=range(0,13911)
full_df = pd.DataFrame(index = index, columns=['DOC NR','CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'])

# Iterate over the list of named entities we extracted earlier, filling in the appropriate cells in the dataframe with a comma-separated list of entities of each type found in the current index
for index, doc in enumerate(entities):
    for entity, entity_type in doc:
        if pd.isna(full_df.at[index, entity_type]):
            full_df.at[index, entity_type] = entity
        else:
            full_df.at[index, entity_type] = full_df.at[index, entity_type] + ", {}".format(entity)