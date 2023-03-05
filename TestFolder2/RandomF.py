# Function to extract common nouns
# Documentation: This function takes in a spacy document object and extracts common nouns from it. It removes stop words and punctuation before extracting the nouns.

def extract_common_nouns(doc):
    nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == 'NOUN']
    word_freq = Counter(nouns)
    common_nouns = word_freq.most_common(10)
    return common_nouns


# Function to sanitize names
# Documentation: This function takes in a text string and replaces all recognized PERSON entities in it with "[REDACTED]".

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


# Function to sanitize locations
# Documentation: This function takes in a text string and replaces all recognized GPE (Geo-Political Entity) entities in it with "[REDACTED]".

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


# Function to tokenize text using spaCy
# Documentation: This function takes in a sentence and tokenizes it using spaCy. It then lemmatizes words, removes stop words and punctuation before returning the tokens.

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


# Function to extract top topics from a given LDA model
# Documentation: This function takes in an LDA model and a vectorizer and extracts the top N topics based on importance.

def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])


# Function to extract named entities and create a dataframe
# Documentation: This function takes in a dataframe and extracts named entities from the 'Description' column using spaCy. It then creates a new dataframe with different columns for each type of named entity recognized, and fills them out with the corresponding entities from the original dataframe.

entities = []
for text in nlp.pipe(iter(bestdf['Description']), batch_size = 1000, n_threads=-1):  
  entities.append([(X.text, X.label_) for X in text.ents])

index = range(0,13911)
full_df = pd.DataFrame(index=index, columns=['DOC NR','CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'])

for index, doc in enumerate(entities):
    for entity, entity_type in doc:
        if pd.isna(full_df.at[index, entity_type]):
            full_df.at[index, entity_type] = entity
        else:
            full_df.at[index, entity_type] = full_df.at[index, entity_type] + ", {}".format(entity)