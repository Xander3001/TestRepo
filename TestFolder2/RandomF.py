# This code selects all the nouns from a Spacy analyzed "doc" that are not stop words or punctuation marks, and creates a list of them called "nouns".
# It then uses Counter from the collections module to count the frequency of each noun and creates a dictionary of the most common nouns. 
# It then prints the ten most common nouns. 

nouns = [ token.text for token in doc if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN']
word_freq = Counter(nouns)
common_nouns = word_freq.most_common(10)
print(common_nouns)


# This code selects all the verbs from a Spacy analyzed "doc" that are not punctuation marks, and creates a list of them called "verbs".
# It then uses Counter from the collections module to count the frequency of each verb and prints the ten most common verbs.

verbs = [ token.text for token in doc if token.is_punct !=True and token.pos_ == 'VERB']
print(Counter(verbs).most_common(10))


# This function receives a string of text and uses Spacy to merge named entities (such as person or location) into a single token.
# It creates a list of tokens from the text, and if the entity type is "PERSON", it replaces the token with the string "[REDACTED]".
# Finally, it joins the tokens back into a string and returns it.

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
  

# This function receives a string of text and uses Spacy to merge named entities (such as person or location) into a single token.
# It creates a list of tokens from the text, and if the entity type is "GPE" (geopolitical entity), it replaces the token with the string "[REDACTED]".
# Finally, it joins the tokens back into a string and returns it.

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


# This function receives a sentence and uses Spacy's parser to tokenize it. 
# It then lemmatizes the words (i.e. turns them into their root form) and removes stop words and punctuation marks.
# Finally, it joins the remaining words back into a string and returns it.

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


# This function receives a topic modeling model, a vectorizer, and an integer "top_n".
# It loops through each topic in the model, prints the topic number, and the top_n words and their weights in that topic.

def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 


# This code creates an empty Pandas dataframe with columns for named entity types (e.g. PERSON, ORG, LOC) and an index range of 0 to 13910.
# It then loops through each entity in each text in the "bestdf" dataframe after analyzing with Spacy, and appends the entity and its type to the corresponding row and column of the dataframe.


entities = []
for text in nlp.pipe(iter(bestdf['Description']), batch_size = 1000, n_threads=-1):  
  entities.append([(X.text, X.label_) for X in text.ents])
  print([(X.text, X.label_) for X in text.ents])


index=range(0,13911)
full_df = pd.DataFrame(index = index, columns=['DOC NR','CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE', 'LANGUAGE', 'LAW', 'WORK_OF_ART', 'EVENT', 'PRODUCT', 'LOC', 'GPE', 'ORG', 'FAC', 'NORP', 'PERSON'])


for index, doc in enumerate(entities):
    for entity, entity_type in doc:
        if pd.isna(full_df.at[index, entity_type]):
            full_df.at[index, entity_type] = entity
        else:
            full_df.at[index, entity_type] = full_df.at[index, entity_type] + ", {}".format(entity)