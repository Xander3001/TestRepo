

nouns = [ token.text for token in doc if token.is_stop != True and token.is_punct !=True and token.pos_ == 'NOUN']
word_freq = Counter(nouns)
common_nouns = word_freq.most_common(10)
print(common_nouns)


verbs = [ token.text for token in doc if token.is_punct !=True and token.pos_ == 'VERB']
print(Counter(verbs).most_common(10))


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
  
  
  
  
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 


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

The code above contains different functions and scripts:

1. `nouns` function: This function extracts nouns from a given spacy `doc` object and returns the 10 most common nouns using `Counter` from the `collections` module.

2. `verbs` function: This function extracts verbs from a given spacy `doc` object and returns the 10 most common verbs using `Counter` from the `collections` module.

3. `sanitize_names` function: This function takes in a string of text and redacts all PERSON entities (people's names) using "[REDACTED]" and returns the sanitized text.

4. `sanitize_locations` function: This function takes in a string of text and redacts all GPE entities (locations) using "[REDACTED]" and returns the sanitized text.

5. `spacy_tokenizer` function: This function tokenizes a given sentence using spacy's default parser, lemmatizes the words, removes stop words and punctuations, and returns the tokens joined as a string.

6. `selected_topics` function: This function takes in a trained LDA model and a vectorizer, and prints the top N words for each topic in the model.

7. `entities` script: This script extracts entity information (text and label) from each document in a pandas dataframe using spacy's `nlp.pipe` method and appends the results to a list.

8. `full_df` script: This script initializes an empty dataframe with specific columns and index, and fills the dataframe with entity information extracted from each document using the `entities` list. If there are multiple entities with the same type in a document, they are concatenated separated by a comma.