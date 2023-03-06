This code performs several natural language processing tasks using the spaCy library. 

1. It identifies the most common nouns and verbs in a given text, as well as sanitizes any names or locations that are identified within the text.
2. The `spacy_tokenizer` function performs lemmatization, stopword and punctuation removal to tokenize a given sentence.
3. The `selected_topics` function identifies the most important topics within a given model, based on the importance of the features.
4. The code creates a pandas dataframe called `full_df` by iterating over the `Description` column in a pandas dataframe called `bestdf`. It identifies entities within the text using spaCy's named entity recognition and categorizes them by entity type. If an entity of a particular entity type already exists in the dataframe, the code adds it to the existing cell, separated by a comma.