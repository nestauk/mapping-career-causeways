import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import mapping_career_causeways.text_cleaning_utils as text_cleaning_utils

def tfidf_keywords(p, dataframe, text_field, stopwords, N=10):
    """
    Fast method to generate keywords characterising each cluster

    Parameters
    ----------
    p (list or nd.array):
        Cluster integer labels
    dataframe (pandas.DataFrame):
        Dataframe with information about the clustered nodes.
    text_field (string):
        Column name of the 'dataframe' that contains the text corpus
        to be used for keyword extraction.
    stopwords (list of strings):
        Specific words which should be excluded from the text corpus.
    N (int)
        Number of keywords to use

    Returns
    -------
    tfidf_keywords (list of strings):
        Strings containing cluster keywords
    tfidf_keywords_ (list of strings):
        Strings containing the cluster number and cluster keywords
    """

    # Collect text for each cluster & remove custom stopwords
    cluster_text = []
    for c in range(len(np.unique(p))):
        t=" ".join(dataframe.loc[p==c][text_field].to_list())
        for stopword in stopwords:
            t=re.sub(stopword,'',t)
        cluster_text.append(t)

    # Further clean the text (see 'text_cleaning_utils' for more details)
    clust_descriptions_clean = []
    for descr in cluster_text:
        text = text_cleaning_utils.clean_text(descr)
        text = text_cleaning_utils.remove_stopwords(text)
        text = text_cleaning_utils.lemmatise(text)
        clust_descriptions_clean.append(text)

    # Find keywords using tf-idf vectors
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(clust_descriptions_clean)
    names = vectorizer.get_feature_names()
    Data = vectors.todense().tolist()

    # Create a dataframe with the results
    df = pd.DataFrame(Data, columns=names)

    tfidf_keywords = []
    for i, row in df.iterrows():
        tfidf_keywords.append(row.sort_values(ascending=False)[:N].index.to_list())

    # Final outputs: string with the cluster number and N keywords per cluster
    tfidf_keywords_ = ["["+str(i)+"] "+", ".join(x) for i, x in enumerate(tfidf_keywords)]

    # Final outputs: string with N keywords per cluster
    tfidf_keywords = [", ".join(x) for i, x in enumerate(tfidf_keywords)]

    return tfidf_keywords, tfidf_keywords_
