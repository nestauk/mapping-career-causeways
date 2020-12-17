import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()

# Load stopwords from nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

def clean_text(text):
    """
    Removes apostrophes, non-alphabet letters, newlines and sets
    everything to lowercase

    Parameters
    ----------
    text (string):
        Input text string

    Returns
    -------
    text (string):
        Cleaned text string
    """

    # Remove apostrophes
    text = re.sub("\'", "", text)
    # Leave only alphabet letters
    text = re.sub("[^a-zA-Z]"," ", text)
    # Remove new-lines
    text = re.sub("\\n", "", text)
    # Convert to lower-case
    text = text.lower()
    return text

def remove_stopwords(text):
    """
    Uses the provided list of stopwords and removes them from the input text string
    """
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def lemmatise(text):
    """
    Lemmatises text using WordNetLemmatizer from nltk
    """
    return " ".join(lemma.lemmatize(word) for word in text.split())


def clean_table(df, id_column_name=None, id_prefix=''):

    """
    Cleans up a dataframe by setting all text to lower case and
    replacing spaces in column titles with underscores;
    can also create a new id column (supposed to be used as the key).

    Parameters
    ----------
    df (DataFrame):
        Dataframe to be cleaned
    id_column_name (str):
        Name of the id column (if None, then no new column is created)
    id_prefix (str):
        Prefix to add to the id number

    Returns:
    --------
    df (DataFrame):
        Cleaned dataframe
    """

    # Create a new column with IDs
    if id_column_name:
        df[id_column_name] = df.index.values
        df[id_column_name] = df[id_column_name].apply(lambda x: id_prefix + str(x))

    # Clean column names
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()

    # Clean column text
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    return df

def split_titles(text):
    """
    Splits text that is divided by slashes
    """
    text = text.split('/')
    text = [re.sub("\n","",t.strip()) for t in text]
    return text
