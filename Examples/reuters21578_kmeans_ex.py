from bs4 import BeautifulSoup
import requests
import tarfile
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import io
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Download Reuters-21578 dataset
# reuters_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
# response = requests.get(reuters_url)
# tar_file = tarfile.open(fileobj=io.BytesIO(response.content),
#                         mode="r:gz")

# Extract the contents
extract_path = "Data\\reuters"
# tar_file.extractall(extract_path)

# Define paths to the dataset
sgm_dir = extract_path

# Download the punkt and stopwords package if not available
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Add 'reuter' to the stopwords as it is always present in each article
stop_words.add('reuter')


# Load the dataset
documents = []
for file_name in os.listdir(sgm_dir):
    if file_name.endswith(".sgm"):
        with open(os.path.join(sgm_dir, file_name), "r", encoding="latin-1") as file:
            content = file.read()
            documents.append(content)


articles = []
for doc in documents:
    # Parse the document as html, the relevant information is within the body tags
    doc_parsed = BeautifulSoup(doc, 'html.parser').find_all('body')

    for article in doc_parsed:
        try:
            articles.append(article.string)
        except:
            pass


# Tokenize the articles and remove the stopwords
tokenized_articles = [word_tokenize(article.lower()) for article in articles]
filtered_articles = [
    [word for word in article if word.isalnum() and word not in stop_words]
    for article in tokenized_articles
]

# Convert to text string for tf-idf feature extraction
text_articles = [" ".join(art) for art in filtered_articles]

# Display some of the articles
for i in range(3):
    print(text_articles[i])

# Transform to tf-idf matrix
vectorizer = TfidfVectorizer(max_features=5000,
                            ngram_range =(1,3),
                            lowercase=False)
tfidf_matrix = vectorizer.fit_transform(text_articles)
print(tfidf_matrix.shape)