from bs4 import BeautifulSoup
import requests
import tarfile
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import io

# Download Reuters-21578 dataset
reuters_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
response = requests.get(reuters_url)
tar_file = tarfile.open(fileobj=io.BytesIO(response.content),
                        mode="r:gz")

# Extract the contents
extract_path = "Data\\reuters"
tar_file.extractall(extract_path)

# Define paths to the dataset
sgm_dir = extract_path

# Load the dataset
documents = []
for file_name in os.listdir(sgm_dir):
    if file_name.endswith(".sgm"):
        with open(os.path.join(sgm_dir, file_name), "r", encoding="latin-1") as file:
            content = file.read()
            documents.append(content)

# Tokenize and remove stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Load the dataset
documents = []
for file_name in os.listdir(sgm_dir):
    if file_name.endswith(".sgm"):
        with open(os.path.join(sgm_dir, file_name), "r", encoding="latin-1") as file:
            content = file.read()
            documents.append(content)

# Parse the documents as HTML
documents_parsed = BeautifulSoup(documents[0], 'html.parser')

# All the articles are between the reuters tag
articles_set = documents_parsed.find_all('reuters')

# The body of the article contains most of the information
articles = []
for doc in articles_set:
    try:
        articles.append(doc.body.string)
    except:
        pass


# Tokenize the articles and remove the stopwords
tokenized_articles = [word_tokenize(article.lower()) for article in articles]
filtered_articles = [
    [word for word in article if word.isalnum() and word not in stop_words]
    for article in tokenized_articles
]

# Display some of the articles
for i in range(3):
    print(filtered_articles[i])