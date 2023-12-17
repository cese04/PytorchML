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




