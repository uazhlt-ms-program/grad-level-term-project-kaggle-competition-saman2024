import nltk
from nltk.tag import HiddenMarkovModelTagger
import train_test_split

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load your dataset
data = pd.read_csv('train.csv')

# Handle missing values
data.dropna(subset=['TEXT'], inplace=True)
data['TEXT'].fillna('', inplace=True)

# Split data into train and test sets with a different random state
X_train, X_test, y_train, y_test = train_test_split(data['TEXT'], data['LABEL'], test_size=0.1, random_state=72)

# Train the Hidden Markov Model (HMM) POS tagger
tagger = HiddenMarkovModelTagger.train([nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in X_train])

# Evaluate the model
accuracy = tagger.evaluate([nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in X_test])
print("Accuracy (Viterbi algorithm):", accuracy)