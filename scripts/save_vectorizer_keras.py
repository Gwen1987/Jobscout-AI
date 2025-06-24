import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization


# Load the vectorizer from the pickled config + vocab (or from memory)
with open("models/jobscout_vectorizer_config.pkl", "rb") as f:
    vectorizer_config = pickle.load(f)

with open("models/jobscout_vectorizer_vocab.txt", "r", encoding="utf-8") as f:
    vocab = f.read().splitlines()

    # Read and clean vocab
with open("models/jobscout_vectorizer_vocab.txt", "r", encoding="utf-8") as f:
    vocab = f.read().splitlines()

# Remove duplicates and empty strings
vocab = sorted(set([word for word in vocab if word.strip() != ""]))



# Reconstruct the vectorizer from saved config and vocab
vectorizer = TextVectorization.from_config(vectorizer_config)
vectorizer.set_vocabulary(vocab)

# Wrap it in a model
vec_input = Input(shape=(1,), dtype="string")
vec_output = vectorizer(vec_input)
vectorizer_model = Model(inputs=vec_input, outputs=vec_output)

# Save the full model
vectorizer_model.save("models/jobscout_vectorizer.keras")

print("✅ Vectorizer wrapped and saved as .keras successfully.")
