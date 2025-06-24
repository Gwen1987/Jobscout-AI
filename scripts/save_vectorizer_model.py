from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import pickle
from tensorflow.keras.layers import TextVectorization

# === Load config
with open("models/jobscout_vectorizer_config.pkl", "rb") as f:
    config = pickle.load(f)

# === Load and clean vocabulary
with open("models/jobscout_vectorizer_vocab.txt", "r", encoding="utf-8") as f:
    raw_vocab = f.read().splitlines()

seen = set()
vocab = []
for word in raw_vocab:
    if word and word not in seen:
        seen.add(word)
        vocab.append(word)

# === Rebuild vectorizer
vectorizer = TextVectorization.from_config(config)
vectorizer.set_vocabulary(vocab)

# === Wrap and save as Keras model
input_text = Input(shape=(1,), dtype="string", name="text_input")
output_vector = vectorizer(input_text)
vectorizer_model = Model(inputs=input_text, outputs=output_vector)

vectorizer_model.save("models/jobscout_vectorizer.keras")
print("✅ Saved cleaned vectorizer as models/jobscout_vectorizer.keras")
