import json
from character_data import char2index, normalize
from keras.models import model_from_json
from keras.preprocessing import sequence

with open('my_model_architecture.json') as f:
    model = model_from_json(f.read())

model.load_weights('my_model_weights.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def score_sentence(sentence):
    "Score a single sentence."
    vectorized = [char2index(char) for char in normalize(sentence)[:100]]
    padded = sequence.pad_sequences([vectorized], maxlen=100)
    prediction = model.predict(padded, batch_size=1)
    return float(prediction)

def score_sentences(sentences):
    "Score multiple sentences."
    vectorized = [[char2index(char) for char in normalize(s)[:100]] for s in sentences]
    padded = sequence.pad_sequences(vectorized, maxlen=100)
    predictions = model.predict(padded, batch_size=len(sentences))
    return predictions

def test_model():
    "Test the model using some example sentences."
    dutch = ['Een man loopt op straat.', 'Ik zie een peuter met een schep.', 'Hij staat in de keuken.']
    english = ['A man crossing the street', "A toddler with a shovel.", "He's in the kitchen."]
    garbage = ['aasdg gdasdf asdf', 'asd trh afd asg', 'sdfasdf']
    
    for sentence in dutch + english + garbage:
        score= score_sentence(sentence)
        print(score, '\t' + sentence)

if __name__ == "__main__":
    test_model()
