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
    vectorized = [char2index(char) for char in normalize(sentence)]
    padded = sequence.pad_sequences([vectorized], maxlen=100)
    prediction = model.predict(padded, batch_size=1)
    return float(prediction)

def test_model():
    dutch = ['Een man loopt op straat.', 'Ik zie een peuter met een schep.', 'Hij staat in de keuken.']
    english = ['A man crossing the street', "A toddler with a shovel.", "He's in the kitchen."]
    garbage = ['aasdg gdasdf asdf', 'asd trh afd asg', 'sdfasdf']
    
    for sentence in dutch + english + garbage:
        score= score_sentence(sentence)
        print(score, '\t' + sentence)
