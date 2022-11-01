import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import string as string_lib



print("Tensorflow Version: ", tf.__version__) # Tested with Tensorflow 2.9.2
# Disable GPU
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



# Data preprocessing

print("Data loading...")

raw_data = pd.read_csv('go_emotions_dataset.csv', sep=',', header=None).to_numpy()

keys = np.delete(raw_data[0], [0, 1, 2], 0)
data = np.delete(raw_data[1:], [0, 2], 1)


#print(keys)
#print(data)

def preprocess(string):
    # Strip URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    string = url_pattern.sub(r'', string) 

    string = re.sub("\S*@\S*\s?", "", string)     # Strip Emails

    string = re.sub("\s+", " ", string)           # Strip newlines

    #string = re.sub("\'", "", string)            # Strip Single Quotes
    
    #string = re.sub("\s..\s", " ", string)         # Strip words with 2 letters
    #string = re.sub("\s.\s", " ", string)          # Strip words with 1 letters
    
    string = string.translate(str.maketrans('', '', string_lib.punctuation)) # Strip Punctuation
    
    string = string.lower()
    
    return string

#X = np.array([preprocess(x[0]) for x in data])
#y = np.delete(data, 0, 1).astype(float)

X = []
y = []

for i in range(len(data)):
    if(data[i][-1] != 1): # don't include "neutral" to avoid overfitting
        X.append(preprocess(data[i][0]))
        y.append(data[i][1:])

X = np.array(X).astype(str)
y = np.array(y).astype(float)
text_dataset = tf.data.Dataset.from_tensor_slices(X) # for adapting
        
print(float(len(y))/len(data))

print("Data loaded.")



# Create Model

max_words = 20000 # number of words to tokenize
max_len = 300     # We allow up to 300 words per string. The largest in our dataset (after preprocessing) is 703 words

tokenizer = keras.layers.TextVectorization( # Vectorize Layer tokenizes words
 max_tokens=max_words,
 output_mode='int',
 output_sequence_length=max_len)

tokenizer.adapt(text_dataset.batch(64)) # adapt to the dataset of words


model = keras.models.Sequential()

model.add(tf.keras.Input(shape=(1,), dtype=tf.string))         # Takes a single string as an input
model.add(tokenizer);                                          # The tokenizer. String -> Vector
model.add(keras.layers.Embedding(max_words, 300))
model.add(keras.layers.Conv1D(260, 8, activation="relu"))      # Hidden sliding window. Dropout to reduce overfitting
model.add(keras.layers.MaxPooling1D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation="tanh"))          # Hidden Layer
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(len(keys), activation="softmax")) # Output Layer

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

if(os.path.isdir("autoSave")):
   print("Loading saved model...")
   model = keras.models.load_model("autoSave")
   print("Loaded.")

checkpoint = tf.keras.callbacks.ModelCheckpoint("autoSave")
# model.fit(X, y, epochs=100, batch_size=10, callbacks=[checkpoint])

def test(string): # Finds the top prediction
    return (keys[np.argmax(model.predict([preprocess(string)]))], np.max(model.predict([preprocess(string)]))) # returns (emotion, percentage certainty)

def guessEmotion(string): # Finds the top 3 predictions
    prediction = model.predict([preprocess(string)]).flatten()
    for i in range(3):
        best = np.argmax(prediction)
        bestPercent = np.max(prediction)
        print("Emotion ", i+1, ": ", keys[best], " (", bestPercent*100, "%)")
        prediction[best] = -100 # Look for next-best prediction on the next loop pass
    

print("Use `guessEmotion('string')` to predict the most prominent emotion(s) in the string.")