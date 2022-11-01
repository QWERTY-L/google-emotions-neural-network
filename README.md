# GoEmotions Neural Network
A neural network to detect emotions using google's GoEmotions dataset

## Data

The data is from here: https://www.kaggle.com/datasets/debarshichanda/goemotions

It was preprocessed using a custom preprocessor (`preprocess(string)`). In the future, it would be nice to add lemmatization to the preprocessing. The data has more of some emotions than others, which causes some bias. A dropout was used to mitigate this memorization.


## Model Structure

- Input Layer: Tokenized string
- 1D Convolutional Layer: kernal size of 8 (looks at 8 words at a time); filter size of 260
- Traditional Dense Layer: 100 Neurons
- Output layer: 28-element array of probabilities (softmax). Maps to each of the 28 emotions present in this dataset.


## Usage

To train, use `model.fit(X, y, epochs=20, batch_size=10, callbacks=[checkpoint])`.


To detect the emotions in a phrase, use `guessEmotion("TEXT")`
