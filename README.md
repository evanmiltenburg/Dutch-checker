# Dutch language model

This repository contains code and weights for an LSTM model that checks whether
a sequence of characters is likely to be Dutch or not. The LSTM code is mostly
copied from the Keras GitHub [here](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py).

My contribution is the application of this code to perform a language identification task.
For the mechanics behind this, see `character_data.py`. See `use_model.py` for how to use this model.

## Requirements

* Numpy
* Keras
* Theano

## Training the model

* Download the Wikipedia dumps for the languages you're interested in. (In my case: English and Dutch.)
* Use the [Wikiextractor](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor) script to extract the text.
* Rename the folders, so that you have a folder called `dutch_text` and a folder called `english_text` with the extracted texts.
* Use `plainwiki.py` to generate plain text versions.
* Run `THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python char_lstm.py` to train the model.
* Run `python use_model.py` to test.
