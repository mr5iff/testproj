# Q3b. Tagging prediction

## Install
````
python3 -m venv env
source env/bin/activate
pip3 install -r requirement.txt
````


## Work files
* `model_final.ipynb`: final model
* `output/testset_with_tags.csv`: offsite-tagging-test-set (1).csv with the tags predicted by the model.
* `model/model_final.h5`: the Keras model of the best run. Can be loaded by `keras.models.load_model('./model/model_final.h5')`

## 1. How well does your model perform?
Around 91.0% validation accuracy after 20 epochs. Also see output/testset_with_tags.csv for the predicted tags of the test set.

## 2. How did you choose the parameters of the final model?
[hyperas](https://github.com/maxpumperla/hyperas) is used for tuning the hyperparameters.

hyperparameters being tuned:

* `batch_size`: choice([128, 256, 512]) -> 256
* `lstm_units`: choice([64, 128, 256, 512]) -> 256
* `dense_units`: choice([64, 128, 256, 512]) -> 64
* `optimizer`: choice(['rmsprop', 'adam', 'adagrad', 'nadam', 'adadelta']) -> 'adam'

## 3. On a high level, please explain your final model’s structure, and how it predicts tags from the article text
### Transform raw data
* text column (features): raw text with html -> text only -> tokenize the article into word tokens (by [jieba](https://github.com/fxsjy/jieba)) -> word embedding (from [polyglot](https://github.com/aboSamoor/polyglot)) (first 150 words, embedding size 64)
* tags (labels): raw tags string -> label index (e.g. 0, 1, 2) -> one hot encoding (e.g. [1, 0, 0], [0, 1, 0], [0, 0, 1])

The input data is wrapped in class `TextClassificationDataSet`.

### Graph
word embedding (first 150 words) -> a RNN with LSTM cells and 3 fully connected layers -> softmax of the 3 tags

We use [hyperas](https://github.com/maxpumperla/hyperas) to search for the best hyperparameters (best_run), and then train the model with 20 epochs.