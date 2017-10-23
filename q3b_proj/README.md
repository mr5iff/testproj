# Q3b. Tagging prediction

We created a Keras RNN classifer which can classify an article (with html) into 3 possible tags given (梁振英, 足球, 美國大選).

## Install
````
python3 -m venv env
source env/bin/activate
pip3 install -r requirement.txt
````

## Usage Example (example.py)
````python
# -*- coding: utf-8 -*-
from api.get_classifer import ArticleClassifer

clf = ArticleClassifer()
pred_str = clf.predict_tag('利物浦重賽擊敗乙組仔　英足盃過關 英格蘭足總盃第三圈今晨重賽，貴為英超勁旅的利物浦上場被乙組仔埃克斯特尷尬逼和，多獲一次機會的紅軍不敢再有差池。先有近期回勇的「威爾斯沙維」祖阿倫10分鐘開紀錄，加上兩個小將舒爾奧祖，及祖奧迪西拿下半場各入一球，以3比0擊敗對手，總算在主場挽 <p style=""text-align: justify;"">英格蘭足總盃第三圈今晨重賽，貴為英超勁旅的利物浦上場被乙組仔埃克斯特尷尬逼和，多獲一次機會的紅軍不敢再有差池。先有近期回勇的「威爾斯沙維」祖阿倫10分鐘開紀錄，加上兩個小將舒爾奧祖，及祖奧迪西拿下半場各入一球，以3比0擊敗對手，總算在主場挽回面子，下一圈對手為韋斯咸。</p> <p style=""text-align: justify;"">另一場英超球隊對壘，今季異軍突起的李斯特城戰至第三圈就宣告畢業。熱刺憑韓國前鋒孫興<U+615C>上半場遠射破網先開紀錄，換邊後此子助攻予中場查迪尼建功，令球隊以兩球輕取李斯特城，第四圈將面對英甲的高車士打。</p>')

print(pred_str) # '足球'
````

## Useful files
* `model_final.ipynb`: the training notebook of the final model (quite messy...)
* `output/testset_with_tags.csv`: offsite-tagging-test-set (1).csv with the tags predicted by the model.
* `api/get_classifer.py`: wrapped the trained model in a ArticleClassifer class
* `example.py`: example of how to use the ArticleClassifer class
* `api/model_final.h5`: the Keras model of the best run. Can be loaded by `keras.models.load_model('./model/model_final.h5')`

---

## Discussion

### 1. How well does your model perform?
Around 91.0% validation accuracy after 20 epochs. Also see output/testset_with_tags.csv for the predicted tags of the test set.

### 2. How did you choose the parameters of the final model?
[hyperas](https://github.com/maxpumperla/hyperas) is used for tuning the hyperparameters.

hyperparameters being tuned:

* `batch_size`: choice([128, 256, 512]) -> 256
* `lstm_units`: choice([64, 128, 256, 512]) -> 256
* `dense_units`: choice([64, 128, 256, 512]) -> 64
* `optimizer`: choice(['rmsprop', 'adam', 'adagrad', 'nadam', 'adadelta']) -> 'adam'

MAX_WORD_COUNT = 150 (hand picked)
EMBEDDING_SIZE = 64 (size of the word embedding vector, using the pre-trained chinese word embedding from polyglot. We can train our own word embedding vectors with better localization if there is more training text and time)

### 3. On a high level, please explain your final model’s structure, and how it predicts tags from the article text
#### Transform raw data
* text column (features): raw text with html -> text only -> tokenize the article into word tokens (by [jieba](https://github.com/fxsjy/jieba)) -> word embedding (from [polyglot](https://github.com/aboSamoor/polyglot)) (first 150 words, embedding size 64)
* tags (labels): raw tags string -> label index (e.g. 0, 1, 2) -> one hot encoding (e.g. [1, 0, 0], [0, 1, 0], [0, 0, 1])

The input data is wrapped in class `TextClassificationDataSet`.

#### Graph
word embedding (first 150 words) -> a RNN with LSTM cells and 3 fully connected layers -> softmax of the 3 tags

We use [hyperas](https://github.com/maxpumperla/hyperas) to search for the best hyperparameters (best_run), and then train the model with 20 epochs.