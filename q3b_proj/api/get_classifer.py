# -*- coding: utf-8 -*-
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
import numpy as np
from bs4 import BeautifulSoup

from polyglot.text import Text
from polyglot.mapping import Embedding
import os.path

# paths for the model etc
package_directory = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(package_directory, 'model_final.h5')
CLASSES_PATH = os.path.join(package_directory, 'tag_classes.npy')
WORD_EMBEDDING_PATH = os.path.join(package_directory, 'polyglot/embeddings2/zh/embeddings_pkl.tar.bz2')

# fixed as per the trained model
MAX_WORD_COUNT = 150
EMBEDDING_SIZE = 64


class ArticleClassifer(object):
    def __init__(self, model_path=MODEL_PATH, classes_path=CLASSES_PATH, word_embedding=WORD_EMBEDDING_PATH):
        self.model = load_model(model_path)
        self.classes_ = np.load(classes_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = self.classes_

        self.embeddings = self._load_word_embeddings(word_embedding)

    def predict_tag(self, input_text):
        """
        Arg:
            input_text<str>: the full article to be classified (could be mixed with html)

        Aeturns:
            tag<str>: the tag being predicted
        """
        clean_text = self._get_clean_text(input_text)
        word_embeddings = self._article2vecs_simple(clean_text, embeddings=self.embeddings, max_word_count=MAX_WORD_COUNT)
        pred = self.model.predict_classes(word_embeddings)
        return str(self.label_encoder.inverse_transform(pred)[0])

    def _parse_text(self, text):
        """convert text to polyglot parsed Text object"""
        if isinstance(text, str):
            text_parsed = Text(text)
        else:
            text_parsed = text
        return text_parsed

    def _get_clean_text(self, text):
        """remove html tags in text"""
        return BeautifulSoup(text, "html5lib").text

    def _article2vecs_simple(self, article_text, embeddings, max_word_count):
        """convert article to word embedding vectors"""
        if isinstance(article_text, str):
            article_parsed = self._parse_text(article_text)

        sentences_words_embedding = sequence.pad_sequences([[embeddings.get(word) for word in article_parsed.words if embeddings.get(word) is not None]], maxlen=max_word_count, truncating='post', dtype='float32')
        return sentences_words_embedding

    def _load_word_embeddings(self, word_embedding):
        """load polyglot word embedding (chinese)"""
        if isinstance(word_embedding, Embedding):
            return word_embedding
        else:
            return Embedding.load(word_embedding)


if __name__ == '__main__':
    """usage example"""
    clf = ArticleClassifer()
    pred_str = clf.predict_tag('利物浦重賽擊敗乙組仔　英足盃過關 英格蘭足總盃第三圈今晨重賽，貴為英超勁旅的利物浦上場被乙組仔埃克斯特尷尬逼和，多獲一次機會的紅軍不敢再有差池。先有近期回勇的「威爾斯沙維」祖阿倫10分鐘開紀錄，加上兩個小將舒爾奧祖，及祖奧迪西拿下半場各入一球，以3比0擊敗對手，總算在主場挽 <p style=""text-align: justify;"">英格蘭足總盃第三圈今晨重賽，貴為英超勁旅的利物浦上場被乙組仔埃克斯特尷尬逼和，多獲一次機會的紅軍不敢再有差池。先有近期回勇的「威爾斯沙維」祖阿倫10分鐘開紀錄，加上兩個小將舒爾奧祖，及祖奧迪西拿下半場各入一球，以3比0擊敗對手，總算在主場挽回面子，下一圈對手為韋斯咸。</p> <p style=""text-align: justify;"">另一場英超球隊對壘，今季異軍突起的李斯特城戰至第三圈就宣告畢業。熱刺憑韓國前鋒孫興<U+615C>上半場遠射破網先開紀錄，換邊後此子助攻予中場查迪尼建功，令球隊以兩球輕取李斯特城，第四圈將面對英甲的高車士打。</p>')
    assert type(pred_str) == str
    assert pred_str == '足球'
    assert pred_str != '梁振英'
