# encoding=utf8
import fasttext
import jieba
import re


def fasttext_predic(inquiry):
    classifier = fasttext.load_model("/home/ff/PycharmProjects/goods_type/goods_type/model/classifier.model.bin",
                                     label_prefix="__label__")
    seg_str = re.sub('[a-zA-Z0-9’!"#$%&\'()（）*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', '',
                     ' '.join(jieba.cut(inquiry)))
    print(seg_str)
    t = []
    t.append(seg_str)
    type = classifier.predict_proba(t)
    type = str(type[0][0][0])
    t.clear()
    return type
