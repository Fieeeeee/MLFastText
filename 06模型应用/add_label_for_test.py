import fasttext
import datetime
import itertools

start_load_time = datetime.datetime.now()
classifier = fasttext.load_model("best_classifier.model.bin", label_prefix="__label__")
end_load_time = datetime.datetime.now()
print('load model over,time is', (end_load_time - start_load_time).seconds, 's')
result = classifier.test('../data/train_added_label.tsv')
print("this model's precision is", '{:.2%}'.format(result.precision))

start_time = datetime.datetime.now()
with open('../data/seged_test.tsv', encoding='utf-8') as seged_test, \
        open('../data/test_label.txt', 'w') as label_test:
    cnt = 0
    for r in seged_test.readlines():
        t = []
        t.append(r)
        type = classifier.predict_proba(t)
        type = str(type[0][0][0])
        label_test.write(type + '\n')
        t.clear()
add_label_time = datetime.datetime.now()
print('predict label over, time is', (add_label_time - start_time).seconds, 's')

with open("../data/test.tsv", encoding='utf-8') as f:
    test_item = [r.rstrip('\n') for r in f.readlines()]
with open("../data/test_label.txt", encoding='utf-8') as f:
    test_label = [r.rstrip('\n') for r in f.readlines()]
result = itertools.zip_longest(test_item, test_label, fillvalue=' ')
with open("../data/test_added_label.tsv", "w", encoding='utf-8') as f:
    [f.write('\t'.join(r) + "\n") for r in result]
end_time = datetime.datetime.now()
print('merge over,time is', (end_time - add_label_time).seconds, 's')
