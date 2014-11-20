import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def QuickConvert(iterable):

	frame = pd.DataFrame(columns=['content','tag'])
	for line in iterable:
		json_object = json.loads(line)
		try:
			new_data = pd.DataFrame({'content':json_object['interaction']['content'],
									'tag': json_object['interaction']['tags']})
			frame = frame.append(new_data)
		except:
			continue
	return frame

with open('./train-fossil.json') as f:
	json_raw = f.readlines()

data = QuickConvert(json_raw)

tfidf_transformer = TfidfVectorizer()
X_train_tfidf = tfidf_transformer.fit_transform(data['content'])

with open('./test-fossil.json') as f:
	json_raw_test = f.readlines()
test_data = QuickConvert(json_raw_test)

clf = MultinomialNB().fit(X_train_tfidf, data['tag'])

X_new_tfidf = tfidf_transformer.transform(test_data["content"])

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(test_data["content"],predicted):
    print('%r => %s' % (doc,category))
