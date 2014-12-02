import numpy as np
import pandas as pd
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold

def QuickConvert(iterable):

	frame = pd.DataFrame(columns=['content','tag'])
	for line in iterable:
		json_object = json.loads(line)
		try:
			new_data = pd.DataFrame({'content':json_object['interaction']['content'],
									'tag': json_object['interaction']['label']},index=[json_object['interaction']['id']])
			frame = frame.append(new_data)
		except:
			continue
	return frame

def ReturnMins(result_list,n):
	'''Return the an index of the n lowest probability labels'''
	decided_prob = [max(result_list[x,:]) for x in xrange(0,result_list.shape[0])]
	mins = []
	for i,x in enumerate(decided_prob):
		if i < n:
			mins.append(min(decided_prob))
			decided_prob.remove(min(decided_prob))
		else:
			continue
	#generate location of the minimums and return the list
	decided_prob = [max(result_list[x,:]) for x in xrange(0,result_list.shape[0])]
	min_index = [decided_prob.index(x) for x in mins]
	return min_index

# Generate data and initial seed index

# filename=sys.argv[1]
with open('./fixtures/airlines.json') as f:
	json_raw = f.readlines()

data = QuickConvert(json_raw)
eval_set = np.random.choice(data.index,size=100,replace=False)
eval_data = data.loc[eval_set]
data.drop(eval_set,inplace=True)

seed = np.random.choice(data.index,size=50,replace=False)
seed_index = [data.index.get_loc(x) for x in seed]

# Generate master tfidf for use then transform the evaluation set.

tfidf_transformer = TfidfVectorizer(stop_words='english',ngram_range=(1,3),max_features=3000)
global_tfidf = tfidf_transformer.fit_transform(data['content'])
eval_tfidf = tfidf_transformer.transform(eval_data['content'])

clf = MultinomialNB().fit(global_tfidf[seed_index], data.loc[seed]['tag'])
results = clf.predict_proba(global_tfidf)
mindex = ReturnMins(results,50)

print "Score on mins:",clf.score(global_tfidf[mindex],data.iloc[mindex]['tag']),"Maximum Training Score::",clf.score(global_tfidf,data['tag'])
print "Evaluation Data score:",clf.score(eval_tfidf,eval_data['tag'])
data['Predicted'] = clf.predict(global_tfidf)



# X_new_tfidf = tfidf_transformer.transform(test_data["content"])

# predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(data["content"],predicted):
# 	print('%r => %s' % (doc,category))

