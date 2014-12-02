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

# def ReturnMins(result_list,n):
# '''Return the n lowest decided labels'''
# 	decided_prob = [max(result_list[x,:]) for x in xrange(0,result_list.shape[0])]
# 	mins = []
# 	for i,x in enumerate(decided_prob):
# 		if i < n:
# 			mins.append(min(decided_prob))
# 			decided_prob.remove(min(decided_prob))
# 		else:
# 			continue
# 	#generate location of the minimums and return the list
# 	decided_prob = [max(result_list[x,:]) for x in xrange(0,result_list.shape[0])]
# 	min_index = [decided_prob.index(x) for x in mins]

# filename=sys.argv[1]
with open('./fixtures/airlines.json') as f:
	json_raw = f.readlines()

data = QuickConvert(json_raw)
size = len(data.index)

seed = np.random.choice(size,size=50,replace=False)

tfidf_transformer = TfidfVectorizer()
global_tfidf = tfidf_transformer.fit_transform(data['content'])

clf = MultinomialNB().fit(global_tfidf[seed], data.iloc[seed]['tag'])
results = clf.predict_proba(global_tfidf)


#Return the 50 lowest decided labels
decided_prob = [max(results[x,:]) for x in xrange(0,results.shape[0])]
mins = []
for i,x in enumerate(decided_prob):
	if i < 50:
		mins.append(min(decided_prob))
		decided_prob.remove(min(decided_prob))
	else:
		continue

for x in mins:
	print x
#generate location of the minimums and return the list
decided_prob = [max(results[x,:]) for x in xrange(0,results.shape[0])]
min_index = [decided_prob.index(x) for x in mins]
print "Score on mins:",clf.score(global_tfidf[min_index],data.iloc[min_index]['tag']),"Global Score:",clf.score(global_tfidf,data['tag'])
data['Predicted'] = clf.predict(global_tfidf)
print data.loc[min_index]



# X_new_tfidf = tfidf_transformer.transform(test_data["content"])

# predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(data["content"],predicted):
# 	print('%r => %s' % (doc,category))

