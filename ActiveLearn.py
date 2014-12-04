import numpy as np
import pandas as pd
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

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

def cross_validate_list(X, y,transformer, model, size,n):
    "Scores classifier using kfold cross_validation"
    # derive a set of (random) training and testing indices
    k_fold_indices = StratifiedShuffleSplit(y, n_iter=n,test_size=size, random_state=0)
    k_score_total = []
    # train and score classifier for each slice
    for train_slice, test_slice in k_fold_indices:
    	k_train_tfidf = transformer.fit_transform(X.iloc[train_slice])
        k_test_tfidf = transformer.transform(X.iloc[test_slice])
        new_model = model.fit(k_train_tfidf,y.iloc[train_slice])
        y_pred = new_model.predict(k_test_tfidf)
        y_true = y.iloc[test_slice]
        k_score = f1_score(y_true,y_pred,average='micro')
        k_score_total.append(k_score)

    # return the average accuracy
    return np.array(k_score_total)

def ReturnMins(results_list,data_index,n, master_list):
	'''Return the an index of the n lowest probability labels'''
	results_frame = pd.DataFrame(data=results_list,index=data_index).max(axis=1)
	
	results_frame.drop(results_frame.index[master_list],inplace=True)
	if len(results_frame) < n:
		return [data_index.get_loc(x) for x in results_frame.index]
	min_locations = []
	for i in xrange(0,n):
		min_locations.append(results_frame.idxmin())
		results_frame.drop(results_frame.idxmin(),inplace=True)

	min_index = [ data_index.get_loc(x) for x in min_locations ]
	return min_index


# Generate data and initial seed index

# filename=sys.argv[1]
with open('./fixtures/airlines.json') as f:
	json_raw = f.readlines()

data = QuickConvert(json_raw)



eval_ix_labels = np.random.choice(data.index,size=100,replace=False)
eval_data = data.loc[eval_ix_labels]
data.drop(eval_ix_labels,inplace=True)



# Generate master tfidf for use then transform the evaluation set.
# for n in [100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
	
# 	l = RandomIndex(data)
# 	for iteration in xrange(1,100):

def ActiveLearner(frame,text_column,label_column,seed_size,interval_size,shuffle_iterations,eval_size,cv_folds,cv_shuffle_size):
	'''There are a lot of variables...'''
	shuff_set = StratifiedShuffleSplit(frame.index)
	tenth_size = len(shuff_set)/10*-1
	eval_set = shuff_set[tenth_size:]
	eval_frame = frame.loc[eval_set]
	rand_source = np.delete(shuff_set,np.s_[tenth_size:],None)
	frame.drop(eval_set,inplace=True)
	
	seed_index = [ frame.index.get_loc(x) for x in shuff_set[0:50]]
	rand_index = [ frame.index.get_loc(x) for x in shuff_set[0:50]]
	
	tfidf_transformer = TfidfVectorizer(stop_words='english',ngram_range=(1,4),max_features=4000)
	global_tfidf = tfidf_transformer.fit_transform(frame[str(text_column)])
	active_clf = MultinomialNB().fit(global_tfidf[seed_index], frame.iloc[seed_index][str(label_column)])
	random_clf = MultinomialNB().fit(global_tfidf[rand_index], frame.iloc[rand_index][str(label_column)])
	master_list = [ frame.index.get_loc(x) for x in shuff_set[0:50]]
	for i in xrange(1,30):
		cv_list = cross_validate_list(frame[str(text_column)], frame[str(label_column)],tfidf_transformer, MultinomialNB(),.1,10)
		print "mean:",np.mean(cv_list), "var:", np.var(cv_list)
		results = active_clf.predict_proba(global_tfidf)
		mindex = ReturnMins(results,frame.index,40, master_list)
		for x in mindex:
			master_list.append(x)
		random_start = 50 + (i-1)*40
		
		if random_start > len(rand_source):
			break
		random_end = 50 +i*40
		if random_end <= len(rand_source):
			pass
		else:
			random_end = len(rand_source)+1

		print "start: ", random_start, " end: ", random_end, " idx: ", len(mindex), " total ", len(master_list)

		new_randoms = [frame.index.get_loc(x) for x in rand_source[random_start:random_end] ]
		active_clf.partial_fit(global_tfidf[mindex],frame.iloc[mindex][str(label_column)])
		random_clf.partial_fit(global_tfidf[new_randoms],frame.iloc[new_randoms][str(label_column)])
		# seed_index.extend(mindex)
		# rand_index.extend(new_randoms)
		tfidf_transformer.fit(frame[str(text_column)])
		eval_tfidf = tfidf_transformer.transform(eval_frame[str(text_column)])
		act_predicted = active_clf.predict(eval_tfidf)
		rand_predicted = random_clf.predict(eval_tfidf)
		print "Active:",f1_score(eval_frame[str(label_column)],act_predicted,average='micro'),"Random:",f1_score(eval_frame[str(label_column)],rand_predicted,average='micro')
	
	return active_clf,random_clf,tfidf_transformer


Model1,Model2,Transformer = ActiveLearner(data,'content','tag')
new_tfidf = Transformer.transform(eval_data['content'])

act_predicted = Model1.predict(new_tfidf)
rand_predicted = Model2.predict(new_tfidf)
print "Eval Active:",f1_score(eval_data['tag'],act_predicted,average='micro'),"Eval Random:",f1_score(eval_data['tag'],rand_predicted,average='micro')

# print "Training F1::",f1_score(,"Score on mins:",clf.score(global_tfidf[mindex],data.iloc[mindex][str(label_column)])
# print "Evaluation Data score:",clf.score(eval_tfidf,eval_data[str(label_column)]) 
	

	# # What is the best case scenario? Trained on entire training set:
	# best_clf = MultinomialNB().fit(global_tfidf,data['tag'])
	# print "Score on model trained with all training data:",best_clf.score(eval_tfidf,eval_data['tag'])

#check out expected maximization algorithm w/ SVMs
#lins

# X_new_tfidf = tfidf_transformer.transform(test_data["content"])

# predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(data["content"],predicted):
# 	print('%r => %s' % (doc,category))

