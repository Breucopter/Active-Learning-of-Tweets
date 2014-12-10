import numpy as np
import pandas as pd
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import ShuffleSplit
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
	# # derive a set of (random) training and testing indices
	# # Control for low volumes of a label (need minimum of 2 for each class)
	# checker = y.value_counts()
	# b = checker == 1
	# drop_spot = []
	# for val in checker[b].index:
	# 	# print val
	# 	drop_spot.append(y[y==val].index[:][0])
	# if drop_spot:
	# 	# print drop_spot
	# 	# print y.iloc[drop_spot].index
	# 	# X_new=X.drop(drop_spot,axis=0,inplace=False)
	# 	# y_new=y.drop(drop_spot,axis=0,inplace=False)
	# 	y_new = y.append(y)
	# 	X_new = X.append(X)
	# else:
	# 	X_new=X
	# 	y_new=y
	X_new=pd.Series(data=X,index=None)
	y_new=pd.Series(data=y,index = None)
	X_new=X_new.append(X_new)
	y_new=y_new.append(y_new)
	k_fold_indices = StratifiedShuffleSplit(y_new, n_iter=n,test_size=size, random_state=0)
	# k_fold_indices = ShuffleSplit(len(y), n_iter=n,test_size=size, random_state=0)
	k_score_total = []
	# train and score classifier for each slice
	for train_slice, test_slice in k_fold_indices:
		k_train_tfidf = transformer.fit_transform(X_new.iloc[train_slice])
		k_test_tfidf = transformer.transform(X_new.iloc[test_slice])
		new_model = model.fit(k_train_tfidf.todense(),y_new.iloc[train_slice])
		y_pred = new_model.predict(k_test_tfidf)
		y_true = y_new.iloc[test_slice]
		k_score = f1_score(y_true,y_pred,average='micro')

		k_score_total.append(k_score)
	# return the average accuracy
	return np.array(k_score_total)

	# k_fold_indices = ShuffleSplit(len(y), n_iter=n,test_size=size, random_state=0)
	# k_score_total = []
	# # train and score classifier for each slice
	# for train_slice, test_slice in k_fold_indices:
	# 	k_train_tfidf = transformer.fit_transform(X.iloc[train_slice])
	# 	k_test_tfidf = transformer.transform(X.iloc[test_slice])
	# 	new_model = model.fit(k_train_tfidf,y.iloc[train_slice])
	# 	y_pred = new_model.predict(k_test_tfidf)[0]
	# 	y_true = y.iloc[test_slice]
	# 	k_score = f1_score(y_true,y_pred,average='micro')
	# 	k_score_total.append(k_score)

	# # return the average accuracy
	# return np.array(k_score_total)

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

# Generate master tfidf for use then transform the evaluation set.
# for n in [100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
	
# 	l = RandomIndex(data)
# 	for iteration in xrange(1,100):

def ActiveLearner(customer_frame,text_column,label_column,seed_size,interval_size,shuffle_iterations,eval_runs,eval_size,cv_folds,cv_shuffle_size):
	'''There are a lot of variables...'''


	shuff_set_indices = StratifiedShuffleSplit(customer_frame[str(label_column)],n_iter=shuffle_iterations,test_size=eval_size,)
	run_count = 0
	for train_set,test_set in shuff_set_indices:
		# tenth_size = len(train_set)/10*-
		print test_set.shape
		eval_frame = customer_frame.iloc[test_set]
		frame = customer_frame.iloc[train_set]
		seed = frame.index[0:seed_size]
		seed_index = range(0,seed_size)
		rand_index = range(0,seed_size)

		tfidf_transformer = TfidfVectorizer(stop_words='english',ngram_range=(1,4),max_features=4000)
		global_tfidf = tfidf_transformer.fit_transform(frame[str(text_column)])
		active_clf = MultinomialNB().fit(global_tfidf[seed_index], frame.iloc[seed_index][str(label_column)])
		random_clf = MultinomialNB().fit(global_tfidf[rand_index], frame.iloc[rand_index][str(label_column)])
		master_list = range(0,seed_size)

		eval_tfidf = tfidf_transformer.transform(eval_frame[str(text_column)])
		act_predicted = active_clf.predict(eval_tfidf)
		rand_predicted = random_clf.predict(eval_tfidf)
		run_index = [seed_size]
		Active_F1_List = [f1_score(eval_frame[str(label_column)],act_predicted,average='micro')]
		Random_F1_List = [f1_score(eval_frame[str(label_column)],rand_predicted,average='micro')]
		initial_cv = cross_validate_list(frame.iloc[master_list][str(text_column)], frame.iloc[master_list][str(label_column)],tfidf_transformer, MultinomialNB(),cv_shuffle_size,cv_folds).mean()
		active_cv_list = [np.mean(initial_cv)]
		random_cv_list = [np.mean(initial_cv)]

		for i in xrange(1,eval_runs):

			start = seed_size + (i-1)*interval_size
			if start >= frame.shape[0]:
				run_count += 1
				# print "End run ",run_count
				break
			end = seed_size +i*interval_size
			if end <= frame.shape[0]:
				pass
			else:
				end = frame.shape[0]
			new_randoms = range(start,end)

			#Calculate a CV score for the models on the currently labeled data
			active_cv = cross_validate_list(frame.iloc[master_list][str(text_column)], frame.iloc[master_list][str(label_column)],tfidf_transformer, MultinomialNB(),cv_shuffle_size,cv_folds)
			random_cv = cross_validate_list(frame.iloc[range(0,end)][str(text_column)], frame.iloc[range(0,end)][str(label_column)],tfidf_transformer, MultinomialNB(),cv_shuffle_size,cv_folds)
			
			active_cv_list.append(np.mean(active_cv))
			random_cv_list.append(np.mean(random_cv))

			# print "mean:",np.mean(active_cv_list), "var:", np.var(active_cv_list)
			results = active_clf.predict_proba(global_tfidf)
			mindex = ReturnMins(results,frame.index,interval_size, master_list)
			[ master_list.append(x) for x in mindex ]

			#Section for determining next section of array to train the random model comparison


			

			active_clf.partial_fit(global_tfidf[mindex],frame.iloc[mindex][str(label_column)])
			random_clf.partial_fit(global_tfidf[new_randoms],frame.iloc[new_randoms][str(label_column)])

			tfidf_transformer.fit(frame[str(text_column)])
			eval_tfidf = tfidf_transformer.transform(eval_frame[str(text_column)])
			act_predicted = active_clf.predict(eval_tfidf)
			rand_predicted = random_clf.predict(eval_tfidf)
			
			Active_F1_List.append(f1_score(eval_frame[str(label_column)],act_predicted,average='micro'))
			Random_F1_List.append(f1_score(eval_frame[str(label_column)],rand_predicted,average='micro'))
			run_index.append(end)
			if i == eval_runs-1:
				run_count += 1
		if run_count == 1:
			active_frame = pd.DataFrame(data=Active_F1_List,index=run_index,columns=["Run 1"])
			random_frame = pd.DataFrame(data=Random_F1_List,index=run_index,columns=["Run 1"])
			active_cv_frame = pd.DataFrame(data=active_cv_list,index=run_index,columns=["Run 1"])
			random_cv_frame = pd.DataFrame(data=random_cv_list,index=run_index,columns=["Run 1"])
		else:
			column_title = "Run " + str(run_count)
			active_frame[column_title] = Active_F1_List
			random_frame[column_title] = Random_F1_List
			active_cv_frame[column_title] = active_cv_list
			random_cv_frame[column_title] = random_cv_list
	return active_frame,active_cv_frame,random_frame,random_cv_frame
			# print "Active:",f1_score(eval_frame[str(label_column)],act_predicted,average='micro'),"Random:",f1_score(eval_frame[str(label_column)],rand_predicted,average='micro')
		

	# # What is the best case scenario? Trained on entire training set:
	# best_clf = MultinomialNB().fit(global_tfidf,data['tag'])
	# print "Score on model trained with all training data:",best_clf.score(eval_tfidf,eval_data['tag'])

# with open('./fixtures/airlines.json') as f:
# 	json_raw = f.readlines()

# data = QuickConvert(json_raw)

# ActiveLearner(data,'content','tag',50,50,20,20,.1,6,10)

#check out expected maximization algorithm w/ SVMs
#lins

# X_new_tfidf = tfidf_transformer.transform(test_data["content"])

# predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(data["content"],predicted):
# 	print('%r => %s' % (doc,category))

