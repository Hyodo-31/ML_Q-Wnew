import csv
import random

# scikit-learnの分類器構築関係のメソッド
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# scikit-learnの，正解率とか精度とか出すためのメソッド
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import model_selection
#from sklearn import cross_validation

# 数値計算のためのライブラリ．実質，numpy.array()というのものを使うためだけにインポートしてます．
# newArray = numpy.array(oldArray)という風に使うと，oldArrayが，numpy形式の配列に変換されて，newArrayに代入されます．
# numpy形式の配列は，[1, 2, 3] + [4, 5, 6]というような形で足し算ができて，この場合，結果は[5, 7, 9]になります．
import numpy

from operator import itemgetter

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def extractSelectedParameters(row, selectedParameters):
	newRow = []
	selectedIndice = [p[0] for p in selectedParameters]
	for i in range(selectedParameters[0][0], len(row)):
		if i in selectedIndice:
			newRow.append(row[i])
	return newRow


# 特徴量になる列名
selectedParameters = [
	(5 , 'time'),
	(6 , 'distance'),
	(7 , 'averageSpeed'),
	# (8 , 'maxSpeed'), #
	# (9 , 'thinkingTime'), #
	# (10 , 'answeringTime'), #
	# (11, 'totalStopTime'), #
	(12, 'maxStopTime '),
	# (13, 'totalDDIntervalTime '), #
	(14, 'maxDDIntervalTime '),
	(15, 'maxDDTime'),
	# (16, 'minDDTime'), #
	(17, 'DDCount'),
	(18, 'groupingDDCount'),
	(19, 'xUTurnCount'),
	(20, 'yUTurnCount'),
]


#2次パラメータの読み込み
#['uid','wid','自信度','解答日時','チェック',2次パラメータ～]
fread = open('outputdatagroup.csv', 'r')
rows = csv.reader(fread)
# for od in rows:
# 	print(od)
print('test')

# データベースのデータをもとに，訓練データを作る．
labels = [] #ラベル（自信度）を入れる配列
labels2 = []
labels3 = []
labels4 = []
newlabels = []
features = [] #特徴量を入れる配列
features2 = []
features3 = []
features4 = []
newfeatures = []
counter = []
newcounter = []
avg_accuracy = []
avg_p_no = []
avg_r_no = []
avg_f_no = []
avg_p_yes = []
avg_r_yes = []
avg_f_yes = []


# 正例，負例の数をカウントしつつ訓練データを作る
nTrue  = 0
nFalse = 0
nFalse2 = 0
nData  = 0
nerror = 0
number = 0

for row in rows:
	if int(row[0]) in [65,66,67,68,69,70710096,70711031,70810029,70810063,70810072,70810100,70811012,70811036,
70811039,70811049,70811066,70811071,70812006,70812016,70812049,70812055,30810106,30814218,30814807,30814811,30814914,60810015,60810026,60810043,60810092,60710027,60710074,60610008,60811006,60811009,60811010,60811028,60811044,60811050,60811057,60811068,60811075,90710037]\
			and int(row[1]) in [22,68,46,191,32,54,67,184,45,89,127,129,141,143,147,160,162,176,181,186,59,60,99,61,139,76,92,58,138,161]:
		if int(row[2]) in [0,1,3] : # row[2]には自信度が入っている．自信度が，0, 1のいずれかなら，何もせずにスルーする．
			nerror += 1
			pass
		elif int(row[2]) in [2] and int(row[4]) in [0,1]: # 自信度が2の場合のみ，迷いありとみなす． 元は[1,2,3] チェックなしのみを扱うときはrow[4]は[0]，両方は[0,1]にする．
			nTrue += 1
			labels.append(1)
			features.append(extractSelectedParameters(list(row), selectedParameters)) # row[1:]は，1行以降の要素．すなわち，特徴量．
		elif int(row[2]) in [4]: # 自信度が4の場合のみ，迷いなしとみなす．
			nFalse += 1
			labels.append(0)
			features.append(extractSelectedParameters(list(row), selectedParameters))

#print("データ数" + str(len(labels)))
print("最初のなしの数 " + str(nFalse) + "  割合" + str((1.0*nFalse)/len(labels)))
print("最初のありの数 " + str(nTrue)  + "  割合" + str((1.0*nTrue)/len(labels)))
print('nerror: ' + str(nerror))

for p in range(len(labels)):	#labelsとfeaturesに混ぜて入れたデータをありとなしに分ける
	if labels[p] in [1]:
		labels2.append(1)
		features2.append(features[p])
	elif labels[p] in [0]:
		labels3.append(0)
		features3.append(features[p])
		counter.append(number)
		number += 1

print("ありの数となしの数は同じ " + str(nTrue) +"個ずつ")
 
for q in range(10):	#10回繰り返す
	#print('機械学習: ' + str(q+1) + '回目')

	#print("---counterの要素数")
	#print(len(counter))
	for x  in labels3:	#labels3に入れたデータを何度も使うので3は残しておきたい．毎回ループの初めにlabels4にうつす
		labels4.append(x)
	for y in features3:	#同様にfeatures3もfeatures4にうつす
		features4.append(y)
	for z in counter:	#同様にcounterをnewcounterにうつす
		newcounter.append(z)

	#print("---初めのnewcounterの要素数")
	#print(len(newcounter))

	for s in labels2:	#labels4とfeatures4からランダムでありの数とおなじだけなしのデータを取り出しnewlabelsとnewfeaturesに格納
		a = random.choice(newcounter)
		#print('---a')
		#print(a)
		b = labels4.pop(a)
		newlabels.append(b)
		c = features4.pop(a)
		newfeatures.append(c)
		newcounter.pop()
		#print("---popしたあとのnewcounterの要素数")
		#print(len(newcounter))

	#nFalse2 = len(newlabels) #なしのデータ数はこの時点でのnewlabelsの長さ

	for t in labels2:	#なしのデータが入っているnew～にありのデータを付け加える
		newlabels.append(t)
	for u in features2:
		newfeatures.append(u)
	

	# 後で使うメソッドが，ただの配列ではなく，numpy形式の配列を引数に取るので，変換しておく．
	labels = numpy.array(newlabels)
	features = numpy.array(newfeatures)
	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

	
	#print("なしの数 " + str(nFalse2))
	#print("ありの数 " + str(nTrue))

	#次のループのために空にしておく
	labels4 = []
	features4 = []
	newcounter = []
	newlabels = []
	newfeatures = []


	# n交差できるように標本を分割
	n_folds = 10
	#skf = cross_validation.StratifiedKFold(labels, n_folds=n_folds, random_state=0) # 後で，このskfというものをfor ... inで辿っていくことで，うまく交差検定できる．
	skf = model_selection.StratifiedKFold(n_splits=n_folds).split(features,labels)
	# 正解率と，迷いなし，迷いありそれぞれの適合率，再現率，F値の総和がここに格納される．
	sum_accuracy = 0
	sum_precision_no = 0
	sum_recall_no    = 0
	sum_f_measure_no = 0
	sum_precision_yes = 0
	sum_recall_yes    = 0
	sum_f_measure_yes = 0
	# 特徴量ごとの重要度の総和が，このnumpy形式の配列に格納される．
	# 特徴量の個数（特徴ベクトルの要素数）だけ，0が並んだ状態で，配列が初期化されている．
	sum_feature_importance = numpy.array([0]*len(features[0]))


	# 交差検定のために分割された学習データ，実験データを使って，実際に，学習，分類していく．
	for train_index, test_index in skf:
		# features_train: 学習データの特徴量
		# features_test : 実験データの特徴量
		# labels_train  : 学習データのラベル
		# labels_test   : 実験データのラベル
		features_train, features_test = features[train_index], features[test_index]
		labels_train,   labels_test   = labels[train_index],   labels[test_index]

		# ランダムフォレストのモデル（分類器生成器）を作る
		rfc = RandomForestClassifier(random_state=0)
		# 学習する．
		rfc.fit(features_train, labels_train)
		# 分類する．
		labels_pred = rfc.predict(features_test)

		# サポートベクターマシンを使いたい時は，上の，ランダムフォレスト関係の部分をコメントアウトして，↓の部分を使う．
		# svc = svm.SVC()
		# svc.fit(features_train, labels_train)
		# labels_pred = svc.predict(features_test)

		# 分類正解率を求めて，総和を保存する変数に足し込む．
		accuracy = accuracy_score(labels_test, labels_pred)
		sum_accuracy += accuracy
		# 適合率，再現率，f値を求めて，総和を保存する変数に足し込む．
		# sには，サンプルの個数が入っているけど，今回は使わない．
		# p, r, fは，3つとも，配列．pは，[迷いなしの場合の適合率, 迷いありの適合率]の2要素からなる配列．再現率，f値も同様．
		p, r, f, s = metrics.precision_recall_fscore_support(labels_test, labels_pred)
		sum_precision_no  += p[0]
		sum_recall_no     += r[0]
		sum_f_measure_no  += f[0]
		sum_precision_yes += p[1]
		sum_recall_yes    += r[1]
		sum_f_measure_yes += f[1]
		# 特徴量の重要度を求めて，総和を保存する配列に足し込む．
		sum_feature_importance = sum_feature_importance + numpy.array(rfc.feature_importances_)

	#10交差検定した結果（正解率，適合率，再現率，F値）を毎回配列に格納．正解率等はそれぞれの値の総和を，交差検定を行った回数(10)で割っている．
	avg_accuracy.append(sum_accuracy / n_folds)
	avg_p_no.append(sum_precision_no / n_folds)
	avg_r_no.append(sum_recall_no / n_folds)
	avg_f_no.append(sum_f_measure_no / n_folds)
	avg_p_yes.append(sum_precision_yes / n_folds)
	avg_r_yes.append(sum_recall_yes / n_folds)
	avg_f_yes.append(sum_f_measure_yes / n_folds)

	#1回ずつの正解率等を表示したかったら以下のコメントアウトをはずす
	# 正解率等を出力．それぞれの値の総和を，交差検定を行った回数で割る．
	#print('----results')
	#print('avg_accuracy(正解率)　　　　　　:'+str(sum_accuracy     /n_folds))
	#print('avg_p_no(適合率_迷いなし)    :'+str(sum_precision_no /n_folds))
	#print('avg_r_no(再現率_迷いなし)    :'+str(sum_recall_no    /n_folds))
	#print('avg_f_no(F値_迷いなし) 　　    :'+str(sum_f_measure_no /n_folds))
	#print('avg_p_yes(適合率_迷いあり)    :'+str(sum_precision_yes/n_folds))
	#print('avg_r_yes(再現率_迷いなし)   :'+str(sum_recall_yes   /n_folds))
	#print('avg_f_yes(F値_迷いなし)   　　:'+str(sum_f_measure_yes/n_folds))
	
	# 特徴量の重要度を出力．値が大きいものから順にソートして表示．
print('----features ordered by importance')
for featureId in sorted( zip(range(len(sum_feature_importance)), sum_feature_importance), key=itemgetter(1), reverse=True):
	print(selectedParameters[featureId[0]][1]+': '+str(featureId[1]))

	#if q in [9]:
	#	print('end')
	# ####


#10回の正解率等の平均値を算出し，表示
print('----results(10times)')
print('avg_accuracy(正解率)　　　　:'+str(sum(avg_accuracy)/len(avg_accuracy)))
print('avg_p_no(適合率_迷いなし)   :'+str(sum(avg_p_no)/len(avg_p_no)))
print('avg_r_no(再現率_迷いなし)   :'+str(sum(avg_r_no)/len(avg_r_no)))
print('avg_f_no(F値_迷いなし) 　　 :'+str(sum(avg_f_no)/len(avg_f_no)))
print('avg_p_yes(適合率_迷いあり)  :'+str(sum(avg_p_yes)/len(avg_p_yes)))
print('avg_r_yes(再現率_迷いあり)  :'+str(sum(avg_r_yes)/len(avg_r_yes)))
print('avg_f_yes(F値_迷いあり)   　:'+str(sum(avg_f_yes)/len(avg_f_yes)))
