import numpy 
import csv

# from operator import itemgetter

#linedatamousesの読み込み
#['uid','wid','time','X','Y','DD','DPos','hLabel','Label','hesitate','understand','date','check']
#データが多すぎるのでファイルを２つに分けてあります．
readers = []
fread1 = open('inputdatasu1.csv', 'r')
readers.append(csv.reader(fread1))
fread2 = open('inputdatasu2.csv', 'r')
readers.append(csv.reader(fread2))

print('test')

parametersPerQuestion = []

for i in range(0, 2):
	# uid, widごとにデータを区切る
	# dataは，2次元配列．data[uid][wid]という形で参照する．
	# data[uid][wid]には，配列が格納されている．配列の内容は，1次のパラメタ．timeでソートされている．
	data = {}
	tmpParametersPerQuestion = [] #2次のパラメータを入れる配列

	testCount = 0
	for row in readers[i]:
		testCount += 1
		row[2] = int(row[2])
		row[3] = int(row[3])
		row[4] = int(row[4])
		row[5] = int(row[5])

		# --- 【修正】SQL順序に基づくデータ読み込み ---
		# SQLのSELECT順序:
		# 0:UID, 1:WID, 2:Time, 3:X, 4:Y, 5:DD, 6:DPos, 7:hLabel, 8:Label
		# 9:register_stick, 10:register_stick_count, 11:stick_now, 12:stick_number1
		# 13:stick_number2, 14:stick_number_same, 15:stick_composition_count, 16:word_now
		# 17:repel, 18:repel_count, 19:back, 20:back_count, 21:NOrder
		# 22:left_X, 23:right_X, 24:Y, 25:incorrect_stick_now, 26:stick_move, 27:stick_same
		# 28:hesitate, 29:Understand, 30:Date, 31:check

		# 新しい特徴量用データの取得
		reg_stick_count = int(row[10]) if row[10] != '' else 0
		stick_comp_count = int(row[15]) if row[15] != '' else 0
		repel_cnt = int(row[18]) if row[18] != '' else 0
		back_cnt = int(row[20]) if row[20] != '' else 0
		# SQLにincorrect_stick(フラグ)がないため、incorrect_stick_now(個数状態)を取得
		incorrect_now = int(row[25]) if row[25] != '' else 0 
		stk_same = int(row[27]) if row[27] != '' else 0

		# 既存データの取得位置変更 (テーブル結合により後ろに移動したため)
		understand_val = row[29]
		date_val = row[30]
		check_val = row[31]

		if str(row[0]) not in data.keys():
			data[str(row[0])] = {}
		if str(row[1]) not in data[str(row[0])]:
			data[str(row[0])][str(row[1])] = []
		
		# 辞書へ格納
		data[str(row[0])][str(row[1])].append({
			'time': row[2], 'x': row[3], 'y': row[4], 'dd': row[5], 
			'hLabel': row[7], 'label': row[8], 
			'understand': understand_val, 'date': date_val, 'check': check_val,
			# 新規追加分
			'reg_stick_count': reg_stick_count,
			'stick_comp_count': stick_comp_count,
			'repel_count': repel_cnt,
			'back_count': back_cnt,
			'incorrect_stick_now': incorrect_now, # 状態量として保存
			'stick_same': stk_same
		})
	print(testCount)
	# パラメタを計算していく
	for user in data.keys():
		for question in data[user].keys():
			params = data[user][question] # 時間でソートされた1次のパラメタが格納されている配列．
			# 自信度と日付とチェックは最初に取得しておく．
			understand = params[0]['understand']
			date = params[0]['date']
			check = params[0]['check']
			# time(解答にかかった時間)は，最後の要素の時刻から最初の要素の時刻を引けば算出できる．
			time = params[-1]['time'] - params[0]['time']
			# パラメタの初期値をゼロにしておく．
			distance = 0
			averageSpeed = 0
			maxSpeed = 0
			answeringTime = 0
			totalStopTime = 0
			maxStopTime = 0
			totalDDIntervalTime = 0
			maxDDIntervalTime = 0
			maxDDTime = 0
			minDDTime = 10000000000000000
			DDCount = 0
			xUTurnCount = 0
			yUTurnCount = 0
			groupingDDCount = 0

			# --- 【修正2】新しい特徴量変数の初期化 ---
			totalGroupFormed = 0      # 単語群ができた回数
			maxGroupDuration = 0      # 単語群の最大維持時間
			minGroupDuration = 10000000000 # 単語群の最小維持時間
			
			# 単語数ごとの単語群作成数 (2語, 3語, 4語, 5語以上)
			groupSizeCounts = {2: 0, 3: 0, 4: 0, 5: 0} 
			
			timeToFirstGroup = -1     # 初めて単語群が作られるまでの時間
			
			totalRepelCount = 0       # 弾かれた総回数
			totalBackCount = 0        # 枠外に戻された総回数
			totalIncorrectStick = 0   # 間違った結合をした回数
			totalStickSame = 0        # 同じ単語群を作成した回数

			# グループ維持時間計算用の一時変数
			lastGroupFormTime = -1
			
			# ここから下は，パラメタを計算するために補助的に使う変数．
			startTime = -1    # 問題の解答を開始した時刻．（最初のクリック）
			lastDragTime = -1 # 直前にドラッグを開始した時刻
			lastDropTime = -1 # 直前にドロップを行った時刻
			lastXDirection = 0 # 直前のx軸の方向．1なら右方向，-1なら左方向．
			lastYDirection = 0 # 直前のy軸の方向．1なら右方向，-1なら左方向．
			continuingStopTime = 0 # マウスの静止が継続している時間
			# 1つ1つデータを見ていって計算していく．
			for i in range(len(params)-1): # 今の行とその次の行の比較によって求めるパラメータがあるので，outofrangeになるのを防ぐためにlen-1にする
				# 距離，速度，最大速度
				currentCoord = numpy.array([params[i]['x'], params[i]['y']])
				nextCoord    = numpy.array([params[i+1]['x'], params[i+1]['y']])
				distance += numpy.sqrt(numpy.power(currentCoord-nextCoord, 2).sum())
				speed = distance*1.0 / (params[i+1]['time']-params[i]['time'])
				if params[i+1]['time']-params[i]['time'] != 0 and speed > maxSpeed:
					maxSpeed = speed
				# 解答開始時刻
				if startTime == -1 and params[i]['dd'] == 2:
					startTime = params[i]['time']
				# 総静止時間，最大静止時間
				if params[i]['x'] == params[i+1]['x'] and params[i]['y'] == params[i+1]['y']:
					stopTime = params[i+1]['time']-params[i]['time']
					totalStopTime += stopTime
					continuingStopTime += stopTime
					if continuingStopTime > maxStopTime:
						maxStopTime = continuingStopTime
				else:
					continuingStopTime = 0
				# DD間時間
				if params[i]['dd'] == 2: # drag時
					lastDragTime = params[i]['time']
					if lastDropTime != -1:
						DDIntervalTime = params[i]['time'] - lastDropTime
						if DDIntervalTime > maxDDIntervalTime:
							maxDDIntervalTime = DDIntervalTime
						totalDDIntervalTime += DDIntervalTime
				# DD時間
				if params[i]['dd'] == 1: # drop時
					lastDropTime = params[i]['time']
					DDTime = params[i]['time'] - lastDragTime
					if DDTime > maxDDTime:
						maxDDTime = DDTime
					if DDTime < minDDTime:
						minDDTime = DDTime
					DDCount += 1
				# グルーピング回数 矩形選択して動かしたやつ（違うところに移動させてないのも含む）
				if params[i]['dd'] == 2 and '#' in params[i]['label']:
					groupingDDCount += 1
				# 差の定義（Uターンのフラグと閾値の判断に使用）
				xUTurnDist = params[i+1]['x']-params[i]['x']
				yUTurnDist = params[i+1]['y']-params[i]['y']
				# x方向のUターン
				if xUTurnDist < 0:
					xDirection = 1
				elif xUTurnDist > 0:
					xDirection = -1
				else:
					xDirection = 0
				#　方向転換の有無の判断と閾値の判断
				if ((xDirection == 1 and lastXDirection == -1) or (xDirection == -1 and lastXDirection == 1)) and ((xUTurnDist < -5) or (xUTurnDist > 5)):
					xUTurnCount += 1
					lastXDirection = xDirection
				elif xDirection != 0:
					lastXDirection = xDirection
				# y方向のUターン．（y方向のUターンとほぼおなじなので，関数化した方がキレイ）
				if yUTurnDist < 0:
					yDirection = 1
				elif yUTurnDist > 0:
					yDirection = -1
				else:
					yDirection = 0
				if ((yDirection == 1 and lastYDirection == -1) or (yDirection == -1 and lastYDirection == 1)) and ((yUTurnDist < -5) or (yUTurnDist > 5)):
					yUTurnCount += 1
					lastYDirection = yDirection
				elif yDirection != 0:
					lastYDirection = yDirection
				# 1. 単語群作成・更新に関するカウント
				if params[i]['reg_stick_count'] == 1:
					totalGroupFormed += 1
					
					# 初めて単語群が作られた時間
					if timeToFirstGroup == -1 and startTime != -1:
						timeToFirstGroup = params[i]['time'] - startTime
					
					# 単語数ごとのカウント
					comp_count = params[i]['stick_comp_count']
					if comp_count >= 5:
						groupSizeCounts[5] += 1
					elif comp_count in groupSizeCounts:
						groupSizeCounts[comp_count] += 1

					# 単語群維持時間の計算
					# 「今回更新された時間」と「前回更新された時間」の差分を維持時間とする
					if lastGroupFormTime != -1:
						duration = params[i]['time'] - lastGroupFormTime
						if duration > maxGroupDuration:
							maxGroupDuration = duration
						if duration < minGroupDuration:
							minGroupDuration = duration
					
					lastGroupFormTime = params[i]['time']

				# 2. その他のカウント系
				if params[i]['repel_count'] == 1:
					totalRepelCount += 1
				
				if params[i]['back_count'] == 1:
					totalBackCount += 1
				
				if params[i+1]['incorrect_stick_now'] > params[i]['incorrect_stick_now']:
					# 次の時刻で誤答数が増えていれば、誤った結合をしたとみなす
					diff = params[i+1]['incorrect_stick_now'] - params[i]['incorrect_stick_now']
					totalIncorrectStick += diff

				if params[i]['stick_same'] > 0: # 回数が入っているので加算するか、1以上ならカウントするか
					totalStickSame += 1 # ここでは事象の発生回数として+1します（値そのものを足すなら += params[i]['stick_same']）

			# 上のループはlen-1しているので解答の最後の一行には入らない
			# そのため解答の最後の一行分は別にここで計算（が，これを行うことはないと思う）
			# DD間時間
			if params[-1]['dd'] == 2: # drag時
				lastDragTime = params[-1]['time']
				if lastDropTime != -1:
					DDIntervalTime = params[-1]['time'] - lastDropTime
					if DDIntervalTime > maxDDIntervalTime:
						maxDDIntervalTime = DDIntervalTime
					totalDDIntervalTime += DDIntervalTime
			# DD時間
			if params[-1]['dd'] == 1: # drop時
				lastDropTime = params[-1]['time']
				DDTime = params[-1]['time'] - lastDragTime
				if DDTime > maxDDTime:
					maxDDTime = DDTime
				if DDTime < minDDTime:
					minDDTime = DDTime
				DDCount += 1
			#グルーピング回数 矩形選択して動かしたやつ（違うところに移動させてないのも含む）
			if params[-1]['dd'] == 2 and '#' in params[-1]['label']:
				groupingDDCount += 1

			# 平均速度
			averageSpeed = distance*1.0 / time
			# 解答時間
			thinkingTime = startTime - params[0]['time'] #最初の単語をクリックするまでの時間
			answeringTime = params[-1]['time'] - startTime #最初の単語をクリックしてから決定までの時間
   
            # 最小時間が初期値のままなら0にする
			if minGroupDuration == 10000000000:
				minGroupDuration = 0
			if timeToFirstGroup == -1: # 一度も作らなかった場合
				timeToFirstGroup = 0 # もしくはansweringTimeを入れるなど調整

			tmpParametersPerQuestion.append([user,question,understand,date,check,time,distance,averageSpeed,maxSpeed,thinkingTime,answeringTime,totalStopTime,
				maxStopTime,totalDDIntervalTime,maxDDIntervalTime,maxDDTime,minDDTime,DDCount,groupingDDCount,xUTurnCount,yUTurnCount,
                # -- 追加分 --
				totalGroupFormed,   # 21
				maxGroupDuration,   # 22
				minGroupDuration,   # 23
				groupSizeCounts[2], # 24 (2語グループ)
				groupSizeCounts[3], # 25 (3語グループ)
				groupSizeCounts[4], # 26 (4語グループ)
				groupSizeCounts[5], # 27 (5語以上)
				timeToFirstGroup,   # 28
				totalRepelCount,    # 29
				totalBackCount,     # 30
				totalIncorrectStick,# 31
				totalStickSame      # 32
            ])

	parametersPerQuestion.extend(tmpParametersPerQuestion)			

	# for x in parametersPerQuestion:
	# 	print(x)

print('end')

# csvファイルで出力
fwrite = open('outputdatagroup.csv','w')
writer = csv.writer(fwrite, lineterminator='\n')
writer.writerows(parametersPerQuestion)
fwrite.close()