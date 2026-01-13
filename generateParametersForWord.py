import math
import csv

# linedatamousesの読み込み
#['uid','wid','time','X','Y','DD','DPos','hLabel','Label','hesitate','understand']
fread = open('inputdata/word/inputdata_hesitate_all.csv', 'r')
inputData = csv.reader(fread)
print('test')

class UTurnChecker: #Uターンの有無の判定
	def __init__(self):
		self.prevdirection = 0
		
	def check(self, coordinates, prevcoordinates): 
		#座標からdirectionを決定
		if (coordinates - prevcoordinates) > 0:
			direction = 1
		elif (coordinates - prevcoordinates) < 0:
			direction = -1
		else:
			direction = 0
		#directionからUターンか判定，directionの更新
		if (self.prevdirection == 1 and direction == -1) or (self.prevdirection == -1 and direction == 1):
			if direction != 0:
				self.prevdirection = direction
			return True
		else:
			if direction != 0:
				self.prevdirection = direction
			return False


class ParameterCalculator:
	def __init__(self, startRow):
		self.userId     = startRow[0]
		self.questionId = startRow[1]
		self.words      = startRow[8].split('#')
		self.hesitateLabel = startRow[9]
		self.understand = startRow[10]

		self.startTime = int(startRow[2])
		self.prevTime  = int(startRow[2])
		self.prevX     = int(startRow[3])
		self.prevY     = int(startRow[4])

		self.uTurnXChecker = UTurnChecker()
		self.uTurnYChecker = UTurnChecker()

		self.moveTime = 0
		self.distance = 0
		self.speed    = 0
		self.stopTime = 0
		self.uTurnX   = 0
		self.uTurnY   = 0

	def calculateForUpdate(self, time, x, y):
		#移動距離の計算
		dx = x - self.prevX
		dy = y - self.prevY
		self.distance += math.sqrt(dx*dx + dy*dy)
		#静止時間の計算
		if x == self.prevX and y == self.prevY:
			self.stopTime += time - self.prevTime
		#Uターンのチェック
		if self.uTurnXChecker.check(x, self.prevX):
			self.uTurnX +=1
		if self.uTurnYChecker.check(y, self.prevY):
			self.uTurnY +=1

	def calculateForFinalize(self, time, x, y):
		#移動時間の計算
		#print('time = ' +str(time))
		#print('startTime = ' +str(self.startTime))
		self.moveTime = time - self.startTime
		#移動速度の計算
		if self.moveTime == 0:
			print(count)
		else:
			self.speed = self.distance / self.moveTime


	def update(self, row):
		time = row[2]
		x = row[3]
		y = row[4]
		#パラメタ計算
		self.calculateForUpdate(time, x, y)
		#状態の更新
		self.prevTime = time
		self.prevX = x
		self.prevY = y

	def finalize(self, row):
		time = row[2]
		x = row[3]
		y = row[4]
		#パラメタ計算
		self.calculateForUpdate(time, x, y)
		self.calculateForFinalize(time, x, y)

	def getUserIdAndQuestionId(self):
		return [self.userId, self.questionId]

	def getWords(self):
		return self.words

	def getHesitateLabel(self):
		return self.hesitateLabel

	def getUnderstand(self):
		return self.understand

	def getParameter(self):
		return [self.moveTime, self.distance, self.speed, self.stopTime, self.uTurnX, self.uTurnY]

class ParameterIntegrator: #1つの解答ごと
	def __init__(self):
		self.parameters = []
		self.DDCOUNT           = 5
		self.TOTALTIME         = 6 #合計，最大，最小，平均，（合計，最大，最小，平均）を解答時間で割った比
		self.TOTALDISTANCE     = 14 # 同上
		self.TOTALSPEED        = 22 # 同上
		self.TOTALSTOPTIME     = 30 # 同上
		self.TOTALUTURNX       = 38 # 同上
		self.TOTALUTURNY       = 46 # 同上
		self.FIRSTDDCOUNT      = 54
		self.LASTDDCOUNT       = 55
		
	def updateTotalMaxMinAverage(self, value, DDCount, parameter, index): #更新
		#合計
		parameter[index] += value
		#最大
		if parameter[index+1] < value: 
			parameter[index+1] = value
		#最小
		if parameter[index+2] > value:
			parameter[index+2] = value
		#平均
		parameter[index+3] = parameter[index] / DDCount
		
	def integrate(self, userId, questionId, wordId, parameterPerDD, DDCountPerQuestion, hesitateLabel, understand):
		time     = parameterPerDD[0]
		distance = parameterPerDD[1]
		speed    = parameterPerDD[2]
		stopTime = parameterPerDD[3]
		uTurnX   = parameterPerDD[4]
		uTurnY   = parameterPerDD[5]
		for parameter in self.parameters:
			if wordId == parameter[2]:
				parameter[self.DDCOUNT] += 1 #DD回数				
				self.updateTotalMaxMinAverage(time,     parameter[self.DDCOUNT], parameter, self.TOTALTIME)#時間
				self.updateTotalMaxMinAverage(distance, parameter[self.DDCOUNT], parameter, self.TOTALDISTANCE)#距離
				self.updateTotalMaxMinAverage(speed,    parameter[self.DDCOUNT], parameter, self.TOTALSPEED)#速度
				self.updateTotalMaxMinAverage(stopTime, parameter[self.DDCOUNT], parameter, self.TOTALSTOPTIME)#静止時間
				self.updateTotalMaxMinAverage(uTurnX,   parameter[self.DDCOUNT], parameter, self.TOTALUTURNX)#UTurnx
				self.updateTotalMaxMinAverage(uTurnY,   parameter[self.DDCOUNT], parameter, self.TOTALUTURNY)#UTurny
				parameter[self.LASTDDCOUNT]  = DDCountPerQuestion
				break
		else: #breakしたらこのelseには入らない
			hesitate = False
			if str(wordId) in hesitateLabel.split('#'):
				hesitate = True
			self.parameters.append([userId, questionId, wordId, 
			hesitate, #迷った単語
			understand, #自信度
			1, #DD回数
			time, time, time, time, 0, 0, 0, 0, #（合計，最大，最小，平均），解答時間で割った（同左）
			distance, distance, distance, distance, 0, 0, 0, 0,
			speed, speed, speed, speed,  0, 0, 0, 0,
			stopTime, stopTime, stopTime, stopTime, 0, 0, 0, 0,
			uTurnX, uTurnX, uTurnX, uTurnX, 0, 0, 0, 0,
			uTurnY, uTurnY, uTurnY, uTurnY, 0, 0, 0, 0,
			DDCountPerQuestion, DDCountPerQuestion])

	def getParameters(self):
		return self.parameters

DDParameter         = ParameterIntegrator()
DDIntervalParameter = ParameterIntegrator()
lastDDtoDecision    = ParameterIntegrator()

begin = -1
limit = 100000000
count = 0

endRowOfPrevDD = False #一つ前のDDの最後の行（解答の開始位置を知るためにほしい）
prevRow = False #一つ前の行

final = []
draggingWords = 0
DDCountPerQuestion = 0 
totalPerQuestion = [0]*6 #time,distance,speed,stoptime,uTurnX,uTurnYの分
parameterCalculator = False

for row in inputData: #csvファイルの中身を1行ずつみていく
	if not parameterCalculator:
		parameterCalculator = ParameterCalculator(row)
	if count == 0:
		#print(prevRow)
		#この時点でprevrow=False
		prevRow = row
		#この時点でprevrow=[  , ,  ,  , ....]
		#print(prevRow)
	if count > begin and count < limit:
		row[2] = int(row[2]) #time
		row[3] = int(row[3]) #X
		row[4] = int(row[4]) #Y
		row[5] = int(row[5]) #DD

		if row[0:2] != prevRow[0:2]: #解答が切り替わったとき
			#print("解答が切り替わったとき")
			parameterCalculator.finalize(prevRow)
			for word in draggingWords: #labelの配列内をループ
				lastDDtoDecision.integrate(row[0], row[1], word, parameterCalculator.getParameter(), DDCountPerQuestion, row[9], row[10])
				#rint("labelの配列内をループ")
			for i in range(0,6):
				totalPerQuestion[i] += parameterCalculator.getParameter()[i]

			if 0 not in totalPerQuestion:
				#比の計算
				for (p1, p2) in zip(DDParameter.getParameters(), DDIntervalParameter.getParameters()):
					for i in range(0,6):
						for j in range(0,4):
							p1[(8*i+6+j)+4] = p1[8*i+6+j] / totalPerQuestion[i]
							p2[(8*i+6+j)+4] = p2[8*i+6+j] / totalPerQuestion[i]
				#DDParameterとDDIntervalParameterをまとめる
				for (p1, p2) in zip(DDParameter.getParameters(), DDIntervalParameter.getParameters()): 
					if p1[0:3] == p2[0:3]:
						final.append([])
						final[-1].extend(p1)
						final[-1].extend(p2[5:])
			else:
				print("out")

			totalPerQuestion = [0]*6
			endRowOfPrevDD = row 
			DDCountPerQuestion = 0
			parameterCalculator = ParameterCalculator(row)
			DDParameter         = ParameterIntegrator()
			DDIntervalParameter = ParameterIntegrator()
			lastDDtoDecision    = ParameterIntegrator()

		elif row[5] == 2: #ドラッグ開始
			#print("ドラッグ開始")
			draggingWords = row[8].split('#')
			parameterCalculator.finalize(row)
			for word in draggingWords: #labelの配列内をループ
				DDIntervalParameter.integrate(row[0], row[1], word, parameterCalculator.getParameter(), DDCountPerQuestion, row[9], row[10])
			for i in range(0,6):
				totalPerQuestion[i] += parameterCalculator.getParameter()[i]
			DDCountPerQuestion += 1
			parameterCalculator = ParameterCalculator(row)
	
		elif row[5] == 1: #ドラッグ終了
			#print("ドラッグ終了")
			parameterCalculator.finalize(row)
			for word in draggingWords: #labelの配列内をループ
				DDParameter.integrate(row[0], row[1], word, parameterCalculator.getParameter(), DDCountPerQuestion, row[9], row[10])
			for i in range(0,6):
				totalPerQuestion[i] += parameterCalculator.getParameter()[i]
			endRowOfPrevDD = row
			parameterCalculator = ParameterCalculator(row)

		else: 
			parameterCalculator.update(row)

		prevRow = row

	count += 1
	#if count%10000 == 0:
		#print('check' + str(count/10000))
  

if parameterCalculator and count > 0:
    # 最終行に対して finalize
    parameterCalculator.finalize(prevRow)

    # 最後に draggingWords を持っていた場合、lastDDtoDecision または DDParameter へ integrate
    # どちらに書き込むかは仕様によりますが、
    # 「ドラッグが終了していない」まま終わった場合もあるので、とりあえずここでは lastDDtoDecision にまとめておく例にします。
    for word in draggingWords:
        lastDDtoDecision.integrate(
            prevRow[0], prevRow[1], word,
            parameterCalculator.getParameter(),
            DDCountPerQuestion,
            prevRow[9], prevRow[10]
        )

    # totalPerQuestion を更新
    for i in range(0, 6):
        totalPerQuestion[i] += parameterCalculator.getParameter()[i]

    # ここで 0 を含まなければ出力
    if 0 not in totalPerQuestion:
        # 比の計算
        for (p1, p2) in zip(DDParameter.getParameters(), DDIntervalParameter.getParameters()):
            for i in range(0, 6):
                for j in range(0, 4):
                    p1[(8*i+6+j)+4] = p1[8*i+6+j] / totalPerQuestion[i]
                    p2[(8*i+6+j)+4] = p2[8*i+6+j] / totalPerQuestion[i]

        # 最終的に final に追加
        for (p1, p2) in zip(DDParameter.getParameters(), DDIntervalParameter.getParameters()):
            if p1[0:3] == p2[0:3]:
                final.append([])
                final[-1].extend(p1)
                final[-1].extend(p2[5:])
    else:
        print("out")

# for p in parameterIntegrator.getParameters():
# 	print(p)

# csvファイルで出力
fwrite = open('outputdata/word/outputdata_hesitate_all.csv','w')
writer = csv.writer(fwrite, lineterminator='\n')
writer.writerows(final)
fwrite.close()

print('end')
