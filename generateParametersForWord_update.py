import math
import csv

# ---------------------------------------------------------
# 入力CSVの列インデックス定義 (SQL文と要件に基づき設定)
# ---------------------------------------------------------
# SQL: UID, WID, Time, X, Y, DD, DPos, hLabel, Label, 
#      register_stick, register_stick_count, stick_now, stick_number1, stick_number2, 
#      stick_number_same, stick_composition_count, word_now, repel, repel_count, 
#      back, back_count, NOrder, left_groupword_X, right_groupword_X, groupword_Y, 
#      (★ここにincorrect_stickがあると仮定★), incorrect_stick_now, stick_move, stick_same, 
#      hesitate1, hesitate2, Understand

IDX_UID = 0
IDX_WID = 1
IDX_TIME = 2
IDX_X = 3
IDX_Y = 4
IDX_DD = 5
IDX_LABEL = 8              # ドラッグ中の単語
IDX_REG_STICK = 9          # 結合した単語リスト (1#4#2)
IDX_REG_STICK_COUNT = 10   # 結合フラグ
IDX_COMP_COUNT = 15        # 構成単語数
IDX_GROUP_Y = 24

# 【重要】要件にある「incorrect_stick(フラグ)」がSQL文に見当たらないため、
# 文脈から groupword_Y の次、incorrect_stick_now の前にあると仮定しています。
IDX_INCORRECT_STICK = 25     # 不正解スティックフラグ
IDX_INCORRECT_STICK_NOW = 26 # カウント
IDX_HESITATE1 = 29           # 以降ずれると仮定
IDX_HESITATE2 = 30
IDX_UNDERSTAND = 31

# 入力ファイル名
INPUT_FILE = '単語単位少し迷ったと迷ったの区別あり.csv' 
OUTPUT_FILE = 'outputdata_単語単位少し迷ったと迷ったの区別あり.csv'

# 読み込み処理
try:
    fread = open(INPUT_FILE, 'r', encoding='utf-8-sig') # utf-8-sig等適宜変更
except FileNotFoundError:
    print(f"File {INPUT_FILE} not found. Please check the file name.")
    exit()

inputData = csv.reader(fread)
# ヘッダーがある場合は飛ばす (データのみの場合はコメントアウト)
header_row = next(inputData) 
print('Processing started...')

class UTurnChecker: 
    def __init__(self):
        self.prevdirection = 0
        
    def check(self, coordinates, prevcoordinates): 
        if (coordinates - prevcoordinates) > 0:
            direction = 1
        elif (coordinates - prevcoordinates) < 0:
            direction = -1
        else:
            direction = 0

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
        self.userId          = startRow[IDX_UID]
        self.questionId      = startRow[IDX_WID]
        self.words           = startRow[IDX_LABEL].split('#')
        
        # 列インデックスを使用して取得
        try:
            self.hesitateLabel1  = startRow[IDX_HESITATE1]
            self.hesitateLabel2  = startRow[IDX_HESITATE2]
            self.understand      = startRow[IDX_UNDERSTAND]
        except IndexError:
            # 行データが足りない場合のフォールバック
            self.hesitateLabel1 = ""
            self.hesitateLabel2 = ""
            self.understand = ""

        self.startTime = int(startRow[IDX_TIME])
        self.prevTime  = int(startRow[IDX_TIME])
        self.prevX     = int(startRow[IDX_X])
        self.prevY     = int(startRow[IDX_Y])

        self.uTurnXChecker = UTurnChecker()
        self.uTurnYChecker = UTurnChecker()

        self.moveTime = 0
        self.distance = 0
        self.speed    = 0
        self.stopTime = 0
        self.uTurnX   = 0
        self.uTurnY   = 0

    def calculateForUpdate(self, time, x, y):
        dx = x - self.prevX
        dy = y - self.prevY
        self.distance += math.sqrt(dx*dx + dy*dy)
        if x == self.prevX and y == self.prevY:
            self.stopTime += time - self.prevTime
        if self.uTurnXChecker.check(x, self.prevX):
            self.uTurnX += 1
        if self.uTurnYChecker.check(y, self.prevY):
            self.uTurnY += 1

    def calculateForFinalize(self, time, x, y):
        self.moveTime = time - self.startTime
        if self.moveTime != 0:
            self.speed = self.distance / self.moveTime

    def update(self, row):
        time = int(row[IDX_TIME])
        x    = int(row[IDX_X])
        y    = int(row[IDX_Y])
        self.calculateForUpdate(time, x, y)
        self.prevTime = time
        self.prevX = x
        self.prevY = y

    def finalize(self, row):
        time = int(row[IDX_TIME])
        x    = int(row[IDX_X])
        y    = int(row[IDX_Y])
        self.calculateForUpdate(time, x, y)
        self.calculateForFinalize(time, x, y)

    def getUserIdAndQuestionId(self):
        return [self.userId, self.questionId]
    def getWords(self):
        return self.words
    def getHesitateLabel1(self):
        return self.hesitateLabel1
    def getHesitateLabel2(self):
        return self.hesitateLabel2
    def getUnderstand(self):
        return self.understand
    def getParameter(self):
        return [self.moveTime, self.distance, self.speed, self.stopTime, self.uTurnX, self.uTurnY]

class ParameterIntegrator:
    def __init__(self):
        self.parameters = []
        # 既存特徴量 (各8要素: 合計,最大,最小,平均 + 各Ratio)
        self.DDCOUNT           = 5
        self.TOTALTIME         = 6
        self.TOTALDISTANCE     = 14
        self.TOTALSPEED        = 22
        self.TOTALSTOPTIME     = 30
        self.TOTALUTURNX       = 38
        self.TOTALUTURNY       = 46
        
        # --- 新規特徴量のインデックス定義 ---
        # 既存の比率計算ループ(range(0,6))に影響を与えないよう後ろに追加
        
        # 1. 単語群化回数 (Sum)
        self.GROUP_FORM_COUNT = 54 
        
        # 2. register_stick_count記録時の出現回数 (Sum)
        self.REG_STICK_COUNT_VAL = 55 
        
        # 3. 単語群化経験時の時間 (Sum, Max, Min, Avg) - 4要素
        self.GROUP_FORM_TIME = 56
        
        # 4. 単語群構成単語数 (Max) - 1要素 (最大値のみ保持)
        self.MAX_COMPOSITION = 60
        
        # 5. 不正解(incorrect_stick)経験回数 (Sum)
        self.INCORRECT_COUNT = 61
        
        # 6. 不正解経験時の時間 (Sum, Max, Min, Avg) - 4要素
        self.INCORRECT_TIME = 62

        # 最後にDDCountPerQuestion系を配置
        self.FIRSTDDCOUNT      = 66
        self.LASTDDCOUNT       = 67
        
    def updateTotalMaxMinAverage(self, value, DDCount, parameter, index):
        # 合計
        parameter[index] += value
        # 最大
        if parameter[index+1] < value: 
            parameter[index+1] = value
        # 最小
        if parameter[index+2] > value:
            parameter[index+2] = value
        # 平均
        parameter[index+3] = parameter[index] / DDCount

    def integrate(self, userId, questionId, wordId,
                  parameterPerDD, DDCountPerQuestion,
                  hesitateLabel1, hesitateLabel2, understand,
                  # 新規データを受け取る引数
                  reg_stick_str, reg_stick_count_flag, stick_comp_count, incorrect_stick_flag):

        # 基本パラメータ
        time     = parameterPerDD[0]
        distance = parameterPerDD[1]
        speed    = parameterPerDD[2]
        stopTime = parameterPerDD[3]
        uTurnX   = parameterPerDD[4]
        uTurnY   = parameterPerDD[5]

        # 新規特徴量の判定ロジック
        is_in_group = False
        is_incorrect = False
        
        # reg_stick_str (例: "1#4#2") に wordId が含まれているか
        if str(wordId) in reg_stick_str.split('#'):
            is_in_group = True
            
        # incorrect_stick_flag が 1 かつ グループに含まれているか
        if is_in_group and str(incorrect_stick_flag) == '1':
            is_incorrect = True

        for parameter in self.parameters:
            if wordId == parameter[2]:
                parameter[self.DDCOUNT] += 1 # DD回数
                
                # 既存パラメータ更新
                self.updateTotalMaxMinAverage(time,     parameter[self.DDCOUNT], parameter, self.TOTALTIME)
                self.updateTotalMaxMinAverage(distance, parameter[self.DDCOUNT], parameter, self.TOTALDISTANCE)
                self.updateTotalMaxMinAverage(speed,    parameter[self.DDCOUNT], parameter, self.TOTALSPEED)
                self.updateTotalMaxMinAverage(stopTime, parameter[self.DDCOUNT], parameter, self.TOTALSTOPTIME)
                self.updateTotalMaxMinAverage(uTurnX,   parameter[self.DDCOUNT], parameter, self.TOTALUTURNX)
                self.updateTotalMaxMinAverage(uTurnY,   parameter[self.DDCOUNT], parameter, self.TOTALUTURNY)
                
                # --- 新規特徴量更新 (既存行) ---
                if is_in_group:
                    # 1. 単語群化回数 (Labelでドラッグされ、register_stickに載った回数)
                    # 注: register_stickに載っている時点でDD=1のタイミングとみなす
                    parameter[self.GROUP_FORM_COUNT] += 1
                    
                    # 2. register_stick_count が記録された回数 (要件により、register_stickに含まれ、かつフラグが1なら)
                    if str(reg_stick_count_flag) == '1':
                        parameter[self.REG_STICK_COUNT_VAL] += 1
                    
                    # 3. 単語群化経験時の時間 (最大＆合計)
                    self.updateTotalMaxMinAverage(time, parameter[self.GROUP_FORM_COUNT], parameter, self.GROUP_FORM_TIME)
                    
                    # 4. 構成単語数の最大
                    try:
                        comp_cnt = int(stick_comp_count)
                        if parameter[self.MAX_COMPOSITION] < comp_cnt:
                            parameter[self.MAX_COMPOSITION] = comp_cnt
                    except ValueError:
                        pass

                    # 5 & 6. 不正解経験
                    if is_incorrect:
                        parameter[self.INCORRECT_COUNT] += 1
                        self.updateTotalMaxMinAverage(time, parameter[self.INCORRECT_COUNT], parameter, self.INCORRECT_TIME)

                parameter[self.LASTDDCOUNT] = DDCountPerQuestion
                break
        else: 
            # 新規追加
            if str(wordId) in hesitateLabel1.split('#'):
                hesitate = 3
            elif str(wordId) in hesitateLabel2.split('#'):
                hesitate = 2
            else:
                hesitate = 4
            
            # 初期リスト作成 (0埋め)
            # 既存: 6(Time) + 8*6(各Metrics) = 54要素 (index 0-53)
            # 新規: 
            #  GroupCount(1) + RegCount(1) + GroupTime(4) + MaxComp(1) + IncCount(1) + IncTime(4) = 12要素
            # 合計 66要素 + DDCount系2つ = 68要素
            
            new_params = [0] * 12 
            
            # 該当する場合のみ値をセット
            if is_in_group:
                new_params[0] = 1 # GroupFormCount
                if str(reg_stick_count_flag) == '1':
                    new_params[1] = 1 # RegStickCount
                
                # GroupTime (Sum, Max, Min, Avg)
                new_params[2] = time
                new_params[3] = time
                new_params[4] = time
                new_params[5] = time
                
                try:
                    new_params[6] = int(stick_comp_count) # MaxComp
                except:
                    pass
                
                if is_incorrect:
                    new_params[7] = 1 # IncCount
                    # IncTime
                    new_params[8] = time
                    new_params[9] = time
                    new_params[10] = time
                    new_params[11] = time

            self.parameters.append([
                userId, questionId, wordId, 
                hesitate, understand, 1, # Index 0-5
                # Time (6-13)
                time, time, time, time, 0, 0, 0, 0,
                # Distance (14-21)
                distance, distance, distance, distance, 0, 0, 0, 0,
                # Speed (22-29)
                speed, speed, speed, speed, 0, 0, 0, 0,
                # StopTime (30-37)
                stopTime, stopTime, stopTime, stopTime, 0, 0, 0, 0,
                # UTurnX (38-45)
                uTurnX, uTurnX, uTurnX, uTurnX, 0, 0, 0, 0,
                # UTurnY (46-53)
                uTurnY, uTurnY, uTurnY, uTurnY, 0, 0, 0, 0
            ] + new_params + [DDCountPerQuestion, DDCountPerQuestion])

    def getParameters(self):
        return self.parameters

DDParameter         = ParameterIntegrator()
DDIntervalParameter = ParameterIntegrator()
lastDDtoDecision    = ParameterIntegrator()

begin = -1
limit = 100000000
count = 0

endRowOfPrevDD = False
prevRow = False

final = []
draggingWords = [] # 初期化修正 (0ではなく空リスト推奨)
DDCountPerQuestion = 0 
totalPerQuestion = [0]*6  # 既存の比率計算用
parameterCalculator = False

for row in inputData:
    # 行の要素数が足りない場合のガード
    if len(row) < 25: 
        continue

    if not parameterCalculator:
        parameterCalculator = ParameterCalculator(row)

    if count == 0:
        prevRow = row

    if count > begin and count < limit:
        # 数値変換 (エラー回避)
        try:
            row[IDX_TIME] = int(row[IDX_TIME])
            row[IDX_X]    = int(row[IDX_X])
            row[IDX_Y]    = int(row[IDX_Y])
            row[IDX_DD]   = int(row[IDX_DD])
        except ValueError:
            continue

        # 解答切り替え判定
        if row[IDX_UID:IDX_WID+1] != prevRow[IDX_UID:IDX_WID+1]:
            parameterCalculator.finalize(prevRow)
            
            # 前の解答の残処理
            for word in draggingWords:
                # 注: lastDDtoDecision にも新規パラメータ用の引数を渡す必要があるが
                # 終了直前の状態(prevRow)を使って渡す
                try:
                    inc_stick = prevRow[IDX_INCORRECT_STICK]
                except IndexError:
                    inc_stick = 0

                lastDDtoDecision.integrate(
                    prevRow[IDX_UID], prevRow[IDX_WID], word,
                    parameterCalculator.getParameter(),
                    DDCountPerQuestion,
                    prevRow[IDX_HESITATE1], prevRow[IDX_HESITATE2], prevRow[IDX_UNDERSTAND],
                    prevRow[IDX_REG_STICK], prevRow[IDX_REG_STICK_COUNT], 
                    prevRow[IDX_COMP_COUNT], inc_stick
                )

            # 比率計算 (既存6項目のみ対象)
            for i in range(0,6):
                totalPerQuestion[i] += parameterCalculator.getParameter()[i]

            if 0 not in totalPerQuestion:
                for (p1, p2) in zip(DDParameter.getParameters(), DDIntervalParameter.getParameters()):
                    # 既存の比率計算
                    for i in range(0,6):
                        for j in range(0,4):
                            p1[(8*i+6+j)+4] = p1[8*i+6+j] / totalPerQuestion[i]
                            p2[(8*i+6+j)+4] = p2[8*i+6+j] / totalPerQuestion[i]

                # 結合
                for (p1, p2) in zip(DDParameter.getParameters(), DDIntervalParameter.getParameters()):
                    if p1[0:3] == p2[0:3]:
                        final.append([])
                        final[-1].extend(p1)
                        # p2(Interval)については、新規特徴量の列も含まれてしまうため
                        # 既存仕様通りDDカウント以降だけ足すか、全て足すか。
                        # ここでは既存ロジック通り「5列目以降(DDCountから)」を結合する
                        final[-1].extend(p2[5:])
            
            # リセット
            totalPerQuestion = [0]*6
            endRowOfPrevDD = row
            DDCountPerQuestion = 0
            draggingWords = []

            parameterCalculator = ParameterCalculator(row)
            DDParameter         = ParameterIntegrator()
            DDIntervalParameter = ParameterIntegrator()
            lastDDtoDecision    = ParameterIntegrator()

        # ドラッグ開始 (DD=2)
        elif row[IDX_DD] == 2:
            parameterCalculator.finalize(row)
            draggingWords = row[IDX_LABEL].split('#')
            
            # IntervalParameterにも引数が必要だが、Drag開始時点では
            # Group化などの結果(Drop時の状態)はまだ確定していないことが多い。
            # ただし形式を合わせるため現在のrowの値を渡す（通常は空や0のはず）
            try:
                inc_stick = row[IDX_INCORRECT_STICK]
            except IndexError:
                inc_stick = 0

            for word in draggingWords:
                DDIntervalParameter.integrate(
                    row[IDX_UID], row[IDX_WID], word,
                    parameterCalculator.getParameter(),
                    DDCountPerQuestion,
                    row[IDX_HESITATE1], row[IDX_HESITATE2], row[IDX_UNDERSTAND],
                    row[IDX_REG_STICK], row[IDX_REG_STICK_COUNT], 
                    row[IDX_COMP_COUNT], inc_stick
                )
            
            for i in range(0,6):
                totalPerQuestion[i] += parameterCalculator.getParameter()[i]
            DDCountPerQuestion += 1
            parameterCalculator = ParameterCalculator(row)
    
        # ドロップ (DD=1)
        elif row[IDX_DD] == 1:
            parameterCalculator.finalize(row)
            
            try:
                inc_stick = row[IDX_INCORRECT_STICK]
            except IndexError:
                inc_stick = 0

            for word in draggingWords:
                # ここで新規特徴量の判定が行われる
                DDParameter.integrate(
                    row[IDX_UID], row[IDX_WID], word,
                    parameterCalculator.getParameter(),
                    DDCountPerQuestion,
                    row[IDX_HESITATE1], row[IDX_HESITATE2], row[IDX_UNDERSTAND],
                    # 新規用データ
                    row[IDX_REG_STICK], row[IDX_REG_STICK_COUNT], 
                    row[IDX_COMP_COUNT], inc_stick
                )

            for i in range(0,6):
                totalPerQuestion[i] += parameterCalculator.getParameter()[i]
            endRowOfPrevDD = row
            parameterCalculator = ParameterCalculator(row)

        else:
            parameterCalculator.update(row)

        prevRow = row
    count += 1

# 最終行処理
if parameterCalculator and count > 0:
    parameterCalculator.finalize(prevRow)
    try:
        inc_stick = prevRow[IDX_INCORRECT_STICK]
    except IndexError:
        inc_stick = 0

    for word in draggingWords:
        lastDDtoDecision.integrate(
            prevRow[IDX_UID], prevRow[IDX_WID], word,
            parameterCalculator.getParameter(),
            DDCountPerQuestion,
            prevRow[IDX_HESITATE1], prevRow[IDX_HESITATE2], prevRow[IDX_UNDERSTAND],
            prevRow[IDX_REG_STICK], prevRow[IDX_REG_STICK_COUNT], 
            prevRow[IDX_COMP_COUNT], inc_stick
        )
    for i in range(0, 6):
        totalPerQuestion[i] += parameterCalculator.getParameter()[i]

    if 0 not in totalPerQuestion:
        for (p1, p2) in zip(DDParameter.getParameters(), DDIntervalParameter.getParameters()):
            for i in range(0, 6):
                for j in range(0, 4):
                    p1[(8*i+6+j)+4] = p1[8*i+6+j] / totalPerQuestion[i]
                    p2[(8*i+6+j)+4] = p2[8*i+6+j] / totalPerQuestion[i]

        for (p1, p2) in zip(DDParameter.getParameters(), DDIntervalParameter.getParameters()):
            if p1[0:3] == p2[0:3]:
                final.append([])
                final[-1].extend(p1)
                final[-1].extend(p2[5:])

# ---------------------------------------------------------
# CSV出力 (ヘッダー作成)
# ---------------------------------------------------------
fwrite = open(OUTPUT_FILE, 'w', newline='')
writer = csv.writer(fwrite, lineterminator='\n')

header = ['UserId', 'QuestionId', 'WordId', 'hesitate', 'understand', 'DDcount']
metrics = ['time', 'distance', 'speed', 'stoptime', 'uturnx', 'uturny']
stats = ['total', 'max', 'min', 'average']

# 1. 既存特徴量 (DDParameter)
for m in metrics:
    for s in stats:
        header.append(s + m)
    for s in stats:
        header.append(s + m + 'ratio')

# 2. 新規特徴量 (DDParameterに追加分)
# GroupFormCount (54)
header.append('GroupFormCount')
# RegisterStickCount (55)
header.append('RegisterStickCount')

# GroupFormTime (56-59)
for s in stats:
    header.append(s + 'GroupFormTime')

# MaxComposition (60)
header.append('MaxComposition')

# IncorrectCount (61)
header.append('IncorrectCount')

# IncorrectTime (62-65)
for s in stats:
    header.append(s + 'IncorrectTime')

# DDCountPerQuestion (66, 67)
header.extend(['DDCountPerQuestion', 'DDCountPerQuestion2'])

# 3. Interval特徴量 (結合される側)
# IntervalDDCount
header.append('IntervalDDCount')

# Interval Metrics
for m in metrics:
    metric_name = 'Interval' + m
    for s in stats:
        header.append(s + metric_name)
    for s in stats:
        header.append(s + metric_name + 'ratio')

# Interval側の新規特徴量 (一応枠はあるのでヘッダーもつける)
header.append('IntervalGroupFormCount')
header.append('IntervalRegisterStickCount')
for s in stats:
    header.append(s + 'IntervalGroupFormTime')
header.append('IntervalMaxComposition')
header.append('IntervalIncorrectCount')
for s in stats:
    header.append(s + 'IntervalIncorrectTime')

header.extend(['IntervalDDCountPerQuestion', 'IntervalDDCountPerQuestion2'])

writer.writerow(header)
writer.writerows(final)
fwrite.close()

print('end')