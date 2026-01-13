import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
import numpy as np
import sys
import csv
# 変更: SVM (SVC) を使うためのインポート（必要なら使用）
from sklearn.svm import SVC

def load_and_preprocess_exclude3(csv_file, feature_set, label_column='hesitate', iteration=None):
    """
    【従来の処理】3 を除外し、2⇒1, 4⇒0 で扱う関数。
    
    - ラベル列に 2/3/4 が入っていることを想定し、2⇒1, 4⇒0。
    - 3 は除外(=NaNにして dropna)。
    - デバッグ用に、2/3/4 の個数や最終行数を出力。
    - 1:1 ダウンサンプリングした X_balanced, y_balanced を返す。
    """
    print(f"\n=== [Exclude3] データセット作成: {feature_set['name']}, iteration={iteration} ===")
    
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"CSVの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

    # 列名の整形
    data.columns = data.columns.str.strip().str.lower()

    # 特徴量列の選択
    if feature_set['type'] == 'all_after_understand':
        start_col = 'understand'
        if start_col not in data.columns:
            print(f"エラー: '{start_col}' 列が存在しません。")
            sys.exit(1)
        start_idx = data.columns.get_loc(start_col)
        feature_columns = data.columns[start_idx:].tolist()
    elif feature_set['type'] == 'specific':
        feature_columns = feature_set['columns']
    else:
        print("エラー: 不明な特徴量セットタイプです。")
        sys.exit(1)
    
    # 除外列
    exclude_cols = feature_set.get('exclude_columns', [])
    feature_columns = [c for c in feature_columns if c not in exclude_cols]

    # デバッグ用に 2/3/4 の件数を表示
    label_counts = data[label_column].value_counts(dropna=False)
    print("=== [Exclude3] 元ラベル列の値分布 (2⇒True, 4⇒False, 3は除外) ===")
    for val in [2, 3, 4]:
        count_val = label_counts.get(val, 0)
        print(f"  値 {val}: {count_val} 件")

    # 2 => 1, 4 => 0, 3等 は NaN
    data['label_mapped'] = data[label_column].map({2: 1, 4: 0})
    data = data.dropna(subset=['label_mapped'])  # 3などはここで除外

    data['label_mapped'] = data['label_mapped'].astype(int)

    X_all = data[feature_columns]
    y_all = data['label_mapped']

    # 1:1 ダウンサンプリング
    df = pd.concat([X_all, y_all], axis=1)
    df_true  = df[df['label_mapped'] == 1]
    df_false = df[df['label_mapped'] == 0]
    min_count= min(len(df_true), len(df_false))

    rs = 42 if iteration is None else 42 + iteration

    df_true_down  = resample(df_true,  replace=False, n_samples=min_count, random_state=rs)
    df_false_down = resample(df_false, replace=False, n_samples=min_count, random_state=rs)
    df_balanced   = pd.concat([df_true_down, df_false_down])

    X_balanced = df_balanced[feature_columns]
    y_balanced = df_balanced['label_mapped'].astype(int)

    # デバッグ表示
    print(f"=== [Exclude3] 2/4 のみ残し、1:1にダウンサンプリングした後の行数: {df_balanced.shape[0]} ===")
    print("selected_columns:")
    print(X_balanced)

    return X_balanced, y_balanced, feature_columns


def load_and_preprocess_merge3and4(csv_file, feature_set, label_column='hesitate', iteration=None):
    """
    【追加機能】3 と 4 をまとめて 0(=False)として扱う関数。
    
    - ラベル列に 2/3/4 が入っていることを想定し、2⇒1, (3, 4)⇒0。
    - 3 は除外せず、4 と同じクラス0として扱う。
    - 1:1 ダウンサンプリングした X_balanced, y_balanced を返す。
    """
    print(f"\n=== [Merge3and4] データセット作成: {feature_set['name']}, iteration={iteration} ===")

    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"CSVの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

    # 列名の整形
    data.columns = data.columns.str.strip().str.lower()

    # 特徴量列の選択 (同じロジック)
    if feature_set['type'] == 'all_after_understand':
        start_col = 'understand'
        if start_col not in data.columns:
            print(f"エラー: '{start_col}' 列が存在しません。")
            sys.exit(1)
        start_idx = data.columns.get_loc(start_col)
        feature_columns = data.columns[start_idx:].tolist()
    elif feature_set['type'] == 'specific':
        feature_columns = feature_set['columns']
    else:
        print("エラー: 不明な特徴量セットタイプです。")
        sys.exit(1)
    
    exclude_cols = feature_set.get('exclude_columns', [])
    feature_columns = [c for c in feature_columns if c not in exclude_cols]

    # デバッグ用に 2/3/4 の件数を表示
    label_counts = data[label_column].value_counts(dropna=False)
    print("=== [Merge3and4] 元ラベル列の値分布 (2⇒True, 3/4⇒False) ===")
    for val in [2, 3, 4]:
        count_val = label_counts.get(val, 0)
        print(f"  値 {val}: {count_val} 件")

    # 2 => 1, 3/4 => 0 (それ以外は NaN にしてもよいが、今回は 2/3/4 しか無い想定なら特に除外不要)
    # もしほかの値がある可能性があるなら map に無いものは NaN にし、dropna する。
    data['label_mapped'] = data[label_column].map({2: 1, 3: 0, 4: 0})

    # 万一 map に無い値があった場合に備え dropna
    data = data.dropna(subset=['label_mapped'])

    data['label_mapped'] = data['label_mapped'].astype(int)

    X_all = data[feature_columns]
    y_all = data['label_mapped']

    # 1:1 ダウンサンプリング
    df = pd.concat([X_all, y_all], axis=1)
    df_true  = df[df['label_mapped'] == 1]  # (元2)
    df_false = df[df['label_mapped'] == 0]  # (元3 or 4)
    min_count= min(len(df_true), len(df_false))

    rs = 42 if iteration is None else 42 + iteration

    df_true_down  = resample(df_true,  replace=False, n_samples=min_count, random_state=rs)
    df_false_down = resample(df_false, replace=False, n_samples=min_count, random_state=rs)
    df_balanced   = pd.concat([df_true_down, df_false_down])

    X_balanced = df_balanced[feature_columns]
    y_balanced = df_balanced['label_mapped'].astype(int)

    # デバッグ表示
    print(f"=== [Merge3and4] 2 と 3/4 をまとめ、1:1にダウンサンプリング後の行数: {df_balanced.shape[0]} ===")
    print("selected_columns:")
    print(X_balanced)

    return X_balanced, y_balanced, feature_columns


def do_10fold_evaluation(X, y, iteration, feature_set_name, output_csv):
    """
    1回分の 10-Fold 交差検証を行い、
    (iteration, fold, class_label, precision, recall, f1)を
    CSVに書き込みつつ、foldごとの指標をreturnする。
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # ランダムフォレスト
    model = RandomForestClassifier(random_state=42)
    # SVCを使用する場合はコメントアウト解除
    #model = SVC(random_state=42)

    fold_prec0 = []
    fold_rec0  = []
    fold_f0    = []
    fold_prec1 = []
    fold_rec1  = []
    fold_f1_   = []

    fold_index = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        prec = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=None, zero_division=0)
        f    = f1_score(y_test, y_pred, average=None, zero_division=0)

        fold_prec0.append(prec[0])
        fold_rec0.append(rec[0])
        fold_f0.append(f[0])
        fold_prec1.append(prec[1])
        fold_rec1.append(rec[1])
        fold_f1_.append(f[1])

        # CSV出力 (クラス0 / クラス1)
        with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
            writer = csv.writer(f_out)
            writer.writerow([
                feature_set_name, 
                f"{iteration}", 
                f"{fold_index}", 
                "クラス0(=FALSE)",
                prec[0], rec[0], f[0]
            ])
            writer.writerow([
                feature_set_name, 
                f"{iteration}", 
                f"{fold_index}", 
                "クラス1(=TRUE)",
                prec[1], rec[1], f[1]
            ])

        fold_index += 1
    
    return (fold_prec0, fold_rec0, fold_f0,
            fold_prec1, fold_rec1, fold_f1_)


def main():
    """
    2通りのアプローチでデータセット作成(1:1)→ 10分割交差検定 → CSV出力。
      - (A) 3を除外する (従来の処理)
      - (B) 3と4をまとめる (FALSE側)
    各アプローチを 10回繰り返し、iteration×foldごとの結果と平均を出力。
    """
    output_csv = "outputdata/word_vif/machineLearning_results/evaluation_results_Updated_predictedUnderstand.csv"

    # CSV初期化 (ヘッダ)
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature_set",
            "iteration",
            "fold",
            "class_label",
            "precision",
            "recall",
            "f1"
        ])

    # 読み込むファイル
    csv_file_name = 'Updated_predictedUnderstand.csv'
    csv_dir       = 'outputdata/word_vif/'
    csv_file      = csv_dir + csv_file_name

    # 特徴量セット
    feature_sets = [
        {
            'name': 'understand追加前',
            'type': 'specific',
            'columns': ['totaltime','totaldistance','minspeed','totalstoptime',
                        'totaluturnx','totaluturny','totalintervaltime'],
            'exclude_columns': []
        },
        {
            'name': 'understand追加後',
            'type': 'specific',
            'columns': ['understand','totaltime','totaldistance','minspeed','totalstoptime',
                        'totaluturnx','totaluturny','totalintervaltime'],
            'exclude_columns': []
        }
    ]

    n_repeat = 10

    for feature_set in feature_sets:
        print(f"\n=== 処理対象: {feature_set['name']} ===")

        # --------------------------------------------------
        # (A) 従来どおり: 3は除外, 2⇒1, 4⇒0
        # --------------------------------------------------
        # 結果をまとめるリスト
        iteration_prec0_list_A = []
        iteration_rec0_list_A  = []
        iteration_f0_list_A    = []
        iteration_prec1_list_A = []
        iteration_rec1_list_A  = []
        iteration_f1_list_A    = []

        for i in range(1, n_repeat+1):
            X_bal, y_bal, selected_cols = load_and_preprocess_exclude3(
                csv_file=csv_file,
                feature_set=feature_set,
                label_column='hesitate',
                iteration=i
            )

            # 10fold
            (fold_prec0, fold_rec0, fold_f0,
             fold_prec1, fold_rec1, fold_f1_) = do_10fold_evaluation(
                X_bal, y_bal, i,
                feature_set['name'] + " (excl3)",  # CSVに書き出す際の名前を区別
                output_csv
            )

            avg_p0 = np.mean(fold_prec0)
            avg_r0 = np.mean(fold_rec0)
            avg_f0 = np.mean(fold_f0)
            avg_p1 = np.mean(fold_prec1)
            avg_r1 = np.mean(fold_rec1)
            avg_f1_ = np.mean(fold_f1_)

            print(f"\n[iteration {i}] 10fold平均 (Exclude3)")
            print(f"  クラス0(FALSE): P={avg_p0:.4f}, R={avg_r0:.4f}, F={avg_f0:.4f}")
            print(f"  クラス1(TRUE):  P={avg_p1:.4f}, R={avg_r1:.4f}, F={avg_f1_:.4f}")

            iteration_prec0_list_A.append(avg_p0)
            iteration_rec0_list_A.append(avg_r0)
            iteration_f0_list_A.append(avg_f0)
            iteration_prec1_list_A.append(avg_p1)
            iteration_rec1_list_A.append(avg_r1)
            iteration_f1_list_A.append(avg_f1_)

            # CSVに iteration単位の平均を追加 (fold名: avg_of_10folds)
            with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([
                    feature_set['name'] + " (excl3)",
                    f"iteration_{i}",
                    "avg_of_10folds",
                    "クラス0(FALSE)",
                    avg_p0, avg_r0, avg_f0
                ])
                writer.writerow([
                    feature_set['name'] + " (excl3)",
                    f"iteration_{i}",
                    "avg_of_10folds",
                    "クラス1(TRUE)",
                    avg_p1, avg_r1, avg_f1_
                ])

        # 全iteration(10回)の平均
        final_p0_A = np.mean(iteration_prec0_list_A)
        final_r0_A = np.mean(iteration_rec0_list_A)
        final_f0_A = np.mean(iteration_f0_list_A)
        final_p1_A = np.mean(iteration_prec1_list_A)
        final_r1_A = np.mean(iteration_rec1_list_A)
        final_f1_A = np.mean(iteration_f1_list_A)

        print(f"\n=== (Exclude3) {feature_set['name']} : 全iteration(10回)の平均 ===")
        print(f"  クラス0(FALSE): P={final_p0_A:.4f}, R={final_r0_A:.4f}, F={final_f0_A:.4f}")
        print(f"  クラス1(TRUE):  P={final_p1_A:.4f}, R={final_r1_A:.4f}, F={final_f1_A:.4f}")

        with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
            writer = csv.writer(f_out)
            writer.writerow([
                feature_set['name'] + " (excl3)",
                "final_average",
                "-",
                "クラス0(FALSE)",
                final_p0_A, final_r0_A, final_f0_A
            ])
            writer.writerow([
                feature_set['name'] + " (excl3)",
                "final_average",
                "-",
                "クラス1(TRUE)",
                final_p1_A, final_r1_A, final_f1_A
            ])

        # --------------------------------------------------
        # (B) 新アプローチ: 3 と 4 をまとめて 0 として扱う
        # --------------------------------------------------
        iteration_prec0_list_B = []
        iteration_rec0_list_B  = []
        iteration_f0_list_B    = []
        iteration_prec1_list_B = []
        iteration_rec1_list_B  = []
        iteration_f1_list_B    = []

        for i in range(1, n_repeat+1):
            X_bal, y_bal, selected_cols = load_and_preprocess_merge3and4(
                csv_file=csv_file,
                feature_set=feature_set,
                label_column='hesitate',
                iteration=i
            )

            (fold_prec0, fold_rec0, fold_f0,
             fold_prec1, fold_rec1, fold_f1_) = do_10fold_evaluation(
                X_bal, y_bal, i,
                feature_set['name'] + " (merge3_4)",
                output_csv
            )

            avg_p0 = np.mean(fold_prec0)
            avg_r0 = np.mean(fold_rec0)
            avg_f0 = np.mean(fold_f0)
            avg_p1 = np.mean(fold_prec1)
            avg_r1 = np.mean(fold_rec1)
            avg_f1_ = np.mean(fold_f1_)

            print(f"\n[iteration {i}] 10fold平均 (Merge3and4)")
            print(f"  クラス0(FALSE): P={avg_p0:.4f}, R={avg_r0:.4f}, F={avg_f0:.4f}")
            print(f"  クラス1(TRUE):  P={avg_p1:.4f}, R={avg_r1:.4f}, F={avg_f1_:.4f}")

            iteration_prec0_list_B.append(avg_p0)
            iteration_rec0_list_B.append(avg_r0)
            iteration_f0_list_B.append(avg_f0)
            iteration_prec1_list_B.append(avg_p1)
            iteration_rec1_list_B.append(avg_r1)
            iteration_f1_list_B.append(avg_f1_)

            # CSVに iteration単位の平均を追加
            with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([
                    feature_set['name'] + " (merge3_4)",
                    f"iteration_{i}",
                    "avg_of_10folds",
                    "クラス0(FALSE)",
                    avg_p0, avg_r0, avg_f0
                ])
                writer.writerow([
                    feature_set['name'] + " (merge3_4)",
                    f"iteration_{i}",
                    "avg_of_10folds",
                    "クラス1(TRUE)",
                    avg_p1, avg_r1, avg_f1_
                ])

        # 全iteration(10回)の平均
        final_p0_B = np.mean(iteration_prec0_list_B)
        final_r0_B = np.mean(iteration_rec0_list_B)
        final_f0_B = np.mean(iteration_f0_list_B)
        final_p1_B = np.mean(iteration_prec1_list_B)
        final_r1_B = np.mean(iteration_rec1_list_B)
        final_f1_B = np.mean(iteration_f1_list_B)

        print(f"\n=== (Merge3and4) {feature_set['name']} : 全iteration(10回)の平均 ===")
        print(f"  クラス0(FALSE): P={final_p0_B:.4f}, R={final_r0_B:.4f}, F={final_f0_B:.4f}")
        print(f"  クラス1(TRUE):  P={final_p1_B:.4f}, R={final_r1_B:.4f}, F={final_f1_B:.4f}")

        with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_out:
            writer = csv.writer(f_out)
            writer.writerow([
                feature_set['name'] + " (merge3_4)",
                "final_average",
                "-",
                "クラス0(FALSE)",
                final_p0_B, final_r0_B, final_f0_B
            ])
            writer.writerow([
                feature_set['name'] + " (merge3_4)",
                "final_average",
                "-",
                "クラス1(TRUE)",
                final_p1_B, final_r1_B, final_f1_B
            ])


if __name__ == "__main__":
    main()
