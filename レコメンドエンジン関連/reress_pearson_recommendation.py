# -*- coding: utf-8 -*-

from math import sqrt
import pandas as pd
import numpy as np
import codecs
import time

### pearson main funcs ###
def similarity_score(person1, person2, dataset):
    # 戻り値は person1 と person2 のユークリッド距離

    both_viewed = {}  # 双方に共通のアイテムを取得

    for item in dataset[person1]:
        if item in dataset[person2]:
            both_viewed[item] = 1

    # 共通のアイテムを持っていなければ 0 を返す
    if len(both_viewed) == 0:
        return 0

    # ユークリッド距離の計算
    sum_of_eclidean_distance = []

    for item in dataset[person1]:
        if item in dataset[person2]:
            sum_of_eclidean_distance.append(
                pow(dataset[person1][item] - dataset[person2][item], 2))
    total_of_eclidean_distance = sum(sum_of_eclidean_distance)

    return 1 / (1 + sqrt(total_of_eclidean_distance))

def pearson_correlation(person1, person2, dataset):

    # 両方のアイテムを取得
    both_rated = {}
    for item in dataset[person1]:
        if item in dataset[person2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)

    # 共通のアイテムがあるかチェック、無ければ 0 を返す
    if number_of_ratings == 0:
        return 0

    # 各ユーザーごとのすべての好みを追加
    person1_preferences_sum = sum(
        [dataset[person1][item] for item in both_rated])
    person2_preferences_sum = sum(
        [dataset[person2][item] for item in both_rated])

    # 各ユーザーの好みの値の二乗を計算
    person1_square_preferences_sum = sum(
        [pow(dataset[person1][item], 2) for item in both_rated])
    person2_square_preferences_sum = sum(
        [pow(dataset[person2][item], 2) for item in both_rated])

    # アイテムごとのユーザー同士のレーティングを算出して合計
    product_sum_of_both_users = sum(
        [dataset[person1][item] * dataset[person2][item] for item in both_rated])

    # ピアソンスコアの計算
    numerator_value = product_sum_of_both_users - \
        (person1_preferences_sum * person2_preferences_sum / number_of_ratings)
    denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum, 2) / number_of_ratings) * (
        person2_square_preferences_sum - pow(person2_preferences_sum, 2) / number_of_ratings))
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / denominator_value
        return r

def most_similar_users(person, number_of_users, dataset):
    # 似たユーザーとその類似度を返す
    scores = [(pearson_correlation(person, other_person, dataset), other_person)
              for other_person in dataset if other_person != person]

    # 最高の類似度の人物が最初になるようにソートする
    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]

def user_reommendations(person, dataset):

    # 他のユーザーの加重平均によるランキングから推薦を求める
    totals = {}
    simSums = {}
    for other in dataset:
        # 自分自身は比較しない
        if other == person:
            continue
        sim = pearson_correlation(person, other, dataset)

        # ゼロ以下のスコアは無視する
        if sim <= 0:
            continue
        for item in dataset[other]:

            # まだ所持していないアイテムのスコア
            if item not in dataset[person] or dataset[person][item] == 0:

                # Similrity * スコア
                totals.setdefault(item, 0)
                totals[item] += dataset[other][item] * sim
                # 類似度の和
                simSums.setdefault(item, 0)
                simSums[item] += sim

        # 正規化されたリストを作成

    rankings = [(total / simSums[item], item)
                for item, total in list(totals.items())]
    rankings.sort()
    rankings.reverse()
    # 推薦アイテムを返す
    recommendataions_list = [
        recommend_item for score, recommend_item in rankings]
    return recommendataions_list

### clean reress data ###
"""
ex)
if __name__ == "__main__":
    file_name = "subjects.csv"
    file_name = "subjects_830.csv"
    # gakunen_gakka_code = "16bc"
    gakunen_gakka_code = ""
    # only_gakka_code = "bc"
    only_gakka_code = ""
    # remove_sub = ['英語ディスカッション１','英語ディスカッション２','英語プレゼンテーション','英語リーディング＆ライティング１（Ｒ）','英語リーディング＆ライティング２（Ｗ）','英語ｅラーニング']
    remove_sub = ["＊必）言語Ａ＊","＊必）言語Ｂ＊"]# "＊学びの精神＊","＊多彩な学び，スポ＊"

    df = csv_reader(file_name)
    cleaned_df = cleanning_df(df)
    cleaned_df_as_array, sorted_student_list = remove_noise_data(cleaned_df, gakunen_gakka_code, only_gakka_code, remove_sub)
    my_dataset = generate_dataset_as_dict(student_list = sorted_student_list, array_data = cleaned_df_as_array)

"""

def csv_reader(file_name):
    #unicodeエラーとヘッダー自動割り当てに対処して読み込み
    with codecs.open(file_name, "r", "Shift-JIS", "ignore") as file:
        df = pd.read_csv(file, header=None, delimiter=",")
    return df

def cleanning_df(df):
    #dfを整理
    np_df = np.array(df)
    cleaned_df = pd.DataFrame({
        "student_id":np_df[:,3],
        "subject_name":np_df[:,4],
        "grade":np_df[:,7],
        "category":np_df[:,13]
    })
    cleaned_df_values=cleaned_df["student_id"].values
    gakunen_gakka = []
    only_gakka=[]
    for i in range(len(cleaned_df_values)):
        gakunen_gakka.append(cleaned_df_values[i][:4])
        only_gakka.append(cleaned_df_values[i][2:4])
    cleaned_df = pd.DataFrame({
        "student_id":np_df[:,3],
        "subject_name":np_df[:,4],
        "grade":np_df[:,7],
        "category":np_df[:,13],
        'gakunen_gakka':gakunen_gakka,
        'only_gakka':only_gakka,
    })
    return cleaned_df

def remove_noise_data(df, gg_code, g_code, remove_sub):
    array_data = np.array(df)# ID、授業名、成績データ、カテゴリーのarray化
    if gg_code != "":
        array_data = np.delete(array_data, np.where(array_data[:,1] != gg_code),axis=0)
    if g_code != "":
        array_data = np.delete(array_data, np.where(array_data[:,3] != g_code),axis=0)        

    if len(remove_sub) > 0:
        for i,w in enumerate(remove_sub):
            array_data = np.delete(array_data, np.where(array_data[:,0] ==w),axis=0)# カテゴリ名
        
    array_data = np.delete(array_data, np.where(array_data[:,2] ==0),axis=0)
    cleaned_df = pd.DataFrame(array_data)
    cleaned_df.columns = ["category",'gakunen_gakka',"grade",'only_gakka',"student_id", "subject_name"]
    sorted_student_list = cleaned_df["student_id"].value_counts().index.tolist()#uniqueなIDを取り出してリスト化
    return array_data, sorted_student_list
    
def generate_dataset_as_dict(student_list, array_data):
    person = []
    dics = []
    for stu_id in student_list:
        one_student_data = np.delete(array_data, np.where(array_data[:,4] != stu_id),axis=0)
        if len(one_student_data)>0:
            keys = one_student_data[:,5]
            values = one_student_data[:,2]
            dic = dict(zip(keys,values))
            person.append(one_student_data[0,4])
            dics.append(dic)
    dataset = dict(zip(person,dics))
    return dataset

### generate dataset by using above 4 funcs
def pearson_dataset_generator(pearson_params):
    file_name = pearson_params[0]
    gakunen_gakka_code = pearson_params[1]
    only_gakka_code = pearson_params[2]
    remove_sub = pearson_params[3]
    
    df = csv_reader(file_name)
    cleaned_df = cleanning_df(df)
    cleaned_df_as_array, sorted_student_list = remove_noise_data(cleaned_df, gakunen_gakka_code, only_gakka_code, remove_sub)
    my_dataset = generate_dataset_as_dict(student_list = sorted_student_list, array_data = cleaned_df_as_array)
    return my_dataset


### recommendation func ###
"""
ex)
if __name__ == "__main__":
    base_user = "16bc046c"
    main_recommendation(base_user, my_dataset)
"""

def main_recommendation(base_user, dataset, recommend_count=20, eclidean_count=20):

    dataset = dataset
    print("base_user「" + str(base_user) + "」のデータセット")
    print(dataset[base_user])
    score_list = []
    user_list = []
    for score,user in most_similar_users(base_user, 3, dataset):
        score_list.append(score)
        user_list.append(user)

    print("=========================")
    print("base_userに似た人ベスト 10", most_similar_users(base_user, 10, dataset))    
    print("=========================")
    print("base_userに似た人ベスト 3の詳細")    
    for i,user in enumerate(user_list):
        print("{0}とbase_userのピアソン距離：{1}".format(user,score_list[i]))
        print(dataset[user])  
        print("=========================")

    print("base_user「" + base_user + "」におすすめの授業")
    recommendataions_list = user_reommendations(base_user, dataset)
    for i in recommendataions_list[:recommend_count]:
        print(i)

    print("=========================")
    print("base_userとピアソン距離が近いユーザーとのユークリッド距離：")
    #similarity_score(base_user, user_list[0], dataset)
    compare_students = [i for i in dataset]
    scores = []
    for i in compare_students:
        scores.append([similarity_score(base_user, i, dataset),i])
    scores.sort()
    scores.reverse()
    if len(scores)>eclidean_count:
        for i in scores[0:eclidean_count]:
            print(i)
    else:
        for i in scores:
            print(i)
    return

"""
ex)
if __name__ == "__main__":
    pick_up_list = ['16bc046c','<similar_student_id>','<similar_student_id>','<similar_student_id>','<similar_student_id>','<similar_student_id>','<similar_student_id>','<similar_student_id>','<similar_student_id>']
    lda_dataset = lda_data_recommendation(my_dataset, pick_up_list)

"""
def lda_data_recommendation(dataset, pick_up_list):
    base_user = pick_up_list[0]
    lda_dataset = dict(dataset)

    dataset_students = [i for i in dataset]
    delete_list = [i for i in dataset_students if i not in pick_up_list]
    for i in delete_list:
        del lda_dataset[i]
    main_recommendation(base_user, lda_dataset)
    return
