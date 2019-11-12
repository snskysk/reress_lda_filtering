# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import codecs
from sklearn.decomposition import NMF
from reress_data_cleaner import *
from reress_matrix_factorization import *


# ================================= SVD ================================
def reduct_matrix(matrix, k):
    # SVDで分解
    # full_metrices=Falseとすると、uがM×M(正方行列)ではなく、M×Nとなる
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    
    # k次元まで圧縮
    s_k = np.array(s)
    s_k[k:] = 0
    sigma_k = np.diag(s_k)
    
    return np.dot(u, np.dot(sigma_k, v))

# ================================= NMF ================================

def reduct_matrix_nmf(matrix, k):
    # NMFで分解
    model = NMF(n_components=k, init='random', random_state=1)
    w = model.fit_transform(matrix)
    h = model.components_
    
    return np.dot(w, h)

# ================================= 共通 ================================

def predict_ranking(user_index, reducted_matrix, original_matrix, n):
    # 対象ユーザのSVD評価値
    reducted_vector = reducted_matrix[:, user_index]
    
    # 評価済みのアイテムの値は0にする
    filter_vector = original_matrix[:, user_index] == 0
    predicted_vector = reducted_vector * filter_vector

    # 上位n個のアイテムのインデックスを返す
    return [(i, predicted_vector[i]) for i in np.argsort(predicted_vector)[::-1][:n]], predicted_vector

def print_ranking(user_ids, matrix_row_list, matrix_col_list,reducted_matrix, rating_matrix, k):
    predicted_vectors = []
    for user_id in user_ids:
        num = [n for n,i in enumerate(matrix_col_list) if i == user_id]
        predicted_ranking, predicted_vector = predict_ranking(num[0], reducted_matrix, rating_matrix, k)
        predicted_vectors.append(predicted_vector)
        print('User: {}:'.format(user_id))
        for item_id, rating in predicted_ranking:
            num = [n for n,i in enumerate(matrix_row_list) if i == item_id]
            # アイテムID, 授業タイトル, 予測した評価値を表示
            print('{}: {} [{}]'.format(item_id, matrix_row_list[item_id], rating))
    return predicted_vectors


# ===================== evaluate ========================


# 生徒ごとのデータを、指定された数で分割してデータフレームとして作成、三次元配列
def split_data_as_df(df, student_list, split_num):
    df_as_array = np.array(df)
    new_array = []
    for s in student_list:
        subjects = []
        for i in df_as_array:
            if i[0] == s:
                subjects.append(i)
        subjects = np.array(subjects)
        # if len(subjects)>split_num:
        subjects = split_list(subjects, split_num)
        subject_as_some_df = []
        for split_subjects in subjects:
            subject_as_some_df.append(pd.DataFrame(np.array(split_subjects)))
                    
        new_array.append(subject_as_some_df)
    return new_array


# split_data_as_df　の内部で使う
# split_num = 6
# your_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# [[1, 7, 13], [2, 8, 14], [3, 9], [4, 10], [5, 11], [6, 12]]
def split_list(your_list, split_num):
    res = []
    small_res = []
    roop_count = len(your_list)//split_num
    for i in range(split_num):
        pick_up_num = []
        for n in range(len(your_list)):
            if (n + split_num) % split_num == i:
                pick_up_num.append(n)
        small_res = [your_list[k] for k in pick_up_num]
        res.append(small_res)
    return res


# 様々なパターンでデータを組み替え
# rearranged_train_dfs = [train_df_0, train_df_1, train_df_2...]
# rearranged_test_dfs = [test_df_0, test_df_1, test_df_2...](評価値を持つデータ)
def rearranging_dfs(splited_dataset):
    rearranged_train_dfs = []
    rearranged_test_dfs = []

    fold_count = len(splited_dataset[0])
    base_train_data_index = [i for i in range(fold_count)]
    for i in range(fold_count):
        train_data_index = base_train_data_index.copy()
        train_data_index.remove(i)# 学習に使うデータ（フレーム）のインデックス ex[0, 2, 3, 4]
        test_data_index = i

        # ===============================
        big_flag = 0
        test_data_flag = 0
        for d in splited_dataset:
            flag = 0
            for f in train_data_index:
                if flag == 0:
                    base_df = d[f]
                    flag = 1
                else:# 学生単位でデータフレームを結合
                    if d[f].empty:
                        pass
                    else:
                        base_df = pd.concat([base_df, d[f]],axis=0)

            if test_data_flag == 0:
                test_data_base_df = d[test_data_index]
                test_data_flag = 1
            else:# 全体のtestdata のデータフレームを結合
                if d[test_data_index].empty:
                    pass
                else:
                    test_data_base_df = pd.concat([test_data_base_df, d[test_data_index]],axis=0)

            if big_flag == 0:
                big_base_df = base_df
                big_flag = 1
            else:# 全体のデータフレームを結合
                big_base_df = pd.concat([big_base_df, base_df],axis=0)
        big_base_df.columns = ["0_stu_id","1_sub_name","2_score"]
        test_data_base_df.columns = ["0_stu_id","1_sub_name","2_score"]
        rearranged_train_dfs.append(big_base_df)
        rearranged_test_dfs.append(test_data_base_df)
    return rearranged_train_dfs, rearranged_test_dfs


# ユーザーごとの推定値行列をDF集合のリストとして返す
def get_predict_df(stu_ids, reducted_matrix, rating_matrix, matrix_col_list, matrix_row_list):
    predict_dfs = []
    for my_id in stu_ids:
        num = [n for n,i in enumerate(matrix_col_list) if i == my_id]
        predicted_ranking, predicted_vector = predict_ranking(num[0], reducted_matrix, rating_matrix, 20)
        my_predict = pd.DataFrame({
            "0_sub":matrix_row_list,# sub_name
            "1_predict_score":predicted_vector,# 推定値
        })
        predict_dfs.append(my_predict)
    return predict_dfs


def my_nmf_evaluation_without_params(rearranged_train_dfs, rearranged_test_dfs):

    cul_count = 0
    cul_sum = 0
    my_count = 0
    my_sum = 0
    result = []

    for n,rearranged_train_df in enumerate(rearranged_train_dfs):
        # print(train_df)
        test_df = rearranged_test_dfs[n]

        # 授業名を行、user_idを列とする行列(欠測値は0埋め)
        rating_matrix = rearranged_train_df.pivot(index='1_sub_name', columns='0_stu_id', values='2_score').fillna(0)
        matrix_row_list = list(rating_matrix.index)
        matrix_col_list = list(rating_matrix.columns)
        rating_matrix = rating_matrix.as_matrix()

        reducted_matrix_nmf = reduct_matrix_nmf(rating_matrix, k=100)
        # =================stu_ids
        stu_ids = []
        for stu_id in list(np.array(rearranged_train_df)[:,0]):
            if stu_id not in stu_ids:
                stu_ids.append(stu_id)

        # =================untrain_sub_list
        test_df_as_array = np.array(test_df)
        untrain_sub_list = []
        for stu_id in stu_ids:
            untrain_data_list = []
            for one_test_data in test_df_as_array:
                if one_test_data[0] == stu_id:
                    untrain_data_list.append([one_test_data[1],one_test_data[2]])
            untrain_sub_list.append(np.array(untrain_data_list))

        # ================== ユーザーごとの予測値ベクトルを取得
        predict_dfs = get_predict_df(stu_ids, reducted_matrix_nmf, rating_matrix, matrix_col_list, matrix_row_list)

        for p,untrain_data in enumerate(untrain_sub_list):
            try:
                one_users_untrain_sub = list(untrain_data[:,0])
                one_users_untrain_score = list(untrain_data[:,1])
                one_users_dict = dict(zip(one_users_untrain_sub,one_users_untrain_score))
                my_predict = predict_dfs[p]
                my_predict = np.array(my_predict)

                a = [(i[0],i[1],one_users_dict[i[0]]) for n,i in enumerate(my_predict) if i[0] in one_users_untrain_sub]    
                for i in a:
                    result.append(i)
                    cul_count += 1
                    cul_sum += (float(i[1])-float(i[2]))**2
                if stu_ids[p] == "16bc046c":
                    print("16bc046c")
                    print(one_users_dict)
                    for i in a:
                        my_count += 1
                        my_sum += (float(i[1])-float(i[2]))**2
                        print(i)
            except:
                pass
        print("==========================")

    print("RMSE:{0}".format((cul_sum/cul_count)**(1/2)))
    print("my_RMSE:{0}".format((my_sum/my_count)**(1/2)))
    return result


def my_nmf_evaluation(rearranged_train_dfs, rearranged_test_dfs):

    cul_count = 0
    cul_sum = 0
    my_count = 0
    my_sum = 0
    result = []
    conf_v = 0.14951114
    intercept_v = 2.24686714

    for n,rearranged_train_df in enumerate(rearranged_train_dfs):
        # print(train_df)
        test_df = rearranged_test_dfs[n]

        # 授業名を行、user_idを列とする行列(欠測値は0埋め)
        rating_matrix = rearranged_train_df.pivot(index='1_sub_name', columns='0_stu_id', values='2_score').fillna(0)
        matrix_row_list = list(rating_matrix.index)
        matrix_col_list = list(rating_matrix.columns)
        rating_matrix = rating_matrix.as_matrix()

        reducted_matrix_nmf = reduct_matrix_nmf(rating_matrix, k=100)
        # =================stu_ids
        stu_ids = []
        for stu_id in list(np.array(rearranged_train_df)[:,0]):
            if stu_id not in stu_ids:
                stu_ids.append(stu_id)

        # =================untrain_sub_list
        test_df_as_array = np.array(test_df)
        untrain_sub_list = []
        for stu_id in stu_ids:
            untrain_data_list = []
            for one_test_data in test_df_as_array:
                if one_test_data[0] == stu_id:
                    untrain_data_list.append([one_test_data[1],one_test_data[2]])
            untrain_sub_list.append(np.array(untrain_data_list))

        # ================== ユーザーごとの予測値ベクトルを取得
        predict_dfs = get_predict_df(stu_ids, reducted_matrix_nmf, rating_matrix, matrix_col_list, matrix_row_list)

        for p,untrain_data in enumerate(untrain_sub_list):
            try:
                one_users_untrain_sub = list(untrain_data[:,0])
                one_users_untrain_score = list(untrain_data[:,1])
                one_users_dict = dict(zip(one_users_untrain_sub,one_users_untrain_score))
                my_predict = predict_dfs[p]
                my_predict = np.array(my_predict)

                # a = [(i[0],i[1],one_users_dict[i[0]]) for n,i in enumerate(my_predict) if i[0] in one_users_untrain_sub]    
                a = [(i[0], conf_v * (i[1]+1) + intercept_v, one_users_dict[i[0]]) for n,i in enumerate(my_predict) if i[0] in one_users_untrain_sub]    
                for i in a:
                    result.append(i)
                    cul_count += 1
                    cul_sum += (float(i[1])-float(i[2]))**2
                if stu_ids[p] == "16bc046c":
                    print("16bc046c")
                    print(one_users_dict)
                    for i in a:
                        my_count += 1
                        my_sum += (float(i[1])-float(i[2]))**2
                        print(i)
            except:
                pass
        print("==========================")

    print("RMSE:{0}".format((cul_sum/cul_count)**(1/2)))
    print("my_RMSE:{0}".format((my_sum/my_count)**(1/2)))
    return result


def my_svd_evaluation(rearranged_train_dfs, rearranged_test_dfs):

    cul_count = 0
    cul_sum = 0
    my_count = 0
    my_sum = 0
    result = []


    for n,rearranged_train_df in enumerate(rearranged_train_dfs):
        # print(train_df)
        test_df = rearranged_test_dfs[n]

        # 授業名を行、user_idを列とする行列(欠測値は0埋め)
        rating_matrix = rearranged_train_df.pivot(index='1_sub_name', columns='0_stu_id', values='2_score').fillna(0)
        matrix_row_list = list(rating_matrix.index)
        matrix_col_list = list(rating_matrix.columns)
        rating_matrix = rating_matrix.as_matrix()

        reducted_matrix = reduct_matrix(rating_matrix, k=100)
        # =================stu_ids
        stu_ids = []
        for stu_id in list(np.array(rearranged_train_df)[:,0]):
            if stu_id not in stu_ids:
                stu_ids.append(stu_id)

        # =================untrain_sub_list
        test_df_as_array = np.array(test_df)
        untrain_sub_list = []
        for stu_id in stu_ids:
            untrain_data_list = []
            for one_test_data in test_df_as_array:
                if one_test_data[0] == stu_id:
                    untrain_data_list.append([one_test_data[1],one_test_data[2]])
            untrain_sub_list.append(np.array(untrain_data_list))

        # ================== ユーザーごとの予測値ベクトルを取得
        predict_dfs = get_predict_df(stu_ids, reducted_matrix, rating_matrix, matrix_col_list, matrix_row_list)

        for p,untrain_data in enumerate(untrain_sub_list):
            try:
                one_users_untrain_sub = list(untrain_data[:,0])
                one_users_untrain_score = list(untrain_data[:,1])
                #one_users_id = stu_ids[p]
                #one_users_untrain_score = []
                #for sub in one_users_untrain_sub:
                #    sub_score = rating_matrix[matrix_row_list.index(sub),matrix_col_list.index(one_users_id)]
                #    one_users_untrain_score.append(sub_score)
                one_users_dict = dict(zip(one_users_untrain_sub,one_users_untrain_score))
                my_predict = predict_dfs[p]
                my_predict = np.array(my_predict)

                #a = [(i[0],i[1],one_users_untrain_score[[k for k,row in enumerate(untrain_data) if row == i[0]][0]]) for n,i in enumerate(my_predict) if i[0] in one_users_untrain_sub]
                a = [(i[0],i[1],one_users_dict[i[0]]) for n,i in enumerate(my_predict) if i[0] in one_users_untrain_sub]    
                for i in a:
                    result.append(i)
                    cul_count += 1
                    cul_sum += (float(i[1])-float(i[2]))**2
                if stu_ids[p] == "16bc046c":
                    print("16bc046c")
                    print(one_users_dict)
                    for i in a:
                        my_count += 1
                        my_sum += (float(i[1])-float(i[2]))**2
                        print(i)
            except:
                pass
        print("==========================")

    print("RMSE:{0}".format((cul_sum/cul_count)**(1/2)))
    print("my_RMSE:{0}".format((my_sum/my_count)**(1/2)))
    return result


def my_evaluation_check_for_bias(rearranged_train_dfs, rearranged_test_dfs):

    cul_count = 0
    cul_sum = 0
    my_count = 0
    my_sum = 0
    pred_value_sum = 0
    measured_value_sum = 0
    result = []
    conf_v = 0.14951114
    intercept_v = 2.24686714


    for n,rearranged_train_df in enumerate(rearranged_train_dfs):
        # print(train_df)
        test_df = rearranged_test_dfs[n]

        # 授業名を行、user_idを列とする行列(欠測値は0埋め)
        rating_matrix = rearranged_train_df.pivot(index='1_sub_name', columns='0_stu_id', values='2_score').fillna(0)
        matrix_row_list = list(rating_matrix.index)
        matrix_col_list = list(rating_matrix.columns)
        rating_matrix = rating_matrix.as_matrix()

        reducted_matrix_nmf = reduct_matrix_nmf(rating_matrix, k=100)
        # =================stu_ids
        stu_ids = []
        for stu_id in list(np.array(rearranged_train_df)[:,0]):
            if stu_id not in stu_ids:
                stu_ids.append(stu_id)

        # =================untrain_sub_list
        test_df_as_array = np.array(test_df)
        untrain_sub_list = []
        for stu_id in stu_ids:
            untrain_data_list = []
            for one_test_data in test_df_as_array:
                if one_test_data[0] == stu_id:
                    untrain_data_list.append([one_test_data[1],one_test_data[2]])
            untrain_sub_list.append(np.array(untrain_data_list))

        # ================== ユーザーごとの予測値ベクトルを取得
        predict_dfs = get_predict_df(stu_ids, reducted_matrix_nmf, rating_matrix, matrix_col_list, matrix_row_list)

        for p,untrain_data in enumerate(untrain_sub_list):
            try:
                one_users_untrain_sub = list(untrain_data[:,0])
                one_users_untrain_score = list(untrain_data[:,1])
                one_users_dict = dict(zip(one_users_untrain_sub,one_users_untrain_score))
                my_predict = predict_dfs[p]
                my_predict = np.array(my_predict)

                #a = [(i[0],i[1],one_users_untrain_score[[k for k,row in enumerate(untrain_data) if row == i[0]][0]]) for n,i in enumerate(my_predict) if i[0] in one_users_untrain_sub]
                a = [(i[0],conf_v * (i[1] + 1) + intercept_v,one_users_dict[i[0]]) for n,i in enumerate(my_predict) if i[0] in one_users_untrain_sub]    
                
                for i in a:
                    result.append(i)
                    cul_count += 1
                    cleaned_pred_value = float(i[1])
                    #if cleaned_pred_value > 4:
                    #    cleaned_pred_value = 4
                    cul_sum += (cleaned_pred_value-float(i[2]))**2
                    pred_value_sum += float(i[1])
                    measured_value_sum += float(i[2])
                if stu_ids[p] == "16bc046c":
                    print("16bc046c")
                    print(one_users_dict)
                    for i in a:
                        my_count += 1
                        cleaned_pred_value = float(i[1])
                        #if cleaned_pred_value > 4:
                        #    cleaned_pred_value = 4
                        my_sum += (cleaned_pred_value-float(i[2]))**2
                        print(i)
            except:
                pass
        print("==========================")

    print("RMSE:{0}".format((cul_sum/cul_count)**(1/2)))
    print("my_RMSE:{0}".format((my_sum/my_count)**(1/2)))
    print(measured_value_sum/pred_value_sum)
    return result


def mf_dataset_generator_as_df(mf_data_params):
    file_name = mf_data_params[0]
    gakunen_gakka_code = mf_data_params[1]
    only_gakka_code = mf_data_params[2]
    remove_sub = mf_data_params[3]

    df = csv_reader(file_name)
    cleaned_df = cleanning_df(df)
    cleaned_df_as_array, sorted_student_list = remove_noise_data(cleaned_df, gakunen_gakka_code, only_gakka_code, remove_sub)
    df = generate_dataset(cleaned_df_as_array)
    df, sorted_student_list = delete_over_two_subject_and_nan(df, sorted_student_list)
    return df, sorted_student_list


def matrix_factorization_processor(df, user_ids, mode="NMF", demention=50, recommend_count=20):
    # 授業名を行、user_idを列とする行列(欠測値は0埋め)
    rating_matrix = df.pivot(index='1_sub_name', columns='0_stu_id', values='2_score').fillna(0)
    matrix_row_list = list(rating_matrix.index)
    matrix_col_list = list(rating_matrix.columns)
    rating_matrix = rating_matrix.as_matrix()
    if mode == "SVD":
        reducted_matrix = reduct_matrix(rating_matrix, k=demention)
        print('MAE: {}'.format(np.average(np.absolute(rating_matrix - reducted_matrix))))
    else:
        reducted_matrix = reduct_matrix_nmf(rating_matrix, k=demention)
        print('MAE: {}'.format(np.average(np.absolute(rating_matrix - reducted_matrix))))

    predicted_vectors = print_ranking(user_ids, matrix_row_list, matrix_col_list, reducted_matrix, rating_matrix, recommend_count)
    return predicted_vectors



