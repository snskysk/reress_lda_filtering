# -*- coding: utf-8 -*-

from reress_pearson_recommendation import *
from reress_lda_sort_student import *
from reress_matrix_factorization import *
from reress_data_cleaner import *

# ベースラインとして採用したピアソン相関によるレコメンドの実行コード
def baseline_pearson_processor():
    # ピアソン相関用のパラメータ設定
    file_name = "subjects_830.csv"
    gakunen_gakka_code = ""
    only_gakka_code = ""
    remove_sub = ["＊必）言語Ａ＊","＊必）言語Ｂ＊","＊学びの精神＊","＊多彩な学び，スポ＊"]
    pearson_params = [file_name, gakunen_gakka_code, only_gakka_code, remove_sub]

    # ピアソン相関用のデータセットを生成
    my_dataset = pearson_dataset_generator(pearson_params)
    base_user_list = ["16bc046c"]# example
    base_user = base_user_list[0]

    # reress_pearson_recommendation.py より レコメンド処理実行
    main_recommendation(base_user, my_dataset, recommend_count=20, eclidean_count=10)

# 最良のレリバンスが得られた非負値行列因子分解によるレコメンドの実行コード
def nmf_single_processor():
    # nmf用のパラメータ
    file_name = "subjects_830.csv"
    gakunen_gakka_code = ""
    only_gakka_code = ""
    remove_sub = ["＊必）言語Ａ＊","＊必）言語Ｂ＊","＊学びの精神＊","＊多彩な学び，スポ＊"]
    mf_data_params = [file_name, gakunen_gakka_code, only_gakka_code, remove_sub]

    # 非負値行列因子分解用のデータセットを生成
    df, sorted_student_list = mf_dataset_generator_as_df(mf_data_params)
    user_ids = ["16bc046c"]# example

    mode="NMF"# SVDも指定可能
    demention = 50
    # reress_matrix_factorization.py より レコメンド処理実行
    predicted_vectors = matrix_factorization_processor(df, user_ids, mode, demention, recommend_count=20)

# 新たに提唱したLDAフィルタリングの実行コード
def lda_and_nmf():
    ### LDA用のパラメータ入力
    file_name = "subjects_830.csv"
    test_student_num = 1
    base_user_list = ["16bc046c"]# example
    random_user_or_not = base_user_list[0]

    gensim_dict_load_or_not = 0
    corpus_load_or_not = 0
    corpus_tfidf_load_or_not = 0
    model_load_or_not = 0
    topics = 80
    RELATE_STUDENT_NUM = 50
    MIN_SIMILARITY = 0.6
    params_as_list = [file_name, test_student_num, random_user_or_not, gensim_dict_load_or_not, corpus_load_or_not, corpus_tfidf_load_or_not, model_load_or_not, topics, RELATE_STUDENT_NUM, MIN_SIMILARITY]

    ### reress_lda_sort_student.py より インスタンスを生成しLDA実行
    rlda = ReressLda()
    lda, relate_df, sorted_result = rlda.lda_pick_up_student(params_as_list)

    ### NMF用のデータセット生成
    gakunen_gakka_code = ""
    only_gakka_code = ""
    remove_sub = ["＊必）言語Ａ＊","＊必）言語Ｂ＊","＊学びの精神＊","＊多彩な学び，スポ＊"]
    mf_data_params = [file_name, gakunen_gakka_code, only_gakka_code, remove_sub]
    df, sorted_student_list = mf_dataset_generator_as_df(mf_data_params)

    ### NMF用に作ったデータセットを LDAで絞り込んだユーザーのみのデータに変換する
    df, lda_sorted_student_list = sort_df_by_lda_data(df, sorted_result)

    ### NMF実行
    print("Matrix Factorization start...")
    user_ids = [random_user_or_not]
    mode="NMF"#SVD
    demention = 50
    predicted_vectors = matrix_factorization_processor(df, user_ids, mode, demention, recommend_count=20)
