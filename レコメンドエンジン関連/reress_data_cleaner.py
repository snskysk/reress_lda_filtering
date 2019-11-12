# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import codecs
import collections



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
        "score":np_df[:,7],
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
        "score":np_df[:,7],
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
        array_data = np.delete(array_data, np.where(array_data[:,2] != g_code),axis=0)        

    if len(remove_sub) > 0:
        for i,w in enumerate(remove_sub):
            array_data = np.delete(array_data, np.where(array_data[:,0] ==w),axis=0)# カテゴリ名
        
    for n,i in enumerate(array_data):
        if i[3] == 0:
            array_data[n,3] = 2
    cleaned_df = pd.DataFrame(array_data)
    cleaned_df.columns = ["category",'gakunen_gakka',"score",'only_gakka',"student_id", "subject_name"]
    sorted_student_list = cleaned_df["student_id"].value_counts().index.tolist()#uniqueなIDを取り出してリスト化
    return array_data, sorted_student_list

def split_data2train_test_by_id(array_data, sorted_student_list, test_id):
    train_stu_id = []
    train_stu_data = []
    test_stu_id = []
    test_stu_data = []
    for i in array_data:
        if i[4] in test_id:
            if i[4] not in test_stu_id:
                test_stu_id.append(i[4])
            test_stu_data.append(i)
        else:
            if i[4] not in train_stu_id:
                train_stu_id.append(i[4])
            train_stu_data.append(i)
    train_stu_data = np.array(train_stu_data)
    test_stu_data = np.array(test_stu_data)
    return train_stu_data, train_stu_id, test_stu_data, test_stu_id

def split_data2train_test_by_ratio(array_data, sorted_student_list, test_ratio):
    # array_data[1]=gakunen_gakka array_data[2]=gakka array_data[4]=id
    test_num = (len(sorted_student_list)//10)*test_ratio
    pick_up_num = [np.random.randint(0,len(sorted_student_list)) for i in range(test_num)]
    pick_up_num = sorted(pick_up_num, reverse=True)
    train_stu_id = []
    train_stu_data = []
    test_stu_id = []
    test_stu_data = []
    for n,i in enumerate(array_data):
        if n in pick_up_num:
            if i[4] not in test_stu_id:
                test_stu_id.append(i[4])
            test_stu_data.append(i)
        else:
            if i[4] not in train_stu_id:
                train_stu_id.append(i[4])
            train_stu_data.append(i)
            
    train_stu_data = np.array(train_stu_data)
    test_stu_data = np.array(test_stu_data)
    return train_stu_data, train_stu_id, test_stu_data, test_stu_id

def generate_dataset(array):
    df = pd.DataFrame({
        "0_stu_id":array[:,4],
        "1_sub_name":array[:,5],
        "2_score":array[:,3],
    })
    return df


# columns = ["0_stu_id","1_sub_name","2_score"]のデータにおいて
def delete_over_two_subject_and_nan(df, student_list):
    df_as_array = np.array(df)
    new_array = []
    for s in student_list:
        box = []
        for i in df_as_array:
            if i[0] == s:
                box.append(i[1])
        box = list(set(box))
        for i in df_as_array:
            if i[0] == s:
                if i[1] in box:
                    new_array.append(i)
                    box.remove(i[1])
                else:
                    pass
                if len(box) == 0:
                    break
    new_array = np.array(new_array)

    # NULL値を削除
    delete_list = []
    for s in student_list:
        box = []
        for i in new_array:
            if i[0] == s:
                box.append(i[1])
                if list(set(box))[0] is np.nan:
                    delete_list.append(s)

    for d in delete_list:
        student_list.remove(d)
        new_array = np.delete(new_array, np.where(new_array[:,1] ==d),axis=0)
    
    df = pd.DataFrame(new_array)
    df.columns = ["0_stu_id","1_sub_name","2_score"]
    return df ,student_list

def sort_df_by_lda_data(df, student_list):
    df_as_array = np.array(df)
    new_array = []
    for i in df_as_array:
        if i[0] in student_list:
            new_array.append(i)
    new_array = np.array(new_array)
    df = pd.DataFrame(new_array)
    df.columns = ["0_stu_id","1_sub_name","2_score"]
    return df ,student_list

