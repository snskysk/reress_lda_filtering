# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import codecs
from operator import itemgetter
import my_utils
from gensim import corpora, models
import gensim
import pickle

class ReressLda():
    def __init__(self, msg="Re:Re:ss Lda Processer"):
        self.msg = msg

    def csv_reader(self, file_name):
        # dataの読み込み。本来はPostgreSQL上に保存してあるが、今回はローカルに引っ張ってきたCSVデータを利用
        # unicodeエラーとヘッダー自動割り当てに対処して読み込み
        with codecs.open(file_name, "r", "Shift-JIS", "ignore") as file:
            df = pd.read_csv(file, header=None, delimiter=",")
        return df

    def cleaning_df(self, df):
        df = np.array(df)
        cleaned_list = []
        for i in df:
            if i[13] == "＊必）言語Ａ＊" or i[13] == "＊必）言語Ｂ＊":
                pass
            elif i[13] == "＊学びの精神＊" or i[13] == "＊多彩な学び，スポ＊":
                pass
            elif i[3][:2] == "15":
                pass
            elif i[3][2:4] == "ｈｍ":
                pass
            else:
                cleaned_list.append(i)
        df = pd.DataFrame(cleaned_list)
        return df

    def get_student_list(self, df):
        np_df = np.array(df)
        sort_students = sorted(np_df, key=itemgetter(3),reverse=False)
        sort_students = [i[3] for i in sort_students]
        student_list = list(set(sort_students))
        return student_list

    def get_train_and_test_student_list(self, df, test_student_num, random_user_or_not):
        np_df = np.array(df)
        sort_students = sorted(np_df, key=itemgetter(3),reverse=False)
        sort_students = [i[3] for i in sort_students]
        student_list = list(set(sort_students))
        if random_user_or_not == 1:
            random_num = np.random.randint(len(student_list))
            test_student_list = [student_list[random_num]]
            train_student_list = [i for i in student_list if i != test_student_list[0]]
        else:
            test_student_list = [random_user_or_not]
            train_student_list = [i for i in student_list if i != test_student_list[0]]    
        return train_student_list, test_student_list

    def generate_document_with_mecab(self, student_list, df):
        # mecabで分かち書きした文書群を生成
        np_df = np.array(df)
        docs = []
        for i in student_list:
            each_sub = [sub[4] for sub in np_df if sub[3]==i]
            each_sub_as_text = "".join(str(each_sub))
            each_sub_as_text = each_sub_as_text.replace("　","")
            docs.append(my_utils.stems(each_sub_as_text))
        return docs

    def generate_dict_for_gensim(self, docs, load_or_not):
        # gensimで文書中の出現単語と単語IDをマッピングするための辞書を生成
        if load_or_not == 0:  # 新たに学習し保存する場合
            dictionary = gensim.corpora.Dictionary(docs)
            dictionary.save_as_text('./data/text.dict')
        else:  # 過去の辞書をロードする場合
            dictionary = gensim.corpora.Dictionary.load_from_text('./data/text.dict')
        return dictionary

    def generate_corpus(self, docs, dictionary, load_or_not):
        # コーパスの生成
        if load_or_not == 0:  # 新たに学習し保存する場合
            corpus = [dictionary.doc2bow(doc) for doc in docs]
            gensim.corpora.MmCorpus.serialize('./data/text.mm', corpus)
        else:  # 過去のコーパスをロードする場合
            corpus = gensim.corpora.MmCorpus('./data/text.mm')
        return corpus

    def corpus_into_tfidf(self, corpus, load_or_not):
        # コーパスにtfidf処理を施す
        if load_or_not == 0:  # 新たに学習し保存する場合
            tfidf = gensim.models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]

            with open('./data/corpus_tfidf.dump', mode='wb') as f:
                pickle.dump(corpus_tfidf, f)
        else:  # 過去のコーパスをロードする場合
            with open('./data/corpus_tfidf.dump', mode='rb') as f:
                corpus_tfidf = pickle.load(f)
        return corpus_tfidf

    def train_lda_model(self, corpus, dictionary, topics, load_or_not):
        # 分類器を学習させる
        if load_or_not == 0:  # 新たに学習し保存する場合
            lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary,
                                        num_topics=topics, minimum_probability=0.001,
                                        passes=20, update_every=0, chunksize=10000)
            lda.save('./data/lda.model')
        else:  # 過去のモデルをロードする場合
            lda = gensim.models.LdaModel.load('./data/lda.model')
        return lda

    def test_lda_model(self, lda, dictionary, df, test_student_list):
        # 基準となるユーザーのLDAベクトルを算出し、分類結果も出力する
        np_df = np.array(df)
        for i in test_student_list:
            each_sub = [sub[4] for sub in np_df if sub[3]==i]
            each_sub_as_text = "".join(str(each_sub))
            vec = dictionary.doc2bow(my_utils.stems(each_sub_as_text))
            print("Base Users Data...")
            print(i, each_sub_as_text)
            print("Base Users LDA Vector")
            print(lda[vec])
        return lda[vec]

    def compare_student_vecs_generator(self, lda, dictionary, df, train_student_list):
        # 基準ユーザー以外のLDAベクトルを算出する
        np_df = np.array(df)
        compare_student_vecs = []
        for i in train_student_list:
            each_sub = [sub[4] for sub in np_df if sub[3]==i]
            each_sub_as_text = "".join(str(each_sub))
            vec = dictionary.doc2bow(my_utils.stems(each_sub_as_text))
            compare_student_vecs.append(lda[vec])
        return compare_student_vecs

    def similarity_calculator(self, base_student_id, base_student_vecs, compare_student_id, compare_student_vecs, topics):
        # 基準となるユーザーのLDAベクトルを、コサイン類似度計算用のベクトルに変換する
        base_vecs = [0 for i in range(topics)]
        for i,value in base_student_vecs:
            base_vecs[i] = value
        array_base_vecs = np.array(base_vecs)

        # 基準ユーザー以外のLDAベクトルを、コサイン類似度計算用のベクトルに変換する
        compare_student_similarity = []
        for r in compare_student_vecs:
            compare_vecs = [0 for i in range(topics)]
            for i,value in r:
                compare_vecs[i] = value
            compare_student_similarity.append(self.cos_sim(array_base_vecs, np.array(compare_vecs)))
        similarity_df = pd.DataFrame({
            "student_id":compare_student_id,
            "similarity":compare_student_similarity,
        })
        return similarity_df

    def cos_sim(self, v1, v2):
        # コサイン類似度計算用の関数
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def lda_pick_up_student(self, params_as_list):
        file_name = params_as_list[0]
        test_student_num = params_as_list[1]
        random_user_or_not = params_as_list[2]
        gensim_dict_load_or_not = params_as_list[3]
        corpus_load_or_not = params_as_list[4]
        corpus_tfidf_load_or_not = params_as_list[5]
        model_load_or_not = params_as_list[6]
        topics = params_as_list[7]
        RELATE_STUDENT_NUM = params_as_list[8]
        MIN_SIMILARITY = params_as_list[9]

        # データ読み込み→ランダムで基準ユーザーと基準以外を決定→ユーザーごとに履修データを文書化し分かち書き文書群を生成
        df = self.csv_reader(file_name)
        df = self.cleaning_df(df)
        # student_list = get_student_list(df)
        student_list, test_student_list = self.get_train_and_test_student_list(df, test_student_num, random_user_or_not)
        docs = self.generate_document_with_mecab(student_list, df)
        
        # gensim内で利用する辞書を生成→コーパスの生成→コーパスにtfidf処理を施す→分類器を学習させる
        dictionary = self.generate_dict_for_gensim(docs, gensim_dict_load_or_not)
        corpus = self.generate_corpus(docs, dictionary, corpus_load_or_not)
        corpus = self.corpus_into_tfidf(corpus, corpus_tfidf_load_or_not)
        print("LDA train start...")
        lda = self.train_lda_model(corpus, dictionary, topics, model_load_or_not)
        print("Done...")
        print("LDA test start...")

        # 基準ユーザーのLDAベクトルを算出、分類結果も出力→基準ユーザー以外のLDAベクトルを算出→コサイン類似度計算
        base_student_vecs = self.test_lda_model(lda, dictionary, df, test_student_list)    
        compare_student_vecs = self.compare_student_vecs_generator(lda, dictionary, df, student_list)
        similarity_df = self.similarity_calculator(test_student_list, base_student_vecs, student_list, compare_student_vecs, topics)
        
        relate_student_list_as_df = similarity_df[similarity_df.similarity > MIN_SIMILARITY].sort_values(by="similarity", ascending=False).head(RELATE_STUDENT_NUM)
        list_for_recommendation = [i for i in test_student_list]
        list_for_recommendation.extend(relate_student_list_as_df["student_id"].values)
        print("Done...")
    
        return lda, relate_student_list_as_df, list_for_recommendation



if __name__ == "__main__":
    file_name = "subjects.csv"
    file_name = "subjects_830.csv"
    test_student_num = 1
    random_user_or_not = 1
    random_user_or_not = "16bc046c"

    gensim_dict_load_or_not = 0
    corpus_load_or_not = 0
    corpus_tfidf_load_or_not = 0
    model_load_or_not = 0
    topics = 50

    # データ読み込み→ランダムで基準ユーザーと基準以外を決定→ユーザーごとに履修データを文書化し分かち書き文書群を生成
    df = csv_reader(file_name)
    df = cleaning_df(df)
    # student_list = get_student_list(df)
    student_list, test_student_list = get_train_and_test_student_list(df, test_student_num, random_user_or_not)
    docs = generate_document_with_mecab(student_list, df)
    
    # gensim内で利用する辞書を生成→コーパスの生成→コーパスにtfidf処理を施す→分類器を学習させる
    dictionary = generate_dict_for_gensim(docs, gensim_dict_load_or_not)
    corpus = generate_corpus(docs, dictionary, corpus_load_or_not)
    corpus = corpus_into_tfidf(corpus, corpus_tfidf_load_or_not)
    lda = train_lda_model(corpus, dictionary, topics, model_load_or_not)
    
    for i in range(topics):
        print('tpc_{0}: {1}'.format(i, lda.print_topic(i)[0:80]+'...'))

    print("=========================")
    # 基準ユーザーのLDAベクトルを算出、分類結果も出力→基準ユーザー以外のLDAベクトルを算出→コサイン類似度計算
    base_student_vecs = test_lda_model(lda, dictionary, df, test_student_list)    
    compare_student_vecs = compare_student_vecs_generator(lda, dictionary, df, student_list)
    similarity_df = similarity_calculator(test_student_list, base_student_vecs, student_list, compare_student_vecs, topics)
    RELATE_STUDENT_NUM = 50
    MIN_SIMILARITY = 0.6
    relate_store_list = similarity_df[similarity_df.similarity > MIN_SIMILARITY].sort_values(by="similarity", ascending=False).head(RELATE_STUDENT_NUM)
    print("=========================")
    print(relate_store_list.head())
    print("=========================")
    list_for_recommendation = [i for i in test_student_list]
    list_for_recommendation.extend(relate_store_list["student_id"].values)
    print(list_for_recommendation)