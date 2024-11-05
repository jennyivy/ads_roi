import pandas as pd
import os
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from collections import defaultdict
import re
import category_encoders as ce
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sqlalchemy import create_engine, select
from pandas import read_sql_query
from sqlalchemy.sql import text
import pymssql
import time

def create_folder_if_not_exists(folder_path):  
    """  
    创建文件夹，如果文件夹已经存在则不创建。  
  
    参数:  
    folder_path (str): 文件夹的路径。  
    """  
    if not os.path.exists(folder_path):  
        os.makedirs(folder_path)  
        print(f"文件夹 '{folder_path}' 创建成功。")  
    else:  
        print(f"文件夹 '{folder_path}' 已经存在，无需创建。") 

def get_content_feature(df):

    df['标题素材_id_cnt'] = df['标题素材id近1天的展示次数'].apply(lambda x:len(x) if x else None)
    df['标题素材id近1天的展示次数_avg'] = df['标题素材id近1天的展示次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的展示次数_max'] = df['标题素材id近1天的展示次数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的展示次数_sum'] = df['标题素材id近1天的展示次数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的消耗_avg'] = df['标题素材id近1天的消耗'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的消耗_max'] = df['标题素材id近1天的消耗'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的消耗_sum'] = df['标题素材id近1天的消耗'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的点击次数_avg'] = df['标题素材id近1天的点击次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的点击次数_max'] = df['标题素材id近1天的点击次数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的点击次数_sum'] = df['标题素材id近1天的点击次数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的点击率_avg'] = df['标题素材id近1天的点击率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的点击率_max'] = df['标题素材id近1天的点击率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的转化数_avg'] = df['标题素材id近1天的转化数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的转化数_max'] = df['标题素材id近1天的转化数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的转化数_sum'] = df['标题素材id近1天的转化数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的转化率_avg'] = df['标题素材id近1天的转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的转化率_max'] = df['标题素材id近1天的转化率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的直接成交订单数_avg'] = df['标题素材id近1天的直接成交订单数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的直接成交订单数_max'] = df['标题素材id近1天的直接成交订单数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的直接成交订单数_sum'] = df['标题素材id近1天的直接成交订单数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的直接成交金额_avg'] = df['标题素材id近1天的直接成交金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的直接成交金额_max'] = df['标题素材id近1天的直接成交金额'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的直接成交金额_sum'] = df['标题素材id近1天的直接成交金额'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的直接支付ROI_avg'] = df['标题素材id近1天的直接支付ROI'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的直接支付ROI_max'] = df['标题素材id近1天的直接支付ROI'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的播放数_avg'] = df['标题素材id近1天的播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的播放数_max'] = df['标题素材id近1天的播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的播放数_sum'] = df['标题素材id近1天的播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的3s播放数_avg'] = df['标题素材id近1天的3s播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的3s播放数_max'] = df['标题素材id近1天的3s播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的3s播放数_sum'] = df['标题素材id近1天的3s播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的3s完播率_avg'] = df['标题素材id近1天的3s完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的3s完播率_max'] = df['标题素材id近1天的3s完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的25%进度播放数_avg'] = df['标题素材id近1天的25%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的25%进度播放数_max'] = df['标题素材id近1天的25%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的25%进度播放数_sum'] = df['标题素材id近1天的25%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的25%进度完播率_avg'] = df['标题素材id近1天的25%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的25%进度完播率_max'] = df['标题素材id近1天的25%进度完播率'].apply(lambda x:max(x) if x else None)   
    df['标题素材id近1天的50%进度播放数_avg'] = df['标题素材id近1天的50%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的50%进度播放数_max'] = df['标题素材id近1天的50%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的50%进度播放数_sum'] = df['标题素材id近1天的50%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的50%进度完播率_avg'] = df['标题素材id近1天的50%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的50%进度完播率_max'] = df['标题素材id近1天的50%进度完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的75%进度播放数_avg'] = df['标题素材id近1天的75%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的75%进度播放数_max'] = df['标题素材id近1天的75%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的75%进度播放数_sum'] = df['标题素材id近1天的75%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的75%进度完播率_avg'] = df['标题素材id近1天的75%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的75%进度完播率_max'] = df['标题素材id近1天的75%进度完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的播放完成数_avg'] = df['标题素材id近1天的播放完成数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的播放完成数_max'] = df['标题素材id近1天的播放完成数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近1天的播放完成数_sum'] = df['标题素材id近1天的播放完成数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近1天的完播率_avg'] = df['标题素材id近1天的完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近1天的完播率_max'] = df['标题素材id近1天的完播率'].apply(lambda x:max(x) if x else None) 
    df['标题素材id近7天的展示次数_avg'] = df['标题素材id近7天的展示次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的展示次数_max'] = df['标题素材id近7天的展示次数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的展示次数_sum'] = df['标题素材id近7天的展示次数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的消耗_avg'] = df['标题素材id近7天的消耗'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的消耗_max'] = df['标题素材id近7天的消耗'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的消耗_sum'] = df['标题素材id近7天的消耗'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的点击次数_avg'] = df['标题素材id近7天的点击次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的点击次数_max'] = df['标题素材id近7天的点击次数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的点击次数_sum'] = df['标题素材id近7天的点击次数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的点击率_avg'] = df['标题素材id近7天的点击率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的点击率_max'] = df['标题素材id近7天的点击率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的转化数_avg'] = df['标题素材id近7天的转化数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的转化数_max'] = df['标题素材id近7天的转化数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的转化数_sum'] = df['标题素材id近7天的转化数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的转化率_avg'] = df['标题素材id近7天的转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的转化率_max'] = df['标题素材id近7天的转化率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的直接成交订单数_avg'] = df['标题素材id近7天的直接成交订单数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的直接成交订单数_max'] = df['标题素材id近7天的直接成交订单数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的直接成交订单数_sum'] = df['标题素材id近7天的直接成交订单数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的直接成交金额_avg'] = df['标题素材id近7天的直接成交金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的直接成交金额_max'] = df['标题素材id近7天的直接成交金额'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的直接成交金额_sum'] = df['标题素材id近7天的直接成交金额'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的直接支付ROI_avg'] = df['标题素材id近7天的直接支付ROI'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的直接支付ROI_max'] = df['标题素材id近7天的直接支付ROI'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的播放数_avg'] = df['标题素材id近7天的播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的播放数_max'] = df['标题素材id近7天的播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的播放数_sum'] = df['标题素材id近7天的播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的3s播放数_avg'] = df['标题素材id近7天的3s播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的3s播放数_max'] = df['标题素材id近7天的3s播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的3s播放数_sum'] = df['标题素材id近7天的3s播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的3s完播率_avg'] = df['标题素材id近7天的3s完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的3s完播率_max'] = df['标题素材id近7天的3s完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的25%进度播放数_avg'] = df['标题素材id近7天的25%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的25%进度播放数_max'] = df['标题素材id近7天的25%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的25%进度播放数_sum'] = df['标题素材id近7天的25%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的25%进度完播率_avg'] = df['标题素材id近7天的25%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的25%进度完播率_max'] = df['标题素材id近7天的25%进度完播率'].apply(lambda x:max(x) if x else None)   
    df['标题素材id近7天的50%进度播放数_avg'] = df['标题素材id近7天的50%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的50%进度播放数_max'] = df['标题素材id近7天的50%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的50%进度播放数_sum'] = df['标题素材id近7天的50%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的50%进度完播率_avg'] = df['标题素材id近7天的50%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的50%进度完播率_max'] = df['标题素材id近7天的50%进度完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的75%进度播放数_avg'] = df['标题素材id近7天的75%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的75%进度播放数_max'] = df['标题素材id近7天的75%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的75%进度播放数_sum'] = df['标题素材id近7天的75%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的75%进度完播率_avg'] = df['标题素材id近7天的75%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的75%进度完播率_max'] = df['标题素材id近7天的75%进度完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的播放完成数_avg'] = df['标题素材id近7天的播放完成数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的播放完成数_max'] = df['标题素材id近7天的播放完成数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近7天的播放完成数_sum'] = df['标题素材id近7天的播放完成数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近7天的完播率_avg'] = df['标题素材id近7天的完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近7天的完播率_max'] = df['标题素材id近7天的完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的展示次数_avg'] = df['标题素材id近30天的展示次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的展示次数_max'] = df['标题素材id近30天的展示次数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的展示次数_sum'] = df['标题素材id近30天的展示次数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的消耗_avg'] = df['标题素材id近30天的消耗'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的消耗_max'] = df['标题素材id近30天的消耗'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的消耗_sum'] = df['标题素材id近30天的消耗'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的点击次数_avg'] = df['标题素材id近30天的点击次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的点击次数_max'] = df['标题素材id近30天的点击次数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的点击次数_sum'] = df['标题素材id近30天的点击次数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的点击率_avg'] = df['标题素材id近30天的点击率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的点击率_max'] = df['标题素材id近30天的点击率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的转化数_avg'] = df['标题素材id近30天的转化数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的转化数_max'] = df['标题素材id近30天的转化数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的转化数_sum'] = df['标题素材id近30天的转化数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的转化率_avg'] = df['标题素材id近30天的转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的转化率_max'] = df['标题素材id近30天的转化率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的直接成交订单数_avg'] = df['标题素材id近30天的直接成交订单数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的直接成交订单数_max'] = df['标题素材id近30天的直接成交订单数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的直接成交订单数_sum'] = df['标题素材id近30天的直接成交订单数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的直接成交金额_avg'] = df['标题素材id近30天的直接成交金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的直接成交金额_max'] = df['标题素材id近30天的直接成交金额'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的直接成交金额_sum'] = df['标题素材id近30天的直接成交金额'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的直接支付ROI_avg'] = df['标题素材id近30天的直接支付ROI'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的直接支付ROI_max'] = df['标题素材id近30天的直接支付ROI'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的播放数_avg'] = df['标题素材id近30天的播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的播放数_max'] = df['标题素材id近30天的播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的播放数_sum'] = df['标题素材id近30天的播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的3s播放数_avg'] = df['标题素材id近30天的3s播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的3s播放数_max'] = df['标题素材id近30天的3s播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的3s播放数_sum'] = df['标题素材id近30天的3s播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的3s完播率_avg'] = df['标题素材id近30天的3s完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的3s完播率_max'] = df['标题素材id近30天的3s完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的25%进度播放数_avg'] = df['标题素材id近30天的25%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的25%进度播放数_max'] = df['标题素材id近30天的25%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的25%进度播放数_sum'] = df['标题素材id近30天的25%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的25%进度完播率_avg'] = df['标题素材id近30天的25%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的25%进度完播率_max'] = df['标题素材id近30天的25%进度完播率'].apply(lambda x:max(x) if x else None)   
    df['标题素材id近30天的50%进度播放数_avg'] = df['标题素材id近30天的50%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的50%进度播放数_max'] = df['标题素材id近30天的50%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的50%进度播放数_sum'] = df['标题素材id近30天的50%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的50%进度完播率_avg'] = df['标题素材id近30天的50%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的50%进度完播率_max'] = df['标题素材id近30天的50%进度完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的75%进度播放数_avg'] = df['标题素材id近30天的75%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的75%进度播放数_max'] = df['标题素材id近30天的75%进度播放数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的75%进度播放数_sum'] = df['标题素材id近30天的75%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的75%进度完播率_avg'] = df['标题素材id近30天的75%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的75%进度完播率_max'] = df['标题素材id近30天的75%进度完播率'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的播放完成数_avg'] = df['标题素材id近30天的播放完成数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的播放完成数_max'] = df['标题素材id近30天的播放完成数'].apply(lambda x:max(x) if x else None)
    df['标题素材id近30天的播放完成数_sum'] = df['标题素材id近30天的播放完成数'].apply(lambda x:sum(x) if x else None)
    df['标题素材id近30天的完播率_avg'] = df['标题素材id近30天的完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['标题素材id近30天的完播率_max'] = df['标题素材id近30天的完播率'].apply(lambda x:max(x) if x else None)
    
    df['视频素材_id_cnt'] = df['视频素材id近1天的展示次数'].apply(lambda x:len(x) if x else None)
    df['视频素材id近1天的展示次数_avg'] = df['视频素材id近1天的展示次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的展示次数_max'] = df['视频素材id近1天的展示次数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的展示次数_sum'] = df['视频素材id近1天的展示次数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的消耗_avg'] = df['视频素材id近1天的消耗'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的消耗_max'] = df['视频素材id近1天的消耗'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的消耗_sum'] = df['视频素材id近1天的消耗'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的点击次数_avg'] = df['视频素材id近1天的点击次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的点击次数_max'] = df['视频素材id近1天的点击次数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的点击次数_sum'] = df['视频素材id近1天的点击次数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的点击率_avg'] = df['视频素材id近1天的点击率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的点击率_max'] = df['视频素材id近1天的点击率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的转化数_avg'] = df['视频素材id近1天的转化数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的转化数_max'] = df['视频素材id近1天的转化数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的转化数_sum'] = df['视频素材id近1天的转化数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的转化率_avg'] = df['视频素材id近1天的转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的转化率_max'] = df['视频素材id近1天的转化率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的直接成交订单数_avg'] = df['视频素材id近1天的直接成交订单数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的直接成交订单数_max'] = df['视频素材id近1天的直接成交订单数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的直接成交订单数_sum'] = df['视频素材id近1天的直接成交订单数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的直接成交金额_avg'] = df['视频素材id近1天的直接成交金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的直接成交金额_max'] = df['视频素材id近1天的直接成交金额'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的直接成交金额_sum'] = df['视频素材id近1天的直接成交金额'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的直接支付ROI_avg'] = df['视频素材id近1天的直接支付ROI'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的直接支付ROI_max'] = df['视频素材id近1天的直接支付ROI'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的播放数_avg'] = df['视频素材id近1天的播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的播放数_max'] = df['视频素材id近1天的播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的播放数_sum'] = df['视频素材id近1天的播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的3s播放数_avg'] = df['视频素材id近1天的3s播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的3s播放数_max'] = df['视频素材id近1天的3s播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的3s播放数_sum'] = df['视频素材id近1天的3s播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的3s完播率_avg'] = df['视频素材id近1天的3s完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的3s完播率_max'] = df['视频素材id近1天的3s完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的25%进度播放数_avg'] = df['视频素材id近1天的25%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的25%进度播放数_max'] = df['视频素材id近1天的25%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的25%进度播放数_sum'] = df['视频素材id近1天的25%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的25%进度完播率_avg'] = df['视频素材id近1天的25%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的25%进度完播率_max'] = df['视频素材id近1天的25%进度完播率'].apply(lambda x:max(x) if x else None)   
    df['视频素材id近1天的50%进度播放数_avg'] = df['视频素材id近1天的50%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的50%进度播放数_max'] = df['视频素材id近1天的50%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的50%进度播放数_sum'] = df['视频素材id近1天的50%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的50%进度完播率_avg'] = df['视频素材id近1天的50%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的50%进度完播率_max'] = df['视频素材id近1天的50%进度完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的75%进度播放数_avg'] = df['视频素材id近1天的75%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的75%进度播放数_max'] = df['视频素材id近1天的75%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的75%进度播放数_sum'] = df['视频素材id近1天的75%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的75%进度完播率_avg'] = df['视频素材id近1天的75%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的75%进度完播率_max'] = df['视频素材id近1天的75%进度完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的播放完成数_avg'] = df['视频素材id近1天的播放完成数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的播放完成数_max'] = df['视频素材id近1天的播放完成数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近1天的播放完成数_sum'] = df['视频素材id近1天的播放完成数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近1天的完播率_avg'] = df['视频素材id近1天的完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近1天的完播率_max'] = df['视频素材id近1天的完播率'].apply(lambda x:max(x) if x else None) 
    df['视频素材id近7天的展示次数_avg'] = df['视频素材id近7天的展示次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的展示次数_max'] = df['视频素材id近7天的展示次数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的展示次数_sum'] = df['视频素材id近7天的展示次数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的消耗_avg'] = df['视频素材id近7天的消耗'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的消耗_max'] = df['视频素材id近7天的消耗'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的消耗_sum'] = df['视频素材id近7天的消耗'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的点击次数_avg'] = df['视频素材id近7天的点击次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的点击次数_max'] = df['视频素材id近7天的点击次数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的点击次数_sum'] = df['视频素材id近7天的点击次数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的点击率_avg'] = df['视频素材id近7天的点击率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的点击率_max'] = df['视频素材id近7天的点击率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的转化数_avg'] = df['视频素材id近7天的转化数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的转化数_max'] = df['视频素材id近7天的转化数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的转化数_sum'] = df['视频素材id近7天的转化数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的转化率_avg'] = df['视频素材id近7天的转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的转化率_max'] = df['视频素材id近7天的转化率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的直接成交订单数_avg'] = df['视频素材id近7天的直接成交订单数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的直接成交订单数_max'] = df['视频素材id近7天的直接成交订单数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的直接成交订单数_sum'] = df['视频素材id近7天的直接成交订单数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的直接成交金额_avg'] = df['视频素材id近7天的直接成交金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的直接成交金额_max'] = df['视频素材id近7天的直接成交金额'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的直接成交金额_sum'] = df['视频素材id近7天的直接成交金额'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的直接支付ROI_avg'] = df['视频素材id近7天的直接支付ROI'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的直接支付ROI_max'] = df['视频素材id近7天的直接支付ROI'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的播放数_avg'] = df['视频素材id近7天的播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的播放数_max'] = df['视频素材id近7天的播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的播放数_sum'] = df['视频素材id近7天的播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的3s播放数_avg'] = df['视频素材id近7天的3s播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的3s播放数_max'] = df['视频素材id近7天的3s播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的3s播放数_sum'] = df['视频素材id近7天的3s播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的3s完播率_avg'] = df['视频素材id近7天的3s完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的3s完播率_max'] = df['视频素材id近7天的3s完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的25%进度播放数_avg'] = df['视频素材id近7天的25%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的25%进度播放数_max'] = df['视频素材id近7天的25%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的25%进度播放数_sum'] = df['视频素材id近7天的25%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的25%进度完播率_avg'] = df['视频素材id近7天的25%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的25%进度完播率_max'] = df['视频素材id近7天的25%进度完播率'].apply(lambda x:max(x) if x else None)   
    df['视频素材id近7天的50%进度播放数_avg'] = df['视频素材id近7天的50%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的50%进度播放数_max'] = df['视频素材id近7天的50%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的50%进度播放数_sum'] = df['视频素材id近7天的50%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的50%进度完播率_avg'] = df['视频素材id近7天的50%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的50%进度完播率_max'] = df['视频素材id近7天的50%进度完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的75%进度播放数_avg'] = df['视频素材id近7天的75%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的75%进度播放数_max'] = df['视频素材id近7天的75%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的75%进度播放数_sum'] = df['视频素材id近7天的75%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的75%进度完播率_avg'] = df['视频素材id近7天的75%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的75%进度完播率_max'] = df['视频素材id近7天的75%进度完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的播放完成数_avg'] = df['视频素材id近7天的播放完成数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的播放完成数_max'] = df['视频素材id近7天的播放完成数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近7天的播放完成数_sum'] = df['视频素材id近7天的播放完成数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近7天的完播率_avg'] = df['视频素材id近7天的完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近7天的完播率_max'] = df['视频素材id近7天的完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的展示次数_avg'] = df['视频素材id近30天的展示次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的展示次数_max'] = df['视频素材id近30天的展示次数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的展示次数_sum'] = df['视频素材id近30天的展示次数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的消耗_avg'] = df['视频素材id近30天的消耗'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的消耗_max'] = df['视频素材id近30天的消耗'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的消耗_sum'] = df['视频素材id近30天的消耗'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的点击次数_avg'] = df['视频素材id近30天的点击次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的点击次数_max'] = df['视频素材id近30天的点击次数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的点击次数_sum'] = df['视频素材id近30天的点击次数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的点击率_avg'] = df['视频素材id近30天的点击率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的点击率_max'] = df['视频素材id近30天的点击率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的转化数_avg'] = df['视频素材id近30天的转化数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的转化数_max'] = df['视频素材id近30天的转化数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的转化数_sum'] = df['视频素材id近30天的转化数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的转化率_avg'] = df['视频素材id近30天的转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的转化率_max'] = df['视频素材id近30天的转化率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的直接成交订单数_avg'] = df['视频素材id近30天的直接成交订单数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的直接成交订单数_max'] = df['视频素材id近30天的直接成交订单数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的直接成交订单数_sum'] = df['视频素材id近30天的直接成交订单数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的直接成交金额_avg'] = df['视频素材id近30天的直接成交金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的直接成交金额_max'] = df['视频素材id近30天的直接成交金额'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的直接成交金额_sum'] = df['视频素材id近30天的直接成交金额'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的直接支付ROI_avg'] = df['视频素材id近30天的直接支付ROI'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的直接支付ROI_max'] = df['视频素材id近30天的直接支付ROI'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的播放数_avg'] = df['视频素材id近30天的播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的播放数_max'] = df['视频素材id近30天的播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的播放数_sum'] = df['视频素材id近30天的播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的3s播放数_avg'] = df['视频素材id近30天的3s播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的3s播放数_max'] = df['视频素材id近30天的3s播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的3s播放数_sum'] = df['视频素材id近30天的3s播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的3s完播率_avg'] = df['视频素材id近30天的3s完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的3s完播率_max'] = df['视频素材id近30天的3s完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的25%进度播放数_avg'] = df['视频素材id近30天的25%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的25%进度播放数_max'] = df['视频素材id近30天的25%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的25%进度播放数_sum'] = df['视频素材id近30天的25%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的25%进度完播率_avg'] = df['视频素材id近30天的25%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的25%进度完播率_max'] = df['视频素材id近30天的25%进度完播率'].apply(lambda x:max(x) if x else None)   
    df['视频素材id近30天的50%进度播放数_avg'] = df['视频素材id近30天的50%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的50%进度播放数_max'] = df['视频素材id近30天的50%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的50%进度播放数_sum'] = df['视频素材id近30天的50%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的50%进度完播率_avg'] = df['视频素材id近30天的50%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的50%进度完播率_max'] = df['视频素材id近30天的50%进度完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的75%进度播放数_avg'] = df['视频素材id近30天的75%进度播放数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的75%进度播放数_max'] = df['视频素材id近30天的75%进度播放数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的75%进度播放数_sum'] = df['视频素材id近30天的75%进度播放数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的75%进度完播率_avg'] = df['视频素材id近30天的75%进度完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的75%进度完播率_max'] = df['视频素材id近30天的75%进度完播率'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的播放完成数_avg'] = df['视频素材id近30天的播放完成数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的播放完成数_max'] = df['视频素材id近30天的播放完成数'].apply(lambda x:max(x) if x else None)
    df['视频素材id近30天的播放完成数_sum'] = df['视频素材id近30天的播放完成数'].apply(lambda x:sum(x) if x else None)
    df['视频素材id近30天的完播率_avg'] = df['视频素材id近30天的完播率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['视频素材id近30天的完播率_max'] = df['视频素材id近30天的完播率'].apply(lambda x:max(x) if x else None)
    
    return df


def type_convert(ss):
    if not ss:
        return None
    else:
        s_list = ss.split(',')
        return [float(i) for i in s_list]

def get_content_feature_data():
    server = '192.168.1.198'
    database = 'WIN_DOUYIN'
    username = 'winc_yuxiaojiao'
    password = 'AtdQEPV4EaovDwBgoD0a'

    # 执行查询并读取数据
    increment_query = "SELECT * FROM 投流模型_特征表2 where 日期>'2024-07-01' and 日期<='2024-10-24'"  # 替换为实际的表名和查询
    connect = pymssql.connect(server, username, password, database)
    cursor = connect.cursor()

    cursor.execute(increment_query)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
    # 打印列名
    res = cursor.fetchall() 
    df1 = pd.DataFrame(res, columns=column_names)
    df1.drop(['insert_time','日期', '标题素材id','视频素材id'], axis=1, inplace =True)
    df1 = df1.drop_duplicates()
    print(df1['直播间ID'].head(5))
    ops_col =[]
    for col in df1.columns:
        if '素材' in col:
            ops_col.append(col)
            df1[col] = df1[col].apply(type_convert)

    df1 = get_content_feature(df1)
    df1.drop(ops_col, axis=1, inplace=True)
    # 查看数据
    print(df1.shape)
    '''
    cursor.execute(query2)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
    res = cursor.fetchall()
    df2 = pd.DataFrame(res, columns=column_names)
    df2.drop(['insert_time','日期','标题素材id','视频素材id'], axis=1, inplace = True)
    df2 = df2.drop_duplicates()
   
    ops_col =[]
    for col in df2.columns:
        if '素材' in col:
            ops_col.append(col)
            df2[col] = df2[col].apply(type_convert)

    df2 = get_content_feature(df2)
    df2.drop(ops_col, axis=1, inplace=True)
    # 查看数据
    print(df2.shape)
    
    cursor.execute(query3)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
    res = cursor.fetchall()
    df3 = pd.DataFrame(res, columns=column_names)
    df3.drop(['insert_time','日期','标题素材id','视频素材id'], axis=1, inplace = True)
    df3 = df3.drop_duplicates()
    ops_col =[]
    for col in df3.columns:
        if '素材' in col:
            ops_col.append(col)
            df3[col] = df3[col].apply(type_convert)

    df3 = get_content_feature(df3)
    df3.drop(ops_col, axis=1, inplace=True)

    df = pd.concat([df1, df2,df3])
    df = df.drop_duplicates()
    print(df.shape)
    df.to_csv('./直播间计划/content_feature_08_02.csv', index=False, encoding='utf-8')
    '''
    return df1





if __name__=='__main__':
    server = '192.168.1.198'
    database = 'WIN_DOUYIN'
    username = 'winc_yuxiaojiao'
    password = 'AtdQEPV4EaovDwBgoD0a'

    # 执行查询并读取数据
    query1 = "SELECT * FROM 投流模型_特征表2 where 日期>='2023-11-13' and 日期<='2024-03-01'"  # 替换为实际的表名和查询
    query2 = "SELECT * FROM 投流模型_特征表2 where 日期>'2024-03-01' and 日期<='2024-06-01'"  # 替换为实际的表名和查询
    query3 = "SELECT * FROM 投流模型_特征表2 where 日期>'2024-06-01' and 日期<='2024-09-02'"  # 替换为实际的表名和查询
    connect = pymssql.connect(server, username, password, database)
    cursor = connect.cursor()
    cursor.execute(query1)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
    # 打印列名
    res = cursor.fetchall() 
    df1 = pd.DataFrame(res, columns=column_names)
    df1.drop(['insert_time','日期', '标题素材id','视频素材id'], axis=1, inplace =True)
    df1 = df1.drop_duplicates()
    print(df1['直播间ID'].head(5))
    ops_col =[]
    for col in df1.columns:
        if '素材' in col:
            ops_col.append(col)
            df1[col] = df1[col].apply(type_convert)

    df1 = get_content_feature(df1)
    df1.drop(ops_col, axis=1, inplace=True)
    # 查看数据
    print(df1.shape)

    cursor.execute(query2)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
    res = cursor.fetchall()
    df2 = pd.DataFrame(res, columns=column_names)
    df2.drop(['insert_time','日期', '标题素材id','视频素材id'], axis=1, inplace = True)
    df2 = df2.drop_duplicates()
   
    ops_col =[]
    for col in df2.columns:
        if '素材' in col:
            ops_col.append(col)
            df2[col] = df2[col].apply(type_convert)

    df2 = get_content_feature(df2)
    df2.drop(ops_col, axis=1, inplace=True)
    # 查看数据
    print(df2.shape)
    

    cursor.execute(query3)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
    res = cursor.fetchall()
    df3 = pd.DataFrame(res, columns=column_names)
    df3.drop(['insert_time','日期', '标题素材id','视频素材id'], axis=1, inplace = True)
    df3 = df3.drop_duplicates()
    ops_col =[]
    for col in df3.columns:
        if '素材' in col:
            ops_col.append(col)
            df3[col] = df3[col].apply(type_convert)

    df3 = get_content_feature(df3)
    df3.drop(ops_col, axis=1, inplace=True)
    
    df = pd.concat([df1, df2,df3])
    df = df.drop_duplicates()
    print(df.shape)
  
    create_folder_if_not_exists('../data/直播间计划')
    df.to_csv('../data/直播间计划/content_feature_08_02.csv', index=False, encoding='utf-8')

  



