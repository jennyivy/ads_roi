import pandas as pd
import os
from collections import defaultdict
import re
import numpy as np
from sqlalchemy import create_engine, select
from pandas import read_sql_query
from sqlalchemy.sql import text
import pymssql
import time



def product_feature_ops(df):

    df['product_id_cnt'] = df['价格'].apply(lambda x:len(x) if x else None)
    df['product_价格_avg'] = df['价格'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_价格_max'] = df['价格'].apply(lambda x:max(x) if x else None)
    df['product_价格_min'] = df['价格'].apply(lambda x:min(x) if x else None)
    df['product_库存_avg'] = df['库存'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_库存_min'] = df['库存'].apply(lambda x:min(x) if x else None)
    df['product_近1日金额汇总_sum'] = df['近1日金额汇总'].apply(lambda x:sum(x) if x else None)
    df['product_近1日金额汇总_avg'] = df['近1日金额汇总'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近1日金额汇总_max'] = df['近1日金额汇总'].apply(lambda x:max(x) if x else None)
    df['product_近1日金额汇总_min'] = df['近1日金额汇总'].apply(lambda x:min(x) if x else None)
    df['product_近1日商品次数_sum'] = df['近1日商品次数'].apply(lambda x:sum(x) if x else None)
    df['product_近1日商品次数_avg'] = df['近1日商品次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近1日平均金额_max'] = df['近1日平均金额'].apply(lambda x:max(x) if x else None)
    df['product_近1日平均金额_min'] = df['近1日平均金额'].apply(lambda x:min(x) if x else None)
    df['product_近1日平均金额_avg'] = df['近1日平均金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近1日最大曝光-点击转化率_max'] = df['近1日最大曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近1日最大曝光-点击转化率_min'] = df['近1日最大曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近1日最大曝光-点击转化率_avg'] = df['近1日最大曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近1日最小曝光-点击转化率_max'] = df['近1日最小曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近1日最小曝光-点击转化率_min'] = df['近1日最小曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近1日最小曝光-点击转化率_avg'] = df['近1日最小曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近1日平均曝光-点击转化率_max'] = df['近1日平均曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近1日平均曝光-点击转化率_min'] = df['近1日平均曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近1日平均曝光-点击转化率_avg'] = df['近1日平均曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近1日最大点击-成交转化率_max'] = df['近1日最大点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近1日最大点击-成交转化率_min'] = df['近1日最大点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近1日最大点击-成交转化率_avg'] = df['近1日最大点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近1日最小点击-成交转化率_max'] = df['近1日最小点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近1日最小点击-成交转化率_min'] = df['近1日最小点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近1日最小点击-成交转化率_avg'] = df['近1日最小点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近1日平均点击-成交转化率_max'] = df['近1日平均点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近1日平均点击-成交转化率_min'] = df['近1日平均点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近1日平均点击-成交转化率_avg'] = df['近1日平均点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日金额汇总_max'] = df['近7日金额汇总'].apply(lambda x:max(x) if x else None) 
    df['product_近7日金额汇总_min'] = df['近7日金额汇总'].apply(lambda x:min(x) if x else None)
    df['product_近7日金额汇总_avg'] = df['近7日金额汇总'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日商品次数_max'] = df['近7日商品次数'].apply(lambda x:max(x) if x else None)
    df['product_近7日商品次数_min'] = df['近7日商品次数'].apply(lambda x:min(x) if x else None)
    df['product_近7日商品次数_avg'] = df['近7日商品次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日平均金额_max'] = df['近7日平均金额'].apply(lambda x:max(x) if x else None)
    df['product_近7日平均金额_min'] = df['近7日平均金额'].apply(lambda x:min(x) if x else None)
    df['product_近7日平均金额_avg'] = df['近7日平均金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日最大曝光-点击转化率_max'] = df['近7日最大曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近7日最大曝光-点击转化率_min'] = df['近7日最大曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近7日最大曝光-点击转化率_avg'] = df['近7日最大曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日最小曝光-点击转化率_max'] = df['近7日最小曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近7日最小曝光-点击转化率_min'] = df['近7日最小曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近7日最小曝光-点击转化率_avg'] = df['近7日最小曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日平均曝光-点击转化率_max'] = df['近7日平均曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近7日平均曝光-点击转化率_min'] = df['近7日平均曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近7日平均曝光-点击转化率_avg'] = df['近7日平均曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日最大点击-成交转化率_max'] = df['近7日最大点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近7日最大点击-成交转化率_min'] = df['近7日最大点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近7日最大点击-成交转化率_avg'] = df['近7日最大点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日最小点击-成交转化率_max'] = df['近7日最小点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近7日最小点击-成交转化率_min'] = df['近7日最小点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近7日最小点击-成交转化率_avg'] = df['近7日最小点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近7日平均点击-成交转化率_max'] = df['近7日平均点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近7日平均点击-成交转化率_min'] = df['近7日平均点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近7日平均点击-成交转化率_avg'] = df['近7日平均点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日金额汇总_max'] = df['近30日金额汇总'].apply(lambda x:max(x) if x else None)
    df['product_近30日金额汇总_min'] = df['近30日金额汇总'].apply(lambda x:min(x) if x else None)
    df['product_近30日金额汇总_avg'] = df['近30日金额汇总'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日商品次数_max'] = df['近30日商品次数'].apply(lambda x:max(x) if x else None)
    df['product_近30日商品次数_min'] = df['近30日商品次数'].apply(lambda x:min(x) if x else None)
    df['product_近30日商品次数_avg'] = df['近30日商品次数'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日平均金额_max'] = df['近30日平均金额'].apply(lambda x:max(x) if x else None)
    df['product_近30日平均金额_min'] = df['近30日平均金额'].apply(lambda x:min(x) if x else None)
    df['product_近30日平均金额_avg'] = df['近30日平均金额'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日最大曝光-点击转化率_max'] = df['近30日最大曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近30日最大曝光-点击转化率_min'] = df['近30日最大曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近30日最大曝光-点击转化率_avg'] = df['近30日最大曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日最小曝光-点击转化率_max'] = df['近30日最小曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近30日最小曝光-点击转化率_min'] = df['近30日最小曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近30日最小曝光-点击转化率_avg'] = df['近30日最小曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日平均曝光-点击转化率_max'] = df['近30日平均曝光-点击转化率'].apply(lambda x:max(x) if x else None)
    df['product_近30日平均曝光-点击转化率_min'] = df['近30日平均曝光-点击转化率'].apply(lambda x:min(x) if x else None)
    df['product_近30日平均曝光-点击转化率_avg'] = df['近30日平均曝光-点击转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日最大点击-成交转化率_max'] = df['近30日最大点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近30日最大点击-成交转化率_min'] = df['近30日最大点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近30日最大点击-成交转化率_avg'] = df['近30日最大点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日最小点击-成交转化率_max'] = df['近30日最小点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近30日最小点击-成交转化率_min'] = df['近30日最小点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近30日最小点击-成交转化率_avg'] = df['近30日最小点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
    df['product_近30日平均点击-成交转化率_max'] = df['近30日平均点击-成交转化率'].apply(lambda x:max(x) if x else None)
    df['product_近30日平均点击-成交转化率_min'] = df['近30日平均点击-成交转化率'].apply(lambda x:min(x) if x else None)
    df['product_近30日平均点击-成交转化率_avg'] = df['近30日平均点击-成交转化率'].apply(lambda x:sum(x)/len(x) if x else None)
   
    return df

def type_convert(ss):
    if not ss:
        return None
    else:
        try:
            s_list = ss.split(',')
            return [float(i) for i in s_list]
        except:
            return None
    

if __name__=='__main__':
    server = '192.168.1.198'
    database = 'WIN_DOUYIN'
    username = 'winc_yuxiaojiao  '
    password = 'AtdQEPV4EaovDwBgoD0a'

    connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    # 创建引擎
    engine = create_engine(connection_string)

    # 执行查询并读取数据
    query = "SELECT * FROM 投流模型_特征表 where 开始时间>='2024-03-13' and 用户名 like '%好奇%'"  # 替换为实际的表名和查询
    df_ads_plan = pd.read_sql_query(query, engine)
    df_ads_plan = df_ads_plan.drop_duplicates()
    # 查看数据
  #  print(df_ads_plan.shape)
  #  print(df_ads_plan.columns.tolist())

    sel_list = ['直播间ID','计划id','价格','库存','近1日金额汇总','近1日商品次数','近1日平均金额','近1日最大曝光-点击转化率',
                '近1日最小曝光-点击转化率','近1日平均曝光-点击转化率','近1日最大点击-成交转化率','近1日最小点击-成交转化率','近1日平均点击-成交转化率',
                '近7日金额汇总','近7日商品次数','近7日平均金额','近7日最大曝光-点击转化率','近7日最小曝光-点击转化率','近7日平均曝光-点击转化率',
                '近7日最大点击-成交转化率','近7日最小点击-成交转化率','近7日平均点击-成交转化率','近30日金额汇总','近30日商品次数','近30日平均金额',
                '近30日最大曝光-点击转化率','近30日最小曝光-点击转化率','近30日平均曝光-点击转化率','近30日最大点击-成交转化率','近30日最小点击-成交转化率','近30日平均点击-成交转化率']
    
    s_list = ['价格','库存','近1日金额汇总','近1日商品次数','近1日平均金额','近1日最大曝光-点击转化率',
                '近1日最小曝光-点击转化率','近1日平均曝光-点击转化率','近1日最大点击-成交转化率','近1日最小点击-成交转化率','近1日平均点击-成交转化率',
                '近7日金额汇总','近7日商品次数','近7日平均金额','近7日最大曝光-点击转化率','近7日最小曝光-点击转化率','近7日平均曝光-点击转化率',
                '近7日最大点击-成交转化率','近7日最小点击-成交转化率','近7日平均点击-成交转化率','近30日金额汇总','近30日商品次数','近30日平均金额',
                '近30日最大曝光-点击转化率','近30日最小曝光-点击转化率','近30日平均曝光-点击转化率','近30日最大点击-成交转化率','近30日最小点击-成交转化率','近30日平均点击-成交转化率']
    
    df_product_data = df_ads_plan[sel_list]
    time1= time.time()
    print(time1)
    
    for col in s_list:
        df_product_data[col] = df_product_data[col].apply(type_convert)

    product_feature_df = product_feature_ops(df_product_data)
    product_feature_df.drop(s_list, axis=1, inplace=True)
    print(time.time()-time1)
    print(product_feature_df.head(100))

