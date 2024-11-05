import pandas as pd
import os
from sqlalchemy import create_engine, select
from pandas import read_sql_query
import pymssql
from sqlalchemy.sql import text
from get_content_feature import *

def extract_data_from_sql_server(**kwargs):
    server = '192.168.1.198'
    database = 'WIN_DOUYIN'
    username = 'winc_yuxiaojiao'
    password = 'AtdQEPV4EaovDwBgoD0a'

    # 执行查询并读取数据
    query = """
        SELECT * FROM 投流模型_特征表 
        where CONVERT(DECIMAL(18,3),[消耗(元)])>0 
        and 广告类型='通投广告' 
        and 开始时间>'2024-07-01' 
        and 开始时间<='2024-10-24' 
        and 用户名 like '%好奇%'
        """
    
    connect = pymssql.connect(server, username, password, database)
    cursor = connect.cursor()
    cursor.execute(query)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
# 打印列名
    res = cursor.fetchall() 
    df = pd.DataFrame(res, columns=column_names)
 #   print(df.columns.tolist())

    del_list_1 = []
    '''
    sub_str =['计划ID','计划id']
    for col in df.columns:
        if  any(sub_string in col for sub_string in sub_str):
            del_list_1.append(col)     
    
    del_list = ['用户名','价格','库存','近1日金额汇总','近1日商品次数','近1日平均金额','近1日最大曝光-点击转化率','近1日最小曝光-点击转化率',
                '近1日平均曝光-点击转化率','近1日最大点击-成交转化率','近1日最小点击-成交转化率','近1日平均点击-成交转化率',
                '近7日金额汇总','近7日商品次数','近7日平均金额','近7日最大曝光-点击转化率','近7日最小曝光-点击转化率','近7日平均曝光-点击转化率',
                '近7日最大点击-成交转化率','近7日最小点击-成交转化率','近7日平均点击-成交转化率','近30日金额汇总','近30日商品次数',
                '近30日平均金额','近30日最大曝光-点击转化率','近30日最小曝光-点击转化率','近30日平均曝光-点击转化率','近30日最大点击-成交转化率',
                '近30日最小点击-成交转化率','近30日平均点击-成交转化率','广告类型'] + del_list_1
    '''

    del_list = ['用户名', '广告类型'] + del_list_1

    del_list  = [ele for ele in del_list if ele!='计划id']
    df.drop(del_list,axis=1,inplace=True)
    return df

if __name__=='__main__':
    df = extract_data_from_sql_server()
    
    df['直播间ID'] = df['直播间ID'].astype(str)
    df['计划id'] = df['计划id'].astype(str)
    df_content = get_content_feature_data()
    print(df_content['直播间ID'].head(50))
    df_content['直播间ID'] = df_content['直播间ID'].astype(str)
    df_content['计划id'] = df_content['计划id'].astype(str)
    print(df_content[['直播间ID', '计划id']].head(10))
    df = pd.merge(df,df_content,on=['直播间ID', '计划id'],how='left')

    print(df.shape)
    create_folder_if_not_exists('../data/直播间计划')
    df.to_csv('../data/直播间计划/haoqi_bi_data.csv',index=False, encoding='utf-8')