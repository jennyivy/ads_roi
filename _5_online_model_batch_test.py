import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from joblib import dump, load
import pickle
import time
from sqlalchemy import create_engine, select
from pandas import read_sql_query
import pymssql
from sqlalchemy.sql import text
from online_feature_ops import *
from batch_feature_ops import *
import json
import xgboost as xgb
import shap
import sys
sys.path.append('C:/Users/admin/Desktop/赢销通/金佰利/直播间计划')


# 全局设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_test_data():
    server = '192.168.1.198'
    database = 'WIN_DOUYIN'
    username = 'winc_yuxiaojiao'
    password = 'AtdQEPV4EaovDwBgoD0a'

    # 执行查询并读取数据
  #  query = "SELECT * FROM 投流模型_特征表 where 计划id = '1807009853679732' and insert_time>='2024-08-21' and insert_time<'2024-08-21 23:59:59' and 用户名 like '%好奇%'"  # 替换为实际的表名和查询
    query = "SELECT * FROM 投流模型_特征表 where 广告类型 ='通投广告' and 开始时间>='2024-10-24' and 开始时间<'2024-10-24 23:59:59' and 用户名 like '%好奇%'"  # 替换为实际的表名和查询
    connect = pymssql.connect(server, username, password, database)
    cursor = connect.cursor()
    cursor.execute(query)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
# 打印列名
    res = cursor.fetchall() 
  #  print(column_names)
  #  print(res)
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
                '近30日最小点击-成交转化率','近30日平均点击-成交转化率','id','评估状态','评估结果','评估时间','广告类型'] + del_list_1
    '''
    del_list = ['用户名','id','评估状态','评估结果','评估时间','广告类型'] + del_list_1
    df.drop(del_list,axis=1,inplace=True)

    query3 = "SELECT * FROM 投流模型_特征表2 where 日期>='2024-10-24' and 日期<'2024-10-25'"
    cursor.execute(query3)
    column_names = [i[0] for i in cursor.description]
    # 打印列名
    res = cursor.fetchall() 
    df_content = pd.DataFrame(res, columns=column_names)
    df_content.drop(['insert_time','日期', '标题素材id','视频素材id','id'], axis=1, inplace =True)
    df_content = df_content.drop_duplicates()
    df = pd.merge(df, df_content, on=['直播间ID','计划id'], how='left')
    
    return df

def label_ops(row):
    ss_roi = row.get('广告支付ROI')
    ss_consump = row.get('消耗(元)')
    if float(ss_roi)>=1.75 and float(ss_consump)>=250:
        return 1
    else:
        return 0
   

def batch_test_ops(df):
    df = batch_feature_encode(df)
    target_encode_list = ['千川id','抖音号', '主播id_投放设置-智能优惠券','主播id_投放设置-优化周期','主播id_投放设置-优化目标',
                          '主播id_投放设置-投放方式','主播id_推广方式','主播id_营销场景','定向人群-行为关键词','定向人群-行为类目词',
                          '定向人群-兴趣分类','定向人群-兴趣关键词','定向人群-抖音号分类','定向人群-抖音号达人',
                        '添加创意-创意分类','添加创意-创意标签']

    loaded_target_encoder_consumption = load('./直播间计划/haoqi_target_encoder_encode_by_消耗(元).pkl')
    loaded_target_encoder_roi = load('./直播间计划/haoqi_target_encoder_encode_by_广告支付ROI.pkl')
    df[[f'{ele}_encode_by_消耗(元)' for ele in target_encode_list]] = loaded_target_encoder_consumption.transform(df[target_encode_list])
    df[[f'{ele}_encode_by_广告支付ROI' for ele in target_encode_list]] = loaded_target_encoder_roi.transform(df[target_encode_list])
    
    num_cols = df.select_dtypes(include = [np.number])
    del_list =['千川id', '计划ID','抖音号','直播间ID','平均千次展现费用(元)','广告支付ROI','消耗(元)','广告点击率', '广告成单率', '广告类型',
               '广告成交金额(元)','广告成单数','广告平均成单金额(元)','广告直接结算金额(7天)(元)', '广告直接结算ROI(7天)','label','id','INSERT_TIME']
    num_list_2 = [ele for ele in num_cols.columns if ele not in del_list]
    df = df[num_list_2]
    return df



if __name__=='__main__':
   
    df = get_test_data()
    df['label'] = df.apply(label_ops, axis=1) 
    del_list =['insert_time','平均千次展现费用(元)','广告直接结算ROI(7天)','广告直接结算金额(7天)(元)','广告支付ROI','标题素材id','视频素材id','广告类型']
    df = df.drop(del_list,axis=1)
    print(df.shape)
    feature_col = [ele for ele in df.columns if ele not in ['label','消耗(元)','广告成交金额(元)']]
    pkl_file = './直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/model_input_feature_list.pkl'
    with open(pkl_file, 'wb') as feature_list_file:
          pickle.dump(feature_col, feature_list_file)

    
    row = df[feature_col].iloc[0]
    # 将这一行转换为JSON字符串
    row_json = row.to_json(orient='records')
    #  print(row_json)
    row_dict = json.loads(row_json)
    my_dict = dict(zip(feature_col, row_dict))
    # print(my_dict)
    
    
    with open('./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/test_09_30_1807616292937801.json', 'w', encoding='utf-8') as file:
    # 使用json.dump()将字典写入文件
    # indent参数用于美化输出，表示缩进的空格数
        json.dump(my_dict, file, ensure_ascii=False, indent=4)

    json_file ='./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/test_09_30_1807616292937801.json'
    with open(json_file, 'rb') as file:
        data = json.load(file)
    df_test = pd.DataFrame(data, index=[0])
    df_test = feature_process(df_test, 'self_defined')
    feature_col = [ele for ele in df_test.columns if ele not in ['label','消耗(元)','广告成交金额(元)']]
    feature_df = pd.DataFrame(feature_col, columns=['feature_col'])
    feature_df.to_csv('./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/test_data_feature.csv')
    df_test.to_csv('./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/test_data.csv')
  #  print(df[df['投放设置-投放时间']==''].count())

    model_type='self_defined'
    df_托管_原始输入 = df[df['推广方式']=='自定义']
    df = feature_process(df, model_type)
    df_托管 = df[df['推广方式']==0]
    df_自定义 = df[df['推广方式']==1]
    print(df_托管.shape)
    print(df_自定义.shape)
    # 加载模型
    pkl_file = f'./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/haoqi_lgb_for_online_pred_{model_type}.pkl'
    with open(pkl_file, 'rb') as model_file:
        model = pickle.load(model_file)   

    pred_proba_托管 = model.predict(df_自定义[feature_col])
  
    print(df_托管[feature_col].shape)
    y_pred_托管 = (pred_proba_托管 >= 0.0816).astype(int)
    df_自定义['pred_label'] = y_pred_托管
    df_自定义_avg_roi = df_自定义['广告成交金额(元)'].sum() / df_自定义['消耗(元)'].sum()
    df_自定义_0= df_自定义[df_自定义['pred_label']==0]
    df_自定义_1= df_自定义[df_自定义['pred_label']==1]
    df_自定义_0_avg_roi = df_自定义_0['广告成交金额(元)'].sum() / df_自定义_0['消耗(元)'].sum()
    df_自定义_1_avg_roi = df_自定义_1['广告成交金额(元)'].sum() / df_自定义_1['消耗(元)'].sum()
    print(f'自定义ROI: {round(df_自定义_avg_roi,2)}, 自定义_pred_0_roi: {round(df_自定义_0_avg_roi,2)}, 自定义_pred_1_roi: {round(df_自定义_1_avg_roi,2)}')
    print(f'自定义消耗: {round(df_自定义["消耗(元)"].mean(), 2)}, 自定义_pred_0_消耗: {round(df_自定义_0["消耗(元)"].mean(), 2)}, 自定义_pred_1_消耗: {round(df_自定义_1["消耗(元)"].mean(),2)}')
    print(f'自定义广告成交金额: {round(df_自定义["广告成交金额(元)"].mean(), 2)}, 自定义_pred_0_广告成交金额: {round(df_自定义_0["广告成交金额(元)"].mean(), 2)}, 自定义_pred_1_广告成交金额: {round(df_自定义_1["广告成交金额(元)"].mean(), 2)}')
    print(classification_report(df_自定义['label'], y_pred_托管))
    print('自定义pred_0_消耗:', df_自定义_0['消耗(元)'].sum())
    print('自定义pred_0_广告成交金额', df_自定义_0["广告成交金额(元)"].sum())
    print('自定义pred_1_消耗:', df_自定义_1["消耗(元)"].sum())
    print('自定义pred_1_广告成交金额', df_自定义_1["广告成交金额(元)"].sum())
    print('自定义Pred_0_次数:', df_自定义_0['pred_label'].count())
    print('自定义Pred_1_次数:', df_自定义_1['pred_label'].count())
    pred_prob_test = model.predict(df_test)
    print(pred_prob_test)

    df_托管_原始输入['pred_label'] = y_pred_托管

  #  df_托管.to_excel('./直播间计划/haoqi_1012_wth_planid_wth_content/托管计划_0921.xlsx', index=False)
  #  df_托管_原始输入.to_excel('./直播间计划/haoqi_1012_wth_planid_wth_content/托管计划_0921_原始输入.xlsx', index=False)

   