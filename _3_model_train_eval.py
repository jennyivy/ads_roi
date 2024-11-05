import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, classification_report,roc_auc_score
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from joblib import dump, load
import pickle
import mlflow
import time
import shap
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
from get_content_feature import create_folder_if_not_exists


# 使用时间戳生成版本号
current_time = time.localtime()
model_version = time.strftime("%Y%m%d%H%M%S", current_time)

# 全局设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def label_ops_self_defined(row):
    ss_roi = row.get('广告支付ROI')
    ss_consump = row.get('消耗(元)')
    if float(ss_roi)>=1.75 and float(ss_consump)>=250:
        return 1
    else:
        return 0

def label_ops_custody(row):
    ss_roi = row.get('广告支付ROI')
    ss_consump = row.get('消耗(元)')
    if float(ss_roi)>=1.86 and float(ss_consump)>=500:
        return 1
    else:
        return 0 

   
def model_train_binary(X_train, y_train, X_test, y_test, ratio, model_type):
    #  feature_col=[ele for ele in X_train.columns if ele not in ['sample_weight']]
    #  train_sample_weight = X_train['sample_weight']
    #  test_sample_weight = X_test['sample_weight']
    #  X_train = X_train[feature_col]
    #  X_test = X_test[feature_col]
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, y_test)

    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'max_depth':5,
    'lambda_l1':25,
    'lambda_l2':25   
    # 'is_unbalance': True
    # 'scale_pos_weight':10
    }

    model = lgb.train(params, train_data, num_boost_round=1000,valid_sets=valid_data)
    # 模型保存
    
    output = open(f'../model/直播间计划/haoqi_lgb_for_online_pred_{model_type}.pkl', 'wb')
    pickle.dump(model, output)
    y_pred_proba_train  = model.predict(X_train, num_iteration=model.best_iteration)
    y_pred_train = (y_pred_proba_train >= ratio).astype(int)

    auc = roc_auc_score(y_train, y_pred_proba_train)
    print(f'AUC: {auc}')
    print(classification_report(y_train, y_pred_train))

    y_pred_proba  = model.predict(X_test, num_iteration=model.best_iteration)
   
    y_pred = (y_pred_proba >= ratio).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f'AUC: {auc}')
    print(classification_report(y_test, y_pred))

    importance = model.feature_importance()
    all_idx = importance.argsort()[::-1]
    feature_imp_df = pd.DataFrame()
    columns_list =[]
    feature_imp =[]
    for fidx in all_idx:
        columns_list.append(X_train.columns[fidx])
        feature_imp.append(importance[fidx])

    feature_imp_df['feature_names'] = columns_list
    feature_imp_df['feature_importance'] = feature_imp
    feature_imp_df.to_csv(f'../model/直播间计划/haoqi_feature_importance_for_online_{model_type}.csv')
    # Get the index of the top 10 most important features
    top20_idx = importance.argsort()[-20:][::-1]
    print("Top 20 feature importances:")
    for fidx in top20_idx:
        print(f"{X_train.columns[fidx]}: {importance[fidx]}")

    # Plot the top 10 feature importances
    plt.barh(range(20), importance[top20_idx], align='center')
    plt.yticks(range(20), X_train.columns[top20_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature', fontsize=4)
    plt.title('Top 20 Feature Importances')
    plt.show()

    explainer = shap.TreeExplainer(model)
    shap_values_test = explainer.shap_values(X_test)
    shap_values_array = np.array(shap_values_test)
    shap_mean = np.mean(shap_values_array[1], axis=0)
    feature_names = X_test.columns
    sorted_idx = np.argsort(np.abs(shap_mean))[::-1]  # 降序排序
    feature_imp_df = pd.DataFrame()
    features_cols = []
    feature_imp =[]
    feature_imp_df['feature_importance'] = []
    print("Feature importance based on SHAP values:")
    for i in sorted_idx:
        features_cols.append(feature_names[i])
        feature_imp.append(shap_mean[i])

    feature_imp_df['feature_names'] = features_cols
    feature_imp_df['feature_importance'] = feature_imp
    feature_imp_df.to_csv(f'../data/直播间计划/haoqi_shap_feature_importance_for_online_{model_type}.csv')
  
    return model 


def model_exp_shap(model, X_train, y_train, X_test, y_test):
    explainer = shap.TreeExplainer(model)
    shap_values_test = explainer.shap_values(X_test)
    plt.figure(figsize=(15, 15))
    # 可视化整体样本的SHAP值
   # sample_shap_values = explainer.shap_values(X_test.iloc[0, :])
    # 可视化单个样本的SHAP值
   
    shap.summary_plot(shap_values_test[1],X_test)
    plt.show()


    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values_test[1][0,:], X_test.iloc[0,:], matplotlib=True, show=False)
    plt.show()

    shap.dependence_plot('直播时长', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('投放设置-支付ROI目标', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('预算(元)', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-预估用户覆盖', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('ads_hour', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('添加创意-创意形式_直播间画面', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('添加创意-创意形式_视频', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-抖音号达人_encode_by_消耗(元)', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('median_30d_groupby_千川id_aggby_广告点击率', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('sum_1d_groupby_千川id_aggby_广告成单数', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('day_of_week', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('is_holiday', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-新客_不限', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

  #  shap.dependence_plot('ads_end_time', shap_values_test[1], X_test, interaction_index = None)
  #  plt.show()

    shap.dependence_plot('7day_单均成交金额_groupby_主播id', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-行为关键词_encode_by_广告支付ROI', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-行为类目词_encode_by_广告支付ROI', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('max_1d_groupby_主播id_aggby_广告成单数', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('30day_单均成交金额_groupby_千川id', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('user_interact_电商互动行为', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('user_interact_资讯互动行为', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-行为天数', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-兴趣分类_encode_by_消耗(元)', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-兴趣分类_encode_by_广告支付ROI', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-兴趣关键词_encode_by_广告支付ROI', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-兴趣关键词_encode_by_消耗(元)', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-抖音号分类_encode_by_消耗(元)', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-抖音号分类_encode_by_广告支付ROI', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-抖音号达人_encode_by_消耗(元)', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('定向人群-抖音号达人_encode_by_广告支付ROI', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('添加创意-创意分类_encode_by_消耗(元)', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('添加创意-创意分类_encode_by_广告支付ROI', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('添加创意-创意标签_encode_by_消耗(元)', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    shap.dependence_plot('添加创意-创意标签_encode_by_广告支付ROI', shap_values_test[1], X_test, interaction_index = None)
    plt.show()

    return 


if __name__=='__main__':
    data_file ='../data/直播间计划/haoqi_bi_data_encoded.csv'
    df = pd.read_csv(data_file)
    print(df.shape)
    df['预算(元)'] = df['预算(元)'].astype(float)
 #   df['sample_weight'] = df['month'].map({1:0.1,2:0.1,3:0.1,4:0.05,5:0.15,6:0.225,7:0.225,12:0.05})
    df['千川id'] = df['千川id'].astype(str)
    df['抖音号'] = df['抖音号'].astype(str)
    df_托管 = df[df['推广方式']==0]
    df_自定义 = df[df['推广方式']==1]
    df_托管['label'] = df_托管.apply(label_ops_custody, axis=1)
    df_自定义['label'] = df_自定义.apply(label_ops_self_defined, axis=1)

    model_types = ['self_defined','custody']
    data_types = [df_自定义, df_托管]
    create_folder_if_not_exists('../model/直播间计划')
    for model_type, df in zip(model_types, data_types):
        feature_cols =[ele for ele in df.columns if ele not in ['label']]
        X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df['label'], test_size=0.2, random_state=42)
    

        target_encode_list = ['千川id','抖音号', '主播id_投放设置-智能优惠券','主播id_投放设置-优化周期','主播id_投放设置-优化目标',
                            '主播id_投放设置-投放方式','主播id_推广方式','主播id_营销场景','定向人群-行为关键词','定向人群-行为类目词',
                            '定向人群-兴趣分类','定向人群-兴趣关键词','定向人群-抖音号分类','定向人群-抖音号达人',
                            '添加创意-创意分类','添加创意-创意标签']

        for target in ['消耗(元)','广告支付ROI']:
            target_encoder = ce.TargetEncoder(handle_unknown='ignore')
            target_encoder.fit(X_train[target_encode_list], X_train[target])
            dump(target_encoder, f'../model/直播间计划/haoqi_target_encoder_encode_by_{target}_{model_type}.pkl')      
            X_train[[f'{ele}_encode_by_{target}' for ele in target_encode_list]] = target_encoder.transform(X_train[target_encode_list])
            X_test[[f'{ele}_encode_by_{target}' for ele in target_encode_list]] = target_encoder.transform(X_test[target_encode_list])
            
        num_cols = X_train.select_dtypes(include = [np.number])
        del_list =['千川id', '计划ID','计划id','抖音号','直播间ID','平均千次展现费用(元)','广告支付ROI','消耗(元)','广告点击率', '广告成单率',
                'id_x','id_y','标题素材id','视频素材id','广告成交金额(元)','广告成单数','广告平均成单金额(元)','广告直接结算金额(7天)(元)', 
                '广告直接结算ROI(7天)','label','id','INSERT_TIME','评估状态','评估结果','评估时间']
        
        num_list_2 = [ele for ele in num_cols.columns if ele not in del_list]

        X_train=X_train[num_list_2].round(2)
        X_test=X_test[num_list_2].round(2)
    
        print(y_train.sum()/y_train.count(), y_test.sum()/y_test.count())
        print(X_train.shape)
        ratio = y_train.sum()/y_train.count()
        ratio_dic = {'ratio': ratio}
        output = open(f'../model/直播间计划/class_ratio_{model_type}.pkl', 'wb')
        pickle.dump(ratio_dic, output)
        model_consumption = model_train_binary(X_train, y_train, X_test, y_test, ratio, model_type)
  