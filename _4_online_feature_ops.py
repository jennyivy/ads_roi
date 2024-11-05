import pandas as pd
import os
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
import pickle
import re
from collections import defaultdict
from sqlalchemy import create_engine, select
from pandas import read_sql_query
import pymssql
from sqlalchemy.sql import text
import pandas as pd
from content_feature_ops import *
from get_product_features import *


def encode_with_unknown(encoder, categories):
    transformed = []
    for category in categories:
        if category in encoder.classes_:
            transformed.append(encoder.transform([category])[0])
        else:
            # 为未知类别分配一个特定的索引，例如 -1
            transformed.append(-1)
    return transformed


def time_transform(ss):
    if 'day' in ss or 'days' in ss:
        ss_list = ss.split(',')
        days = int(ss_list[0].split(' ')[0])
        hours_mint_secs = ss_list[1].split(':')
        hours = int(hours_mint_secs[0])
        total_hours = days*24+hours
        return total_hours
    hours_mint_secs = ss.split(':')
    hours = int(hours_mint_secs[0])
    mints = int(hours_mint_secs[1])
    mints_to_hour = float(mints/60)
    total_hours = hours+mints_to_hour
    return round(total_hours,2)

def is_hoiday(ss):
    holiday_date_list = [pd.to_datetime('2023-11-11'), pd.to_datetime('2024-01-01'), pd.to_datetime('2024-02-10'), pd.to_datetime('2024-03-08'),pd.to_datetime('2024-06-18')]  # 节假日和商业大促
    # 包括节日前后各3天
    for holiday_date in holiday_date_list:
        start_date = holiday_date - pd.Timedelta(days=3)
        end_date = holiday_date + pd.Timedelta(days=3)
     # 筛选节日期间的数据
        if ss >= start_date and ss <= end_date:
            return True
    return False

def gender_encode(ss):
    if not ss:
        return pd.Series([1, 1], index=['target_men', 'target_women'])
    elif ss in ['不限','没有设置']:
        return pd.Series([1, 1], index=['target_men', 'target_women'])
    elif ss=='女':
        return pd.Series([0, 1], index=['target_men', 'target_women'])
    else:
        return pd.Series([1, 0], index=['target_men', 'target_women'])
    
def age_encode(ss):
    if not ss:
        return pd.Series([1, 1, 1, 1, 1], index=['age_18_23', 'age_24_30','age_31_40','age_41_49','age_50+'])
    elif ss in ['不限','没有设置']:
        return pd.Series([1, 1, 1, 1, 1], index=['age_18_23', 'age_24_30','age_31_40','age_41_49','age_50+'])
    else:
        age_dic={'18-23':0, '24-30':0, '31-40':0, '41-49':0, '50+':0}
        ss_list=ss.split('、')
        for ele in ss_list:
            age_dic[ele]+=1

        return pd.Series([age_dic['18-23'],age_dic['24-30'] , age_dic['31-40'] , age_dic['41-49'], age_dic['50+']], index=['age_18_23', 'age_24_30','age_31_40','age_41_49','age_50+'])


def network_encode(ss):
    if not ss:
        return pd.Series([1,1,1,1,1], index=['network_wifi','network_2G','network_3G','network_4G','network_5G'])
    elif ss in ['不限','没有设置']:
        return pd.Series([1,1,1,1,1], index=['network_wifi','network_2G','network_3G','network_4G','network_5G'])
    
    network_dic= {'WIFI':0, '2G':0, '3G':0, '4G':0, '5G':0}
    ss_list = ss.split('、')
     
    for ele in ss_list:
        network_dic[ele]+=1
   
    return pd.Series([network_dic['WIFI'],network_dic['2G'] , network_dic['3G'] , network_dic['4G'], network_dic['5G']], index=['network_wifi','network_2G','network_3G','network_4G','network_5G'])

def user_interact_days(ss):
    if not ss:
        return 720
    elif ss in ['不限','没有设置']:
        return 720
    ss_int = int(ss[:-1])
    return ss_int

def user_coverage_est(ss):
    if not ss:
        return 2300000000
    elif ss in ['没有设置','-']:
        return 2300000000
    elif ss=='0':
        return 0
    elif ss.endswith('万'):
      #  print(ss)
        return float(ss[:-1])*10000    
    return float(ss[:-1])*100000000


def user_behavior_context(ss):
    if not ss:
        return pd.Series([0,0,0,0,1], index = ['user_interact_电商互动行为','user_interact_APP推广互动行为','user_interact_资讯互动行为','user_interact_不限','user_interact_没有设置'])
    elif ss=='不限':
        return pd.Series([0,0,0,1,0], index = ['user_interact_电商互动行为','user_interact_APP推广互动行为','user_interact_资讯互动行为','user_interact_不限','user_interact_没有设置'])
    elif ss=='没有设置':
        return pd.Series([0,0,0,0,1], index = ['user_interact_电商互动行为','user_interact_APP推广互动行为','user_interact_资讯互动行为','user_interact_不限','user_interact_没有设置'])
    user_interact_dic={'电商互动行为':0,'APP推广互动行为':0, '资讯互动行为':0}
    ss_list = ss.split('、')
    for ele in ss_list:
        user_interact_dic[ele]+=1
    return pd.Series([user_interact_dic['电商互动行为'], user_interact_dic['APP推广互动行为'],user_interact_dic['资讯互动行为'],0,0], index = ['user_interact_电商互动行为','user_interact_APP推广互动行为', 'user_interact_资讯互动行为','user_interact_不限','user_interact_没有设置'])

def fans_interact_context(ss):

    fans_interact_dic={'关注行为':0,'15天评论':0, '15天点赞':0, '15天分享':0, '30天评论':0, '30天点赞':0, '30天分享':0, '60天评论':0, '60天点赞':0, '60天分享':0,
                       '15天直播观看':0, '15天直播有效观看':0, '15天直播评论':0, '15天直播打赏':0,'15天直播商品点击':0, '15天直播视频下单':0, '15天直播商品下单':0,
                       '30天直播观看':0, '30天直播有效观看':0, '30天直播评论':0, '30天直播打赏':0,'30天直播商品点击':0, '30天直播视频下单':0,'30天直播商品下单':0,
                       '60天直播观看':0, '60天直播有效观看':0, '60天直播评论':0, '60天直播打赏':0,'60天直播商品点击':0, '60天直播视频下单':0, '60天直播商品下单':0,
                       '15天购物车点击':0, '15天购物车下单':0, '30天购物车点击':0, '30天购物车下单':0, '60天购物车点击':0, '60天购物车下单':0, '没有设置':0}
    if not ss or ss=='没有设置':
        return pd.Series(fans_interact_dic.values(), index = fans_interact_dic.keys())
    ss_list = ss.split('、')
    for ele in ss_list:
        fans_interact_dic[ele]+=1
    
    return pd.Series(fans_interact_dic.values(), index = fans_interact_dic.keys())


def ads_hour_encode(row):
    ss = row.get('投放设置-投放时段')
    start_time = int(row.get('开始时间').hour)
    end_time = int(row.get('结束时间').hour)
    ads_hour = row.get('直播时长')
    if end_time < start_time:
        end_time = end_time + 24
    if ss in ['不限','没有设置'] or not ss:
        return pd.Series([start_time, end_time, ads_hour], index=['ads_start_time','ads_end_time','ads_hour'])
    
    given_interval_time = [(start_time, end_time)]
    overlap_interval_time = []
    day_in_week = row.get('day_of_week')
    pattern = r"(?P<day>\星期[一二三四五六日])\s(?P<time>(\d{1,2}:\d{2}~\d{1,2}:\d{2}[,、]?)+)"
    # 使用re.finditer查找所有匹配项
    matches = re.finditer(pattern, ss)

    # 遍历匹配项并提取信息
    interval_time = []
    interval_hour=0
    for match in matches:
        day_of_week = match.group('day')
        time_range = match.group('time')
        if day_of_week ==day_in_week:
            time_intervals_list = time_range.split('、')
            for time_interval in time_intervals_list:
                start_time = time_interval.split('~')[0]
                start_hour = start_time.split(':')[0]
                end_time = time_interval.split('~')[1]
                end_hour = end_time.split(':')[0]
                interval_time.append((int(start_hour), int(end_hour)))    

            直播开始时间 = given_interval_time[0][0]
            直播结束时间 = given_interval_time[0][1]

            for time_interval in interval_time:
                if time_interval[0] < 直播结束时间 and time_interval[1] > 直播开始时间:
                    # 计算重叠区间
                    overlap_start = max(time_interval[0], 直播开始时间)
                    overlap_end = min(time_interval[1], 直播结束时间)
                    overlap_interval_time.append((overlap_start, overlap_end))
                    interval_hour += (overlap_end-overlap_start)

            if overlap_interval_time:
                return pd.Series([interval_time[0][0], interval_time[-1][1], interval_hour], index=['ads_start_time','ads_end_time','ads_hour'])
        
    return pd.Series([start_time, end_time, ads_hour], index=['ads_start_time','ads_end_time','ads_hour'])


def city_encode(ss):
    with open('./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/haoqi_city_dict.pkl', 'rb') as file:  # 'rb' 代表读取二进制模式
        province_city_tier_cnt_dic = pickle.load(file)

    oversea_city =["台湾", "香港", "澳门"]
    first_tier_city = ['北京','上海','广州','深圳']
    second_tier_city = [
    "天津", "石家庄", "唐山", "太原", "沈阳", "大连",
    "长春", "哈尔滨", "南京", "无锡", "苏州", "杭州",
    "宁波", "温州", "合肥", "福州", "厦门", "南昌",
    "济南", "青岛", "淄博", "烟台", "郑州", "武汉",
    "长沙", "佛山", "东莞", "南宁", "重庆", "成都",
    "昆明", "西安"]
    third_tier_city = [
    "邯郸", "保定", "沧州", "廊坊", "呼和浩特", "包头", "鄂尔多斯",
    "鞍山", "吉林", "大庆", "徐州", "常州", "南通", "淮安",
    "盐城", "扬州", "镇江", "泰州", "嘉兴", "绍兴", "金华",
    "台州", "芜湖", "泉州", "漳州", "东营", "潍坊", "济宁",
    "泰安", "威海", "临沂", "德州", "聊城", "滨州", "菏泽",
    "洛阳", "南阳", "宜昌", "襄阳", "岳阳", "常德", "珠海",
    "江门", "湛江", "茂名", "惠州", "中山", "柳州", "海口",
    "绵阳", "贵阳", "宝鸡", "榆林", "兰州", "西宁", "银川",
    "乌鲁木齐"]

    fourth_tier_city = [
    "秦皇岛", "邢台", "张家口", "承德", "衡水", "大同", "长治", "晋城",
    "朔州", "晋中", "运城", "临汾", "吕梁", "赤峰", "通辽", "呼伦贝尔",
    "抚顺", "本溪", "丹东", "锦州", "营口", "辽阳", "盘锦", "铁岭",
    "朝阳", "四平", "通化", "松原", "齐齐哈尔", "牡丹江", "绥化",
    "连云港", "宿迁", "湖州", "衢州", "丽水", "蚌埠", "马鞍山",
    "安庆", "滁州", "阜阳", "宿州", "六安", "莆田", "三明", "南平",
    "龙岩", "宁德", "九江", "赣州", "吉安", "宜春", "枣庄", "日照",
    "开封", "平顶山", "新乡", "焦作", "濮阳", "许昌", "三门峡",
    "商丘", "信阳", "周口", "驻马店", "黄石", "十堰", "荆门",
    "孝感", "荆州", "黄冈", "株洲", "湘潭", "邵阳", "益阳",
    "郴州", "永州", "怀化", "娄底", "韶关", "汕头", "肇庆", "阳江",
    "清远", "揭阳", "桂林", "玉林", "自贡", "攀枝花", "泸州",
    "德阳", "内江", "乐山", "南充", "宜宾", "达州", "资阳", "遵义",
    "毕节", "曲靖", "玉溪", "咸阳", "渭南"]

    other_tier_city={}

    city_dic = {'河北':["石家庄","唐山","秦皇岛","邯郸","邢台","保定","张家口","承德","沧州","廊坊","衡水"], 
                '山西': ["太原", "大同", "阳泉", "长治", "晋城", "朔州", "晋中", "运城", "忻州", "临汾", "吕梁"],
                '内蒙古':["呼和浩特", "包头", "乌海", "赤峰", "通辽", "鄂尔多斯","呼伦贝尔", "巴彦淖尔", "乌兰察布", "兴安", "锡林郭勒", "阿拉善"],
                '辽宁':["沈阳", "大连", "鞍山", "抚顺", "本溪", "丹东","锦州", "营口", "阜新", "辽阳", "盘锦", "铁岭","朝阳", "葫芦岛"],
                '吉林':["长春", "吉林", "四平", "辽源", "通化", "白山","松原", "白城", "延边",'吉林市'],
                '黑龙江':["哈尔滨", "齐齐哈尔", "鸡西", "鹤岗", "双鸭山","大庆", "伊春", "佳木斯", "七台河", "牡丹江","黑河", "绥化", "大兴安岭"],
                '江苏':["南京", "无锡", "徐州", "常州", "苏州", "南通", "连云港", "淮安", "盐城", "扬州", "镇江", "泰州", "宿迁"],
                '浙江': ['杭州', '宁波', '温州', '嘉兴', '湖州', '绍兴', '金华', '衢州', '舟山', '台州', '丽水'],
                '安徽': ["合肥", "芜湖", "蚌埠", "淮南", "马鞍山", "淮北", "铜陵", "安庆", "黄山", "滁州", "阜阳", "宿州", "六安", "亳州", "池州", "宣城"],
                '福建': ["福州", "厦门", "莆田", "三明", "泉州", "漳州", "南平", "龙岩", "宁德"],
                '江西':["南昌", "景德镇", "萍乡", "九江", "新余", "鹰潭", "赣州", "吉安", "宜春", "抚州", "上饶"],
                '山东':['济南', '青岛', '淄博', '枣庄', '东营', '烟台', '潍坊', '济宁', '泰安', '威海', '日照', '莱芜', '临沂', '德州', '聊城', '滨州', '菏泽'],
                '河南':["郑州", "开封", "洛阳", "平顶山", "安阳", "鹤壁", "新乡", "焦作", "濮阳", "许昌", "漯河", "三门峡", "南阳", "商丘", "信阳", "周口", "驻马店", "济源"],
                '湖北':['武汉', '黄石', '十堰', '宜昌', '襄阳', '鄂州', '荆门', '孝感', '荆州', '黄冈', '咸宁', '随州', '恩施', '仙桃', '潜江', '天门', '神农架'],
                '湖南':['长沙', '株洲', '湘潭', '衡阳', '邵阳', '岳阳', '常德', '张家界', '益阳', '郴州', '永州', '怀化', '娄底', '湘西'],
                '广东':['广州', '韶关', '深圳', '珠海', '汕头', '佛山', '江门', '湛江', '茂名', '肇庆', '惠州', '梅州', '汕尾', '河源', '阳江', '清远', '东莞', '中山', '潮州', '揭阳', '云浮'],
                '广西':['南宁', '柳州', '桂林', '梧州', '北海', '防城港', '钦州', '贵港', '玉林', '百色', '贺州', '河池', '来宾', '崇左'],
                '海南':['海口', '三亚', '三沙', '儋州', '五指山', '琼海', '文昌', '万宁', '东方', '定安', '屯昌', '澄迈', '临高', '白沙', '昌江', '乐东', '陵水', '保亭', '琼中'],
                '四川':['成都', '自贡', '攀枝花', '泸州', '德阳', '绵阳', '广元', '遂宁', '内江', '乐山', '南充', '眉山', '宜宾', '广安', '达州', '雅安', '巴中', '资阳', '阿坝', '甘孜', '凉山'],
                '贵州':['贵阳', '六盘水', '遵义', '安顺', '毕节', '铜仁', '黔西南', '黔东南', '黔南'],
                '云南':['昆明', '曲靖', '玉溪', '保山', '昭通', '丽江', '普洱', '临沧', '楚雄', '红河', '文山', '西双版纳', '大理', '德宏', '怒江', '迪庆'],
                '西藏': ['拉萨', '日喀则', '昌都', '林芝', '山南', '那曲', '阿里'],
                '陕西': ["西安", "铜川", "宝鸡", "咸阳", "渭南", "延安", "汉中", "榆林", "安康", "商洛"],
                '甘肃':['兰州', '嘉峪关', '金昌', '白银', '天水', '武威', '张掖', '平凉', '酒泉', '庆阳', '定西', '陇南', '临夏', '甘南'],
                '青海': ['西宁', '海东', '海北', '黄南', '海南', '果洛', '玉树', '海西'],
                '宁夏': ['银川', '石嘴山', '吴忠', '固原', '中卫'],
                '新疆': ['乌鲁木齐', '克拉玛依', '吐鲁番', '哈密', '昌吉', '博尔塔拉', '巴音郭楞', '阿克苏', '克孜勒苏', '喀什', '和田', '伊犁', '塔城', '阿勒泰', '石河子', '阿拉尔', '图木舒克', '五家渠', '北屯', '铁门关', '双河', '可克达拉']
               }

    city_tier_dic = {'first_tier_city_cnt':0, 'second_tier_city_cnt':0, 'third_tier_city_cnt':0, 'fourth_tier_city_cnt':0, 'other_city_cnt':0}
    if ss in ['不限','不限(排除限运地区)','没有设置']:
        for key in province_city_tier_cnt_dic.keys():
            city_tier_dic['first_tier_city_cnt'] += province_city_tier_cnt_dic[key]['first_tier_city_cnt']
            city_tier_dic['second_tier_city_cnt'] += province_city_tier_cnt_dic[key]['second_tier_city_cnt']
            city_tier_dic['third_tier_city_cnt'] += province_city_tier_cnt_dic[key]['third_tier_city_cnt']
            city_tier_dic['fourth_tier_city_cnt'] += province_city_tier_cnt_dic[key]['fourth_tier_city_cnt']
            city_tier_dic['other_city_cnt'] += province_city_tier_cnt_dic[key]['other_city_cnt']

        return pd.Series([city_tier_dic['first_tier_city_cnt'], city_tier_dic['second_tier_city_cnt'], 
                           city_tier_dic['third_tier_city_cnt'], city_tier_dic['fourth_tier_city_cnt'],
                           city_tier_dic['other_city_cnt']], index=['first_tier_city_cnt', 'second_tier_city_cnt', 
                         'third_tier_city_cnt','fourth_tier_city_cnt', 'other_city_cnt'])

    ss_list=ss.split('、')
    for ele in ss_list:
        # 判断是否是省份
        if ele in city_dic:
            city_tier_dic['first_tier_city_cnt'] += province_city_tier_cnt_dic[ele]['first_tier_city_cnt']
            city_tier_dic['second_tier_city_cnt'] += province_city_tier_cnt_dic[ele]['second_tier_city_cnt']
            city_tier_dic['third_tier_city_cnt'] += province_city_tier_cnt_dic[ele]['third_tier_city_cnt']
            city_tier_dic['fourth_tier_city_cnt'] += province_city_tier_cnt_dic[ele]['fourth_tier_city_cnt']
            city_tier_dic['other_city_cnt'] += province_city_tier_cnt_dic[ele]['other_city_cnt']

        # 如果不是省份判断是否城市,如果是城市则增加
        elif ele in first_tier_city:
             city_tier_dic['first_tier_city_cnt']+=1
        elif ele in second_tier_city:
            city_tier_dic['second_tier_city_cnt']+=1
        elif ele in third_tier_city:
            city_tier_dic['third_tier_city_cnt']+=1
        elif ele in fourth_tier_city:
            city_tier_dic['fourth_tier_city_cnt']+=1
        elif '(' and ')' in ele:
            ele = re.findall(r'\((.*?)\)', ele)
            if ele in first_tier_city:
                city_tier_dic['first_tier_city_cnt']+=1
            elif ele in second_tier_city:
                city_tier_dic['second_tier_city_cnt']+=1
            elif ele in third_tier_city:
                city_tier_dic['third_tier_city_cnt']+=1
            elif ele in fourth_tier_city:
                city_tier_dic['fourth_tier_city_cnt']+=1
            else:
                city_tier_dic['other_city_cnt']+=1
        elif '(' in ele:
            ele = ele.split('(')[1]
            if ele in first_tier_city:
                city_tier_dic['first_tier_city_cnt']+=1
            elif ele in second_tier_city:
                city_tier_dic['second_tier_city_cnt']+=1
            elif ele in third_tier_city:
                city_tier_dic['third_tier_city_cnt']+=1
            elif ele in fourth_tier_city:
                city_tier_dic['fourth_tier_city_cnt']+=1
            else:
                city_tier_dic['other_city_cnt']+=1
        elif ')' in ele:
            ele = ele.split(')')[1]
            if ele in first_tier_city:
                city_tier_dic['first_tier_city_cnt']+=1
            elif ele in second_tier_city:
                city_tier_dic['second_tier_city_cnt']+=1
            elif ele in third_tier_city:
                city_tier_dic['third_tier_city_cnt']+=1
            elif ele in fourth_tier_city:
                city_tier_dic['fourth_tier_city_cnt']+=1
            else:
                city_tier_dic['other_city_cnt']+=1
        else:
            city_tier_dic['other_city_cnt']+=1

    return pd.Series([city_tier_dic['first_tier_city_cnt'], city_tier_dic['second_tier_city_cnt'], 
                           city_tier_dic['third_tier_city_cnt'], city_tier_dic['fourth_tier_city_cnt'],
                           city_tier_dic['other_city_cnt']], index=['first_tier_city_cnt', 'second_tier_city_cnt', 
                         'third_tier_city_cnt','fourth_tier_city_cnt', 'other_city_cnt'])            

def feature_process(df, model_type):
    df['开始时间'] = pd.to_datetime(df['开始时间'])
    df['结束时间'] = pd.to_datetime(df['结束时间'])
    print(df['开始时间'])
    df['日期'] = df['开始时间'].dt.date
    df['month'] = df['开始时间'].dt.month
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df['day_of_month'] = df['日期'].dt.day
    df['直播时长'] = df['直播时长'].apply(time_transform)
    df['is_holiday'] = df['日期'].apply(is_hoiday)
    df['day_of_week'] = df['日期'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].map({'Monday':0, 'Tuesday':0, 'Wednesday':0, 'Thursday':0, 'Friday':0, 'Saturday':1, 'Sunday':1})
    df['day_of_week'] = df['day_of_week'].replace({'Monday':'星期一', 'Tuesday':'星期二', 'Wednesday':'星期三', 'Thursday':'星期四','Friday':'星期五','Saturday':'星期六', 'Sunday':'星期日'})
    print(df['day_of_week'])
    Na_list = ['投放设置-优化周期','投放设置-智能优惠券','投放设置-投放时间', 
                 '投放设置-投放日期','投放设置-投放时段','定向人群-定向设置','定向人群-地域','定向人群-性别','定向人群-年龄',
                 '定向人群-行为兴趣意向','定向人群-抖音达人','定向人群-网络','定向人群-新客','定向人群-智能放量',
                 '定向人群-预估用户覆盖','定向人群-用户地域类型','定向人群-行为关键词','定向人群-行为类目词',
                 '定向人群-行为场景','定向人群-行为天数','定向人群-兴趣分类','定向人群-兴趣关键词','定向人群-抖音号分类',
                 '定向人群-抖音号达人','定向人群-粉丝互动行为','添加创意-创意形式','添加创意-创意类型',
                 '添加创意-抖音主页可见性设置','添加创意-创意分类','添加创意-创意标签']
    
    for col in Na_list:
        df[col] = df[col].replace({'':None, " ":None})
        df[col].fillna('没有设置', inplace = True)
    #    print(df[col].isnull().sum())
  #  print(df[df['投放设置-投放时间']==''].count())
  #  print(df['定向人群-网络'].head())
    df['主播id'] = df['主播id'].astype('str')
    df['主播id_投放设置-智能优惠券'] = df['主播id'] +'_'+ df['投放设置-智能优惠券']
    df['主播id_投放设置-优化周期'] = df['主播id']+'_'+ df['投放设置-优化周期']
    df['主播id_投放设置-优化目标'] = df['主播id'] +'_'+ df['投放设置-优化目标']
    df['主播id_投放设置-投放方式'] = df['主播id']+'_'+ df['投放设置-投放方式']
    df['主播id_推广方式'] = df['主播id'] +'_'+ df['推广方式']
    df['主播id_营销场景'] = df['主播id'] +'_'+ df['营销场景']

    df['定向人群-定向设置'] = df['定向人群-定向设置'].replace({'已有定向包':'自定义定向','没有设置':'不限'})
    cat_ops_cols = ['定向人群-行为兴趣意向','定向人群-抖音达人','定向人群-新客']
    df[cat_ops_cols] = df[cat_ops_cols].replace({'没有设置':'不限'})
    df['定向人群-智能放量'] = df['定向人群-智能放量'].replace({'没有设置':'启用'})
    df['添加创意-创意形式'] = df['添加创意-创意形式'].replace({'没有设置':'直播间画面'})
    df['添加创意-抖音主页可见性设置'] = df['添加创意-抖音主页可见性设置'].replace({'没有设置':'仅单次展示可见'})
    df['投放设置-投放时间'] = df['投放设置-投放时间'].replace({'没有设置':'长期投放'})
    df['投放设置-投放日期'] = df['投放设置-投放日期'].replace({'没有设置':'从今天起长期投放'})

    cat_cols= ['定向人群-定向设置', '定向人群-用户地域类型', '营销场景' ,'投放设置-优化目标', '投放设置-优化周期',
               '定向人群-行为兴趣意向', '定向人群-抖音达人', '定向人群-新客', '定向人群-智能放量', '添加创意-创意形式',
               '添加创意-抖音主页可见性设置', '投放设置-投放时间', '投放设置-投放日期']
    
    loaded_one_hot_encoder = load('./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/haoqi_category_one_hot_encoder.joblib')
    encoded_categorical_data = loaded_one_hot_encoder.transform(df[cat_cols]).todense()
    df_encoded = pd.DataFrame(encoded_categorical_data, columns=loaded_one_hot_encoder.get_feature_names_out())
    df = pd.concat([df.drop(cat_cols, axis=1), df_encoded], axis=1)
    try:
        df['投放设置-智能优惠券'] = df['投放设置-智能优惠券'].map({'启用':1, '不启用':0, '暂不支持':0, '没有设置':1})
        df[['first_tier_city_cnt', 'second_tier_city_cnt','third_tier_city_cnt','fourth_tier_city_cnt', 'other_city_cnt']] = df['定向人群-地域'].apply(city_encode)
        df[['target_men','target_women']] = df['定向人群-性别'].apply(gender_encode)
        df[['age_18_23', 'age_24_30','age_31_40','age_41_49','age_50+']] = df['定向人群-年龄'].apply(age_encode)
        df[['network_wifi','network_2G','network_3G','network_4G','network_5G']] = df['定向人群-网络'].apply(network_encode)
        df[['user_interact_电商互动行为','user_interact_APP推广互动行为', 'user_interact_资讯互动行为','user_interact_不限','user_interact_没有设置']] = df['定向人群-行为场景'].apply(user_behavior_context)
        df['定向人群-行为天数'] = df['定向人群-行为天数'].apply(user_interact_days)
        df['定向人群-预估用户覆盖'] = df['定向人群-预估用户覆盖'].apply(user_coverage_est)
        df[['关注行为','15天评论','15天点赞','15天分享','30天评论','30天点赞','30天分享','60天评论','60天点赞','60天分享',
            '15天直播观看','15天直播有效观看','15天直播评论','15天直播打赏','15天直播商品点击','15天直播视频下单','15天直播商品下单',
            '30天直播观看','30天直播有效观看','30天直播评论','30天直播打赏','30天直播商品点击','30天直播视频下单','30天直播商品下单',
            '60天直播观看','60天直播有效观看','60天直播评论','60天直播打赏','60天直播商品点击','60天直播视频下单', '60天直播商品下单',
            '15天购物车点击','15天购物车下单','30天购物车点击','30天购物车下单','60天购物车点击','60天购物车下单','没有设置']] = df['定向人群-粉丝互动行为'].apply(fans_interact_context)
    except:
        print('error')
    with open('./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/haoqi_targeted_user_coverage.pkl', 'rb') as file:  # 'rb' 代表读取二进制模式
        mean_dic = pickle.load(file)
    mean_user_coverage = mean_dic['定向人群-预估用户覆盖均值']
    targeted_roi_mean = mean_dic['投放设置-支付ROI目标均值']
    cpm_mean = mean_dic['平均千次展现费用均值']

    df['定向人群-预估用户覆盖'] = df['定向人群-预估用户覆盖'].replace({0:mean_user_coverage})
    df['投放设置-支付ROI目标'] = df['投放设置-支付ROI目标'].replace({'': None})
    df['投放设置-支付ROI目标'] = df['投放设置-支付ROI目标'].astype('float')
    df['投放设置-支付ROI目标'] = df['投放设置-支付ROI目标'].fillna(targeted_roi_mean)
    df['出价(元)'] = df['出价(元)'].replace({'-': None})
    df['出价(元)'] = df['出价(元)'].fillna(cpm_mean)
    try:
        df[['ads_start_time','ads_end_time', 'ads_hour']] = df.apply(ads_hour_encode, axis=1)
        label_encode_list = ['day_of_week','is_holiday','推广方式','投放设置-投放方式']
        for col in label_encode_list:
            loaded_label_encoder = load(f'./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/haoqi_category_label_encoder_{col}.joblib')
            label_encode_data = encode_with_unknown(loaded_label_encoder, df[col])
            df[col] =  label_encode_data
        

        bool_cols= df.select_dtypes(include='bool')
        for col in bool_cols:
            df[col] = df[col].map({True: 1, False:0})

        df['开始时间'] = df['开始时间'].dt.hour
        df['结束时间'] = df['结束时间'].dt.hour
        df['千川id'] = df['千川id'].astype(str)
        df['抖音号'] = df['抖音号'].astype(str)

        target_encode_list = ['千川id','抖音号', '主播id_投放设置-智能优惠券','主播id_投放设置-优化周期','主播id_投放设置-优化目标',
                            '主播id_投放设置-投放方式','主播id_推广方式','主播id_营销场景','定向人群-行为关键词','定向人群-行为类目词',
                            '定向人群-兴趣分类','定向人群-兴趣关键词','定向人群-抖音号分类','定向人群-抖音号达人',
                            '添加创意-创意分类','添加创意-创意标签']

        loaded_target_encoder_consumption = load(f'./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/haoqi_target_encoder_encode_by_消耗(元)_{model_type}.pkl')
        loaded_target_encoder_roi = load(f'./直播间计划/haoqi_0930_wth_planid_wth_content_wth_product/haoqi_target_encoder_encode_by_广告支付ROI_{model_type}.pkl')
        df[[f'{ele}_encode_by_消耗(元)' for ele in target_encode_list]] = loaded_target_encoder_consumption.transform(df[target_encode_list])
        df[[f'{ele}_encode_by_广告支付ROI' for ele in target_encode_list]] = loaded_target_encoder_roi.transform(df[target_encode_list])
        
        del_list =['抖音号','直播间ID','日期','用户名','SN','主播id','投放设置-投放时段','定向人群-地域','定向人群-性别', '计划id', '计划ID',
                '定向人群-年龄','定向人群-网络','定向人群-行为场景','定向人群-粉丝互动行为','添加创意-创意类型'] + target_encode_list
      
    
        del_list = del_list
        sel_list =[ele for ele in df.columns if ele not in del_list]
        df = df[sel_list]
       
    except:
        print('error 2')

    df = content_feature_gen(df)
    s_list = ['价格','库存','近1日金额汇总','近1日商品次数','近1日平均金额','近1日最大曝光-点击转化率',
                '近1日最小曝光-点击转化率','近1日平均曝光-点击转化率','近1日最大点击-成交转化率','近1日最小点击-成交转化率','近1日平均点击-成交转化率',
                '近7日金额汇总','近7日商品次数','近7日平均金额','近7日最大曝光-点击转化率','近7日最小曝光-点击转化率','近7日平均曝光-点击转化率',
                '近7日最大点击-成交转化率','近7日最小点击-成交转化率','近7日平均点击-成交转化率','近30日金额汇总','近30日商品次数','近30日平均金额',
                '近30日最大曝光-点击转化率','近30日最小曝光-点击转化率','近30日平均曝光-点击转化率','近30日最大点击-成交转化率','近30日最小点击-成交转化率','近30日平均点击-成交转化率']
    for col in s_list:
        df[col] = df[col].apply(type_convert)

    df = product_feature_ops(df)
    df.drop(s_list, axis=1, inplace=True)
    cat_list = df.select_dtypes(include=['object']).columns.tolist()
    #   df[cat_list] = df[cat_list].replace({" ":np.nan,"":np.nan})
    df[cat_list] = df[cat_list].astype('float')
    df = df.round(2)
    return df


def extract_data_from_sql_server():
    server = '192.168.1.198'
    database = 'WIN_DOUYIN'
    username = 'winc_yuxiaojiao'
    password = 'AtdQEPV4EaovDwBgoD0a'

    # 执行查询并读取数据
    query = "SELECT * FROM 投流模型_特征表_线上"  # 替换为实际的表名和查询
    connect = pymssql.connect(server, username, password, database)
    cursor = connect.cursor()
    cursor.execute(query)  # 执行sql语句
    column_names = [i[0] for i in cursor.description]
# 打印列名
    res = cursor.fetchall() 
    df = pd.DataFrame(res, columns=column_names)
  
    del_list_1 = []
    sub_str =['计划ID','计划id']
    for col in df.columns:
        if  any(sub_string in col for sub_string in sub_str):
            del_list_1.append(col)     

    del_list = ['价格','库存','近1日金额汇总','近1日商品次数','近1日平均金额','近1日最大曝光-点击转化率','近1日最小曝光-点击转化率',
                '近1日平均曝光-点击转化率','近1日最大点击-成交转化率','近1日最小点击-成交转化率','近1日平均点击-成交转化率',
                '近7日金额汇总','近7日商品次数','近7日平均金额','近7日最大曝光-点击转化率','近7日最小曝光-点击转化率','近7日平均曝光-点击转化率',
                '近7日最大点击-成交转化率','近7日最小点击-成交转化率','近7日平均点击-成交转化率','近30日金额汇总','近30日商品次数',
                '近30日平均金额','近30日最大曝光-点击转化率','近30日最小曝光-点击转化率','近30日平均曝光-点击转化率','近30日最大点击-成交转化率',
                '近30日最小点击-成交转化率','近30日平均点击-成交转化率','本场直播商品id 列表','评估状态','评估结果'] + del_list_1
    
    df.drop(del_list,axis=1,inplace=True)
  
    return df



if __name__ == '__main__':
    import json
    df = extract_data_from_sql_server()
    row = df.iloc[2]
# 将这一行转换为JSON字符串
    row_json = row.to_json(orient='records')
  #  print(row_json)
    row_dict = json.loads(row_json)
    my_dict = dict(zip(df.columns, row_dict))
   # print(my_dict)
    
    
    with open('./my_dict1.json', 'w', encoding='utf-8') as file:
    # 使用json.dump()将字典写入文件
    # indent参数用于美化输出，表示缩进的空格数
        json.dump(my_dict, file, ensure_ascii=False, indent=4)
    df_test = pd.DataFrame(my_dict, index=[0])
    
    json_file ='./my_dict_positive.json'
    with open(json_file, 'rb') as file:
        data = json.load(file)
    df_test = pd.DataFrame(data, index=[0])
   
    df = feature_process(df_test)
 #   df = feature_process(df)
    # 预测
   
    pkl_file = '../model/直播间计划/haoqi_lgb_for_online_pred.pkl'
    with open(pkl_file, 'rb') as model_file:
        model = pickle.load(model_file)
    pred_proba = model.predict(df)
    y_pred = (pred_proba >= 0.1).astype(int)
    print(y_pred)