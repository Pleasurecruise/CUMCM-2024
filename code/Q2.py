import pandas as pd
from scipy.optimize import linprog
import random

# 读取数据
planting_data = pd.read_csv('data/processed_planting_data.csv')  # 2023年的种植数据
crop_data = pd.read_csv('data/processed_crop_data.csv')
land_data = pd.read_csv('data/processed_land_data.csv')

# 初始化数据
yearly_data = {2023: planting_data}
yearly_profit = {}

# 豆类作物
legume_crops = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '豇豆', '刀豆', '芸豆']

# 定义作物类别
grain_crops = [
    "黄豆",
    "黑豆",
    "红豆",
    "绿豆",
    "爬豆",
    "小麦",
    "玉米",
    "谷子",
    "高粱",
    "黍子",
    "荞麦",
    "南瓜",
    "红薯",
    "莜麦",
    "大麦",
    "水稻"
]

vegetable_crops = [
    "豇豆",
    "刀豆",
    "芸豆",
    "土豆",
    "西红柿",
    "茄子",
    "菠菜",
    "青椒",
    "菜花",
    "包菜",
    "油麦菜",
    "小青菜",
    "黄瓜",
    "生菜",
    "辣椒",
    "空心菜",
    "黄心菜",
    "芹菜",
    "大白菜",
    "白萝卜",
    "红萝卜"
]

mushroom_crops = [
    "榆黄菇",
    "香菇",
    "白灵菇",
    "羊肚菌"
]

for current_year in range(2024, 2031):
    # 读取数据
    planting_data = yearly_data[current_year - 1]  # 使用前一年的种植数据作为基础迭代

    # 添加 land_type 到 planting_data
    planting_data['land_type'] = planting_data['land_id'].apply(lambda x: land_data[land_data['land_id'] == x]['land_type'].values[0])

    # 合并 crop_data 和 planting_data
    merged_data = pd.merge(crop_data, planting_data, on=['crop_name', 'season', 'land_type'])

    print(f"开始计算{current_year}年的最优解")

    # 计算市场需求量（亩产量 * 去年种植面积）
    merged_data['market_demand'] = merged_data['yield'] * merged_data['crop_area']

    # 计算每种作物在每个季节的总市场需求量
    market_demand = merged_data.groupby(['crop_name', 'season'])['market_demand'].sum().reset_index()
    # 单季的作物改为第一季
    market_demand.loc[market_demand['season'] == '单季', 'season'] = '第一季'
    crop_data.loc[crop_data['season'] == '单季', 'season'] = '第一季'

    # 提取参数
    crops = crop_data['crop_name'].unique()
    lands = land_data['land_id'].unique()
    seasons = ["第一季", "第二季"]
    seasons = pd.Series(seasons)

    # 定义变量数量
    num_crops = len(crops)
    num_lands = len(lands)
    num_seasons = len(seasons)

    # 定义目标函数系数
    c = []
    c_discounted = []
    for land in lands:
        for crop in crops:
            for season in seasons:
                crop_info = crop_data[(crop_data['crop_name'] == crop) & (crop_data['season'] == season) & (crop_data['land_type'] == land_data[land_data['land_id'] == land]['land_type'].values[0])]

                if not crop_info.empty:
                    # 处理价格变化
                    if crop in grain_crops:
                        price = crop_info['price_min'].values[0]  # 粮食类作物价格稳定
                    elif crop in vegetable_crops:
                        price = crop_info['price_min'].values[0] * (1 + 0.05 * (current_year - 2023))  # 蔬菜类作物价格每年增长5%
                    elif crop in mushroom_crops:
                        if crop == '羊肚菌':
                            price = crop_info['price_min'].values[0] * (1 - 0.05 * (current_year - 2023))  # 羊肚菌价格每年下降5%
                        else:
                            price = crop_info['price_min'].values[0] * (1 - random.uniform(0.01, 0.05) * (current_year - 2023))  # 其他食用菌价格每年下降1%~5%
                    else:
                        price = random.uniform(crop_info['price_min'].values[0], crop_info['price_max'].values[0])

                    # 处理亩产量变化
                    yield_per_acre = crop_info['yield'].values[0] * (1 + random.uniform(-0.1, 0.1))

                    # 处理种植成本变化
                    cost = crop_info['cost'].values[0] * (1 + 0.05 * (current_year - 2023))

                    c.append(-(price * yield_per_acre - cost))  # 因为 linprog 求最小值，所以这里加负号
                    c_discounted.append(-(0.5 * price * yield_per_acre - cost))  # 50%降价
                else:
                    c.append(0)
                    c_discounted.append(0)
    # 降价后的目标函数系数
    c.extend(c_discounted)

    # A b 存储约束条件
    A = []
    b = []

    # 地块面积限制
    for land in lands:
        for season in seasons:
            row = [0] * (2 * num_crops * num_lands * num_seasons)
            for i, crop in enumerate(crops):
                index = lands.tolist().index(land) * num_crops * num_seasons + i * num_seasons + seasons.tolist().index(season)
                row[index] = 1
                row[index + num_crops * num_lands * num_seasons] = 1  # 超售部分
            A.append(row)
            b.append(land_data[land_data['land_id'] == land]['land_area'].values[0])

    # 市场需求限制
    for crop in crops:
        for season in seasons:
            row = [0] * (2 * num_crops * num_lands * num_seasons)
            for land in lands:
                index = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + seasons.tolist().index(season)
                crop_info = crop_data[(crop_data['crop_name'] == crop) & (crop_data['season'] == season)]
                if not crop_info.empty:
                    yield_per_acre = crop_info['yield'].values[0]
                    row[index] = yield_per_acre
                    row[index + num_crops * num_lands * num_seasons] = yield_per_acre  # 超售部分
            A.append(row)
            demand_info = market_demand[(market_demand['crop_name'] == crop) & (market_demand['season'] == season)]
            if not demand_info.empty:
                if crop in ["小麦", "玉米"]:
                    b.append(demand_info['market_demand'].values[0] * (1 + random.uniform(0.05, 0.1)))  # 小麦和玉米市场需求增长5%~10%
                else:
                    b.append(demand_info['market_demand'].values[0] * (1 + random.uniform(-0.05, 0.05)))  # 其他作物市场需求变化±5%
            else:
                b.append(0)

    # 豆类轮作要求
    if current_year >= 2025:
        previous_years = [current_year - 1, current_year - 2]
        previous_data = pd.concat([yearly_data[y] for y in previous_years if y in yearly_data])

        for land in lands:
            legume_planted = previous_data[(previous_data['land_id'] == land) & (previous_data['crop_name'].isin(legume_crops))]
            if legume_planted.empty:
                for crop in legume_crops:
                    row = [0] * (2 * len(crops) * len(lands) * len(seasons))
                    for season in seasons:
                        index = list(lands).index(land) * len(crops) * len(seasons) + list(crops).index(crop) * len(seasons) + list(seasons).index(season)
                        row[index] = -1
                        row[index + num_crops * num_lands * num_seasons] = -1  # 超售部分
                    A.append(row)
                    b.append(0)

    # 求解线性规划问题
    res = linprog(c, A_ub=A, b_ub=b, method='highs')

    # 输出结果
    if res.success:
        print(f"{current_year}年最优解已找到")
        print(f"最优解：{ -res.fun}")
        yearly_profit[current_year] = -res.fun

        rows = []
        for i, land in enumerate(lands):
            for j, crop in enumerate(crops):
                for k, season in enumerate(seasons):
                    index = i * num_crops * num_seasons + j * num_seasons + k
                    if res.x[index] > 0:
                        rows.append({
                            'land_id': land,
                            'crop_name': crop,
                            'crop_area': res.x[index],
                            'season': season
                        })
                    if res.x[index + num_crops * num_lands * num_seasons] > 0:
                        rows.append({
                            'land_id': land,
                            'crop_name': crop,
                            'crop_area': res.x[index + num_crops * num_lands * num_seasons],
                            'season': season,
                            'discounted': True
                        })

        current_year_data = pd.DataFrame(rows)
        print(f"{current_year}年的种植数据：")
        print(current_year_data.head())
                    
        yearly_data[current_year] = current_year_data
    else:
        print(f"{current_year}年没有找到可行解")
        break

# 保存到excel
template_df = pd.read_excel("code/results_template.xlsx")
# 0 - 53 行 为第一季
land_id_first = template_df["地块名"][0:54]
land_id_first = land_id_first + "_第一季"
# 54 - 81 行 为第二季
land_id_second = template_df["地块名"][54:82]
land_id_second = land_id_second + "_第二季"

template_cols = template_df.columns[2:]
with pd.ExcelWriter('results/result2.xlsx') as writer:
    for year, data in yearly_data.items():
        if (year == 2023):
            continue

        df = pd.DataFrame(index=range(0, len(land_id_first) + len(land_id_second)), columns=template_cols)
        df["地块名"] = pd.concat([land_id_first, land_id_second], ignore_index=True)
        for i, row in data.iterrows():
            land_id = row['land_id']
            crop_name = row['crop_name']
            crop_area = row['crop_area']
            season = row['season']
            land_name_with_season = land_id + "_" + season
            index = df[df["地块名"] == land_name_with_season].index[0]
            df.at[index, crop_name] = crop_area

        # 空缺值填充为 0
        df = df.fillna(0)
        df.to_excel(writer, sheet_name=str(year), index=False)
        print(f"已保存{year}年的结果")

# 保存年度利润
yearly_profit_df = pd.DataFrame(yearly_profit.items(), columns=['year', 'profit'])
yearly_profit_df.to_excel('results/profit2.xlsx', index=False)

print("数据已保存")