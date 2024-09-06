import pandas as pd
from scipy.optimize import linprog

# 读取数据
crop_data = pd.read_csv('processed_crop_data.csv')
land_data = pd.read_csv('processed_land_data.csv')
planting_data = pd.read_csv('processed_planting_data.csv')

# 假设 crop_data, planting_data 和 land_data 是包含所有作物数据、种植面积数据和地块类型数据的 DataFrame
# 添加 land_type 到 planting_data
planting_data['land_type'] = planting_data['land_id'].apply(lambda x: land_data[land_data['land_id'] == x]['land_type'].values[0])

# 合并 crop_data 和 planting_data
merged_data = pd.merge(crop_data, planting_data, on=['crop_name', 'season', 'land_type'])


# 计算市场需求量（亩产量 * 去年种植面积）
merged_data['market_demand'] = merged_data['yield'] * merged_data['crop_area']

# 计算每种作物在每个季节的总市场需求量
market_demand = merged_data.groupby(['crop_name', 'season'])['market_demand'].sum().reset_index()
# 单季的作物改为第一季
market_demand.loc[market_demand['season'] == '单季', 'season'] = '第一季'
crop_data.loc[crop_data['season'] == '单季', 'season'] = '第一季'
# print(market_demand)

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
for land in lands:
    for crop in crops:
        for season in seasons:
            crop_info = crop_data[(crop_data['crop_name'] == crop) & (crop_data['season'] == season) & (crop_data['land_type'] == land_data[land_data['land_id'] == land]['land_type'].values[0])]

            if not crop_info.empty:
                # price = crop_info['price_min'].values[0]
                # price 使用最小值和最大值的平均值
                price = (crop_info['price_min'].values[0] + crop_info['price_max'].values[0]) / 2
                yield_per_acre = crop_info['yield'].values[0]
                cost = crop_info['cost'].values[0]
                c.append(-(price * yield_per_acre - cost))  # 负号用于最大化
            else:
                c.append(0)
# 定义约束矩阵和约束向量
A = []
b = []

# 地块面积限制
for land in lands:
    row = [0] * (num_crops * num_lands * num_seasons)
    for i, crop in enumerate(crops):
        for j, season in enumerate(seasons):
            index = lands.tolist().index(land) * num_crops * num_seasons + i * num_seasons + j
            row[index] = 1
    A.append(row)
    b.append(land_data[land_data['land_id'] == land]['land_area'].values[0])


# 市场需求限制
for crop in crops:
    for season in seasons:
        row = [0] * (num_crops * num_lands * num_seasons)
        for land in lands:
            index = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + seasons.tolist().index(season)
            crop_info = crop_data[(crop_data['crop_name'] == crop) & (crop_data['season'] == season)]
            if not crop_info.empty:
                yield_per_acre = crop_info['yield'].values[0]
                row[index] = yield_per_acre
        A.append(row)
        # 使用新的市场需求量 DataFrame
        demand_info = market_demand[(market_demand['crop_name'] == crop) & (market_demand['season'] == season)]
        if not demand_info.empty:
            b.append(demand_info['market_demand'].values[0])
        else:
            b.append(0)

# print(len(c), len(A), len(b))

# # 轮作要求
# for land in lands:
#     for crop in crops:
#         row = [0] * (num_crops * num_lands * num_seasons)
#         for season in seasons:
#             index = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + seasons.tolist().index(season)
#             row[index] = 1
#         A.append(row)
#         b.append(1)

# # 作物不连续重茬种植
# for land in lands:
#     for crop in crops:
#         for season in range(num_seasons - 1):
#             row = [0] * (num_crops * num_lands * num_seasons)
#             index1 = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + season
#             index2 = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + (season + 1)
#             row[index1] = 1
#             row[index2] = 1
#             A.append(row)
#             b.append(1)

# # 种植地不宜过于分散
# for crop in crops:
#     for season in seasons:
#         row = [0] * (num_crops * num_lands * num_seasons)
#         for land in lands:
#             index = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + seasons.tolist().index(season)
#             row[index] = 1
#         A.append(row)
#         b.append(2) 

# # 单类作物面积不宜过小
# for land in lands:
#     for crop in crops:
#         for season in seasons:
#             row = [0] * (num_crops * num_lands * num_seasons)
#             index = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + seasons.tolist().index(season)
#             row[index] = 1
#             A.append(row)
#             b.append(0.5)  # 假设每种作物每季最小种植面积为0.5亩

# 求解线性规划问题


res = linprog(c, A_ub=A, b_ub=b, method='highs')

# 解析结果
if res.success:
    print("最优解：")
    for i, land in enumerate(lands):
        for j, crop in enumerate(crops):
            for k, season in enumerate(seasons):
                index = i * num_crops * num_seasons + j * num_seasons + k
                if res.x[index] > 0:
                    print(f"地块 {land} 在季节 {season} 种植 {crop} 面积: {res.x[index]:.2f} 亩")
else:
    print("没有找到可行解")