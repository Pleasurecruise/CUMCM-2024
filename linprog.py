import pandas as pd
from scipy.optimize import linprog

planting_data = pd.read_csv('processed_planting_data.csv') # 2023年的种植数据
yearly_data = {
    2023: planting_data
}


crop_data = pd.read_csv('processed_crop_data.csv')
land_data = pd.read_csv('processed_land_data.csv')
# 豆类作物
legume_crops = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '豇豆', '刀豆', '芸豆']

for current_year in range(2024, 2031):
    # 读取数据
    planting_data = yearly_data[current_year - 1] # 使用前一年的种植数据作为基础

    # 假设 crop_data, planting_data 和 land_data 是包含所有作物数据、种植面积数据和地块类型数据的 DataFrame
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
        for season in seasons:
            row = [0] * (num_crops * num_lands * num_seasons)
            for i, crop in enumerate(crops):
                index = lands.tolist().index(land) * num_crops * num_seasons + i * num_seasons + seasons.tolist().index(season)
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

    # 轮作要求

    # for land in lands:
    #     for crop in legume_crops:
    #         row = [0] * (num_crops * num_lands * num_seasons)
    #         for season in seasons:
    #             index = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + seasons.tolist().index(season)
    #             row[index] = 1
    #         A.append(row)
    #         b.append(1)


    # 求解线性规划问题


    res = linprog(c, A_ub=A, b_ub=b, method='highs')

    # # 解析结果
    # if res.success:
    #     print("最优解：")
    #     for i, land in enumerate(lands):
    #         for j, crop in enumerate(crops):
    #             for k, season in enumerate(seasons):
    #                 index = i * num_crops * num_seasons + j * num_seasons + k
    # else:
    #     print("没有找到可行解")
    if res.success:
        print(f"{current_year}年最优解已找到")
        print(f"最优解：{ -res.fun}")
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

        current_year_data = pd.DataFrame(rows)
        print(f"{current_year}年的种植数据：")
        print(current_year_data.head())
                    
        yearly_data[current_year] = current_year_data
    else:
        print(f"{current_year}年没有找到可行解")
        break


# 保存到一个Excel文件的不同工作表中

template_df = pd.read_excel("result1_1.xlsx")
# 0 - 53 行 为第一季
land_id_first = template_df["地块名"][0:54]
land_id_first = land_id_first + "_第一季"
# 54 - 81 行 为第二季
land_id_second = template_df["地块名"][54:82]
land_id_second = land_id_second + "_第二季"

template_cols = template_df.columns[2:]
with pd.ExcelWriter('results.xlsx') as writer:
    for year, data in yearly_data.items():
        if (year == 2023):
            continue

        df = pd.DataFrame(index=range(0, len(land_id_first) + len(land_id_second)), columns=template_cols)
        # 请补全这里
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
        
