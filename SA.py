


import numpy as np
import pandas as pd
from scipy.optimize import linprog

# 加载数据
crop_data = pd.read_csv('data/processed_crop_data.csv')
land_data = pd.read_csv('data/processed_land_data.csv')

# 定义常量
n_crops = len(crop_data)
n_lands = len(land_data)
n_simulations = 10  # 可以先减少模拟次数以便调试

# 提取作物的价格、成本、产量
P = crop_data['price_min'].values
C = crop_data['cost'].values
Y = crop_data['yield'].values
D = crop_data['yield'].values  # 假设市场需求量为前一年产量
C_Y = C / Y  # 成本产量比

#生成随机数的最大值和最小值
P_min = crop_data['price_min'].values
P_max = crop_data['price_max'].values

# 地块面积
A = land_data['land_area'].values


planting_data = pd.read_csv('data/processed_planting_data.csv') # 2023年的种植数据
yearly_data = {
    2023: planting_data
}
current_year = 2024

legume_crops = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '豇豆', '刀豆', '芸豆']

# 目标函数系数定义
def calculate_profit(x, p, y, c):
    return np.sum(p * y * x - c * x)

# 蒙特卡洛模拟函数
def monte_carlo_simulation(n_simulations):
    simulation_results = []
    
    for i in range(n_simulations):
        # 输出模拟进度
        if i % 10 == 0:  # 每10次输出一次进度
            print(f"模拟进度: {i}/{n_simulations}")
        
        # 提取参数
        crops = crop_data['crop_name'].unique()
        lands = land_data['land_id'].unique()
        seasons = ["第一季", "第二季"]
        seasons = pd.Series(seasons)
        print(f"开始计算{current_year}年的最优解")
        # 定义变量数量
        num_crops = len(crops)
        num_lands = len(lands)
        num_seasons = len(seasons)

        planting_data = yearly_data[current_year - 1] # 使用前一年的种植数据作为基础

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

    #定义约束矩阵和约束向量
        X = []
        y = []

        # # 地块面积限制
        # for land in lands:
        #     for season in seasons:
        #         row = [0] * (num_crops * num_lands * num_seasons)
        #         for i, crop in enumerate(crops):
        #             index = lands.tolist().index(land) * num_crops * num_seasons + i * num_seasons + seasons.tolist().index(season)
        #             row[index] = 1
        #         X.append(row)
        #         y.append(land_data[land_data['land_id'] == land]['land_area'].values[0])

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
                X.append(row)
                # 使用新的市场需求量 DataFrame
                demand_info = market_demand[(market_demand['crop_name'] == crop) & (market_demand['season'] == season)]
                if not demand_info.empty:
                    y.append(demand_info['market_demand'].values[0])
                else:
                    y.append(0)
        # # 轮作要求               
        # for land in lands:
        #     for crop in legume_crops:
        #         row = [0] * (num_crops * num_lands * num_seasons)
        #         for season in seasons:
        #             index = lands.tolist().index(land) * num_crops * num_seasons + crops.tolist().index(crop) * num_seasons + seasons.tolist().index(season)
        #             row[index] = 1
        #         X.append(row)
        #         y.append(1)


            # 随机扰动市场需求和价格
            random_D = D # # 市场需求波动范围
            random_P = np.random.uniform(P_min, P_max, size=n_crops) # 价格波动范围
            random_Y = Y #* np.random.uniform(0.9, 1.1, size=n_crops) # 产量波动范围
            
            # 随机种植方案生成
            # random_solution = np.random.uniform(0, A[:, np.newaxis], size=(n_lands, n_crops))
            # print(random_solution)
            random_solution = np.zeros((num_lands, num_crops, num_seasons))

            for land in lands:
                for season in seasons:
                    remaining_area = A[land]
                    for crop in crops:
                        if crop in legume_crops:
                            # 确保豆类作物的种植符合轮作要求
                            if np.random.rand() > 0.5:  # 50%的概率种植豆类作物
                                area = np.random.uniform(0, remaining_area)
                                random_solution[land, crop, season] = area
                                remaining_area -= area
                        else:
                            # 非豆类作物的种植
                            area = np.random.uniform(0, remaining_area)
                            random_solution[land, crop, season] = area
                            remaining_area -= area

            print(random_solution)  # 输出随机种植方案


            # 生成随机种植方案
            # random_solution = np.random.randint(0, 2, size=(n_lands, n_crops))

            # # # 对每个地块的种植方案进行归一化处理
            # # for j in range(n_lands):
            # #     random_solution[j] = random_solution[j] / random_solution[j].sum() * A[j]

            # print(random_solution)  # 输出随机种植方案        
            
            # 检查生产量是否超出市场需求，调整超出的部分降价销售
            production = random_solution * random_Y
            q_normal = np.minimum(production.sum(axis=0), random_D)  # 正常销售
            
        #   q_discount = np.maximum(0, production.sum(axis=0) - random_D)  # 降价销售、
            q_discount = 0
            
            # 计算总收益
            revenue_normal = np.sum(q_normal * random_P)
            revenue_discount = 0
            total_cost = np.sum(random_solution * C_Y)
            
            total_profit = revenue_normal + revenue_discount - total_cost
            simulation_results.append(total_profit)
        
    return simulation_results

# 执行蒙特卡洛模拟
print("开始蒙特卡洛模拟...")
simulation_profits = monte_carlo_simulation(n_simulations)
print(simulation_profits)
print("蒙特卡洛模拟完成")

# 初始化最大值为列表中的第一个元素
max_profit = simulation_profits[0]

# 遍历列表中的每个元素
for profit in simulation_profits:
    if profit > max_profit:
        max_profit = profit

# 打印最大值
print("最大利润:", max_profit)
# # 线性规划求解最优解
# c = np.zeros(n_crops * n_lands + 2 * n_crops)
# for i in range(n_lands):
#     for j in range(n_crops):
#         c[i * n_crops + j] = -C[j]  # 生产成本
# for j in range(n_crops):
#     c[n_crops * n_lands + j] = -P[j]  # 正常销售收益
#     c[n_crops * n_lands + n_crops + j] = -0.5 * P[j]  # 降价销售收益

# # 定义线性规划约束
# A_ub = np.zeros((n_lands, n_crops * n_lands + 2 * n_crops))
# for i in range(n_lands):
#     A_ub[i, i * n_crops:(i + 1) * n_crops] = Y
# b_ub = A

# # 确保 Y 的大小和作物的数量一致
# assert len(Y) == n_crops, "作物产量数据的大小与作物数量不一致。请检查数据表。"

# # 确保 A_eq 的形状符合 n_crops 和 n_lands 的定义
# A_eq = np.zeros((n_crops, n_lands * n_crops + 2 * n_crops))

# # 处理作物的产量限制
# for j in range(n_crops):
#     num_land_slots = len(A_eq[j, j::n_crops])  # 计算每个作物的地块数量
#     assert num_land_slots == n_lands, f"地块数量不匹配：{num_land_slots} != {n_lands}"
#     A_eq[j, j::n_crops] = np.repeat(Y[j], n_lands)  # 将每个作物的产量复制到它对应的地块
#     A_eq[j, n_crops * n_lands + j] = -1  # 正常销售量
#     A_eq[j, n_crops * n_lands + n_crops + j] = -1  # 降价销售量

# b_eq = np.zeros(n_crops)
# A_ub_demand = np.zeros((n_crops, n_crops * n_lands + 2 * n_crops))
# for j in range(n_crops):
#     A_ub_demand[j, n_crops * n_lands + j] = 1
# b_ub_demand = D
# A_ub = np.vstack([A_ub, A_ub_demand])
# b_ub = np.hstack([b_ub, b_ub_demand])

# # 开始线性规划求解
# print("开始线性规划求解...")
# result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

# if result.success:
#     print(f"线性规划求解成功，最优收益为: {-result.fun}")
# else:
#     print("线性规划问题未能成功求解")

# 蒙特卡洛结果分析
mean_profit = np.mean(simulation_profits)
std_profit = np.std(simulation_profits)
print(f"蒙特卡洛模拟平均收益: {mean_profit}")
print(f"蒙特卡洛模拟收益标准差: {std_profit}")

# # 比较线性规划解与蒙特卡洛模拟结果
# if optimal_profit > mean_profit + std_profit:
#     print("线性规划的解在模拟中表现较优。")
# else:
#     print("线性规划的解未在模拟中表现较优，可能需要进一步优化。")
