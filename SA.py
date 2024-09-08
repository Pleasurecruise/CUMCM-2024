


import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linprog

# 加载数据
crop_data = pd.read_csv('data/processed_crop_data.csv')
land_data = pd.read_csv('data/processed_land_data.csv')

# 定义常量
n_crops = len(crop_data)
n_lands = len(land_data)
seasons = ["第一季", "第二季"]
seasons = pd.Series(seasons)
n_seasons = len(seasons)
n_simulations = 10  # 可以先减少模拟次数以便调试

# 提取作物的价格、成本、产量

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

# 目标函数系数定义
def calculate_profit(x, p, y, c):
    return np.sum(p * y * x - c * x)

# 蒙特卡洛模拟函数
def monte_carlo_simulation(n_simulations):
    simulation_results = []
    
    for i in tqdm(range(n_simulations)):
    # 随机种植方案生成
        random_solution = np.zeros((n_lands, n_crops, n_seasons))

        for land in range(n_lands):
            for season in range(n_seasons):  # 确保 season 在 0 到 n_seasons-1 之间
                remaining_area = A[land]
                for crop in range(n_crops):
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
    for i in tqdm(range(n_simulations)):
        


        
        # 随机扰动价格

        random_P = np.random.uniform(0,7, size=n_crops) # 价格波动范围
     
        # 随机种植方案生成
        random_solution = np.random.uniform(0, A[:, np.newaxis], size=(n_lands, n_crops))
        
        # 检查生产量是否超出市场需求，调整超出的部分降价销售
        production = random_solution * Y
        q_normal = np.minimum(production.sum(axis=0), D)  # 正常销售
        
        
        # 计算总收益
        revenue_normal = np.sum(q_normal * (P_max+random_P))

        total_cost = np.sum(random_solution * C_Y)
        
        total_profit = revenue_normal  - total_cost
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
