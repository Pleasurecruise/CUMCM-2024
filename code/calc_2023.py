import pandas as pd
import random

planting_data = pd.read_csv('data/processed_planting_data.csv') # 2023年的种植数据
# 读取之前处理好的数据
crop_data = pd.read_csv('data/processed_crop_data.csv')
land_data = pd.read_csv('data/processed_land_data.csv')

# 添加 land_type 到 planting_data
planting_data['land_type'] = planting_data['land_id'].apply(lambda x: land_data[land_data['land_id'] == x]['land_type'].values[0])

merged_data_2023 = pd.merge(planting_data, crop_data, on=['crop_name', 'season', 'land_type'])

merged_data_2023['profit'] = (merged_data_2023['price_min'] * merged_data_2023['yield'] - merged_data_2023['cost']) * merged_data_2023['crop_area']

total_profit_2023 = merged_data_2023['profit'].sum()

print(f"2023年的总利润（最低单价）为: {total_profit_2023} 元")

merged_data_2023['profit'] = (merged_data_2023['price_max'] * merged_data_2023['yield'] - merged_data_2023['cost']) * merged_data_2023['crop_area']

total_profit_2023 = merged_data_2023['profit'].sum()

print(f"2023年的总利润（最高单价）为: {total_profit_2023} 元")

# price 使用平均值
merged_data_2023['profit'] = ((merged_data_2023['price_min'] + merged_data_2023['price_max']) / 2 * merged_data_2023['yield'] - merged_data_2023['cost']) * merged_data_2023['crop_area']

total_profit_2023 = merged_data_2023['profit'].sum()

print(f"2023年的总利润（平均单价）为: {total_profit_2023} 元")

# price 使用随机值
merged_data_2023['random_price'] = merged_data_2023.apply(
    lambda row: random.uniform(row['price_min'], row['price_max']), axis=1
)
merged_data_2023['profit'] = (merged_data_2023['random_price'] * merged_data_2023['yield'] - merged_data_2023['cost']) * merged_data_2023['crop_area']

total_profit_2023 = merged_data_2023['profit'].sum()

print(f"2023年的总利润（随机单价）为: {total_profit_2023} 元")
