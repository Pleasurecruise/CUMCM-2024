import pandas as pd

# 读取Excel文件
land_df = pd.read_excel('data/data_land.xlsx')
crop_df = pd.read_excel('data/data_crop.xlsx')
planting_df = pd.read_excel('data/data_2023.xlsx')

# 检查缺失值
print("Land Data Missing Values:\n", land_df.isnull().sum())
print("Crop Data Missing Values:\n", crop_df.isnull().sum())
print("Planting Data Missing Values:\n", planting_df.isnull().sum())

# 数据清洗和转换
# 处理地块数据
land_df['land_area'] = land_df['land_area'].astype(float)

# 处理作物数据（价格范围转换为最低价格和最高价格）
crop_df[['yield', 'cost']] = crop_df[['yield', 'cost']].astype(float)
crop_df['price_min'] = crop_df['price_range'].apply(lambda x: float(x.split('-')[0]))
crop_df['price_max'] = crop_df['price_range'].apply(lambda x: float(x.split('-')[1]))
crop_df.drop(columns=['price_range'], inplace=True)

# 处理2023年种植数据
planting_df['crop_area'] = planting_df['crop_area'].astype(float)

# 保存数据
land_df.to_csv('data/processed_land_data.csv', index=False)
crop_df.to_csv('data/processed_crop_data.csv', index=False)
planting_df.to_csv('data/processed_planting_data.csv', index=False)

print("Data saved.")