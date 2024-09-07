import pandas as pd
df1 = pd.read_excel("result1_1.xlsx")
print(df1["地块名"])
# 0 - 53 行
print(df1["地块名"][0:54])
# 54 - 81 行
print(df1["地块名"][54:82])


# 读取列名 去除前2列
print(df1.columns[2:])