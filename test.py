import random
import pandas as pd

s1 = pd.Series([1,1,1,1,1])
s1 = s1 * random.random()
print(s1)