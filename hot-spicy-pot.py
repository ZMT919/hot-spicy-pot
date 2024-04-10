# 麻辣香锅菜品推荐
import pandas as pd
from numpy import *
from sklearn.feature_extraction.text import TfidfVectorizer

# 推荐函数，输出与其最相似的10个菜品
def content_based_recommendation(name):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1])
    sim_scores = sim_scores[1:11]
    food_indices = [i[0] for i in sim_scores]
    return food['name'].iloc[food_indices]
# 读取数据
print('step1:读取数据...')
food = pd.read_csv('hot-spicy pot.csv')
print(food.head(10))

# 将菜品的描述构成TF-IDF向量
print('step2:构造TF-IDF...')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(food['taste'])
print(tfidf_matrix.shape)

# 计算两个菜品的余弦相似度
print('step3:计算余弦相似度...')
from sklearn.metrics.pairwise import pairwise_distances
cosine_sim = pairwise_distances(tfidf_matrix, metric='cosine')

# 根据菜名及其特点进行推荐
print('step4:推荐菜品...')
# 建立索引方便使用菜品进行数据访问
indices = pd.Series(food.index, index=food['name']).drop_duplicates()
result = content_based_recommendation('celery')
print(result)
