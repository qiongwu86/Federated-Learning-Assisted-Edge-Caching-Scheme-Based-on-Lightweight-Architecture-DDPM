import numpy as np
import random
from utils import count_top_items
from collections import Counter

def thompson_sampling(args, data_set, test_dataset, cachesize):
    """
    TS算法
    对于不同的缓存大小，执行TS时，得到的beta分布的参数不同，所以对于每个缓存大小都要跑一边TS
    :param args:
    :param data_set: 数据集
    :param test_dataset: 测试集，用来测试缓存命中，以更新beta参数
    :param cachesize: 缓存大小
    :return:
    """
    # 初始化beta分布系数, [1,1]*电影数目*客户数目
    movie_in_dataset = np.unique(data_set[:, 1].astype(np.uint16))
    beta_hits_losses = np.array([[1, 1]] * max(movie_in_dataset))
    # 初始化推荐电影列表
    recommend_movies = []

    for epoch in range(args.epochs):

        recommend_movies = []

        for idx in range(args.clients_num):
            prob_movies = []

            for movie_id in movie_in_dataset:
                prob_movies.append([movie_id, random.betavariate(beta_hits_losses[movie_id - 1, 0],
                                                                 beta_hits_losses[movie_id - 1, 1])])
            prob_movies.sort(key=lambda x: x[1], reverse=True)
            recommend_movie_i = [prob_movies[i][0] for i in range(cachesize)]
            recommend_movies.append(recommend_movie_i)
        recommend_movies = count_top_items(cachesize, recommend_movies)

        movies_request = test_dataset[:, 1]
        count = Counter(movies_request)
        for movie_id in recommend_movies:

            if movie_id in count.keys():
                beta_hits_losses[movie_id - 1, 0] += round(count[movie_id]/max(count.values())*5)
            else:
                beta_hits_losses[movie_id - 1, 1] += 1
    return recommend_movies