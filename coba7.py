import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import json

filename = 'dataset.csv'
column_names = ['userID', 'productID', 'rating', 'timestamp']
df = pd.read_csv(filename, names=column_names, header=None)

df = df.dropna()
df['date'] = pd.to_datetime(df['timestamp'], unit='s')

user_mapper = {user: idx for idx, user in enumerate(df['userID'].unique())}
product_mapper = {product: idx for idx, product in enumerate(df['productID'].unique())}
user_inverse_mapper = {idx: user for user, idx in user_mapper.items()}
product_inverse_mapper = {idx: product for product, idx in product_mapper.items()}

user_index = df['userID'].map(user_mapper)
product_index = df['productID'].map(product_mapper)

sparse_matrix = csr_matrix((df['rating'], (user_index, product_index)), shape=(len(user_mapper), len(product_mapper)))

train_data = df
train_user_index = train_data['userID'].map(user_mapper)
train_product_index = train_data['productID'].map(product_mapper)
train_sparse_matrix = csr_matrix((train_data['rating'], (train_user_index, train_product_index)), shape=(len(user_mapper), len(product_mapper)))

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(train_sparse_matrix.T)

def recommend_products(user_id, sparse_matrix, user_mapper, product_mapper, product_inverse_mapper, knn, df, n_recommendations=8, time_threshold=pd.Timestamp.now() - pd.Timedelta(days=365)):
    user_idx = user_mapper[user_id]
    user_ratings = sparse_matrix[user_idx, :].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]
    
    recommendations = {}
    
    for item_idx in rated_items:
        distances, indices = knn.kneighbors(sparse_matrix.T[item_idx], n_neighbors=n_recommendations+1)
        similar_items = indices.flatten()[1:]  # Menghilangkan produk itu sendiri
        similar_items_scores = distances.flatten()[1:]
        
        user_item_timestamp = df[(df['userID'] == user_id) & (df['productID'] == product_inverse_mapper[item_idx])]['date'].values[0]
        
        for similar_item_idx, score in zip(similar_items, similar_items_scores):
            similar_item = product_inverse_mapper[similar_item_idx]
            if similar_item not in recommendations:
                recommendations[similar_item] = [score, user_item_timestamp]
            else:
                recommendations[similar_item][0] += score
                recommendations[similar_item][1] = min(recommendations[similar_item][1], user_item_timestamp)
    
    recommendations = sorted(recommendations.items(), key=lambda x: (x[1][0], -pd.Timestamp(x[1][1]).timestamp()), reverse=True)
    
    recommendations = recommendations[:n_recommendations]
    
    filtered_recommendations = [item for item in recommendations if item[1][1] > time_threshold]
    
    if not filtered_recommendations:
        filtered_recommendations = recommendations
    
    avg_ratings = {item[0]: df[df['productID'] == item[0]]['rating'].mean() for item in filtered_recommendations}
    
    return [(item[0], avg_ratings[item[0]]) for item in filtered_recommendations]

if __name__ == '__main__':
    import sys
    user_id = sys.argv[1]
    recommendations = recommend_products(user_id, train_sparse_matrix, user_mapper, product_mapper, product_inverse_mapper, knn, train_data)
    print(json.dumps(recommendations))
