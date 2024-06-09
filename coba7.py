import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca file CSV tanpa header
filename = 'dataset.csv'
column_names = ['userID', 'productID', 'rating', 'timestamp']
df = pd.read_csv(filename, names=column_names, header=None)

# Memeriksa kebersihan data
print("Jumlah nilai kosong per kolom:\n", df.isnull().sum())

# Menghapus baris dengan nilai kosong
df = df.dropna()

# Ubah kolom 'timestamp' ke format datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Membuat user dan produk mapping
user_mapper = {user: idx for idx, user in enumerate(df['userID'].unique())}
product_mapper = {product: idx for idx, product in enumerate(df['productID'].unique())}

user_inverse_mapper = {idx: user for user, idx in user_mapper.items()}
product_inverse_mapper = {idx: product for product, idx in product_mapper.items()}

# Mapping IDs ke index
user_index = df['userID'].map(user_mapper)
product_index = df['productID'].map(product_mapper)

# Membuat matriks sparse
sparse_matrix = csr_matrix((df['rating'], (user_index, product_index)), shape=(len(user_mapper), len(product_mapper)))

# Membagi data menjadi set pelatihan dan set pengujian
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Membuat matriks sparse untuk data pelatihan
train_user_index = train_data['userID'].map(user_mapper)
train_product_index = train_data['productID'].map(product_mapper)
train_sparse_matrix = csr_matrix((train_data['rating'], (train_user_index, train_product_index)), shape=(len(user_mapper), len(product_mapper)))

# Menggunakan Nearest Neighbors untuk mencari tetangga terdekat
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(train_sparse_matrix.T)

# Mendefinisikan fungsi rekomendasi
def recommend_products(user_id, sparse_matrix, user_mapper, product_mapper, product_inverse_mapper, model_knn, df, n_recommendations=5, time_threshold=pd.Timestamp.now() - pd.Timedelta(days=365)):
    user_idx = user_mapper[user_id]
    user_ratings = sparse_matrix[user_idx, :].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]
    
    recommendations = {}
    
    for item_idx in rated_items:
        distances, indices = model_knn.kneighbors(sparse_matrix.T[item_idx], n_neighbors=n_recommendations+1)
        similar_items = indices.flatten()[1:]  # Menghilangkan produk itu sendiri
        similar_items_scores = distances.flatten()[1:]
        
        user_item_timestamp = df[(df['userID'] == user_id) & (df['productID'] == product_inverse_mapper[item_idx])]['datetime'].values[0]
        
        for similar_item_idx, score in zip(similar_items, similar_items_scores):
            similar_item = product_inverse_mapper[similar_item_idx]
            if similar_item not in recommendations:
                recommendations[similar_item] = [score, user_item_timestamp]
            else:
                recommendations[similar_item][0] += score
                recommendations[similar_item][1] = min(recommendations[similar_item][1], user_item_timestamp)
    
    recommendations = sorted(recommendations.items(), key=lambda x: (x[1][0], -pd.Timestamp(x[1][1]).timestamp()), reverse=True)
    
    recommendations = recommendations[:n_recommendations]
    
    # Filter recommendations by time threshold
    filtered_recommendations = [item for item in recommendations if item[1][1] > time_threshold]
    
    # If no recommendations meet the time threshold, use all recommendations
    if not filtered_recommendations:
        filtered_recommendations = recommendations
    
    # Calculate average ratings
    avg_ratings = {item[0]: df[df['productID'] == item[0]]['rating'].mean() for item in filtered_recommendations}
    
    return [(item[0], avg_ratings[item[0]]) for item in filtered_recommendations]

# Contoh penggunaan fungsi rekomendasi
user_id = 'A2BSSBK0B4FL26'
recommendations = recommend_products(user_id, train_sparse_matrix, user_mapper, product_mapper, product_inverse_mapper, model_knn, train_data)
print(recommendations)

# Visualisasi data
def visualize_recommendations(recommendations):
    if not recommendations:
        print("No recommendations to display.")
        return
    
    product_ids, avg_ratings = zip(*recommendations)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(product_ids), y=list(avg_ratings))
    plt.xlabel('Product ID')
    plt.ylabel('Average Rating')
    plt.title('Recommended Products and Their Average Ratings')
    plt.show()

# Visualisasi rekomendasi
visualize_recommendations(recommendations)

# Evaluasi kinerja model
def evaluate_model(test_data, train_sparse_matrix, user_mapper, product_mapper, product_inverse_mapper, model_knn, df, k=5):
    y_true = []
    y_pred = []
    
    for _, row in test_data.iterrows():
        user_id = row['userID']
        actual_product_id = row['productID']
        
        if user_id in user_mapper and actual_product_id in product_mapper:
            recommendations = recommend_products(user_id, train_sparse_matrix, user_mapper, product_mapper, product_inverse_mapper, model_knn, df, n_recommendations=k)
            recommended_product_ids = [rec[0] for rec in recommendations]
            
            y_true.append(1)
            y_pred.append(1 if actual_product_id in recommended_product_ids else 0)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Evaluasi model
evaluate_model(test_data, train_sparse_matrix, user_mapper, product_mapper, product_inverse_mapper, model_knn, train_data)
