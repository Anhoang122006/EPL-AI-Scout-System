import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class EPLEngine:
    def __init__(self, csv_path):
        """Khởi tạo Bếp trưởng AI và tự động tải dữ liệu"""
        self.df = pd.read_csv(csv_path)
        self._prepare_data()

    def _prepare_data(self):
        """Quy trình ETL: Làm sạch -> Chuẩn hóa -> Phân cụm AI -> Giảm chiều PCA"""
        # 1. Làm sạch dữ liệu rỗng và ép kiểu về số
        self.df.replace('N/a', np.nan, inplace=True)
        cols_to_exclude = ['STT', 'Tên cầu thủ', 'Nation', 'Pos', 'Squad', 'Age', 'Born']
        self.numeric_cols = [c for c in self.df.columns if c not in cols_to_exclude]

        for col in self.numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(0)

        # 2. Chuẩn hóa Z-Score (StandardScaler)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.df[self.numeric_cols])

        # 3. Phân cụm K-Means (Chốt K=4 từ biểu đồ Elbow)
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df['Cluster'] = self.kmeans.fit_predict(self.X_scaled)

        # 4. Giảm chiều PCA xuống 2D để vẽ bản đồ chiến thuật
        pca2 = PCA(n_components=2)
        coords2d = pca2.fit_transform(self.X_scaled)
        self.df['PC1'] = coords2d[:, 0]
        self.df['PC2'] = coords2d[:, 1]

    def get_player_stats(self, player_name):
        """Tìm kiếm chỉ số chi tiết của 1 cầu thủ theo tên gần đúng"""
        res = self.df[self.df['Tên cầu thủ'].str.contains(player_name, case=False, na=False)]
        return res.iloc[0] if not res.empty else None

    def recommend_similar_players(self, player_name, top_n=5):
        """Thuật toán AI: Tìm Top 5 cầu thủ có lối chơi giống nhất (Cosine Similarity)"""
        idx_list = self.df.index[self.df['Tên cầu thủ'].str.contains(player_name, case=False, na=False)].tolist()
        if not idx_list:
            return None

        target_idx = idx_list[0]
        target_vector = self.X_scaled[target_idx].reshape(1, -1)

        # Đo góc tương đồng giữa cầu thủ mục tiêu với toàn bộ giải đấu
        sim_scores = cosine_similarity(target_vector, self.X_scaled)[0]

        df_sim = self.df.copy()
        df_sim['Similarity'] = sim_scores * 100  # Quy đổi ra thang điểm 100%

        # Loại bỏ chính cầu thủ đang xét ra khỏi danh sách gợi ý
        df_sim = df_sim[df_sim.index != target_idx]

        # Sắp xếp từ cao xuống thấp và lấy Top N
        return df_sim.sort_values(by='Similarity', ascending=False).head(top_n)