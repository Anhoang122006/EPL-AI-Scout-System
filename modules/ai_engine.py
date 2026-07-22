import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class EPLEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self._prepare_data()

        # In danh sách các cột số liệu mà AI đang dùng để so sánh
        print("=== DANH SÁCH TÊN CÁC CHỈ SỐ ===")
        print(self.numeric_cols)
        print("==================================")

    def _prepare_data(self):
        """Quy trình ETL: Làm sạch -> Chuẩn hóa -> Phân cụm AI -> Giảm chiều PCA"""
        self.df.replace('N/a', np.nan, inplace=True)
        cols_to_exclude = ['STT', 'Tên cầu thủ', 'Nation', 'Pos', 'Squad', 'Age', 'Born']
        self.numeric_cols = [c for c in self.df.columns if c not in cols_to_exclude]

        for col in self.numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(0)

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.df[self.numeric_cols])

        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.df['Cluster'] = self.kmeans.fit_predict(self.X_scaled)

        pca2 = PCA(n_components=2)
        coords2d = pca2.fit_transform(self.X_scaled)
        self.df['PC1'] = coords2d[:, 0]
        self.df['PC2'] = coords2d[:, 1]

    def get_player_stats(self, player_name):
        res = self.df[self.df['Tên cầu thủ'].str.contains(player_name, case=False, na=False)]
        return res.iloc[0] if not res.empty else None

    def recommend_similar_players(self, player_name, top_n=5):
        idx_list = self.df.index[self.df['Tên cầu thủ'].str.contains(player_name, case=False, na=False)].tolist()
        if not idx_list:
            return None

        target_idx = idx_list[0]

        # Dùng toàn bộ dữ liệu chuẩn hóa hiện có (Số phút, bàn thắng, thẻ phạt...)
        weighted_X = self.X_scaled.copy()

        target_vector = weighted_X[target_idx].reshape(1, -1)
        sim_scores = cosine_similarity(target_vector, weighted_X)[0]

        df_sim = self.df.copy()
        df_sim['Similarity'] = sim_scores * 100

        # =================================================================
        # BƯỚC CẢI TIẾN: Lọc vị trí khắt khe
        # =================================================================
        target_pos = self.df.loc[target_idx, 'Pos']

        if target_pos == 'MF':
            # Nếu cầu thủ là 'MF' thuần túy, CHỈ lọc những người có đúng vị trí 'MF'
            # Loại bỏ ngay lập tức những người lai tiền đạo như 'MF,FW' hay 'FW,MF'
            df_sim = df_sim[df_sim['Pos'] == 'MF']
        else:
            # Nếu là vị trí khác, áp dụng cách lọc tương đối
            primary_pos = target_pos.split(',')[0] if isinstance(target_pos, str) else target_pos
            df_sim = df_sim[df_sim['Pos'].str.contains(primary_pos, na=False)]
        # =================================================================

        # Loại bỏ chính cầu thủ mục tiêu ra khỏi kết quả tìm kiếm
        df_sim = df_sim[df_sim.index != target_idx]

        return df_sim.sort_values(by='Similarity', ascending=False).head(top_n)