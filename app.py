import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from modules.ai_engine import EPLEngine

# 1. Cấu hình trang Web Dashboard
st.set_page_config(
    page_title="EPL AI Scout Platform",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ EPL AI Scout: Hệ thống Phân tích & Tuyển trạch Cầu thủ")
st.markdown("---")


# 2. Khởi tạo AI Engine (Dùng @st.cache_resource để lưu vào RAM, không tính lại mỗi lần bấm nút)
@st.cache_resource
def load_engine():
    return EPLEngine("data/ThongKe_NgoaiHangAnh.csv")


engine = load_engine()

# 3. Tạo 3 Tab chức năng chính
tab1, tab2, tab3 = st.tabs([
    "📊 So sánh Radar (Nâng cấp Bài 3)",
    "🤖 AI Scouting - Gợi ý thay thế (Mới)",
    "🗺️ Bản đồ Chiến thuật PCA (Nâng cấp Bài 5)"
])

# =====================================================================
# TAB 1: SO SÁNH RADAR CHART (ĐÃ CHUẨN HÓA THANG ĐIỂM 100)
# =====================================================================
with tab1:
    st.subheader("🔍 So sánh trực quan phong độ giữa 2 cầu thủ")
    col1, col2 = st.columns(2)

    player_list = sorted(engine.df['Tên cầu thủ'].unique())
    with col1:
        p1_name = st.selectbox("Chọn Cầu thủ 1 (Khung xanh):", player_list,
                               index=player_list.index("Enzo Fernández") if "Enzo Fernández" in player_list else 0)
    with col2:
        p2_name = st.selectbox("Chọn Cầu thủ 2 (Khung đỏ):", player_list,
                               index=player_list.index("Mikel Merino") if "Mikel Merino" in player_list else 1)

    stats_to_plot = ['Gls', 'Ast', 'Min', 'CrdY', 'CrdR', 'MP']

    s1 = engine.get_player_stats(p1_name)
    s2 = engine.get_player_stats(p2_name)

    if s1 is not None and s2 is not None:
        # 1. TÍNH ĐIỂM CHUẨN HÓA (0 - 100) VÀ LẤY SỐ LIỆU THÔ ĐỂ HIỂN THỊ HOVER
        r1_scaled = []
        r2_scaled = []
        hover_text1 = []
        hover_text2 = []

        for k in stats_to_plot:
            max_val = engine.df[k].max()
            max_val = max_val if max_val > 0 else 1  # Tránh chia cho 0

            # Quy đổi ra thang điểm 100
            r1_scaled.append((s1[k] / max_val) * 100)
            r2_scaled.append((s2[k] / max_val) * 100)

            # Tạo chú thích số thực tế khi di chuột
            hover_text1.append(f"{k}: {s1[k]} (Đạt {r1_scaled[-1]:.1f}% đỉnh giải)")
            hover_text2.append(f"{k}: {s2[k]} (Đạt {r2_scaled[-1]:.1f}% đỉnh giải)")

        # 2. VẼ BIỂU ĐỒ RADAR
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r1_scaled,
            theta=stats_to_plot,
            fill='toself',
            name=p1_name,
            line_color='#1f77b4',
            hoverinfo='text',
            hovertext=hover_text1
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=r2_scaled,
            theta=stats_to_plot,
            fill='toself',
            name=p2_name,
            line_color='#d62728',
            hoverinfo='text',
            hovertext=hover_text2
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),  # Khóa trục từ 0 đến 100 điểm
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# =====================================================================
# TAB 2: AI SCOUTING (RECOMMENDATION SYSTEM)
# =====================================================================
with tab2:
    st.subheader("💡 AI Tìm kiếm cầu thủ có lối chơi tương đồng (Replace Finder)")
    st.markdown(
        "Thuật toán **Cosine Similarity** sẽ đo lường độ tương đồng của hơn 30 chỉ số để tìm ra những cầu thủ có phong cách thi đấu giống hệt mục tiêu của bạn.")

    target_player = st.selectbox("Chọn cầu thủ mục tiêu bạn muốn tìm người thay thế:", player_list, key="ai_target")
    top_k = st.slider("Số lượng gợi ý:", min_value=3, max_value=10, value=5)

    if st.button("🚀 Kích hoạt AI Tuyển trạch", type="primary"):
        recs = engine.recommend_similar_players(target_player, top_n=top_k)
        if recs is not None and not recs.empty:
            st.success(f"Dưới đây là Top {top_k} cầu thủ có lối chơi giống với **{target_player}** nhất:")

            # Chọn các cột quan trọng để hiển thị cho đẹp
            display_cols = ['Tên cầu thủ', 'Squad', 'Pos', 'Age', 'Cluster', 'Similarity']
            # Kiểm tra xem các cột có tồn tại trong DataFrame không
            valid_cols = [c for c in display_cols if c in recs.columns]

            st.dataframe(
                recs[valid_cols].style.format({'Similarity': '{:.1f}%'}, na_rep="N/A"),
                use_container_width=True
            )
        else:
            st.warning("Không tìm thấy dữ liệu phù hợp!")

# =====================================================================
# TAB 3: BẢN ĐỒ PHÂN CỤM PCA 2D
# =====================================================================
with tab3:
    st.subheader("🗺️ Bản đồ Phân cụm Chiến thuật Cầu thủ Ngoại hạng Anh")
    st.markdown(
        "Thuật toán **K-Means ($K=4$)** và **PCA** nén không gian dữ liệu xuống 2 chiều (PC1, PC2). Mỗi chấm tròn là 1 cầu thủ, màu sắc đại diện cho nhóm phong cách thi đấu.")

    fig_pca = px.scatter(
        engine.df,
        x='PC1',
        y='PC2',
        color=engine.df['Cluster'].astype(str),
        hover_name='Tên cầu thủ',
        hover_data=['Squad', 'Pos', 'Gls', 'Ast'],
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'color': 'Cụm AI (Cluster)'}
    )
    fig_pca.update_traces(marker=dict(size=10, opacity=0.8))
    fig_pca.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=550)

    st.plotly_chart(fig_pca, use_container_width=True)