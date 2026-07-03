import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ai_engine import EPLEngine


def main():
    print("⏳ Đang tính toán AI và gom 2 đồ thị vào 1 màn hình...")
    engine = EPLEngine("data/ThongKe_NgoaiHangAnh.csv")

    # 1. TẠO KHUNG CHIA 2 CỘT SONG SONG TRÊN CÙNG 1 TRANG
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "polar"}, {"type": "xy"}]],
        subplot_titles=("1. Radar So sánh 100 điểm", "2. Bản đồ Chiến thuật PCA")
    )

    # 2. VẼ RADAR VÀO CỘT BÊN TRÁI (col=1)
    p1, p2 = "Enzo Fernández", "Mikel Merino"
    s1, s2 = engine.get_player_stats(p1), engine.get_player_stats(p2)
    stats = ['Gls', 'Ast', 'Min', 'CrdY', 'CrdR', 'MP']
    r1 = [(s1[k] / (engine.df[k].max() or 1)) * 100 for k in stats]
    r2 = [(s2[k] / (engine.df[k].max() or 1)) * 100 for k in stats]

    fig.add_trace(go.Scatterpolar(r=r1, theta=stats, fill='toself', name=p1), row=1, col=1)
    fig.add_trace(go.Scatterpolar(r=r2, theta=stats, fill='toself', name=p2), row=1, col=1)

    # 3. VẼ BẢN ĐỒ PCA VÀO CỘT BÊN PHẢI (col=2)
    for c in sorted(engine.df['Cluster'].unique()):
        df_c = engine.df[engine.df['Cluster'] == c]
        fig.add_trace(
            go.Scatter(
                x=df_c['PC1'], y=df_c['PC2'], mode='markers',
                marker=dict(size=9, opacity=0.8), name=f"Cụm {c}",
                text=df_c['Tên cầu thủ'], hoverinfo='text+name'
            ),
            row=1, col=2
        )

    fig.update_layout(height=650, title_text="⚽ DASHBOARD PHÂN TÍCH SONG SONG",
                      polar=dict(radialaxis=dict(range=[0, 100])))
    fig.show()  # Chỉ mở đúng 1 trang web hiện cả 2 hình!


if __name__ == "__main__":
    main()