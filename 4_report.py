import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

SAVE_DIR = "outputs"
EXT_DIR = os.path.join(SAVE_DIR, "external")

INPUT_PATH = os.path.join(EXT_DIR, "latest_input.png")
MASK_PATH = os.path.join(EXT_DIR, "latest_mask.png")
SKELETON_PATH = os.path.join(SAVE_DIR, "skeleton.npy")
GRAPH_PATH = os.path.join(SAVE_DIR, "road_graph.pkl")


def load_required():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("latest_input.png bulunamadı. Önce infer_external_image.py çalıştır.")
    if not os.path.exists(MASK_PATH):
        raise FileNotFoundError("latest_mask.png bulunamadı. Önce infer_external_image.py çalıştır.")
    if not os.path.exists(SKELETON_PATH):
        raise FileNotFoundError("skeleton.npy bulunamadı. Önce 2_skeleton.py çalıştır.")
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError("road_graph.pkl bulunamadı. Önce 3_graph.py çalıştır.")

    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(f"Girdi okunamadı: {INPUT_PATH}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Maske okunamadı: {MASK_PATH}")

    skeleton = np.load(SKELETON_PATH)

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    return img, mask, skeleton, G


if __name__ == "__main__":
    img, mask, skeleton, G = load_required()

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.25)

    title_kw = dict(fontsize=11, color="white", pad=8, fontweight="bold")
    DARK_BG = "#16213e"

    # =====================================================
    # Panel 1 - Uydu görüntüsü
    # =====================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title("1. Uydu Görüntüsü", **title_kw)
    ax1.axis("off")
    ax1.set_facecolor(DARK_BG)

    # =====================================================
    # Panel 2 - Yol maskesi
    # =====================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap="gray")
    ax2.set_title("2. Yol Maskesi (Model Çıktısı)", **title_kw)
    ax2.axis("off")
    ax2.set_facecolor(DARK_BG)

    # =====================================================
    # Panel 3 - Skeleton
    # =====================================================
    ax3 = fig.add_subplot(gs[0, 2])
    skel_rgb = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    skel_rgb[skeleton > 0] = [255, 80, 80]
    ax3.imshow(skel_rgb)
    ax3.set_title("3. Yol İskeleti", **title_kw)
    ax3.axis("off")
    ax3.set_facecolor(DARK_BG)

    # =====================================================
    # Panel 4 - Graph overlay
    # =====================================================
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(img)
    ax4.set_facecolor(DARK_BG)

    pos = nx.get_node_attributes(G, "pos")
    degrees = dict(G.degree())

    if pos:
        for u, v in G.edges():
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                ax4.plot([x1, x2], [y1, y2], "b-", lw=1.0, alpha=0.7)

        xs = [pos[n][0] for n in G.nodes() if n in pos]
        ys = [pos[n][1] for n in G.nodes() if n in pos]
        node_colors = ["#e74c3c" if degrees.get(n, 0) >= 3 else "#f1c40f"
                       for n in G.nodes() if n in pos]
        ax4.scatter(xs, ys, c=node_colors, s=20, zorder=5)

    ax4.set_title("4. Graph Üst Katmanı", **title_kw)
    ax4.axis("off")

    # =====================================================
    # Panel 5 - Spring layout graph
    # =====================================================
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.set_facecolor(DARK_BG)

    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        largest_cc = G.subgraph(max(components, key=len)).copy()

        # Betweenness centrality
        if largest_cc.number_of_nodes() > 1:
            centrality = nx.betweenness_centrality(largest_cc, normalized=True)
            max_c = max(centrality.values()) if centrality else 1.0
        else:
            centrality = {n: 0.0 for n in largest_cc.nodes()}
            max_c = 1.0

        node_colors = []
        for n in largest_cc.nodes():
            c = centrality.get(n, 0.0) / (max_c + 1e-9)
            node_colors.append((1.0, 1 - c * 0.85, 0.1 + c * 0.1))

        layout = nx.spring_layout(largest_cc, seed=42, k=0.8)

        nx.draw_networkx(
            largest_cc,
            pos=layout,
            ax=ax5,
            node_size=70,
            node_color=node_colors,
            edge_color="#7f8c8d",
            width=1.2,
            with_labels=False
        )

    ax5.set_title("5. Yol Ağı Grafiği  (koyu=kritik kavşak  açık=uç nokta)", **title_kw)
    ax5.axis("off")

    # =====================================================
    # Panel 6 - Metrik tablosu
    # =====================================================
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.set_facecolor(DARK_BG)
    ax6.axis("off")

    components = list(nx.connected_components(G)) if G.number_of_nodes() > 0 else []
    density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
    avg_degree = sum(d for _, d in G.degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0.0

    metrics = [
        ("Node sayısı", f"{G.number_of_nodes()}"),
        ("Edge sayısı", f"{G.number_of_edges()}"),
        ("Bağlı bileşen", f"{len(components)}"),
        ("Graph yoğunluğu", f"{density:.4f}"),
        ("Ort. bağlantı derecesi", f"{avg_degree:.2f}"),
        ("Yol pikseli", f"{int(skeleton.sum()):,}"),
        ("Yol yoğunluğu", f"%{skeleton.sum()/(skeleton.shape[0]*skeleton.shape[1])*100:.2f}"),
    ]

    ax6.text(
        0.5, 0.97, "Graph Analiz Metrikleri",
        transform=ax6.transAxes,
        ha="center", va="top",
        fontsize=12, color="white", fontweight="bold"
    )

    for i, (label, value) in enumerate(metrics):
        y = 0.82 - i * 0.115

        ax6.text(
            0.15, y, label,
            transform=ax6.transAxes,
            ha="left", va="center",
            fontsize=10, color="#bdc3c7"
        )

        ax6.text(
            0.85, y, value,
            transform=ax6.transAxes,
            ha="right", va="center",
            fontsize=11, color="#2ecc71", fontweight="bold"
        )

        ax6.plot(
            [0.05, 0.95], [y - 0.04, y - 0.04],
            color="#2c3e50", linewidth=0.5,
            transform=ax6.transAxes
        )

    # =====================================================
    # Başlık
    # =====================================================
    fig.suptitle(
        "Derin Öğrenme ile Uydu Görüntülerinden Yol Segmentasyonu ve Yol Ağı Analizi",
        fontsize=15, color="white", fontweight="bold", y=0.98
    )

    out_path = os.path.join(SAVE_DIR, "final_report.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

    print(f"\nFinal rapor kaydedildi -> {out_path}")
print("\n✔ Proje tamamlandı!")
