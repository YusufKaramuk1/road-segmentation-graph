import os
import pickle
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt

SAVE_DIR = "outputs"
EXT_DIR = os.path.join(SAVE_DIR, "external")

SKELETON_PATH = os.path.join(SAVE_DIR, "skeleton.npy")
INPUT_PATH = os.path.join(EXT_DIR, "latest_input.png")


# =========================================================
# KOMŞU BULMA
# =========================================================
def get_neighbors(y, x, img):
    neighbors = []
    h, w = img.shape

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue

            ny, nx_ = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx_ < w:
                if img[ny, nx_] > 0:
                    neighbors.append((ny, nx_))

    return neighbors


# =========================================================
# NODE BULMA
# endpoint: 1 komşu
# junction: 3 ve üstü komşu
# =========================================================
def find_nodes(skeleton):
    nodes = []

    h, w = skeleton.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue

            neigh = get_neighbors(y, x, skeleton)
            if len(neigh) == 1 or len(neigh) >= 3:
                nodes.append((y, x))

    return nodes


# =========================================================
# NODE'ları yakınsa birleştir (cluster merge)
# =========================================================
def merge_close_nodes(nodes, radius=6):
    merged = []
    used = set()

    for i, (y1, x1) in enumerate(nodes):
        if i in used:
            continue

        cluster = [(y1, x1)]
        used.add(i)

        for j, (y2, x2) in enumerate(nodes):
            if j in used:
                continue

            dist = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
            if dist <= radius:
                cluster.append((y2, x2))
                used.add(j)

        cy = int(np.mean([p[0] for p in cluster]))
        cx = int(np.mean([p[1] for p in cluster]))
        merged.append((cy, cx))

    return merged


# =========================================================
# NODE HARİTASI
# =========================================================
def build_node_map(nodes):
    node_map = {}
    for idx, (y, x) in enumerate(nodes):
        node_map[(y, x)] = idx
    return node_map


# =========================================================
# EN YAKIN NODE KONTROLÜ
# =========================================================
def nearest_node(y, x, node_map, radius=2):
    for (ny, nx_), idx in node_map.items():
        if abs(ny - y) <= radius and abs(nx_ - x) <= radius:
            return idx
    return None


# =========================================================
# EDGE TAKİBİ
# =========================================================
def trace_edges(skeleton, nodes):
    G = nx.Graph()

    for i, (y, x) in enumerate(nodes):
        G.add_node(i, pos=(x, y))

    node_map = build_node_map(nodes)
    visited_paths = set()

    for start_idx, (sy, sx) in enumerate(nodes):
        neighbors = get_neighbors(sy, sx, skeleton)

        for ny, nx_ in neighbors:
            path = [(sy, sx)]
            prev = (sy, sx)
            curr = (ny, nx_)

            while True:
                path.append(curr)

                cy, cx = curr
                near_idx = nearest_node(cy, cx, node_map, radius=2)

                if near_idx is not None and near_idx != start_idx:
                    edge = tuple(sorted((start_idx, near_idx)))
                    if edge not in visited_paths:
                        visited_paths.add(edge)
                        G.add_edge(start_idx, near_idx, weight=len(path))
                    break

                neigh = get_neighbors(cy, cx, skeleton)
                neigh = [p for p in neigh if p != prev]

                if len(neigh) == 0:
                    break

                # bir sonraki piksel
                prev = curr
                curr = neigh[0]

                # güvenlik
                if len(path) > 5000:
                    break

    return G


# =========================================================
# GÖRSELLEŞTİRME
# =========================================================
def visualize_graph(img, G, save_path):
    plt.figure(figsize=(12, 10))
    plt.imshow(img)

    pos = nx.get_node_attributes(G, "pos")

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1)

    for node, (x, y) in pos.items():
        degree = G.degree[node]
        if degree >= 3:
            plt.scatter(x, y, c='red', s=18)
        else:
            plt.scatter(x, y, c='yellow', s=14)

    plt.title(f"Graph (node={G.number_of_nodes()}, edge={G.number_of_edges()})")
    plt.axis("off")
    plt.savefig(save_path, dpi=150)
    plt.show()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(SKELETON_PATH):
        raise FileNotFoundError("Önce python 2_skeleton.py çalıştır!")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("latest_input.png bulunamadı. Önce infer_external_image.py çalıştır!")

    skeleton = np.load(SKELETON_PATH)
    img = cv2.imread(INPUT_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Skeleton yüklendi: {skeleton.shape}")

    raw_nodes = find_nodes(skeleton)
    print(f"Ham node sayısı: {len(raw_nodes)}")

    nodes = merge_close_nodes(raw_nodes, radius=6)
    print(f"Birleştirilmiş node sayısı: {len(nodes)}")

    G = trace_edges(skeleton, nodes)

    print(f"Edge sayısı: {G.number_of_edges()}")

    with open(os.path.join(SAVE_DIR, "road_graph.pkl"), "wb") as f:
        pickle.dump(G, f)

    print("Graph kaydedildi -> outputs/road_graph.pkl")

    visualize_graph(img, G, os.path.join(SAVE_DIR, "graph.png"))

    print("✔ Graph tamamlandı")
    print("Sıradaki: python 4_report.py")