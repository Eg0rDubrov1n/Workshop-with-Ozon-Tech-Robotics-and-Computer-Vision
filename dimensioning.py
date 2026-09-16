"""
Минимальный рабочий пайплайн определения габаритов объекта по облаку точек.

Реализует алгоритмы из приложенных статей и в тех же библиотеках, что
указаны в отчёте (раздел 5 "Вычислитель и алгоритм определения габаритов"):

  - Этап 1: выпуклая оболочка (Convex Hull) — scipy.spatial.ConvexHull
    (обёртка над Qhull), как в отчёте.
  - Этап 2: минимальный охватывающий прямоугольник по методу
    вращающихся калиперов (Rotating Calipers), Toussaint, 1983,
    опирается на теорему Freeman & Shapira (1975): сторона
    прямоугольника минимальной площади всегда коллинеарна одному
    из рёбер выпуклой оболочки, поэтому достаточно перебрать только
    углы рёбер оболочки, а не все углы от 0 до 90°.
  - Этап 3: высота — независимый замер по оси Z.

Геометрия подаётся в виде STL-модели. Загрузка через Open3D (как в
отчёте), либо, если Open3D недоступен, через встроенный лёгкий
бинарный/ASCII STL-парсер на NumPy.

Запуск:
    python dimensioning.py path/to/model.stl
    python dimensioning.py path/to/model.stl --max-points=50000
    python dimensioning.py path/to/model.stl --no-viz
"""

from __future__ import annotations

import os
import struct
import sys
import warnings

import numpy as np
from scipy.spatial import ConvexHull

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# --------------------------------------------------------------------------
# Загрузка геометрии
# --------------------------------------------------------------------------

def _read_stl_numpy(path: str) -> np.ndarray:
    """Читает STL (binary ИЛИ ascii) без внешних зависимостей.
    Возвращает (N,3) массив вершин треугольников (с дубликатами)."""
    with open(path, "rb") as f:
        head = f.read(84)

    # --- Признак ASCII: файл начинается с 'solid' и в первых ~1 КБ есть 'facet'
    with open(path, "rb") as f:
        probe = f.read(1024)
    is_ascii = probe[:5].lower() == b"solid" and b"facet" in probe.lower()

    # Некоторые экспортёры пишут 'solid' и в бинарный STL — тогда размер файла
    # ровно 84 + n_tri * 50, и это точно бинарный вариант.
    if is_ascii:
        file_size = os.path.getsize(path)
        n_tri_le = int.from_bytes(head[80:84], "little")
        if 84 + n_tri_le * 50 == file_size:
            is_ascii = False

    if is_ascii:
        return _read_ascii_stl_numpy(path)
    return _read_binary_stl_numpy(path)


def _read_binary_stl_numpy(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        f.read(80)
        (n_tri,) = struct.unpack("<I", f.read(4))

        expected = 84 + n_tri * 50
        actual = os.path.getsize(path)
        if expected != actual:
            raise ValueError(
                f"Файл не похож на бинарный STL: ожидалось {expected} байт "
                f"по заголовку ({n_tri} треугольников), а на диске {actual}. "
                f"Похоже, это ASCII STL — пересохраните модель или проверьте формат."
            )

        # Читаем весь блок треугольников разом.
        raw = f.read(n_tri * 50)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(n_tri, 50)
        # Смещение вершин внутри записи: 12 (нормаль) .. 48 (3*3 float32)
        verts_bytes = arr[:, 12:48].copy()
        verts = verts_bytes.view(np.float32).reshape(n_tri * 3, 3).astype(np.float64)
    return verts


def _read_ascii_stl_numpy(path: str) -> np.ndarray:
    """Быстрый парсер ASCII STL.

    Читает файл целиком, оставляет только строки 'vertex x y z',
    дальше разбирает все числа одним вызовом C-парсера np.fromstring.
    """
    with open(path, "rb") as f:
        raw = f.read()

    chunks = []
    for ln in raw.split(b"\n"):
        s = ln.lstrip()
        if s.startswith(b"vertex"):
            # 'vertex 1.2 3.4 5.6' -> '1.2 3.4 5.6'
            chunks.append(s.split(None, 1)[1])
    if not chunks:
        raise ValueError(f"ASCII STL не содержит вершин: {path}")

    joined = b"\n".join(chunks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = np.fromstring(joined, sep=" ")
    return arr.reshape(-1, 3).astype(np.float64)


def decimate(points: np.ndarray, max_points: int = 100_000,
             seed: int = 0) -> np.ndarray:
    """Случайное прореживание облака.

    Для габаритов крайние точки сохраняются практически наверняка, а
    ConvexHull и kNN ускоряются на порядки.
    """
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), max_points, replace=False)
    return points[idx]


def load_mesh_and_points(path: str, sample_points: int = 100_000,
                          max_points: int = 100_000):
    """Возвращает (mesh, points, triangles).

    mesh      — Open3D TriangleMesh или None (если Open3D нет);
    points    — (N,3) облако точек для измерения (уже прореженное);
    triangles — (T,3,3) треугольники STL (только для numpy-фолбэка и
                matplotlib-визуализации), иначе None.
    """
    if HAS_OPEN3D:
        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.has_vertices():
            raise ValueError(f"Не удалось загрузить меш из {path}")
        mesh.compute_vertex_normals()
        n_samples = min(sample_points, max_points)
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_samples)
        return mesh, np.asarray(pcd.points), None
    else:
        verts = _read_stl_numpy(path)
        triangles = verts.reshape(-1, 3, 3)
        pts = decimate(verts, max_points=max_points)
        return None, pts, triangles


def load_points_from_stl(path: str, sample_points: int = 100_000) -> np.ndarray:
    """Совместимость со старым API: только точки."""
    _, pts, _ = load_mesh_and_points(path, sample_points)
    return pts


def remove_outliers(points: np.ndarray, nb_neighbors: int = 20,
                     std_ratio: float = 2.0) -> np.ndarray:
    """Этап 0 (частично): фильтрация выбросов.

    Для STL-геометрии используем быстрый O(N) отсев по перцентилям bbox:
    все вершины настоящие, выбросов почти нет, а попарные kNN на миллионах
    точек — избыточны. Open3D-вариант оставлен для шумных сканов.
    """
    if HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return np.asarray(pcd_clean.points)

    if len(points) < 10:
        return points

    lo = np.percentile(points, 0.05, axis=0)
    hi = np.percentile(points, 99.95, axis=0)
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    return points[mask]


# --------------------------------------------------------------------------
# Этап 2: Rotating Calipers — минимальный охватывающий прямоугольник
# --------------------------------------------------------------------------

def min_area_rectangle(hull_points_2d: np.ndarray) -> dict:
    """
    Rotating Calipers (Toussaint, 1983) для минимального по площади
    охватывающего прямоугольника выпуклого многоугольника.

    hull_points_2d: (H,2) вершины выпуклой оболочки, упорядоченные
                     (как их возвращает scipy ConvexHull).

    Возвращает словарь: width, height, area, angle (рад), center, corners.
    """
    n = len(hull_points_2d)
    if n < 3:
        mins = hull_points_2d.min(axis=0)
        maxs = hull_points_2d.max(axis=0)
        w, h = maxs - mins
        return {"width": float(w), "height": float(h), "area": float(w * h),
                "angle": 0.0, "center": (mins + maxs) / 2, "corners": None}

    edges = np.roll(hull_points_2d, -1, axis=0) - hull_points_2d
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
    candidate_angles = np.unique(np.mod(edge_angles, np.pi / 2))

    best = None
    for theta in candidate_angles:
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])
        rotated = hull_points_2d @ R.T
        mins = rotated.min(axis=0)
        maxs = rotated.max(axis=0)
        w, h = maxs - mins
        area = w * h
        if best is None or area < best["area"]:
            rect_rot = np.array([
                [mins[0], mins[1]],
                [maxs[0], mins[1]],
                [maxs[0], maxs[1]],
                [mins[0], maxs[1]],
            ])
            corners = rect_rot @ R
            best = {"width": float(w), "height": float(h), "area": float(area),
                    "angle": float(theta), "center": corners.mean(axis=0),
                    "corners": corners}
    return best


# --------------------------------------------------------------------------
# Основной пайплайн: Этапы 1-3 из отчёта
# --------------------------------------------------------------------------

def measure_dimensions(points: np.ndarray) -> dict:
    """
    points: (N,3) облако точек объекта (Z — вертикальная ось / высота).
    Возвращает габариты (длина, ширина, высота) и вспомогательные данные.
    """
    if len(points) < 3:
        raise ValueError("Недостаточно точек для измерения (нужно >= 3)")

    z = points[:, 2]
    height = float(z.max() - z.min())

    xy = points[:, :2]

    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]

    rect = min_area_rectangle(hull_pts)

    length = max(rect["width"], rect["height"])
    width = min(rect["width"], rect["height"])

    return {
        "length": length,
        "width": width,
        "height": height,
        "rect_angle_deg": np.degrees(rect["angle"]),
        "rect_area": rect["area"],
        "rect_corners_xy": rect["corners"],
        "hull_vertices_xy": hull_pts,
        "n_points": len(points),
    }


def check_tolerance(measured: float, nominal: float,
                     tol_percent: float = 0.05, tol_min_mm: float = 5.0) -> bool:
    """Допуск по ТЗ: ±max(5% от размера, 5 мм) (Этап 4 отчёта)."""
    tol = max(tol_percent * nominal, tol_min_mm)
    return abs(measured - nominal) <= tol


# --------------------------------------------------------------------------
# 3D-визуализация: объект + минимальный параллелепипед + подписи размеров
# --------------------------------------------------------------------------

def _box_geometry(corners_xy: np.ndarray, z_min: float, z_max: float):
    """Строит 8 углов и 12 рёбер параллелепипеда по контуру XY и диапазону Z."""
    bottom = np.hstack([corners_xy, np.full((4, 1), z_min)])
    top    = np.hstack([corners_xy, np.full((4, 1), z_max)])
    corners = np.vstack([bottom, top])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return corners, edges


def _dimension_labels(corners: np.ndarray, result: dict):
    """Подписи трёх габаритов по серединам рёбер нижнего основания и вертикали."""
    L, W, H = result["length"], result["width"], result["height"]

    d01 = float(np.linalg.norm(corners[1] - corners[0]))
    d12 = float(np.linalg.norm(corners[2] - corners[1]))
    long_is_01 = d01 >= d12

    label_01 = f"L = {L:.1f} мм" if long_is_01 else f"W = {W:.1f} мм"
    label_12 = f"W = {W:.1f} мм" if long_is_01 else f"L = {L:.1f} мм"

    return [
        ((corners[0] + corners[1]) * 0.5, label_01),
        ((corners[1] + corners[2]) * 0.5, label_12),
        ((corners[0] + corners[4]) * 0.5, f"H = {H:.1f} мм"),
    ]


def _visualize_open3d(points, mesh, corners, edges, labels) -> None:
    """Объект (меш или облако) + параллелепипед поверх + 3D-подписи размеров."""
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(
        "Габариты: объект и охватывающий параллелепипед", 1280, 720
    )

    if mesh is not None:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLitTransparency"
        mat.base_color = [0.70, 0.72, 0.78, 0.45]
        vis.add_geometry("object", mesh, mat)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.65, 0.65, 0.68])
        vis.add_geometry("object", pcd)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([[1.0, 0.1, 0.1]] * len(edges))

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2.5
    vis.add_geometry("box", ls, line_mat)

    for pos, text in labels:
        vis.add_3d_label(pos, text)

    vis.reset_camera_to_default()
    app.add_window(vis)
    app.run()


def _visualize_matplotlib(points, triangles, corners, edges, labels) -> None:
    """Фолбэк: меш + параллелепипед + подписи на matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    except ImportError:
        print("matplotlib не установлен — 3D-вывод пропущен.")
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if triangles is not None and len(triangles) > 0:
        tris = triangles
        if len(tris) > 5000:
            idx = np.random.default_rng(0).choice(len(tris), 5000, replace=False)
            tris = tris[idx]
        poly = Poly3DCollection(tris, facecolor="lightsteelblue",
                                edgecolor="none", alpha=0.35)
        ax.add_collection3d(poly)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=1, c="gray", alpha=0.35)

    segs = [[corners[a], corners[b]] for a, b in edges]
    ax.add_collection3d(Line3DCollection(segs, colors="red", linewidths=1.8))

    for pos, text in labels:
        ax.text(pos[0], pos[1], pos[2], text, color="darkred", fontsize=10)

    all_pts = np.vstack([points, corners])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    r = float((maxs - mins).max()) / 2.0 * 1.1
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel("X, мм"); ax.set_ylabel("Y, мм"); ax.set_zlabel("Z, мм")
    ax.set_title("Объект и минимальный охватывающий параллелепипед")
    plt.tight_layout()
    plt.show()


def visualize_dimensions(points: np.ndarray, result: dict,
                         mesh=None, triangles=None) -> None:
    """Объект + минимальный параллелепипед поверх + подписи размеров."""
    corners_xy = result.get("rect_corners_xy")
    if corners_xy is None:
        print("Параллелепипед не построен: вырожденный случай.")
        return

    z = points[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    corners, edges = _box_geometry(corners_xy, z_min, z_max)
    labels = _dimension_labels(corners, result)

    if HAS_OPEN3D:
        _visualize_open3d(points, mesh, corners, edges, labels)
    else:
        _visualize_matplotlib(points, triangles, corners, edges, labels)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_viz = "--no-viz" not in sys.argv

    max_pts = 100_000
    for a in sys.argv[1:]:
        if a.startswith("--max-points="):
            max_pts = int(a.split("=", 1)[1])

    stl_path = positional[0] if positional else "test_box.stl"

    print(f"Загрузка модели: {stl_path} "
          f"(Open3D {'найден' if HAS_OPEN3D else 'не найден, используется numpy-фолбэк'})")
    print(f"Лимит точек после прореживания: {max_pts}")

    mesh, pts, triangles = load_mesh_and_points(
        stl_path, sample_points=max_pts, max_points=max_pts
    )
    pts = remove_outliers(pts)

    result = measure_dimensions(pts)

    print(f"\nТочек в облаке после фильтрации: {result['n_points']}")
    print(f"Длина  (L): {result['length']:.2f} мм")
    print(f"Ширина (W): {result['width']:.2f} мм")
    print(f"Высота (H): {result['height']:.2f} мм")
    print(f"Угол разворота охватывающего прямоугольника: "
          f"{result['rect_angle_deg']:.2f}°")

    if show_viz:
        visualize_dimensions(pts, result, mesh=mesh, triangles=triangles)