import numpy as np
import trimesh
import os
import sys
import io
from scipy.spatial import cKDTree
from collections import defaultdict
from trimesh.ray import ray_pyembree
from typing import List, Sequence
from trimesh.ray.ray_pyembree import RayMeshIntersector
from zipfile import ZipFile
from sorting_points_spirally import sort_points_spirally


PLATFORM_OFFSET  = 2.5


def load_stl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден!")
    return trimesh.load(file_path)

def find_overhangs(mesh, angle_threshold=110, vertical_threshold=0.1):
    normals = mesh.face_normals
    z_axis = np.array([0, 0, 1])
    norms = np.linalg.norm(normals, axis=1)
    norms[norms < 1e-6] = 1e-6  # минимальный предел для деления
    cos_angles = np.dot(normals, z_axis) / norms
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angles))
    
    return np.where((angles > angle_threshold)  & (normals[:, 2] < 0) & 
                   (np.abs(normals[:, 2]) >= vertical_threshold))[0], angles
    

def calculate_support_params(angle_deg, mesh):
    """Вычисляет параметры опоры (spacing и thickness) в зависимости от угла и размеров модели.
    Масштабирует spacing под площадь XY, ограничивая его в разумных пределах.
    
    new_points — желаемое количество точек для новой модели.
    """
    # 1. Нормализуем угол в диапазоне [0, 180]
    angle_deg = np.clip(angle_deg, 0, 180)
    t = angle_deg / 180.0

    # 2. Параметры текущей модели
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    bbox_size = bounds_max - bounds_min
    bbox_area_xy = bbox_size[0] * bbox_size[1]

    # 3. Эталонные параметры
    ref_area_xy = 326.4
    ref_points = 154
    target_density = ref_points / ref_area_xy  # ≈0.47 pts/unit²
    print(target_density)

    # 4. Базовый spacing эталона
    spacing_ref = 0.8  # ≈1.46
    print(spacing_ref)
    # Рассчитываем новое количество точек с сохранением плотности
    new_points = int(target_density * bbox_area_xy)
    print(new_points)

    # 5. Масштабируем spacing с учётом площади и количества точек (гибридная формула)
    spacing_scaled = spacing_ref * np.sqrt( (ref_area_xy / bbox_area_xy) * (new_points / ref_points) )
    print(spacing_scaled)
    # 6. Ограничения на spacing
    spacing_min = 0.4
    spacing_max = 3.0
    spacing_scaled = np.clip(spacing_scaled, spacing_min, spacing_max)
    print(spacing_scaled)

    # 7. Интерполяция spacing по углу в пределах ±20% от масштабированного spacing
    spacing_low = 0.8 * spacing_scaled
    spacing_high = 1.2 * spacing_scaled
    spacing = spacing_low + t * (spacing_high - spacing_low)
    print(spacing)

    # 8. Толщина (fixed range из твоего примера)
    thickness_min = 0.23
    thickness_max = 0.28
    thickness = thickness_min + t * (thickness_max - thickness_min)

    return spacing, thickness





def generate_support_points(mesh, overhang_faces, angles, min_distance=0.7):
    """
    Генерирует опорные точки на нависающих поверхностях модели mesh,
    используя сетки XY с разным шагом (spacing), соответствующим углу нависающих граней.
    """
    support_points = []
    point_params = []
    point_spacings = []

    # Получаем параметры для каждой нависающей грани
    spacings, thicknesses = calculate_support_params(angles[overhang_faces], mesh)

    # --- Группировка граней по spacing (округление уменьшает число групп)
    face_groups = defaultdict(list)
    for i, face_idx in enumerate(overhang_faces):
        spacing_rounded = round(spacings[i], 2)  # группировка с точностью 0.01
        face_groups[spacing_rounded].append((face_idx, spacings[i], thicknesses[i]))

    z_min = mesh.bounds[0][2] - 10.0  # стартовая высота для лучей

    for spacing_key, grouped in face_groups.items():
        faces_in_group = [x[0] for x in grouped]
        spacings_group = [x[1] for x in grouped]
        thicknesses_group = [x[2] for x in grouped]

        # Создание сабмеша только из этих граней
        submesh = mesh.submesh([faces_in_group], only_watertight=False, append=True)

        # Границы XY этой группы
        overhang_vertices = mesh.vertices[mesh.faces[faces_in_group].flatten()]
        x_min, y_min = np.min(overhang_vertices[:, :2], axis=0)
        x_max, y_max = np.max(overhang_vertices[:, :2], axis=0)

        # Сетка XY с spacing этой группы
        print(spacing_key)
        x_grid = np.arange(x_min, x_max + spacing_key, spacing_key)
        y_grid = np.arange(y_min, y_max + spacing_key, spacing_key)
        grid_points = np.array([[x, y] for x in x_grid for y in y_grid])

        # Трассировка лучей вверх (+Z)
        for xy in grid_points:
            ray_origin = np.array([xy[0], xy[1], z_min])
            ray_direction = np.array([0, 0, 1])

            locations, _, index_tri = submesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_direction]
            )

            if len(locations) > 0:
                # Сортировка пересечений по Z
                sorted_hits = sorted(zip(locations, index_tri), key=lambda hit: hit[0][2])

                for location, local_face_idx in sorted_hits:
                    global_face_idx = faces_in_group[local_face_idx]
                    angle = angles[global_face_idx]
                    thickness = thicknesses_group[local_face_idx]
                    spacing = spacings_group[local_face_idx]

                    support_points.append(location)
                    point_params.append((angle, thickness))
                    point_spacings.append(spacing)
                    break  # берём только первое пересечение

    if support_points:
        support_points = np.vstack(support_points)
        point_spacings = np.array(point_spacings)
    
    if support_points is not None:
        total_points = len(support_points)
        
        # Bounding box всей модели
        bounds_min = mesh.bounds[0]
        bounds_max = mesh.bounds[1]
        bbox_size = bounds_max - bounds_min
        bbox_volume = np.prod(bbox_size)
        bbox_area_xy = bbox_size[0] * bbox_size[1]

        print("\n=== Модель: bounding box ===")
        print(f"X: {bounds_min[0]:.2f} → {bounds_max[0]:.2f} ({bbox_size[0]:.2f})")
        print(f"Y: {bounds_min[1]:.2f} → {bounds_max[1]:.2f} ({bbox_size[1]:.2f})")
        print(f"Z: {bounds_min[2]:.2f} → {bounds_max[2]:.2f} ({bbox_size[2]:.2f})")
        print(f"Объём: {bbox_volume:.2f} units³")
        print(f"Площадь XY: {bbox_area_xy:.2f} units²")

        print(f"\nВсего опорных точек: {total_points}")
        print(f"Плотность точек по XY: {total_points / bbox_area_xy:.2f} pts/unit²")
        print(f"Плотность точек по объёму: {total_points / bbox_volume:.4f} pts/unit³")

        # По группам spacing
        unique_spacings, counts = np.unique(point_spacings, return_counts=True)
        print("\nРазбивка по spacing:")
        for spacing, count in zip(unique_spacings, counts):
            print(f"  Spacing {spacing:.2f} → {count} точек")


    return support_points, point_params, point_spacings, mesh.submesh([overhang_faces], only_watertight=False, append=True)



def group_support_points(base_points, supports_info, point_spacings, model_mesh,
                         rect_size=(3, 2), min_height=1.0, min_support_length=2.0):
    """
    Группирует опорные точки по прямоугольным ячейкам, используя индивидуальный grid_spacing.
    """
    print("group_support_points")

    if (
        not isinstance(base_points, (list, np.ndarray)) 
        or len(base_points) == 0 
        or not isinstance(base_points[0], (list, np.ndarray))
    ):
        raise ValueError("base_points должны быть непустым списком или массивом точек с координатами (X, Y).")

    valid_indices = []
    valid_points = []
    valid_spacings = []

    for i, base_point in enumerate(base_points):
        height = supports_info[i]["height"]
        if height >= min_height and base_point is not None:
            valid_indices.append(i)
            valid_points.append(base_point)
            valid_spacings.append(point_spacings[i])

    if len(valid_points) == 0:
        return []

    valid_points = np.array(valid_points)
    valid_spacings = np.array(valid_spacings)

    if valid_points.ndim != 2 or valid_points.shape[1] < 2:
        raise ValueError("Каждая точка в base_points должна иметь хотя бы два элемента (X, Y).")

    xy = valid_points[:, :2]

    # Индивидуальные координаты ячеек по X и Y, с учётом локального spacing
    grid_indices_x = np.round(xy[:, 0] / valid_spacings).astype(int)
    grid_indices_y = np.round(xy[:, 1] / valid_spacings).astype(int)

    # Делим на размер прямоугольной группы (в ячейках)
    cell_indices = np.stack([
        grid_indices_x // rect_size[0],
        grid_indices_y // rect_size[1],
    ], axis=1)

    groups_dict = defaultdict(list)
    for idx, cell in enumerate(cell_indices):
        groups_dict[tuple(cell)].append(idx)

    # Трассировщик
    try:
        intersector = RayMeshIntersector(model_mesh)
    except:
        intersector = None

    def line_intersects_mesh(p1, p2, steps=100):
        """Сэмплируем отрезок от p1 до p2 и проверяем попадание внутрь меша."""
        points = np.linspace(p1, p2, steps)
        return model_mesh.contains(points).any()

    filtered_groups = []
    for group in groups_dict.values():
        if len(group) < min_support_length:
            continue

        # Проверка пересечений
        passes_check = True
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = supports_info[valid_indices[group[i]]]["center"]
                b = supports_info[valid_indices[group[j]]]["center"]
                if line_intersects_mesh(a, b):
                    passes_check = False
                    break
            if not passes_check:
                break

        if passes_check:
            final_group = [valid_indices[i] for i in group]
            final_group.sort()
            filtered_groups.append(final_group)

    return filtered_groups


def regroup_unassigned_points(base_points, groups, supports_info, model_mesh,
                              min_height=2.0, max_distance=2.5, max_group_size=6):
    """
    Второй этап группировки: оставшиеся точки либо присоединяются к ближайшим группам,
    либо образуют свои собственные малые группы.
    Перед добавлением в группу проверяется, не проходит ли прямая между точками через модель.
    """
    from trimesh.ray.ray_pyembree import RayMeshIntersector  # для fallback при необходимости

    print("regroup_unassigned_points")

    assigned = set(i for group in groups for i in group)
    unassigned = [i for i in range(len(base_points)) if i not in assigned and base_points[i] is not None]

    if len(unassigned) == 0:
        return groups

    points = np.array(base_points)
    new_groups = [set(group) for group in groups]

    def line_intersects_mesh(p1, p2, steps=10):
        """Сэмплируем отрезок от p1 до p2 и проверяем попадание внутрь меша."""
        points_line = np.linspace(p1, p2, steps)
        return model_mesh.contains(points_line).any()

    for idx in unassigned:
        if supports_info[idx]["height"] < min_height:
            continue

        p = points[idx][:2]
        z = points[idx][2]
        best_candidate = None
        min_dist = float('inf')

        for group in new_groups:
            if len(group) >= max_group_size:
                continue
            group_points = [i for i in group if base_points[i] is not None]
            if not group_points:
                continue

            for i in group_points:
                pi = points[i][:2]
                zi = points[i][2]
                dist = np.linalg.norm(p - pi)
                if dist < min_dist and dist <= max_distance:
                    # Проверим, не пересекает ли линия меш
                    a = np.array([*p, z])
                    b = np.array([*pi, zi])
                    if not line_intersects_mesh(a, b):
                        min_dist = dist
                        best_candidate = group

        if best_candidate:
            best_candidate.add(idx)
        else:
            new_groups.append(set([idx]))

    final_groups = [sorted(list(g)) for g in new_groups]
    return final_groups

def prepare_supports(mesh, support_points, point_params):
    """
    Подготавливает параметры для построения опор:
    - Вычисляет base_points
    - Определяет высоту, центр, радиус и цвет каждой опоры
    Возвращает список с параметрами опор и словарь base_points.
    """
    if len(support_points) == 0:
        return None, {}

    platform_z = mesh.bounds[0][2] - PLATFORM_OFFSET
    min_height = 0.01  # минимальная высота поддержки
    base_points = {}

    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        intersector = RayMeshIntersector(mesh)
    except ImportError:
        intersector = mesh.ray

    supports_info = []

    for point_index, point in enumerate(support_points):
        _, thickness = point_params[point_index]

        if point[2] <= platform_z + min_height:
            print(f"Точка {point_index} пропущена: слишком низко ({point[2]} <= {platform_z + min_height})")
            continue

        ray_origin = point + np.array([0, 0, 0.01])  # Чуть выше точки
        ray_direction = np.array([0, 0, -1])

        locations, index_ray, index_tri = intersector.intersects_location(
            [ray_origin], [ray_direction], multiple_hits=True
        )

        valid_locs = [
            loc for loc in locations
            #if platform_z <= loc[2] < point[2] and abs(loc[0] - point[0]) <= 1 and abs(loc[1] - point[1]) <= 1
            if platform_z <= loc[2] < point[2]
        ]

        if valid_locs:
            sorted_locs = sorted(valid_locs, key=lambda loc: point[2] - loc[2])
            
                   
            nearest = None
            for loc in sorted_locs:
                if (point[2] - loc[2] >= thickness / 2 and
                    abs(loc[0] - point[0]) <= 1 and
                    abs(loc[1] - point[1]) <= 1):
                    nearest = loc
                    break

            if nearest is not None:
                z_bottom = nearest[2]
                color = [0, 0, 255, 255]  # Синий — на модель
            else:
                z_bottom = platform_z
                color = [255, 0, 0, 255]  # Красный — на платформу
        else:
            z_bottom = platform_z
            color = [255, 0, 0, 255]

        z_top = point[2]
        height = z_top - z_bottom
        if height < min_height:
            print(f"Точка {point_index} пропущена: высота слишком мала ({height} < {min_height})")
            continue

        center = np.array([point[0], point[1], (z_top + z_bottom) / 2])

        base_points[point_index] = np.array([point[0], point[1], z_bottom])

        supports_info.append({
            "center": center,
            "height": height,
            "radius": thickness / 3,
            "color": color
        })

    return supports_info, base_points

def build_support_cylinders(supports_info):

    """
    Строит цилиндры для опор на основе подготовленных данных.
    """
    if not supports_info:
        return None

    cylinders = []

    for support in supports_info:
        cylinder = trimesh.creation.cylinder(
            radius=support["radius"],
            height=support["height"],
            sections=8,
            transform=trimesh.transformations.translation_matrix(
                support["center"]
            )
        )
        cylinder.visual.vertex_colors = support["color"]
        cylinders.append(cylinder)

    return trimesh.util.concatenate(cylinders)


def create_support_lines(base_points, groups, supports_info, model_mesh,
                        thickness=0.2, max_distance=5.0, min_height=1.0,
                        height_multiplier=2.0, spirally=False):
    print("create_support_lines")
    def create_marker_sphere(position, radius, color):
        
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(position)
        sphere.visual.vertex_colors = color
        return sphere
    # --------------------- Helper Functions ---------------------
    def build_cross_stack(start_point: np.ndarray,
                      direction: np.ndarray,
                      diagonal_length: float,
                      max_offset: float,
                      radius: float,
                      sections: int,
                      color: Sequence[int],
                      mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        """
        Строит вертикальную стопку крестов (два диагональных цилиндра в противоположных направлениях),
        поднимаясь от start_point по оси Z, пока не превысит max_offset.
        """
        # Длина диагонали берётся из нормали direction (должен быть нормирован вектор от первой диагонали!)
        if diagonal_length <= 0:
            return [] 

        # Расстояние по Z между центрами крестов
        cross_span = diagonal_length+0.3

        crosses: List[trimesh.Trimesh] = []
        current_offset = 0.0

        half = 0.5 * diagonal_length * direction
        # print(half)
        while current_offset <= max_offset:
            # Центр креста на высоте
            center = start_point + np.array([0.0, 0.0, current_offset])

        
            # Построим два цилиндра (крест) в этом центре
            for dir_vec in ( direction, -direction ):
                # Концы цилиндра
                half = 0.5 * diagonal_length * dir_vec
                p_start = center - half
                p_end   = center + half
                cyl = create_diagonal_cylinder(p_start, p_end, radius, color)
                if cyl is not None:
                    crosses.append(cyl)

                    # # Добавляем сферы-маркеры в начало и конец цилиндра
                    # marker_start = create_marker_sphere(p_start, radius * 1.2, [255, 0, 0, 255])  # красный
                    # marker_end = create_marker_sphere(p_end, radius * 1.2, [0, 255, 10, 255])    # зелёный

                # crosses.extend([marker_start, marker_end])
            current_offset += cross_span

        return crosses
    def create_diagonal_cylinder(start, end, radius, color):
        """Создает наклонный цилиндр между двумя точками с заданным цветом"""
        if start is None or end is None:
            return None
            
        dir_vec = end - start
        dir_xy = dir_vec[:2]
        dist_xy = np.linalg.norm(dir_xy)
        
        if dist_xy < 1e-6:
            return None
            
        height = dist_xy * np.sqrt(2)
        direction = np.array([dir_xy[0], dir_xy[1], height])
        direction /= np.linalg.norm(direction)
        
        # Создаем и ориентируем цилиндр
        cyl = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=8
        )
        rot_axis = np.cross([0, 0, 1], direction)
        rot_angle = np.arccos(np.clip(np.dot([0, 0, 1], direction), -1.0, 1.0))
        
        if np.linalg.norm(rot_axis) > 1e-6:
            rotation = trimesh.transformations.rotation_matrix(rot_angle, rot_axis)
            cyl.apply_transform(rotation)
            
        # Позиционируем цилиндр
        midpoint = (start + end + np.array([0, 0, height])) / 2
        cyl.apply_translation(midpoint)
        cyl.visual.vertex_colors = color
        
        return cyl

    def get_connection_pattern(group_size):
        """Возвращает шаблон соединений для группы заданного размера"""
        patterns = {
            6: [(0,1), (1,2), (3,4), (4,5), (0,3), (2,5)],
            5: [(0,1), (1,2), (3,4), (1,4), (2,3)],
            4: [(0,1), (2,3), (1,2), (0,3)],
            3: [(0,1), (1,2)],
            2: [(0,1)]
        }
        return patterns.get(group_size, [])

    def process_group_connections(group, support_lines):
        """Обрабатывает соединения внутри группы"""
        if len(group) < 2:
            return support_lines

        points = np.array([base_points[i] for i in group])
        y_med = np.median(points[:, 1])
        
        # Сортируем точки группы
        lower = sorted([i for i in group if base_points[i][1] <= y_med], 
                      key=lambda i: base_points[i][0])
        upper = sorted([i for i in group if base_points[i][1] > y_med],
                      key=lambda i: base_points[i][0])
        ordered = lower + upper
        
        # Создаем соединения по шаблону
        for i, j in get_connection_pattern(len(ordered)):
            if i >= len(ordered) or j >= len(ordered):
                continue
                
            a, b = ordered[i], ordered[j]
            dist_xy = np.linalg.norm(base_points[a][:2] - base_points[b][:2])
            required_height = dist_xy * height_multiplier
            
            # Создаем соединения в обе стороны
            support_lines = create_bidirectional_connection(
                a, b, required_height,
                support_lines,
                radius=thickness/4,
                color=[50, 250, 74, 170]
            )
            
        return support_lines


    def create_bidirectional_connection(a, b, required_height, support_lines, radius, color):
        # Отметим точки a и b цветом
       
        already_connected = False

        # От a к b
        if supports_info[b]["height"] >= required_height:
            start = base_points[a]
            end   = base_points[b]
            # cyl = create_diagonal_cylinder(start, end, radius, color)
            # support_lines.append(cyl)
            already_connected = True

            # Строим стопку крестов над start
            midpoint = (start + end) / 2
            vec = end - start
            diagonal_length = np.linalg.norm(vec)
            direction = vec / diagonal_length
            max_offset=min(supports_info[a]["height"],supports_info[b]["height"])
            cross_stack = build_cross_stack(
                    start_point    = midpoint,
                    direction      = direction,
                    diagonal_length= diagonal_length+0.1,
                    max_offset     = max_offset - 2.0,
                    radius         = radius,
                    sections       = 8,
                    color          = color,
                    mesh           = model_mesh
                )

            support_lines.extend(cross_stack)

        return support_lines


    # def process_remaining_connections(support_lines):
    #     """Обрабатывает соединения между несвязанными точками"""
    #     connected = [False] * len(base_points)
    #     sorted_points = sorted(
    #         [(i, p) for i, p in enumerate(base_points) if p is not None],
    #         key=lambda x: x[1][2]
    #     )

    #     for idx, pos in sorted_points:
    #         if connected[idx]:
    #             continue
                
    #         neighbors = find_unconnected_neighbors(idx, pos, connected)
    #         support_lines = process_neighbor_connections(
    #             idx, pos, neighbors, 
    #             connected, support_lines
    #         )
            
    #     return support_lines

    # def find_unconnected_neighbors(idx, pos, connected):
    #     """Находит неподключенных соседей для указанной точки"""
    #     other_points = [
    #         (i, p) for i, p in enumerate(base_points) 
    #         if i != idx and p is not None
    #     ]
        
    #     if not other_points:
    #         return []
            
    #     # Используем KD-дерево для поиска ближайших соседей
    #     points_array = np.array([p for _, p in other_points])
    #     tree = cKDTree(points_array[:, :2])
    #     dists, indices = tree.query([pos[:2]], k=min(3, len(other_points)))
        
    #     return [
    #         (other_points[i][0], dists[0][j]) 
    #         for j, i in enumerate(indices[0])
    #         if not connected[other_points[i][0]] 
    #         and dists[0][j] <= max_distance
    #     ]

    # def process_neighbor_connections(idx, pos, neighbors, connected, support_lines):
    #     """Обрабатывает соединения с соседями"""
    #     for neighbor_idx, dist in neighbors:
    #         h_idx = supports_info[idx]["height"]
    #         h_ni = supports_info[neighbor_idx]["height"]
    #         required_height = dist * height_multiplier
            
    #         # Создаем соединения если удовлетворяем условиям
    #         if h_ni >= required_height:
    #             cyl = create_diagonal_cylinder(
    #                 pos,
    #                 base_points[neighbor_idx],
    #                 thickness/4,
    #                 [0, 200, 100, 200]
    #             )
    #             if cyl:
    #                 support_lines.append(cyl)
                    
    #         if h_idx >= required_height:
    #             cyl = create_diagonal_cylinder(
    #                 base_points[neighbor_idx],
    #                 pos,
    #                 thickness/4,
    #                 [0, 200, 100, 200]
    #             )
    #             if cyl:
    #                 support_lines.append(cyl)
                    
    #         if h_idx >= required_height or h_ni >= required_height:
    #             connected[idx] = True
    #             connected[neighbor_idx] = True
    #             break
                
    #     return support_lines

    # --------------------- Main Logic ---------------------
    support_lines = []

    if(spirally):
        group = groups[0]
        for i in range(group.size-1):
            a, b = group[i], group[i+1]
            p1, p2 = base_points[a], base_points[b]
            dist_xy = np.linalg.norm(p1[:2] - p2[:2])
            required_height = dist_xy * height_multiplier
            
            # If we'd like to connect all points then we should put required_height=0
            support_lines = create_bidirectional_connection(
                a, b, required_height=required_height,
                support_lines=support_lines,
                radius=thickness/4,
                color=[50, 250, 74, 170]
            )
    else:
        # 1. Внутригрупповые соединения
        for group in groups:
            support_lines = process_group_connections(group, support_lines)

        # 2. Соединения между несвязанными точками
        #support_lines = process_remaining_connections(support_lines)

    return trimesh.util.concatenate(support_lines) if support_lines else None

def generate_tree_supports(base_points, supports_info, group_indices, model_mesh, 
                          branch_length=0.5, angle_deg=30, offset=1.0, thickness=0.2):
    print("generate_tree_supports")
    tree_branches = []
    thickness_branch = thickness / 4
    angle_rad = np.radians(angle_deg)
    horizontal_dist = branch_length * np.sin(angle_rad)
    vertical_dist   = branch_length * np.cos(angle_rad)
    directions = [
        (horizontal_dist, 0, vertical_dist),
        (-horizontal_dist, 0, vertical_dist),
        (0, horizontal_dist, vertical_dist),
        (0, -horizontal_dist, vertical_dist)
    ]
    intersector = RayMeshIntersector(model_mesh)

    def get_transform(start, end):
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return None
        direction = vec / length
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation = np.eye(4)
        else:
            axis = np.cross(z_axis, direction)
            if np.linalg.norm(axis) < 1e-6:
                axis = np.array([1, 0, 0]); angle = np.pi
            else:
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            rotation = trimesh.transformations.rotation_matrix(angle, axis)
        translation = trimesh.transformations.translation_matrix((start + end) / 2)
        return translation.dot(rotation)

    for group in group_indices:
        for idx in group:
            if idx >= len(supports_info):
                continue
            support = supports_info[idx]
            center  = support['center']
            height  = support['height']
            tree_start = np.array([center[0], center[1], center[2] + height/2 - offset])
            if model_mesh.contains([tree_start])[0]:
                continue
            
            saved_tree_start = tree_start
            for dir_vec in directions:
                tree_start = saved_tree_start
                branch_tip = tree_start + np.array(dir_vec)

               

                # --- ЗДЕСЬ ЗАМЕНА intersects_line ---
                vec = branch_tip - tree_start
                length = np.linalg.norm(vec)
                if length < 1e-6:
                    continue
                direction = vec / length
                dist = intersector.intersects_first([tree_start], [direction])[0]
                if not np.isnan(dist) and dist < length:
                    continue
                
                saved_tree_start = tree_start
                while (model_mesh.contains([branch_tip])[0]):
                    tree_start = np.array([tree_start[0], tree_start[1], tree_start[2]- 0.5])
                    branch_tip = tree_start + np.array(dir_vec)
                    vec = branch_tip - tree_start
                    if tree_start[2] < 0:
                        continue
                
                # ------------------------------------

                vertical_locations, _, _ = intersector.intersects_location(
                    ray_origins=[branch_tip],
                    ray_directions=[[0, 0, 1]],
                    multiple_hits=True
                )
                vertical_hit = None
                for loc in vertical_locations:
                    if (vertical_hit is None or loc[2] < vertical_hit[2]):
                        vertical_hit = loc
                if vertical_hit is None or loc[2] <= branch_tip[2] :
                    continue
                # Фильтруем только точки выше branch_tip
                # valid_hits = [loc for loc in vertical_locations if loc[2] > branch_tip[2]]

                # if not valid_hits:
                #     continue  # Нет подходящих точек

                # Выбираем точку с минимальной Z (ближайшую снизу)
                # vertical_hit = min(valid_hits, key=lambda x: x[2])

                # 1) создаём наклонную ветвь (с перекрытием)
                length = np.linalg.norm(vec)
                # строим чуть длиннее:
                over = thickness_branch/2
                end_inc = tree_start + vec*(1 + over/length)
                transform_inc = get_transform(tree_start, end_inc)
                cyl_inc = trimesh.creation.cylinder(radius=thickness_branch, height=length+over, transform=transform_inc, sections=6)
                cyl_inc.visual.vertex_colors = [255,255,0,180]
                tree_branches.append(cyl_inc)

                # 2) добавляем сферу-стык
                joint = trimesh.creation.icosphere(subdivisions=4, radius=thickness_branch*1.1)
                joint.apply_translation(branch_tip)
                joint.visual.vertex_colors = [255,255,0,180]
                tree_branches.append(joint)

                # 3) создаём вертикальную ветвь, стартуя чуть ниже точки branch_tip
                
                start_vert = branch_tip - np.array([0,0, thickness_branch/2])
                r_base = thickness_branch
                
                vert_height = vertical_hit[2] - start_vert[2]

                if vert_height > 0:
                    # Создаем конус в 2 раза выше
                    cone_full = trimesh.creation.cone(
                        radius=r_base,
                        height=vert_height * 2,
                        sections=6
                    )

                    # Высота среза сверху (например, оставим верхние 0.5 * vert_height)
                    cut_height = vert_height   # от основания
                    # cut_height = vertical_hit[2] 
                    
                    # Плоскость среза (нормаль вверх, точка на высоте cut_height)
                    plane_origin = np.array([0, 0, cut_height])
                    plane_normal = np.array([0, 0, -1])

                    # Усечение конуса сверху плоскостью, оставляем нижнюю часть
                    truncated_cone = cone_full.slice_plane(plane_origin, plane_normal)

                    # Смещаем в нужное место
                    truncated_cone.apply_translation(start_vert)

                    truncated_cone.visual.vertex_colors = [255, 255, 0, 180]
                    tree_branches.append(truncated_cone)


    return tree_branches

if __name__ == "__main__":

    file_path = os.path.join(os.path.dirname(__file__), 'cat.stl')
    export_format = sys.argv[1] if len(sys.argv)>1 else "stl"

    try:
        model = load_stl(file_path)
        overhang_faces, angles = find_overhangs(model)
        print(f"Найдено нависающих граней: {len(overhang_faces)}")

        if overhang_faces.size > 0:
            support_points, point_params, grid_spacing, overhang_mesh = generate_support_points(model, overhang_faces, angles)
            supports_info, base_points = prepare_supports(model, support_points, point_params)

            base_points_array = np.array(list(base_points.values()))

            groups_1 = group_support_points(base_points_array, supports_info, grid_spacing, model)
            groups = regroup_unassigned_points(base_points_array, groups_1, supports_info, model)

            spirally = True
            if (spirally):
                sorted_ids_spirally = sort_points_spirally(list(base_points_array), True)
                groups = np.array([sorted_ids_spirally]) 
        
            supports = build_support_cylinders(supports_info)
            support_lines = create_support_lines(base_points_array, groups, supports_info, model, thickness=0.2, min_height=1.0, spirally=spirally)
            tree_supports = generate_tree_supports(base_points_array, supports_info, groups, model, thickness=0.2)

            if supports:
                # Пути для сохранения только опор
                support_only_path = os.path.join(os.path.dirname(__file__), 'supports_only.ply')
                support_only_stl = os.path.join(os.path.dirname(__file__), f'supports_only.{export_format}')
                supported_model_stl = os.path.join(os.path.dirname(__file__), f'supported_model.{export_format}')

                # Собираем только опоры
                support_scene_objects = []

                if supports:
                    support_scene_objects.append(supports)
                if support_lines:
                    support_scene_objects.append(support_lines)
                if tree_supports:
                    support_scene_objects.extend(tree_supports)

                # Экспортируем только поддержки
                if support_scene_objects:
                    print("Start exporting supports")
                    combined_supports = trimesh.util.concatenate(support_scene_objects)
                    combined_supports.export(support_only_path)
                    combined_supports.export(support_only_stl)
                    print(f"Поддержки сохранены в: {support_only_path}")
                    print(f"STL поддержек сохранён в: {support_only_stl}")
                    combined_model_supports = trimesh.util.concatenate([model, combined_supports])
                    combined_model_supports.export(supported_model_stl)
                    print(f"STL модели с поддержками сохранён в: {supported_model_stl}")

                    # Экспортируем меш в байты
                    model_bytes = io.BytesIO()
                    model.export(model_bytes, file_type='stl')

                    support_bytes = io.BytesIO()
                    combined_supports.export(support_bytes, file_type='stl')

                    # Упаковываем в ZIP
                    with ZipFile("out.lamprj", "w") as zipf:
                        zipf.writestr("2/body-0001.stl", model_bytes.getvalue())
                        zipf.writestr("2/support-0002.stl", support_bytes.getvalue())
                    print(f"Lam-project сохранён в: out.lamprj")

                else:
                    print("Нет поддержек для сохранения.")

                # Визуализация: модель + поддержки
                scene = trimesh.Scene()
                scene.add_geometry(model)
                overhang_mesh.visual.face_colors = [255, 0, 0, 100]  # RGBA: красный с прозрачностью
                # Берём оригинальные вершины и грани
                verts = overhang_mesh.vertices
                faces = overhang_mesh.faces

                # Создаём "обратные" грани (переворачиваем порядок вершин)
                reversed_faces = faces[:, [1, 0, 2]]

                # Объединяем оригинальные и обратные
                double_faces = np.vstack([faces, reversed_faces])

                # Создаём новый меш с двойным покрытием
                double_overhang_mesh = trimesh.Trimesh(
                    vertices=verts,
                    faces=double_faces,
                    process=False
                )

                # Задаём цвет
                double_overhang_mesh.visual.face_colors = [255, 0, 0, 100]

                #scene.add_geometry(double_overhang_mesh)
                if supports:
                    scene.add_geometry(supports)
                if support_lines:
                    scene.add_geometry(support_lines)
                if tree_supports:
                    scene.add_geometry(tree_supports)
                scene.show()

            else:
                print("Не удалось создать поддержки")
        else:
            print("Нависающие элементы не обнаружены")
            model.show()

    except Exception as e:
        print(f"Ошибка: {e}")

