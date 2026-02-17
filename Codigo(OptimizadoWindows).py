import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk, simpledialog
from PIL import Image, ImageDraw, ImageTk
from collections import deque
import heapq

# ==================== CONFIGURACIÓN DINÁMICA ====================
class GraphConfig:
    def __init__(self):
        # Parámetros de nodos
        self.min_radius = 15
        self.max_radius = 120
        self.min_area = 500
        self.max_area = 20000
        self.circularity = 0.4
        self.min_dist = 40

        # Parámetros de Hough Circles
        self.hough_param1 = 100
        self.hough_param2 = 30

        # Parámetros de aristas
        self.max_endpoint_dist = 80

        # Parámetros de color rojo
        self.red_tolerance = 15
        self.red_saturation_min = 100

config = GraphConfig()


# ================================================================
#              ANALIZADOR DE TEORÍA DE GRAFOS
# ================================================================
class GraphAnalyzer:
    """Análisis completo de propiedades de grafos según teoría matemática"""
    
    def __init__(self, matrix, edges, nodes, edge_costs=None, edge_multiplicity=None):
        self.matrix = matrix
        self.edges = edges
        self.nodes = nodes
        self.n = len(matrix)
        self.edge_costs = edge_costs if edge_costs else {}
        self.edge_multiplicity = edge_multiplicity if edge_multiplicity else {}
    
    def is_simple_graph(self):
        """Determina si es un grafo simple (sin loops ni aristas paralelas)"""
        has_loops = np.any(np.diag(self.matrix) != 0)
        has_parallel = np.any(self.matrix > 1)
        return not (has_loops or has_parallel), has_loops, has_parallel
    
    def is_directed(self):
        """Determina si el grafo es dirigido (matriz no simétrica)"""
        # Verificar si la matriz es simétrica con tolerancia numérica
        is_symmetric = np.allclose(self.matrix, self.matrix.T)
        
        # Contar aristas asimétricas
        asymmetric_edges = 0
        total_edges = 0
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.matrix[i][j] > 0 or self.matrix[j][i] > 0:
                    total_edges += 1
                    if self.matrix[i][j] != self.matrix[j][i]:
                        asymmetric_edges += 1
        
        return not is_symmetric, asymmetric_edges, total_edges
    
    def is_connected(self):
        """Determina si el grafo es conectado usando BFS"""
        if self.n == 0:
            return False
        
        visited = [False] * self.n
        queue = deque([0])
        visited[0] = True
        count = 1
        
        while queue:
            node = queue.popleft()
            for neighbor in range(self.n):
                if self.matrix[node][neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    count += 1
        
        return count == self.n
    
    def is_complete(self):
        """Determina si es un grafo completo"""
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.matrix[i][j] == 0:
                    return False
        return True
    
    def get_node_degrees(self):
        """Calcula el grado de cada nodo"""
        degrees = []
        for i in range(self.n):
            degree = np.sum(self.matrix[i])
            if self.matrix[i][i] > 0:
                degree += self.matrix[i][i]
            degrees.append(int(degree))
        return degrees
    
    def find_cycles(self):
        """Detecta ciclos en el grafo"""
        def dfs_cycle(node, parent, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            for neighbor in range(self.n):
                if self.matrix[node][neighbor] > 0:
                    if not visited[neighbor]:
                        if dfs_cycle(neighbor, node, visited, rec_stack):
                            return True
                    elif rec_stack[neighbor] and neighbor != parent:
                        return True
            
            rec_stack[node] = False
            return False
        
        visited = [False] * self.n
        rec_stack = [False] * self.n
        
        for i in range(self.n):
            if not visited[i]:
                if dfs_cycle(i, -1, visited, rec_stack):
                    return True
        return False
    
    def dijkstra(self, start, end):
        """Algoritmo de Dijkstra con costos personalizados"""
        if start >= self.n or end >= self.n:
            return None, float('inf')
        
        dist = [float('inf')] * self.n
        dist[start] = 0
        parent = [-1] * self.n
        visited = [False] * self.n
        pq = [(0, start)]
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if visited[u]:
                continue
            visited[u] = True
            
            if u == end:
                break
            
            for v in range(self.n):
                if self.matrix[u][v] > 0 and not visited[v]:
                    # Buscar costo personalizado
                    edge_key = tuple(sorted((u, v)))
                    weight = self.edge_costs.get(edge_key, 1)
                    
                    if dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        parent[v] = u
                        heapq.heappush(pq, (dist[v], v))
        
        if dist[end] == float('inf'):
            return None, float('inf')
        
        path = []
        current = end
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        return path, dist[end]
    
    def bfs_shortest_path(self, start, end):
        """BFS para camino más corto (sin pesos)"""
        if start >= self.n or end >= self.n:
            return None, float('inf')
        
        visited = [False] * self.n
        parent = [-1] * self.n
        queue = deque([start])
        visited[start] = True
        
        while queue:
            node = queue.popleft()
            
            if node == end:
                path = []
                current = end
                while current != -1:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path, len(path) - 1
            
            for neighbor in range(self.n):
                if self.matrix[node][neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        return None, float('inf')
    
    def calculate_diameter(self):
        """Calcula el diámetro del grafo"""
        if not self.is_connected():
            return float('inf')
        
        max_distance = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                _, dist = self.bfs_shortest_path(i, j)
                if dist != float('inf'):
                    max_distance = max(max_distance, dist)
        
        return max_distance
    
    def calculate_efficiency(self):
        """Calcula la eficiencia global del grafo"""
        if self.n <= 1:
            return 0.0
        
        total_efficiency = 0
        count = 0
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                _, dist = self.bfs_shortest_path(i, j)
                if dist != float('inf') and dist > 0:
                    total_efficiency += 1.0 / dist
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_efficiency / count
    
    def get_geodesic_info(self):
        """Información sobre geodésicas"""
        geodesics = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                path, dist = self.bfs_shortest_path(i, j)
                if path:
                    geodesics[(i, j)] = {'path': path, 'distance': dist}
        return geodesics
    
    def is_bipartite(self):
        """Determina si el grafo es bipartito"""
        color = [-1] * self.n
        
        for start in range(self.n):
            if color[start] == -1:
                queue = deque([start])
                color[start] = 0
                
                while queue:
                    node = queue.popleft()
                    
                    for neighbor in range(self.n):
                        if self.matrix[node][neighbor] > 0:
                            if color[neighbor] == -1:
                                color[neighbor] = 1 - color[node]
                                queue.append(neighbor)
                            elif color[neighbor] == color[node]:
                                return False
        return True
    
    def count_triangles(self):
        """Cuenta el número de triángulos"""
        triangles = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                for k in range(j + 1, self.n):
                    if (self.matrix[i][j] > 0 and 
                        self.matrix[j][k] > 0 and 
                        self.matrix[k][i] > 0):
                        triangles += 1
        return triangles


# ================================================================
#              EDITOR DE COSTOS DE ARISTAS
# ================================================================
class EdgeCostEditor:
    def __init__(self, parent, edges, node_label_map, initial_costs=None):
        self.window = tk.Toplevel(parent)
        self.window.title("💰 Editor de Costos de Aristas")
        self.window.geometry("600x500")
        self.window.configure(bg="#1a1a2e")
        
        self.edges = edges
        self.node_label_map = node_label_map
        self.costs = initial_costs if initial_costs else {}
        self.entries = {}
        
        self.create_widgets()
    
    def create_widgets(self):
        # Título
        tk.Label(self.window, text="💰 ASIGNAR COSTOS A LAS ARISTAS",
                font=("Arial", 14, "bold"), fg="#00d4ff", bg="#1a1a2e").pack(pady=10)
        
        tk.Label(self.window, 
                text="Define el costo/peso de cada arista para el algoritmo de Dijkstra\n" +
                     "(distancia, tiempo, dinero, etc.)",
                font=("Arial", 9), fg="#a8dadc", bg="#1a1a2e").pack(pady=5)
        
        # Frame con scroll para las aristas
        canvas_frame = tk.Frame(self.window, bg="#1a1a2e")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg="#16213e", highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#16213e")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Crear entrada para cada arista
        for idx, (i, j) in enumerate(self.edges):
            edge_key = tuple(sorted((i, j)))
            label_i = self.node_label_map.get(i, str(i + 1))
            label_j = self.node_label_map.get(j, str(j + 1))
            
            row_frame = tk.Frame(scrollable_frame, bg="#2d2d44", pady=5, padx=10)
            row_frame.pack(fill="x", padx=5, pady=3)
            
            tk.Label(row_frame, text=f"🔗 Arista {label_i} ↔ {label_j}:",
                    font=("Arial", 10), fg="white", bg="#2d2d44",
                    width=20, anchor="w").pack(side="left", padx=5)
            
            entry = tk.Entry(row_frame, font=("Consolas", 10), width=10,
                           bg="#16213e", fg="#00d4ff", insertbackground="white")
            entry.insert(0, str(self.costs.get(edge_key, 1)))
            entry.pack(side="left", padx=5)
            
            self.entries[edge_key] = entry
        
        # Botones de acción
        button_frame = tk.Frame(self.window, bg="#1a1a2e", pady=10)
        button_frame.pack(fill="x")
        
        tk.Button(button_frame, text="🔄 Resetear Todos a 1",
                 command=self.reset_all, font=("Arial", 10),
                 bg="#ffa500", fg="white", padx=15, pady=5).pack(side="left", padx=10)
        
        tk.Button(button_frame, text="✓ GUARDAR COSTOS",
                 command=self.save_costs, font=("Arial", 10, "bold"),
                 bg="#00d4ff", fg="#1a1a2e", padx=20, pady=5).pack(side="right", padx=10)
    
    def reset_all(self):
        """Resetea todos los costos a 1"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, "1")
    
    def save_costs(self):
        """Guarda los costos y cierra la ventana"""
        try:
            new_costs = {}
            for edge_key, entry in self.entries.items():
                value = entry.get().strip()
                if value:
                    cost = float(value)
                    if cost <= 0:
                        raise ValueError(f"El costo debe ser positivo")
                    new_costs[edge_key] = cost
                else:
                    new_costs[edge_key] = 1.0
            
            self.costs = new_costs
            self.window.destroy()
            
        except ValueError as e:
            messagebox.showerror("❌ Error", f"Valor inválido: {str(e)}\n\nUse números positivos.")
    
    def get_costs(self):
        """Retorna los costos definidos"""
        return self.costs


# ================================================================
#              TABLERO DE DIBUJO
# ================================================================
class DrawingBoard:
    def __init__(self, master, width=800, height=600):
        self.master = master
        self.width = width
        self.height = height
        
        self.canvas = tk.Canvas(master, width=width, height=height, 
                               bg="white", cursor="cross")
        self.canvas.pack()
        
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        self.drawing = False
        self.tool = "node"
        self.node_radius = 20
        self.line_color = "black"
        self.line_width = 3
        
        self.node_count = 0
        self.node_labels = {}
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.old_x = None
        self.old_y = None
    
    def on_click(self, event):
        self.drawing = True
        self.old_x = event.x
        self.old_y = event.y
        
        if self.tool == "node":
            self.draw_node(event.x, event.y)
    
    def on_drag(self, event):
        if self.drawing and self.tool == "line":
            if self.old_x and self.old_y:
                self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                       fill=self.line_color, width=self.line_width,
                                       capstyle=tk.ROUND, smooth=True)
                
                self.draw.line([self.old_x, self.old_y, event.x, event.y],
                              fill=self.line_color, width=self.line_width)
                
                self.old_x = event.x
                self.old_y = event.y
    
    def on_release(self, event):
        self.drawing = False
        self.old_x = None
        self.old_y = None
    
    def draw_node(self, x, y):
        r = self.node_radius
        
        self.node_count += 1
        label = str(self.node_count)
        
        self.node_labels[(x, y)] = label
        
        self.canvas.create_oval(x-r, y-r, x+r, y+r, 
                               fill="red", outline="darkred", width=2)
        
        self.canvas.create_text(x, y, text=label, 
                               font=("Arial", int(r*0.8), "bold"),
                               fill="white")
        
        self.draw.ellipse([x-r, y-r, x+r, y+r], 
                         fill="red", outline="red")
    
    def set_tool(self, tool):
        self.tool = tool
        if tool == "node":
            self.canvas.config(cursor="cross")
        else:
            self.canvas.config(cursor="pencil")
    
    def set_node_size(self, size):
        self.node_radius = size
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.width, self.height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.node_count = 0
        self.node_labels = {}
    
    def get_image_array(self):
        return cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
    
    def get_node_labels(self):
        return self.node_labels


# ================================================================
#              FUNCIONES DE DETECCIÓN
# ================================================================
def preprocess_smart(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    return gray, blurred, enhanced


def filter_red_regions(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    tol = config.red_tolerance
    sat_min = config.red_saturation_min
    
    lower_red1 = np.array([0, sat_min, 100])
    upper_red1 = np.array([10 + tol, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170 - tol, sat_min, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return red_mask


def detect_circles_multimethod(img):
    gray, blurred, enhanced = preprocess_smart(img)
    red_mask = filter_red_regions(img)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=red_mask)
    blurred = cv2.bitwise_and(blurred, blurred, mask=red_mask)
    
    all_circles = []
    
    circles1 = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT, dp=1, minDist=config.min_dist,
        param1=config.hough_param1, param2=config.hough_param2,
        minRadius=config.min_radius, maxRadius=config.max_radius
    )
    
    if circles1 is not None:
        for circle in circles1[0]:
            x, y, r = circle
            area = math.pi * r * r
            if config.min_area <= area <= config.max_area:
                all_circles.append({
                    'x': int(x), 'y': int(y), 'r': int(r),
                    'area': area, 'method': 'hough', 'score': 1.0
                })
    
    circles2 = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=config.min_dist,
        param1=config.hough_param1 - 20, param2=config.hough_param2 - 10,
        minRadius=config.min_radius, maxRadius=config.max_radius
    )
    
    if circles2 is not None:
        for circle in circles2[0]:
            x, y, r = circle
            area = math.pi * r * r
            if config.min_area <= area <= config.max_area:
                all_circles.append({
                    'x': int(x), 'y': int(y), 'r': int(r),
                    'area': area, 'method': 'hough2', 'score': 0.9
                })
    
    binary = red_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (config.min_area <= area <= config.max_area):
            continue
        
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * math.pi * area / (peri * peri)
        
        if circularity < config.circularity:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if not (0.6 <= aspect <= 1.4):
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        r = int(math.sqrt(area / math.pi))
        
        score = circularity * 0.8
        
        all_circles.append({
            'x': cx, 'y': cy, 'r': r,
            'area': area, 'method': 'contour',
            'score': score, 'circularity': circularity
        })
    
    return all_circles


def merge_detections(circles):
    if not circles:
        return []
    
    circles = sorted(circles, key=lambda c: c['score'], reverse=True)
    
    merged = []
    used = set()
    
    for i, circle in enumerate(circles):
        if i in used:
            continue
        
        cluster = [circle]
        
        for j, other in enumerate(circles[i+1:], start=i+1):
            if j in used:
                continue
            
            dist = math.hypot(circle['x'] - other['x'], circle['y'] - other['y'])
            
            if dist < config.min_dist * 0.7:
                cluster.append(other)
                used.add(j)
        
        if len(cluster) > 0:
            avg_x = int(np.mean([c['x'] for c in cluster]))
            avg_y = int(np.mean([c['y'] for c in cluster]))
            avg_r = int(np.mean([c['r'] for c in cluster]))
            avg_area = np.mean([c['area'] for c in cluster])
            max_score = max([c['score'] for c in cluster])
            
            merged.append({
                'x': avg_x, 'y': avg_y, 'r': avg_r,
                'area': avg_area, 'score': max_score,
                'cluster_size': len(cluster)
            })
    
    merged = sorted(merged, key=lambda c: c['score'], reverse=True)
    
    return merged[:30]


def create_mask(shape, nodes):
    mask = np.ones(shape[:2], dtype=np.uint8) * 255
    for node in nodes:
        r = int(node['r'] * 1.5)
        cv2.circle(mask, (node['x'], node['y']), r, 0, -1)
    return mask


def detect_lines(img, mask):
    """Detecta líneas con múltiples métodos para mayor robustez"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(masked)
    
    # Método 1: Detección de bordes estándar
    median = np.median(enhanced[enhanced > 0])
    if median == 0:
        median = 50
    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))
    edges = cv2.Canny(enhanced, lower, upper)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Método 2: Detección de colores oscuros (líneas negras/grises)
    # Crear máscara para colores oscuros
    dark_mask = cv2.inRange(gray, 0, 120)
    dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=mask)
    
    # Adelgazar líneas gruesas
    skeleton = cv2.ximgproc.thinning(dark_mask) if hasattr(cv2, 'ximgproc') else dark_mask
    
    # Combinar ambos métodos
    combined_edges = cv2.bitwise_or(edges, skeleton)
    
    # Detectar líneas con HoughLinesP
    lines = cv2.HoughLinesP(
        combined_edges, rho=1, theta=np.pi/180, threshold=20,
        minLineLength=25, maxLineGap=40
    )
    
    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2-x1, y2-y1)
            if length >= 20:
                segments.append((x1, y1, x2, y2))
    
    # Método adicional: buscar directamente píxeles negros conectados
    if len(segments) < 2:
        # Buscar píxeles oscuros en la imagen original
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        
        # Máscara de píxeles muy oscuros (casi negros)
        black_mask = cv2.inRange(v_channel, 0, 80)
        black_mask = cv2.bitwise_and(black_mask, black_mask, mask=mask)
        
        # Dilatar para conectar píxeles cercanos
        kernel_big = np.ones((5, 5), np.uint8)
        black_mask = cv2.dilate(black_mask, kernel_big, iterations=2)
        
        # Detectar contornos de líneas
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) >= 2:
                # Obtener puntos extremos del contorno
                leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
                rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
                topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
                bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
                
                # Crear líneas desde extremos
                dist1 = math.hypot(leftmost[0]-rightmost[0], leftmost[1]-rightmost[1])
                dist2 = math.hypot(topmost[0]-bottommost[0], topmost[1]-bottommost[1])
                
                if dist1 > 20:
                    segments.append((leftmost[0], leftmost[1], rightmost[0], rightmost[1]))
                if dist2 > 20:
                    segments.append((topmost[0], topmost[1], bottommost[0], bottommost[1]))
    
    return segments


def nearest_node(pt, nodes):
    x, y = pt
    best_idx = None
    best_dist = 1e9
    
    for i, node in enumerate(nodes):
        dist = math.hypot(node['x'] - x, node['y'] - y)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    
    return best_idx if best_dist < config.max_endpoint_dist else None


def build_graph(nodes, segments):
    """Construye el grafo detectando aristas múltiples"""
    from collections import defaultdict
    
    # Diccionario para contar aristas entre cada par de nodos
    edge_count = defaultdict(int)
    
    for (x1, y1, x2, y2) in segments:
        i = nearest_node((x1, y1), nodes)
        j = nearest_node((x2, y2), nodes)
        
        if i is not None and j is not None and i != j:
            # Usar tupla ordenada como clave
            edge_key = tuple(sorted((i, j)))
            edge_count[edge_key] += 1
    
    # Convertir a lista de aristas con multiplicidad
    edges = []
    edge_multiplicity = {}
    
    for edge_key, count in edge_count.items():
        edges.append(edge_key)
        edge_multiplicity[edge_key] = count
    
    return sorted(edges), edge_multiplicity


def adjacency_matrix(nodes, edges, edge_multiplicity=None):
    """Crea matriz de adyacencia considerando aristas múltiples"""
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=int)
    
    if edge_multiplicity:
        # Usar multiplicidad de aristas
        for (i, j), count in edge_multiplicity.items():
            matrix[i][j] = count
            matrix[j][i] = count
    else:
        # Modo simple (una arista por par)
        for i, j in edges:
            matrix[i][j] = 1
            matrix[j][i] = 1
    
    return matrix


def visualize(img, nodes, edges, original_labels=None, edge_multiplicity=None):
    """Visualiza el grafo con aristas múltiples"""
    result = img.copy()
    
    # Dibujar aristas con indicador de multiplicidad
    for i, j in edges:
        x1, y1 = nodes[i]['x'], nodes[i]['y']
        x2, y2 = nodes[j]['x'], nodes[j]['y']
        
        edge_key = tuple(sorted((i, j)))
        multiplicity = edge_multiplicity.get(edge_key, 1) if edge_multiplicity else 1
        
        # Dibujar línea principal
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # Si hay múltiples aristas, dibujar indicador
        if multiplicity > 1:
            # Calcular punto medio
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            
            # Dibujar círculo con el número de aristas
            cv2.circle(result, (mid_x, mid_y), 15, (255, 165, 0), -1)
            cv2.circle(result, (mid_x, mid_y), 15, (0, 0, 0), 2)
            
            # Dibujar número de aristas
            text = str(multiplicity)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(result, text, (mid_x - tw//2, mid_y + th//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Dibujar nodos
    for idx, node in enumerate(nodes):
        x, y, r = node['x'], node['y'], node['r']
        
        cv2.circle(result, (x, y), r, (0, 255, 0), 3)
        
        if original_labels:
            min_dist = float('inf')
            closest_label = str(idx + 1)
            
            for (lx, ly), label in original_labels.items():
                dist = math.hypot(x - lx, y - ly)
                if dist < min_dist:
                    min_dist = dist
                    closest_label = label
            
            if min_dist < r * 2:
                label = closest_label
            else:
                label = str(idx + 1)
        else:
            label = str(idx + 1)
        
        font_scale = 0.8
        thickness = 2
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        cv2.rectangle(result,
                     (x - tw//2 - 4, y - th//2 - 4),
                     (x + tw//2 + 4, y + th//2 + 4),
                     (255, 255, 255), -1)
        
        cv2.putText(result, label, (x - tw//2, y + th//2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    
    return result


# ================================================================
#                VENTANA DE RESULTADOS AVANZADA
# ================================================================
def show_advanced_results(matrix, edges, nodes, original_labels, node_label_map, edge_costs, edge_multiplicity):
    """Muestra ventana con análisis completo del grafo"""
    
    results_window = tk.Toplevel(root)
    results_window.title("📊 Análisis Completo del Grafo")
    results_window.geometry("900x700")
    results_window.configure(bg="#1a1a2e")
    
    notebook = ttk.Notebook(results_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    analyzer = GraphAnalyzer(matrix, edges, nodes, edge_costs, edge_multiplicity)
    
    # ============= PESTAÑA 1: PROPIEDADES BÁSICAS =============
    tab1 = tk.Frame(notebook, bg="#1a1a2e")
    notebook.add(tab1, text="📋 Propiedades Básicas")
    
    text1 = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=100, height=35,
                                      font=("Consolas", 10), bg="#0f0f1e", fg="#00ff00",
                                      insertbackground="white")
    text1.pack(fill="both", expand=True, padx=10, pady=10)
    
    is_simple, has_loops, has_parallel = analyzer.is_simple_graph()
    is_directed, asymmetric_edges, total_edges = analyzer.is_directed()
    is_connected = analyzer.is_connected()
    is_complete = analyzer.is_complete()
    has_cycles = analyzer.find_cycles()
    degrees = analyzer.get_node_degrees()
    is_bipartite = analyzer.is_bipartite()
    triangles = analyzer.count_triangles()
    
    result_text = "═" * 80 + "\n"
    result_text += "                    ANÁLISIS COMPLETO DEL GRAFO\n"
    result_text += "═" * 80 + "\n\n"
    
    result_text += "📊 CARACTERÍSTICAS ESTRUCTURALES\n"
    result_text += "─" * 80 + "\n"
    result_text += f"• Número de nodos (vértices): {analyzer.n}\n"
    result_text += f"• Número de aristas únicas: {len(edges)}\n"
    
    # Contar aristas totales considerando multiplicidad
    total_edges_with_mult = sum(edge_multiplicity.values()) if edge_multiplicity else len(edges)
    if total_edges_with_mult > len(edges):
        result_text += f"• Número total de aristas (con paralelas): {total_edges_with_mult}\n"
        result_text += f"  └─ ⚠ Hay {total_edges_with_mult - len(edges)} aristas paralelas\n"
    
    result_text += f"• Densidad del grafo: {len(edges) / (analyzer.n * (analyzer.n - 1) / 2) * 100:.2f}%\n\n"
    
    result_text += "🔍 CLASIFICACIÓN DEL GRAFO\n"
    result_text += "─" * 80 + "\n"
    result_text += f"• Grafo simple: {'✓ SÍ' if is_simple else '✗ NO'}\n"
    if has_loops:
        result_text += "  └─ Contiene loops (aristas n-n)\n"
    if has_parallel:
        result_text += "  └─ Contiene aristas paralelas\n"
    
    result_text += f"\n• Grafo dirigido: {'✓ SÍ (Dígrafo)' if is_directed else '✗ NO (No dirigido)'}\n"
    if is_directed:
        result_text += f"  └─ Aristas asimétricas: {asymmetric_edges} de {total_edges} ({asymmetric_edges/total_edges*100:.1f}%)\n"
        result_text += f"  └─ Matriz NO simétrica - Algunas aristas tienen dirección única\n"
    else:
        result_text += f"  └─ Matriz simétrica - Todas las aristas son bidireccionales\n"
    
    result_text += f"\n• Grafo conectado: {'✓ SÍ' if is_connected else '✗ NO'}\n"
    result_text += f"• Grafo completo: {'✓ SÍ (Kn)' if is_complete else '✗ NO'}\n"
    result_text += f"• Grafo acíclico: {'✗ NO (Tiene ciclos)' if has_cycles else '✓ SÍ'}\n"
    result_text += f"• Grafo bipartito: {'✓ SÍ' if is_bipartite else '✗ NO'}\n\n"
    
    result_text += "📐 GRADOS DE LOS NODOS\n"
    result_text += "─" * 80 + "\n"
    for i, degree in enumerate(degrees):
        label = node_label_map.get(i, str(i + 1))
        result_text += f"• Nodo {label}: grado = {degree}\n"
    
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    
    result_text += f"\n• Grado máximo: {max_degree}\n"
    result_text += f"• Grado mínimo: {min_degree}\n"
    result_text += f"• Grado promedio: {avg_degree:.2f}\n\n"
    
    result_text += "🔺 PROPIEDADES GEOMÉTRICAS\n"
    result_text += "─" * 80 + "\n"
    result_text += f"• Número de triángulos: {triangles}\n\n"
    
    # Información sobre costos
    if edge_costs:
        result_text += "💰 COSTOS DE ARISTAS DEFINIDOS\n"
        result_text += "─" * 80 + "\n"
        total_cost = sum(edge_costs.values())
        avg_cost = total_cost / len(edge_costs) if edge_costs else 0
        result_text += f"• Total de aristas con costo: {len(edge_costs)}\n"
        result_text += f"• Costo total del grafo: {total_cost:.2f}\n"
        result_text += f"• Costo promedio por arista: {avg_cost:.2f}\n\n"
    
    text1.insert("1.0", result_text)
    text1.config(state=tk.DISABLED)
    
    # ============= PESTAÑA 2: MATRIZ DE ADYACENCIA =============
    tab2 = tk.Frame(notebook, bg="#1a1a2e")
    notebook.add(tab2, text="📐 Matriz de Adyacencia")
    
    text2 = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, width=100, height=35,
                                      font=("Consolas", 10), bg="#0f0f1e", fg="#00d4ff",
                                      insertbackground="white")
    text2.pack(fill="both", expand=True, padx=10, pady=10)
    
    matrix_text = "═" * 80 + "\n"
    matrix_text += "                         MATRIZ DE ADYACENCIA\n"
    matrix_text += "═" * 80 + "\n\n"
    
    matrix_text += "📊 REPRESENTACIÓN MATRICIAL\n"
    matrix_text += "─" * 80 + "\n\n"
    
    header = "    "
    for i in range(analyzer.n):
        label = node_label_map.get(i, str(i + 1))
        header += f"{label:>4}"
    matrix_text += header + "\n"
    matrix_text += "    " + "─" * (4 * analyzer.n) + "\n"
    
    for i, row in enumerate(matrix):
        label_i = node_label_map.get(i, str(i + 1))
        matrix_text += f"{label_i:>3} │"
        for val in row:
            matrix_text += f"{val:>4}"
        matrix_text += "\n"
    
    matrix_text += "\n\n"
    matrix_text += "📋 PROPIEDADES DE LA MATRIZ\n"
    matrix_text += "─" * 80 + "\n"
    matrix_text += f"• Dimensión: {analyzer.n} × {analyzer.n}\n"
    matrix_text += f"• Matriz cuadrada: ✓ SÍ\n"
    
    # Análisis de simetría mejorado
    if is_directed:
        matrix_text += f"• Matriz simétrica: ✗ NO\n"
        matrix_text += f"  └─ Grafo DIRIGIDO detectado\n"
        matrix_text += f"  └─ {asymmetric_edges} de {total_edges} aristas son unidireccionales\n"
        
        # Mostrar aristas asimétricas
        if asymmetric_edges > 0:
            matrix_text += f"\n  📌 Aristas con dirección única:\n"
            for i in range(analyzer.n):
                for j in range(i + 1, analyzer.n):
                    if matrix[i][j] != matrix[j][i]:
                        label_i = node_label_map.get(i, str(i + 1))
                        label_j = node_label_map.get(j, str(j + 1))
                        if matrix[i][j] > 0 and matrix[j][i] == 0:
                            matrix_text += f"     • {label_i} → {label_j} (solo ida)\n"
                        elif matrix[j][i] > 0 and matrix[i][j] == 0:
                            matrix_text += f"     • {label_j} → {label_i} (solo ida)\n"
                        else:
                            matrix_text += f"     • {label_i} ↔ {label_j} (diferente peso)\n"
    else:
        matrix_text += f"• Matriz simétrica: ✓ SÍ\n"
        matrix_text += f"  └─ Grafo NO DIRIGIDO (todas las aristas son bidireccionales)\n"
    
    matrix_text += f"\n• Suma de elementos: {int(np.sum(matrix))}\n"
    matrix_text += f"• Traza (diagonal): {int(np.trace(matrix))}\n"
    
    if np.trace(matrix) > 0:
        matrix_text += f"  └─ ⚠ Hay {int(np.trace(matrix))} loops (aristas de un nodo a sí mismo)\n"
    
    matrix_text += "\n\n"
    
    if edges:
        matrix_text += "🔗 LISTA DE ARISTAS"
        if edge_costs:
            matrix_text += " (CON COSTOS)\n"
        else:
            matrix_text += "\n"
        matrix_text += "─" * 80 + "\n"
        
        for i, j in edges:
            label_i = node_label_map.get(i, str(i + 1))
            label_j = node_label_map.get(j, str(j + 1))
            edge_key = tuple(sorted((i, j)))
            
            multiplicity = edge_multiplicity.get(edge_key, 1) if edge_multiplicity else 1
            
            # Información base de la arista
            edge_info = f"• Nodo {label_i} ↔ Nodo {label_j}"
            
            # Agregar multiplicidad si hay aristas paralelas
            if multiplicity > 1:
                edge_info += f" | ×{multiplicity} aristas paralelas"
            
            # Agregar costo si existe
            if edge_costs and edge_key in edge_costs:
                cost = edge_costs[edge_key]
                edge_info += f" | Costo: {cost:.2f}"
            
            matrix_text += edge_info + "\n"
    
    text2.insert("1.0", matrix_text)
    text2.config(state=tk.DISABLED)
    
    # ============= PESTAÑA 3: DIJKSTRA Y CAMINOS =============
    tab3 = tk.Frame(notebook, bg="#1a1a2e")
    notebook.add(tab3, text="🛣 Dijkstra - Camino Óptimo")
    
    frame3 = tk.Frame(tab3, bg="#1a1a2e")
    frame3.pack(fill="both", expand=True, padx=10, pady=10)
    
    control_frame = tk.Frame(frame3, bg="#16213e", pady=10)
    control_frame.pack(fill="x")
    
    tk.Label(control_frame, text="🎯 ALGORITMO DE DIJKSTRA - CAMINO DE MENOR COSTO",
             font=("Arial", 12, "bold"), fg="#00d4ff", bg="#16213e").pack(pady=5)
    
    if edge_costs:
        tk.Label(control_frame, text="✓ Costos personalizados activos - Se usarán en el cálculo",
                font=("Arial", 9), fg="#00ff00", bg="#16213e").pack()
    else:
        tk.Label(control_frame, text="⚠ Sin costos definidos - Se asumirá costo 1 para todas las aristas",
                font=("Arial", 9), fg="#ffa500", bg="#16213e").pack()
    
    selection_frame = tk.Frame(control_frame, bg="#16213e")
    selection_frame.pack(pady=10)
    
    tk.Label(selection_frame, text="Desde nodo:", font=("Arial", 10),
             fg="white", bg="#16213e").grid(row=0, column=0, padx=5)
    
    start_var = tk.StringVar()
    start_combo = ttk.Combobox(selection_frame, textvariable=start_var, 
                               values=[node_label_map.get(i, str(i+1)) for i in range(analyzer.n)],
                               width=10, state="readonly")
    start_combo.grid(row=0, column=1, padx=5)
    if analyzer.n > 0:
        start_combo.current(0)
    
    tk.Label(selection_frame, text="Hasta nodo:", font=("Arial", 10),
             fg="white", bg="#16213e").grid(row=0, column=2, padx=5)
    
    end_var = tk.StringVar()
    end_combo = ttk.Combobox(selection_frame, textvariable=end_var,
                             values=[node_label_map.get(i, str(i+1)) for i in range(analyzer.n)],
                             width=10, state="readonly")
    end_combo.grid(row=0, column=3, padx=5)
    if analyzer.n > 1:
        end_combo.current(1)
    
    text3 = scrolledtext.ScrolledText(frame3, wrap=tk.WORD, width=100, height=25,
                                      font=("Consolas", 10), bg="#0f0f1e", fg="#00ff00",
                                      insertbackground="white")
    text3.pack(fill="both", expand=True, pady=10)
    
    diameter = analyzer.calculate_diameter()
    efficiency = analyzer.calculate_efficiency()
    
    geodesic_text = "═" * 80 + "\n"
    geodesic_text += "              ALGORITMO DE DIJKSTRA - OPTIMIZACIÓN DE RUTAS\n"
    geodesic_text += "═" * 80 + "\n\n"
    
    geodesic_text += "📏 MÉTRICAS GLOBALES DEL GRAFO\n"
    geodesic_text += "─" * 80 + "\n"
    
    if is_connected:
        geodesic_text += f"• Diámetro del grafo: {diameter}\n"
        geodesic_text += "  (Camino más largo entre cualquier par de nodos)\n\n"
        geodesic_text += f"• Eficiencia global: {efficiency:.4f}\n"
        geodesic_text += "  (Promedio de inversos de distancias geodésicas)\n\n"
    else:
        geodesic_text += "• ⚠ El grafo NO es conectado\n"
        geodesic_text += "  No todos los nodos son alcanzables entre sí\n\n"
    
    if edge_costs:
        geodesic_text += "\n💰 TABLA DE COSTOS DE ARISTAS\n"
        geodesic_text += "─" * 80 + "\n"
        for edge_key, cost in sorted(edge_costs.items()):
            i, j = edge_key
            label_i = node_label_map.get(i, str(i + 1))
            label_j = node_label_map.get(j, str(j + 1))
            geodesic_text += f"• {label_i} ↔ {label_j}: {cost:.2f}\n"
        geodesic_text += "\n"
    
    geodesic_text += "\n💡 Seleccione nodos y presione 'Calcular Camino Óptimo' para encontrar\n"
    geodesic_text += "   la ruta de MENOR COSTO usando el algoritmo de Dijkstra\n"
    
    text3.insert("1.0", geodesic_text)
    text3.config(state=tk.DISABLED)
    
    def calculate_dijkstra():
        """Calcula y muestra el camino óptimo con Dijkstra"""
        try:
            start_label = start_var.get()
            end_label = end_var.get()
            
            if not start_label or not end_label:
                messagebox.showwarning("⚠ Selección incompleta", 
                                      "Por favor seleccione nodo inicial y final")
                return
            
            start_idx = None
            end_idx = None
            
            for idx, label in node_label_map.items():
                if label == start_label:
                    start_idx = idx
                if label == end_label:
                    end_idx = idx
            
            if start_idx is None or end_idx is None:
                messagebox.showerror("❌ Error", "No se pudieron encontrar los nodos")
                return
            
            if start_idx == end_idx:
                messagebox.showinfo("ℹ Mismo nodo", 
                                   f"El nodo inicial y final son el mismo: {start_label}")
                return
            
            # Calcular con Dijkstra
            path, total_cost = analyzer.dijkstra(start_idx, end_idx)
            
            # Mostrar resultado
            text3.config(state=tk.NORMAL)
            text3.delete("1.0", tk.END)
            
            result = "═" * 80 + "\n"
            result += "              RESULTADO DEL ALGORITMO DE DIJKSTRA\n"
            result += "═" * 80 + "\n\n"
            
            result += f"🎯 Origen: Nodo {start_label}\n"
            result += f"🎯 Destino: Nodo {end_label}\n\n"
            
            if path:
                result += "✅ CAMINO ÓPTIMO ENCONTRADO\n"
                result += "─" * 80 + "\n"
                result += f"💰 COSTO TOTAL MÍNIMO: {total_cost:.2f}\n"
                result += f"📏 Número de aristas: {len(path) - 1}\n\n"
                
                result += "📍 Secuencia de nodos (ruta óptima):\n\n"
                path_labels = [node_label_map.get(node, str(node+1)) for node in path]
                result += "   " + " → ".join(path_labels) + "\n\n"
                
                result += "🔗 Detalle del recorrido:\n"
                result += "─" * 80 + "\n"
                
                accumulated_cost = 0
                for i in range(len(path) - 1):
                    node_from = path[i]
                    node_to = path[i + 1]
                    label_from = node_label_map.get(node_from, str(node_from+1))
                    label_to = node_label_map.get(node_to, str(node_to+1))
                    
                    edge_key = tuple(sorted((node_from, node_to)))
                    edge_cost = edge_costs.get(edge_key, 1) if edge_costs else 1
                    accumulated_cost += edge_cost
                    
                    result += f"   Paso {i+1}: {label_from} → {label_to}\n"
                    result += f"           Costo: {edge_cost:.2f} | Acumulado: {accumulated_cost:.2f}\n\n"
                
                result += "═" * 80 + "\n"
                result += "🏆 ESTE ES EL CAMINO DE MENOR COSTO POSIBLE\n"
                result += "─" * 80 + "\n"
                result += "El algoritmo de Dijkstra garantiza que este es el camino óptimo\n"
                result += "que minimiza el costo total entre los nodos seleccionados.\n"
                
                if edge_costs:
                    result += "\n💡 Se utilizaron los costos personalizados que definiste.\n"
                else:
                    result += "\n⚠ Se asumió costo 1 para todas las aristas.\n"
                    result += "  Tip: Usa el botón '💰 Definir Costos' para personalizar.\n"
                
            else:
                result += "❌ NO EXISTE CAMINO\n"
                result += "─" * 80 + "\n"
                result += f"No hay una ruta que conecte {start_label} con {end_label}\n"
                result += "Los nodos están en componentes conexas diferentes.\n"
            
            text3.insert("1.0", result)
            text3.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Error al calcular:\n{str(e)}")
    
    tk.Button(control_frame, text="🔍 Calcular Camino Óptimo (Dijkstra)", 
             command=calculate_dijkstra,
             font=("Arial", 11, "bold"), bg="#00d4ff", fg="#1a1a2e",
             padx=20, pady=8, cursor="hand2").pack(pady=10)
    
    # ============= PESTAÑA 4: LISTA DE ADYACENCIA =============
    tab4 = tk.Frame(notebook, bg="#1a1a2e")
    notebook.add(tab4, text="📝 Lista de Adyacencia")
    
    text4 = scrolledtext.ScrolledText(tab4, wrap=tk.WORD, width=100, height=35,
                                      font=("Consolas", 10), bg="#0f0f1e", fg="#ffd700",
                                      insertbackground="white")
    text4.pack(fill="both", expand=True, padx=10, pady=10)
    
    adj_text = "═" * 80 + "\n"
    adj_text += "                         LISTA DE ADYACENCIA\n"
    adj_text += "═" * 80 + "\n\n"
    
    adj_text += "📋 REPRESENTACIÓN POR LISTAS\n"
    adj_text += "─" * 80 + "\n"
    adj_text += "Para cada nodo, se muestran sus nodos adyacentes:\n\n"
    
    for i in range(analyzer.n):
        label_i = node_label_map.get(i, str(i + 1))
        adjacent = []
        
        for j in range(analyzer.n):
            if matrix[i][j] > 0:
                label_j = node_label_map.get(j, str(j + 1))
                for _ in range(int(matrix[i][j])):
                    adjacent.append(label_j)
        
        adj_text += f"Nodo {label_i:>3}: "
        if adjacent:
            adj_text += " → ".join(adjacent) + " → ∅"
        else:
            adj_text += "∅ (nodo aislado)"
        adj_text += "\n"
    
    adj_text += "\n\n"
    adj_text += "💾 EFICIENCIA DE ALMACENAMIENTO\n"
    adj_text += "─" * 80 + "\n"
    
    total_edges_count = int(np.sum(matrix))
    matrix_space = analyzer.n * analyzer.n
    list_space = analyzer.n + total_edges_count
    
    adj_text += f"• Espacio matriz de adyacencia: {matrix_space} entradas\n"
    adj_text += f"• Espacio lista de adyacencia: {list_space} entradas\n"
    adj_text += f"• Ahorro: {((matrix_space - list_space) / matrix_space * 100):.1f}%\n\n"
    
    if total_edges_count < matrix_space / 3:
        adj_text += "✓ Grafo disperso: Lista de adyacencia es más eficiente\n"
    else:
        adj_text += "⚠ Grafo denso: Matriz de adyacencia puede ser más eficiente\n"
    
    text4.insert("1.0", adj_text)
    text4.config(state=tk.DISABLED)


def open_settings():
    """Abre ventana para ajustar parámetros"""
    settings_win = tk.Toplevel(root)
    settings_win.title("⚙ Configuración Avanzada")
    settings_win.geometry("450x650")
    settings_win.configure(bg="#1a1a2e")
    
    tk.Label(settings_win, text="⚙ AJUSTAR PARÁMETROS", font=("Arial", 14, "bold"),
             fg="#00d4ff", bg="#1a1a2e").pack(pady=15)
    
    frame = tk.Frame(settings_win, bg="#1a1a2e", padx=20)
    frame.pack(fill="both", expand=True)
    
    def create_slider(parent, label, var_name, min_val, max_val, default):
        container = tk.Frame(parent, bg="#1a1a2e")
        container.pack(fill="x", pady=8)
        
        lbl = tk.Label(container, text=label, font=("Arial", 10),
                      fg="#e0e0e0", bg="#1a1a2e", anchor="w")
        lbl.pack(fill="x")
        
        value_label = tk.Label(container, text=str(default), font=("Consolas", 10, "bold"),
                              fg="#00d4ff", bg="#1a1a2e")
        value_label.pack(side="right")
        
        slider = tk.Scale(container, from_=min_val, to=max_val, orient="horizontal",
                         bg="#2d2d44", fg="#e0e0e0", highlightthickness=0, troughcolor="#16213e",
                         activebackground="#00d4ff", length=250)
        slider.set(default)
        slider.pack(side="left", fill="x", expand=True)
        
        def update_val(val):
            value_label.config(text=val)
            setattr(config, var_name, int(float(val)))
        
        slider.config(command=update_val)
        return slider
    
    create_slider(frame, "🔵 Radio mínimo (px)", "min_radius", 5, 50, config.min_radius)
    create_slider(frame, "🔵 Radio máximo (px)", "max_radius", 50, 200, config.max_radius)
    create_slider(frame, "📏 Distancia mínima entre nodos", "min_dist", 20, 100, config.min_dist)
    create_slider(frame, "🎯 Sensibilidad Hough (param2)", "hough_param2", 10, 80, config.hough_param2)
    create_slider(frame, "🔗 Distancia máx línea-nodo", "max_endpoint_dist", 30, 150, config.max_endpoint_dist)
    create_slider(frame, "🔴 Tolerancia de rojo", "red_tolerance", 0, 50, config.red_tolerance)
    create_slider(frame, "🎨 Saturación mínima", "red_saturation_min", 50, 200, config.red_saturation_min)
    
    tk.Button(settings_win, text="✓ GUARDAR Y CERRAR", command=settings_win.destroy,
             font=("Arial", 11, "bold"), bg="#00d4ff", fg="#1a1a2e",
             padx=20, pady=8, cursor="hand2").pack(pady=20)


# ================================================================
#                   PROCESAMIENTO PRINCIPAL
# ================================================================
# Variable global para costos
edge_costs_global = {}

def edit_edge_costs(edges, node_label_map):
    """Abre editor de costos de aristas"""
    global edge_costs_global
    
    editor = EdgeCostEditor(root, edges, node_label_map, edge_costs_global)
    root.wait_window(editor.window)
    edge_costs_global = editor.get_costs()
    
    status_label.config(text=f"✓ Costos actualizados - {len(edge_costs_global)} aristas configuradas")
    messagebox.showinfo("✓ Guardado", 
                       f"Se guardaron los costos de {len(edge_costs_global)} aristas.\n\n" +
                       "Estos costos se usarán en el algoritmo de Dijkstra.")


def process_drawing():
    """Procesa el dibujo actual"""
    global edge_costs_global
    
    try:
        img = board.get_image_array()
        original_labels = board.get_node_labels()
        
        status_label.config(text="🔍 Detectando círculos rojos...")
        root.update()
        
        circles = detect_circles_multimethod(img)
        nodes = merge_detections(circles)
        
        if not nodes:
            messagebox.showwarning(
                "⚠ Sin nodos",
                f"No se detectaron nodos ROJOS.\n\n" +
                f"Asegúrate de dibujar círculos ROJOS.\n\n" +
                f"Configuración actual:\n" +
                f"• Radio: {config.min_radius}-{config.max_radius} px\n" +
                f"• Distancia mín: {config.min_dist} px\n" +
                f"• Tolerancia rojo: {config.red_tolerance}\n" +
                f"• Saturación mín: {config.red_saturation_min}\n\n" +
                f"Ajusta parámetros con el botón ⚙"
            )
            status_label.config(text="❌ Sin nodos rojos detectados")
            return
        
        status_label.config(text=f"✓ {len(nodes)} nodos | Detectando aristas...")
        root.update()
        
        mask = create_mask(img.shape, nodes)
        segments = detect_lines(img, mask)
        edges, edge_multiplicity = build_graph(nodes, segments)
        
        node_label_map = {}
        for idx, node in enumerate(nodes):
            x, y = node['x'], node['y']
            
            min_dist = float('inf')
            closest_label = str(idx + 1)
            
            for (lx, ly), label in original_labels.items():
                dist = math.hypot(x - lx, y - ly)
                if dist < min_dist:
                    min_dist = dist
                    closest_label = label
            
            if min_dist < node['r'] * 2:
                node_label_map[idx] = closest_label
            else:
                node_label_map[idx] = str(idx + 1)
        
        matrix = adjacency_matrix(nodes, edges, edge_multiplicity)
        
        result = visualize(img, nodes, edges, original_labels, edge_multiplicity)
        
        cv2.imshow("✅ Grafo Detectado", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Mostrar info sobre aristas paralelas si existen
        parallel_edges = sum(1 for count in edge_multiplicity.values() if count > 1)
        if parallel_edges > 0:
            messagebox.showinfo(
                "🔗 Aristas Paralelas Detectadas",
                f"Se detectaron {parallel_edges} pares de nodos con aristas múltiples.\n\n" +
                "Las aristas paralelas están marcadas con un círculo naranja\n" +
                "que muestra el número de aristas entre esos nodos.\n\n" +
                "Esto indica un MULTIGRAFO (grafo con aristas paralelas)."
            )
        
        # Preguntar si desea definir costos
        if edges:
            response = messagebox.askyesno(
                "💰 Definir Costos",
                f"Se detectaron {len(edges)} aristas.\n\n" +
                "¿Deseas asignar costos personalizados a las aristas?\n\n" +
                "(Útil para el algoritmo de Dijkstra - optimización de rutas)\n\n" +
                "• SÍ: Abrir editor de costos\n" +
                "• NO: Usar costo 1 para todas las aristas"
            )
            
            if response:
                edit_edge_costs(edges, node_label_map)
        
        show_advanced_results(matrix, edges, nodes, original_labels, node_label_map, edge_costs_global, edge_multiplicity)
        
        total_edges_msg = len(edges)
        if edge_multiplicity:
            total_with_mult = sum(edge_multiplicity.values())
            if total_with_mult > len(edges):
                total_edges_msg = f"{len(edges)} únicas ({total_with_mult} con paralelas)"
        
        status_label.config(text=f"✓ {len(nodes)} nodos, {total_edges_msg} aristas - Análisis completo")
        
    except Exception as e:
        messagebox.showerror("❌ Error", f"Error en el procesamiento:\n{str(e)}\n\nVerifique el dibujo.")
        status_label.config(text="❌ Error en procesamiento")
        import traceback
        traceback.print_exc()


# ================================================================
#                    INTERFAZ PRINCIPAL
# ================================================================
root = tk.Tk()
root.title("🎯 Graph Reader Pro v7 - Con Algoritmo de Dijkstra y Costos")
root.configure(bg="#1a1a2e")

toolbar = tk.Frame(root, bg="#16213e", pady=10)
toolbar.pack(fill="x")

tk.Label(toolbar, text="🎨 HERRAMIENTAS:", font=("Arial", 12, "bold"),
         fg="#00d4ff", bg="#16213e").pack(side="left", padx=10)

tool_frame = tk.Frame(toolbar, bg="#16213e")
tool_frame.pack(side="left", padx=20)

current_tool = tk.StringVar(value="node")

def select_tool(tool):
    current_tool.set(tool)
    board.set_tool(tool)
    if tool == "node":
        tool_label.config(text="🔴 Dibujando: NODOS")
    else:
        tool_label.config(text="✏ Dibujando: LÍNEAS")

tk.Radiobutton(tool_frame, text="🔴 Nodos", variable=current_tool, value="node",
              command=lambda: select_tool("node"), font=("Arial", 10),
              bg="#16213e", fg="white", selectcolor="#2d2d44",
              activebackground="#16213e", activeforeground="#00d4ff").pack(side="left", padx=5)

tk.Radiobutton(tool_frame, text="✏ Líneas", variable=current_tool, value="line",
              command=lambda: select_tool("line"), font=("Arial", 10),
              bg="#16213e", fg="white", selectcolor="#2d2d44",
              activebackground="#16213e", activeforeground="#00d4ff").pack(side="left", padx=5)

size_frame = tk.Frame(toolbar, bg="#16213e")
size_frame.pack(side="left", padx=20)

tk.Label(size_frame, text="📏 Tamaño:", font=("Arial", 9),
         fg="white", bg="#16213e").pack(side="left")

node_size = tk.Scale(size_frame, from_=10, to=50, orient="horizontal",
                     command=lambda v: board.set_node_size(int(v)),
                     bg="#2d2d44", fg="white", highlightthickness=0,
                     troughcolor="#16213e", activebackground="#00d4ff",
                     length=100)
node_size.set(20)
node_size.pack(side="left")

action_frame = tk.Frame(toolbar, bg="#16213e")
action_frame.pack(side="right", padx=10)

tk.Button(action_frame, text="🗑 Limpiar", command=lambda: board.clear_canvas(),
         font=("Arial", 10, "bold"), bg="#ff6b6b", fg="white",
         padx=15, pady=5, cursor="hand2").pack(side="left", padx=5)

tk.Button(action_frame, text="⚙ Config", command=open_settings,
         font=("Arial", 10, "bold"), bg="#ffa500", fg="white",
         padx=15, pady=5, cursor="hand2").pack(side="left", padx=5)

tk.Button(action_frame, text="🔍 ANALIZAR GRAFO", command=process_drawing,
         font=("Arial", 10, "bold"), bg="#00d4ff", fg="#1a1a2e",
         padx=20, pady=5, cursor="hand2").pack(side="left", padx=5)

tool_label = tk.Label(root, text="🔴 Dibujando: NODOS",
                     font=("Arial", 10), fg="#00d4ff", bg="#1a1a2e", pady=5)
tool_label.pack()

board_frame = tk.Frame(root, bg="#1a1a2e", padx=10, pady=10)
board_frame.pack()

board = DrawingBoard(board_frame, width=800, height=600)

instructions = tk.Frame(root, bg="#1a1a2e", pady=10)
instructions.pack()

tips = [
    "💡 Selecciona 🔴 NODOS y haz clic para dibujar círculos rojos (vértices del grafo)",
    "💡 Selecciona ✏ LÍNEAS y arrastra para conectar nodos (aristas del grafo)",
    "💡 Presiona 🔍 ANALIZAR y podrás asignar COSTOS para usar Dijkstra",
    "💡 El algoritmo de Dijkstra encontrará el camino de MENOR COSTO entre nodos"
]

for tip in tips:
    tk.Label(instructions, text=tip, font=("Consolas", 8),
            fg="#a8dadc", bg="#1a1a2e").pack()

status_label = tk.Label(root, text="⚡ Listo - Dibuja tu grafo y usa Dijkstra para optimizar rutas",
                       font=("Consolas", 9), fg="#a8dadc",
                       bg="#0f0f1e", pady=10)
status_label.pack(side="bottom", fill="x")

root.mainloop()