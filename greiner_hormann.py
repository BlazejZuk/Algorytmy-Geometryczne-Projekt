# Algorytm Greinera-Hormanna
# Zalozenia implementacji:
# 1. Wielokaty musza byc proste (nie moga miec dziur i nie moga przecinac samych siebie)
# 2. Krawedzie wielokatow nie nachodza na siebie,
# 3. Wierzcholek jednego wielokata nie lezy na krawedzi drugiego wielokata,
# 4. Zadna para wierzcholkow nie posiada takiej samej wspolrzednej y


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon
from math import dist

eps = 1e-10

class Vertex:
    """ Klasa reprezentujaca wierzcholki wielokata oraz przeciecia. """
    
    def __init__(self, coords, idx = None):
        self.coords = coords 	# wspolrzedne
        self.idx = idx 			# indeks wierzcholka w oryginalnej liscie reprezentujacej wielokat
        self.prev = None 		# wskaznik na poprzedni wierzcholek
        self.next = None 		# wskaznik na nastepny wierzcholek
        self.intersect = False 	# czy wierzcholek powstal w wyniku przeciecia
        self.entry = False 		# czy wierzcholek jest typu "wejscie"
        self.exit = False 		# czy wierzcholek jest typu "wyjscie"
        self.neighbour = None 	# wskaznik na odpowiedni wierzcholek w drugim wielokacie
        self.alpha = None 		# parametr wynikajacy z funkcji param_intersection
        
        
class Polygon:
    """ Klasa reprezentujaca wielokat jako podwojna liste odsylaczowa obiektow Vertex. """
    
    def __init__(self):
        self.head = None
        
    def append(self, new_vertex):
        if not self.head:
            self.head = new_vertex
            new_vertex.prev = new_vertex
            new_vertex.next = new_vertex
            return
        
        tail = self.head.prev
        
        new_vertex.prev = tail
        new_vertex.next = self.head
        self.head.prev = new_vertex
        tail.next = new_vertex
           
    def insert_after(self, vertex, new_vertex):
        new_vertex.prev = vertex
        new_vertex.next = vertex.next
        vertex.next.prev = new_vertex
        vertex.next = new_vertex
        
              
def intersects(s1, s2):
    """ Funkcja zwraca informacje o tym, czy dwa odcinki sie przecinaja ze soba
        s1, s2 - koordynaty odcinkow """
    
    t, r = param_intersection(s1, s2)
    
    return eps <= t <= 1 - eps and eps <= r <= 1 - eps


def segment_intersection(s1, s2):
    """ Funkcja wyznacza wspolrzedne przeciecia dwoch odcinkow.
        Jesli odcinki sie nie przecinaja, funkcja zwraca none.
        s1, s2 - koordynaty odcinkow """
    
    xa, ya = s1[0]
    xb, yb = s1[1]
    
    t, r = param_intersection(s1, s2)
    
    if t < eps or r < eps or t > 1 - eps or r > 1 - eps:
        return None
    
    x_intersect =  xa + t*(xb - xa)
    y_intersect = ya + t*(yb - ya)
    
    return x_intersect, y_intersect


def param_intersection(s1, s2):
    """ Funkcja wyznacza parametry t i r z ukladu dwoch parametrycznych rownan odcinkow.
        s1, s2 - koordynaty odcinkow """
    
    xa, ya = s1[0]
    xb, yb = s1[1]
    xc, yc = s2[0]
    xd, yd = s2[1]
    
    denominator = (xb - xa)*(yd - yc) - (xd - xc)*(yb - ya)
    
    if abs(denominator) < eps: # jesli odcinki sa rownolegle 
        return 2, 2            # to zakladamy ze sie nie przecinaja
    
    t = ((xc - xa)*(yd - yc) - (xd - xc)*(yc - ya)) / denominator
    r = ((xc - xa)*(yb - ya) - (xb - xa)*(yc - ya)) / denominator
    
    return t, r


def in_polygon(vertex, polygon):
    """ Funkcja zwraca informacje czy wierzcholek znajduje sie wewnatrz wielokata.
        vertex - obiekt klasy Vertex, polygon - obiekt klasy Polygon """
    
    max_x = polygon.head.coords[0]
    polygon_vertex = polygon.head.next
    
    while True:
        x = polygon_vertex.coords[0]
        max_x = max(x, max_x)
        
        polygon_vertex = polygon_vertex.next
        
        if polygon_vertex == polygon.head:
            break
    
    x, y = vertex.coords
    
    if x > max_x:
        return False
    
    ray = ((x, y), (max_x + 1, y))
    
    polygon_vertex = polygon.head
    intersect_count = 0
    
    while True:
        polygon_edge = (polygon_vertex.coords, polygon_vertex.next.coords)
        
        if intersects(ray, polygon_edge):
            intersect_count += 1
            
        polygon_vertex = polygon_vertex.next
        
        if polygon_vertex == polygon.head:
            break
            
    return intersect_count % 2


def find_intersections(subject, clip):
    """ Funkcja znajduje przeciecia dwoch wielokatow.
        subject, clip - obiekty klasy Polygon """
    
    subject_intersections = [[] for _ in range(subject.head.prev.idx + 1)]
    clip_intersections = [[] for _ in range(clip.head.prev.idx + 1)]
    intersect_flag = False
    
    subject_vertex = subject.head
    while True:
        clip_vertex = clip.head
        while True:
            subject_edge = (subject_vertex.coords, subject_vertex.next.coords)
            clip_edge = (clip_vertex.coords, clip_vertex.next.coords)

            if intersects(subject_edge, clip_edge):
                intersect_flag = True
                coords = segment_intersection(subject_edge, clip_edge)
                subject_alpha, clip_alpha = param_intersection(subject_edge, clip_edge)
                
                subject_inter = Vertex(coords)
                subject_inter.intersect = True
                subject_inter.alpha = subject_alpha
                    
                clip_inter = Vertex(coords)
                clip_inter.intersect = True
                clip_inter.alpha = clip_alpha
                    
                subject_inter.neighbour = clip_inter
                clip_inter.neighbour = subject_inter
                
                subject_intersections[subject_vertex.idx].append(subject_inter)
                clip_intersections[clip_vertex.idx].append(clip_inter)
            
            clip_vertex = clip_vertex.next

            if clip_vertex == clip.head:
                break
            
        subject_vertex = subject_vertex.next
                
        if subject_vertex == subject.head:
            break
        
    for idx in range(len(subject_intersections)):
        subject_intersections[idx].sort(key=lambda vertex: vertex.alpha)
        
    for idx in range(len(clip_intersections)):
        clip_intersections[idx].sort(key=lambda vertex: vertex.alpha)
        
    return subject_intersections, clip_intersections, intersect_flag


def insert_intersections(subject, clip):
    """ Funkcja wstawia przeciecia do wielokatow.
        subject, clip - obiekty klasy Polygon """
    
    subject_intersections, clip_intersections, intersect_flag = find_intersections(subject, clip)
    
    subject_vertex = subject.head
    
    while True:
        next_vertex = subject_vertex.next
        idx = subject_vertex.idx
        
        if subject_intersections[idx] is not None:
            for vertex in subject_intersections[idx][::-1]:
                subject.insert_after(subject_vertex, vertex)
        
        subject_vertex = next_vertex
        
        if subject_vertex == subject.head:
            break
    
    clip_vertex = clip.head
    
    while True:
        next_vertex = clip_vertex.next
        idx = clip_vertex.idx
        
        if clip_intersections[idx] is not None:
            for vertex in clip_intersections[idx][::-1]:
                clip.insert_after(clip_vertex, vertex)
        
        clip_vertex = next_vertex
        
        if clip_vertex == clip.head:
            break

    mark_intersections(subject, clip)
    mark_intersections(clip, subject)
    
    return intersect_flag


def mark_intersections(polygon, other_polygon):
    """ Funkcja oznacza przeciecia polygon z other_polygon jako wejscie lub wyjscie.
        polygon, other_polygon - obiekty klasy Polygon """
    
    next_is_entry = True
    
    if in_polygon(polygon.head, other_polygon):
        next_is_entry = False
        
    polygon_vertex = polygon.head
    
    while True:
        if polygon_vertex.intersect:
            if next_is_entry:
                polygon_vertex.entry = True
                
            else:
                polygon_vertex.exit = True
                
            next_is_entry = not next_is_entry
        
        polygon_vertex = polygon_vertex.next
        
        if polygon_vertex == polygon.head:
            break
        

def common_part(subject, clip):
    """ Funkcja zwraca iloczyn dwoch wielokatow.
        subject, clip - obiekty klasy Polygon ze wstawionymi przecieciami """
    
    clipped_polygons = []
    unprocessed = set()
    entry_unprocessed = set()
    
    subject_vertex = subject.head
    while True:
        if subject_vertex.intersect:
            unprocessed.add(subject_vertex)
                
        subject_vertex = subject_vertex.next
        
        if subject_vertex == subject.head:     
            break
    
    while unprocessed:    
        current = unprocessed.pop()       
        clipped_polygons.append(Polygon())
        
        while True:
            if current in unprocessed:
                unprocessed.remove(current)
                
            if current.neighbour in unprocessed:
                unprocessed.remove(current.neighbour)
            
            if current.entry: # wejscie -> idziemy do przodu
                while True:
                    clipped_polygons[-1].append(Vertex(current.coords))
                    current = current.next
                    
                    if current.intersect:
                        break
                    
            else:  # wyjscie -> idziemy do tylu
                while True:
                    clipped_polygons[-1].append(Vertex(current.coords))
                    current = current.prev
                    
                    if current.intersect:
                        break
            
            current = current.neighbour # przelaczamy sie na drugi wielokat
            
            if clipped_polygons[-1].head and dist(current.coords, clipped_polygons[-1].head.coords) < eps:
                break
    
    return clipped_polygons


def union(subject, clip):
    """ Funkcja zwraca sume dwoch wielokatow.
        subject, clip - obiekty klasy Polygon ze wstawionymi przecieciami """
    
    clipped_polygons = []
    unprocessed = set()
    entry_unprocessed = set()
    
    subject_vertex = subject.head
    while True:
        if subject_vertex.intersect:
            unprocessed.add(subject_vertex)
                
        subject_vertex = subject_vertex.next
        
        if subject_vertex == subject.head:     
            break
    
    while unprocessed:    
        current = unprocessed.pop()       
        clipped_polygons.append(Polygon())
        
        while True:
            if current in unprocessed:
                unprocessed.remove(current)
                
            if current.neighbour in unprocessed:
                unprocessed.remove(current.neighbour)
            
            if current.entry: # wejscie -> idziemy do tylu
                while True:
                    clipped_polygons[-1].append(Vertex(current.coords))
                    current = current.prev
                    
                    if current.intersect:
                        break
                    
            else:  # wyjscie -> idziemy do przodu
                while True:
                    clipped_polygons[-1].append(Vertex(current.coords))
                    current = current.next
                    
                    if current.intersect:
                        break
            
            current = current.neighbour # przelaczamy sie na drugi wielokat
            
            if clipped_polygons[-1].head and dist(current.coords, clipped_polygons[-1].head.coords) < eps:
                break
    
    return clipped_polygons


def greiner_hormann(subject, clip):
    """ Funkcja zwraca iloczyn oraz sume dwoch wielokatow.
        subject, clip - obiekty klasy Polygon """
    
    intersect_flag = insert_intersections(subject, clip)
    
    if not intersect_flag:
        if in_polygon(subject.head, clip):
            return [subject], [clip], intersect_flag
        
        elif in_polygon(clip.head, subject):
            return [clip], [subject], intersect_flag
        
        else:
            return [], [subject, clip], intersect_flag
    
    return common_part(subject, clip), union(subject, clip), intersect_flag


def draw_two_polygons(threshold=0.2):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_title("Rysowanie dwóch wielokątow")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    polygons = [[], []]
    current_poly = 0
    drawing_active = True

    lines = []
    lines.append(ax.plot([], [], marker='o', markersize=8, markeredgecolor='k', c='k', linewidth=1.5)[0])

    def on_click(event):
        nonlocal current_poly, drawing_active

        if not drawing_active or event.inaxes != ax:
            return

        x, y = event.xdata, event.ydata
        points = polygons[current_poly]
        line = lines[current_poly]

        if points:
            x0, y0 = points[0]
            distance = dist((x, y), (x0, y0))

            if distance < threshold:
                xs = [p[0] for p in points + [points[0]]]
                ys = [p[1] for p in points + [points[0]]]
                line.set_data(xs, ys)
                line.set_color('red')
                fig.canvas.draw_idle()

                if current_poly == 0:
                    current_poly = 1
                    polygons[1] = []
                    lines.append(ax.plot([], [], marker='o', markersize=8, markeredgecolor='k', c='k', linewidth=1.5)[0])
                    
                else:
                    drawing_active = False

                return

        points.append((float(x + np.random.rand() * eps), float(y + np.random.rand() * eps)))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        line.set_data(xs, ys)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return polygons[0], polygons[1]


def show_common_part(subject, clip, intersection_polygons):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Iloczyn wielokątow")

    for idx, poly in enumerate([subject, clip]):
        poly = polygon_to_list(poly)
        colors = ['k', 'r']
        
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        
        ax.plot(xs, ys, c=colors[idx], linewidth=2)


    for poly in intersection_polygons:
        poly = polygon_to_list(poly)
        
        patch = MplPolygon(
            poly,
            closed=True,
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.4,
            linewidth=2,
        )
        ax.add_patch(patch)

    plt.show()


def show_union(subject, clip, union_polygons):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Suma wielokątow")

    for idx, poly in enumerate([subject, clip]):
        poly = polygon_to_list(poly)
        colors = ['k', 'r']
        
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        
        ax.plot(xs, ys, c=colors[idx], linewidth=2)
    
    
    is_outside = [False for _ in range(len(union_polygons))]
    
    for idx, poly in enumerate(union_polygons):
        outside = True
        
        for other_poly in union_polygons:
            if poly != other_poly and in_polygon(poly.head, other_poly):
                outside = False
                
        if outside:
            is_outside[idx] = True

            patch = MplPolygon(
                polygon_to_list(poly),
                closed=True,
                facecolor='g',
                edgecolor='k',
                alpha=0.4,
                linewidth=2,
            )
            ax.add_patch(patch)

    for idx, poly in enumerate(union_polygons):
        if not is_outside[idx]:      
            patch = MplPolygon(
                polygon_to_list(poly),
                closed=True,
                facecolor='w',
                edgecolor='k',
                alpha=1,
                linewidth=2,
            )
            ax.add_patch(patch)

    plt.show()
    

def common_part_steps(subject, clip):
    starts = []
    v = subject.head
    
    while True:
        if v.intersect:
            starts.append(v)
            
        v = v.next
        
        if v == subject.head:
            break

    visited = set()

    for start in starts:
        if start in visited:
            continue

        path = []
        current = start
        start_coords = start.coords
        current_poly = "subject"

        while True:
            new_poly_flag = True
            visited.add(current)
            visited.add(current.neighbour)
            direction = "next" if current.entry else "prev"

            while True:
                path.append(current.coords)         
                yield ("step", current_poly, current.coords, list(path))
                
                if new_poly_flag:
                    yield ("step", current_poly, current.coords, list(path))
                    new_poly_flag = False
                
                current = getattr(current, direction)
                
                if current.intersect:
                    break

            current = current.neighbour
            current_poly = "clip" if current_poly == "subject" else "subject"

            if dist(current.coords, start_coords) < eps:
                path.append(start_coords)
                yield ("step", current_poly, current.coords, list(path))
                yield ("done", list(path))
                break


def animate_common_part(subject, clip):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Konstruowanie iloczynu")

    subj_pts = polygon_to_list(subject)
    clip_pts = polygon_to_list(clip)

    subject_line, = ax.plot(
        [p[0] for p in subj_pts] + [subj_pts[0][0]],
        [p[1] for p in subj_pts] + [subj_pts[0][1]],
        color="k",
        linewidth=2,
        zorder=1,
    )

    clip_line, = ax.plot(
        [p[0] for p in clip_pts] + [clip_pts[0][0]],
        [p[1] for p in clip_pts] + [clip_pts[0][1]],
        color="orange",
        linewidth=2,
        zorder=1,
    )

    def collect(poly):
        entry, exit_ = [], []
        v = poly.head
        
        while True:
            if v.intersect:
                if v.entry:
                    entry.append(v.coords)
                    
                if v.exit:
                    exit_.append(v.coords)
                    
            v = v.next
            
            if v == poly.head:
                break
            
        return entry, exit_

    sub_entry, sub_exit = collect(subject)
    clip_entry, clip_exit = collect(clip)

    sub_entry_sc = ax.scatter(*zip(*sub_entry), c="green", s=80, alpha=0.3, zorder=3)
    sub_exit_sc  = ax.scatter(*zip(*sub_exit),  c="red",   s=80, alpha=0.3, zorder=3)

    clip_entry_sc = ax.scatter(*zip(*clip_entry), c="green", s=80, alpha=0.3, zorder=3)
    clip_exit_sc  = ax.scatter(*zip(*clip_exit),  c="red",   s=80, alpha=0.3, zorder=3)

    active_line, = ax.plot([], [], c="blue", linewidth=4, zorder=4)

    finished_patches = []

    steps = list(common_part_steps(subject, clip))

    def update(i):
        event = steps[i]

        if event[0] == "step":
            _, poly_id, point, path = event

            active_line.set_data(
                [p[0] for p in path],
                [p[1] for p in path],
            )

            if poly_id == "subject":
                subject_line.set_alpha(1.0)
                clip_line.set_alpha(0.3)

                sub_entry_sc.set_alpha(1.0)
                sub_exit_sc.set_alpha(1.0)
                clip_entry_sc.set_alpha(0.0)
                clip_exit_sc.set_alpha(0.0)

            else:
                clip_line.set_alpha(1.0)
                subject_line.set_alpha(0.2)

                clip_entry_sc.set_alpha(1.0)
                clip_exit_sc.set_alpha(1.0)
                sub_entry_sc.set_alpha(0.0)
                sub_exit_sc.set_alpha(0.0)

        elif event[0] == "done":
            _, closed_path = event
            
            if i == len(steps) - 1:
                subject_line.set_alpha(1.0)
                clip_line.set_alpha(1.0)
                clip_entry_sc.set_alpha(0.0)
                clip_exit_sc.set_alpha(0.0)
                sub_entry_sc.set_alpha(0.0)
                sub_exit_sc.set_alpha(0.0)

            active_line.set_data(
                [p[0] for p in closed_path],
                [p[1] for p in closed_path],
            )

            patch = MplPolygon(
                closed_path,
                closed=True,
                facecolor="tab:blue",
                edgecolor="tab:blue",
                alpha=0.4,
                linewidth=2,
                zorder=2,
            )
            
            ax.add_patch(patch)
            finished_patches.append(patch)

            active_line.set_data([], [])

        return [
            active_line,
            subject_line,
            clip_line,
            sub_entry_sc,
            sub_exit_sc,
            clip_entry_sc,
            clip_exit_sc,
            *finished_patches,
        ]

    animate = FuncAnimation(fig, update, frames=len(steps), interval=500, blit=True, repeat=False)
    
    #animate.save('c_iloczyn.gif', writer='pillow', fps=2) 

    plt.show()


def union_steps(subject, clip):
    starts = []
    v = subject.head
    
    while True:
        if v.intersect:
            starts.append(v)
        v = v.next
        if v == subject.head:
            break

    visited = set()

    for start in starts:
        if start in visited:
            continue

        path = []
        current = start
        start_coords = start.coords
        current_poly = "subject"

        while True:
            new_poly_flag = True
            visited.add(current)
            visited.add(current.neighbour)

            direction = "prev" if current.entry else "next"

            while True:
                path.append(current.coords)
                yield ("step", current_poly, current.coords, list(path))
                
                if new_poly_flag:
                    yield ("step", current_poly, current.coords, list(path))
                    new_poly_flag = False

                current = getattr(current, direction)
                if current.intersect:
                    break

            current = current.neighbour
            current_poly = "clip" if current_poly == "subject" else "subject"

            if dist(current.coords, start_coords) < eps:
                path.append(start_coords)
                yield ("step", current_poly, current.coords, list(path))
                yield ("done", list(path))
                break


def animate_union(subject, clip):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Konstruowanie sumy")

    subj_pts = polygon_to_list(subject)
    clip_pts = polygon_to_list(clip)

    subject_line, = ax.plot(
        [p[0] for p in subj_pts] + [subj_pts[0][0]],
        [p[1] for p in subj_pts] + [subj_pts[0][1]],
        color="k", linewidth=2, zorder=1
    )

    clip_line, = ax.plot(
        [p[0] for p in clip_pts] + [clip_pts[0][0]],
        [p[1] for p in clip_pts] + [clip_pts[0][1]],
        color="orange", linewidth=2, zorder=1
    )

    def collect(poly):
        entry, exit_ = [], []
        v = poly.head
        while True:
            if v.intersect:
                if v.entry:
                    entry.append(v.coords)
                if v.exit:
                    exit_.append(v.coords)
            v = v.next
            if v == poly.head:
                break
        return entry, exit_

    sub_entry, sub_exit = collect(subject)
    clip_entry, clip_exit = collect(clip)

    sub_entry_sc = ax.scatter(*zip(*sub_entry), c="green", s=80, alpha=0.3, zorder=3)
    sub_exit_sc  = ax.scatter(*zip(*sub_exit),  c="red",   s=80, alpha=0.3, zorder=3)
    clip_entry_sc = ax.scatter(*zip(*clip_entry), c="green", s=80, alpha=0.3, zorder=3)
    clip_exit_sc  = ax.scatter(*zip(*clip_exit),  c="red",   s=80, alpha=0.3, zorder=3)

    active_line, = ax.plot([], [], c="blue", linewidth=4, zorder=4)

    finished_lines = []
    finished_paths = []

    steps = list(union_steps(subject, clip))
    done_events = [e for e in steps if e[0] == "done"]
    polygon_count = len(done_events)
    
    fill_queue = []
    fill_patches = []

    def update(frame):
        nonlocal fill_queue

        if frame < len(steps):
            event = steps[frame]
            kind = event[0]

            if kind == "step":
                _, poly_id, _, path = event

                active_line.set_data(
                    [p[0] for p in path],
                    [p[1] for p in path],
                )

                if poly_id == "subject":
                    subject_line.set_alpha(1.0)
                    clip_line.set_alpha(0.2)
                    sub_entry_sc.set_alpha(1.0)
                    sub_exit_sc.set_alpha(1.0)
                    clip_entry_sc.set_alpha(0.0)
                    clip_exit_sc.set_alpha(0.0)
                    
                else:
                    clip_line.set_alpha(1.0)
                    subject_line.set_alpha(0.2)
                    clip_entry_sc.set_alpha(1.0)
                    clip_exit_sc.set_alpha(1.0)
                    sub_entry_sc.set_alpha(0.0)
                    sub_exit_sc.set_alpha(0.0)

            elif kind == "done":
                closed_path = event[-1]

                xs = [p[0] for p in closed_path]
                ys = [p[1] for p in closed_path]

                line, = ax.plot(xs, ys, c="blue", linewidth=4, zorder=4)
                finished_lines.append(line)
                finished_paths.append(closed_path)

                active_line.set_data([], [])

                if frame == len(steps) - 1:
                    subject_line.set_alpha(1.0)
                    clip_line.set_alpha(1.0)
                    sub_entry_sc.set_alpha(0.0)
                    sub_exit_sc.set_alpha(0.0)
                    clip_entry_sc.set_alpha(0.0)
                    clip_exit_sc.set_alpha(0.0)

                    polys = [list_to_polygon(p) for p in finished_paths]
                    is_outside = [True] * len(polys)

                    for i, p in enumerate(polys):
                        for j, q in enumerate(polys):
                            if i != j and in_polygon(p.head, q):
                                is_outside[i] = False

                    fill_queue = (
                        [(poly, True) for poly, out in zip(polys, is_outside) if out] +
                        [(poly, False) for poly, out in zip(polys, is_outside) if not out]
                    )

        else:
            fill_idx = frame - len(steps)
            
            if fill_idx < len(fill_queue):
                poly, is_outside = fill_queue[fill_idx]

                patch = MplPolygon(
                    polygon_to_list(poly),
                    closed=True,
                    facecolor="green" if is_outside else "white",
                    edgecolor="k",
                    alpha=0.4 if is_outside else 1.0,
                    linewidth=2,
                    zorder=2 if is_outside else 3,
                )
                ax.add_patch(patch)
                fill_patches.append(patch)

        return []
    
    animate = FuncAnimation(fig, update, frames=len(steps) + polygon_count, interval=500, blit=False, repeat=False)
    
    #animate.save('c_suma.gif', writer='pillow', fps=2) 

    plt.show()



def mark_intersections_steps(polygon, other_polygon):
    next_is_entry = True
    
    if in_polygon(polygon.head, other_polygon):
        next_is_entry = False

    v = polygon.head

    while True:
        yield ("visit", v, next_is_entry)

        if v.intersect:
            yield ("mark", v, next_is_entry)
            next_is_entry = not next_is_entry

        v = v.next
        
        if v == polygon.head:
            yield ("visit", v, next_is_entry)
            break


def animate_entry_exit_marking(polygon, other_polygon, colors = ["k", "orange"]):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Animacja wejście/wyjście")

    for idx, poly in enumerate([polygon, other_polygon]):
        pts = polygon_to_list(poly)
        ax.plot(
            [p[0] for p in pts] + [pts[0][0]],
            [p[1] for p in pts] + [pts[0][1]],
            color=colors[idx],
            linewidth=2,
        )

    inter_coords = []
    v = polygon.head
    
    while True:
        if v.intersect:
            inter_coords.append(v.coords)
            
        v = v.next
        
        if v == polygon.head:
            break

    inter_scatter = ax.scatter(
        [p[0] for p in inter_coords],
        [p[1] for p in inter_coords],
        c="gray",
        s=80,
        zorder=4,
    )

    active_dot = ax.scatter([], [], c="blue", s=120, zorder=6)

    state_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="white", ec="black"),
    )

    steps = list(mark_intersections_steps(polygon, other_polygon))

    entry_pts = []
    exit_pts = []
    
    marked_scatter = ax.scatter(
        [p[0] for p in inter_coords],
        [p[1] for p in inter_coords],
        c="gray",
        s=80,
        zorder=4,
    )
    
    def update(i):
        kind, v, flag = steps[i]
        active_dot.set_offsets([v.coords])

        if kind == "visit":
            state_text.set_text(
                f"Następne przecięcie: {'WEJŚCIE' if flag else 'WYJŚCIE'}"
            )

        elif kind == "mark":
            if flag:
                entry_pts.append(v.coords)
                
            else:
                exit_pts.append(v.coords)

            marked_scatter.set_offsets(entry_pts + exit_pts)
            marked_scatter.set_color(["green"] * len(entry_pts) + ["red"] * len(exit_pts))

        return [active_dot, marked_scatter, state_text]

    
    animate = FuncAnimation(fig, update, frames=len(steps), interval=500, blit=True, repeat=False)
    
    #animate.save('c_przeciecia_czarny.gif', writer='pillow', fps=2) 
    #animate.save('c_przeciecia_pomaranczowy.gif', writer='pillow', fps=2) 

    plt.show()


def list_to_polygon(points):
    """ Funkcja konwertuje liste wspolrzednych na obiekt klasy Polygon """
    
    polygon = Polygon()
    
    for idx, coords in enumerate(points):
        vertex = Vertex(coords, idx)
        polygon.append(vertex)
        
    return polygon


def polygon_to_list(polygon):
    """ Funkcja konwertuje obiekt klasy Polygon na liste wspolrzednych """
    
    points = []
    polygon_vertex = polygon.head
    
    while True:
        points.append((float(polygon_vertex.coords[0]), float(polygon_vertex.coords[1])))
        polygon_vertex = polygon_vertex.next
        
        if polygon_vertex == polygon.head:
            break
        
    return points





"""
# Para A
polygon1 = [(3.4379509379509376, 7.388167388167387), (2.0815295815295816, 6.83982683982684),
            (1.331168831168831, 5.093795093795094), (2.0815295815295816, 3.823953823953824),
            (3.7554112554112558, 3.3910533910533913), (4.866522366522366, 3.93939393939394),
            (5.4004329004329, 4.8484848484848495), (5.357142857142858, 6.34920634920635)]

polygon2 = [(6.352813852813853, 7.619047619047619), (4.765512265512266, 7.330447330447329),
            (3.697691197691198, 6.522366522366523), (3.207070707070707, 5.252525252525253),
            (3.048340548340548, 4.112554112554113), (3.5822510822510822, 2.7994227994227994),
            (5.616883116883116, 2.943722943722944), (7.2186147186147185, 3.6075036075036073),
            (7.70923520923521, 5.079365079365079), (7.406204906204907, 6.464646464646465)]
"""

"""
# Para B
polygon1 = [(9.080086580119268, 4.689754689833734), (8.560606060631665, 6.5800865801762765),
            (7.305194805284093, 7.922077922132967), (6.4249639250170665, 6.551226551233327),
            (6.3528138528344265, 4.761904761978447), (5.530303030310518, 4.848484848580928),
            (5.227272727371828, 6.796536796544711), (4.34704184711288, 7.965367965378032),
            (3.4668109668546654, 6.78210678218855), (3.6399711400445343, 4.8917748918153405),
            (3.0339105340030557, 4.877344877355662), (2.8030303031086676, 6.103896103971944),
            (1.8939393939774043, 7.662337662415092), (1.071428571463015, 6.450216450306445),
            (0.9415584416260855, 4.632034632129706), (4.707792207801157, 1.0678210678629023)]
polygon2 = [(9.527417027500592, 6.839826839856453), (7.435064935080428, 7.417027417083781),
            (4.34704184713914, 7.1861471861868464), (1.9372294373172128, 7.330447330496156),
            (0.45093795097747297, 6.839826839898752), (0.42207792214281536, 4.920634920652418),
            (1.3744588744821622, 4.992784992874971), (2.240259740274676, 5.58441558445221),
            (4.015151515211267, 5.84415584423453), (4.722222222251424, 5.353535353552877),
            (6.771284271300438, 5.800865800900771), (8.041125541137252, 5.151515151609018),
            (9.715007215034385, 5.526695526716456)]
"""

"""
# Para C
polygon1 = [(4.981962482035613, 2.33766233771644), (8.344155844220046, 5.367965368057533),
            (4.794372294439524, 8.268398268443319), (1.5187590188256843, 5.367965368020028)]

polygon2 = [(1.850649350668481, 3.679653679710186), (7.319624819658782, 3.3189033189137036),
            (7.795815295815422, 7.3593073593242), (2.269119769163218, 7.878787878845304)]
"""

"""
# Para D
polygon1 = [(7.507215007232987, 6.681096681104719), (5.400432900485161, 7.792207792269088),
            (3.654401154471501, 8.037518037535524), (2.1969696970141546, 7.301587301602631),
            (1.6919191919213465, 4.935064935117471), (3.322510822601254, 2.366522366550667),
            (6.857864357894077, 3.347763347837203)]
polygon2 = [(3.538961039052852, 3.838383838448931), (3.3513708513798504, 6.3492063492381865),
            (5.905483405560024, 5.310245310310318)]
"""

"""
# Para E
polygon1 = [(3.1637806638181765, 7.056277056317767), (1.5620490620538632, 7.113997114029844),
            (0.8694083694537182, 5.050505050565012), (1.5331890332577636, 3.1168831169159916),
            (3.6399711400517667, 2.8571428571717354), (5.111832611887897, 3.823953824030227),
            (5.602453102469577, 5.122655122725863), (4.3037518037914575, 6.608946608995056)]
polygon2 = [(8.546176046273725, 6.493506493590053), (7.492784992824779, 6.26262626264998),
            (6.136363636420547, 4.819624819645282), (6.237373737471451, 3.2900432900762535),
            (6.944444444483615, 2.1500721501007107), (8.676046176078364, 3.246753246779776),
            (9.253246753280836, 4.502164502189605), (8.921356421378189, 6.075036075102898)]
"""


polygon1, polygon2 = draw_two_polygons()
print("Wielokat 1: ")
print(polygon1)
print("Wielokat 2: ")
print(polygon2)

subject = list_to_polygon(polygon1)
clip = list_to_polygon(polygon2)
intersection_polygons, union_polygons, intersect_flag = greiner_hormann(subject, clip)

print("Iloczyn: ")
for p in intersection_polygons:
    print(polygon_to_list(p))

print("Suma: ")
for p in union_polygons:
    print(polygon_to_list(p))

show_common_part(subject, clip, intersection_polygons)
show_union(subject, clip, union_polygons)

if intersect_flag:
    animate_entry_exit_marking(subject, clip, colors = ["k", "orange"])
    animate_entry_exit_marking(clip, subject, colors = ["orange", "k"])
    animate_common_part(subject, clip)
    animate_union(subject, clip)



