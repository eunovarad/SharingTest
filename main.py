import cv2
import math
import numpy as np

# ---- CONFIG ----
IMAGE_PATH = "PrimaryHeadset/Primary_1.jpg"   # change this
NUM_SETS = 2          # how many (A,B,C,D) vector-angle measurements you want
POINTS_PER_SET = 4     # A, B, C, D -> vector AB, vector CD
WINDOW_NAME = "Zoom: scroll | Pan: right-drag | Click: left-click | r=reset q=done"
VIEW_W, VIEW_H = 1000, 800       # size of the display window
MIN_SCALE, MAX_SCALE = 0.1, 60.0  # bumped up max zoom

points = []          # collected points, in ORIGINAL image coordinates
scale = 1.0           # current zoom level
offset = [0.0, 0.0]   # top-left of the view, in ORIGINAL image coordinates
dragging = False
drag_start = None
offset_start = None

def view_to_original(vx, vy):
    return offset[0] + vx / scale, offset[1] + vy / scale

def clamp_offset(img_w, img_h):
    view_w_orig = VIEW_W / scale
    view_h_orig = VIEW_H / scale
    offset[0] = min(max(offset[0], 0), max(0, img_w - view_w_orig))
    offset[1] = min(max(offset[1], 0), max(0, img_h - view_h_orig))

def mouse_event(event, x, y, flags, param):
    global scale, dragging, drag_start, offset_start
    img_w, img_h = param["w"], param["h"]

    if event == cv2.EVENT_LBUTTONDOWN:
        ox, oy = view_to_original(x, y)
        points.append((ox, oy))
        idx = len(points) - 1
        role = "ABCD"[idx % POINTS_PER_SET]
        set_num = idx // POINTS_PER_SET + 1
        print(f"Point {role}{set_num}: ({ox:.1f}, {oy:.1f})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        dragging = True
        drag_start = (x, y)
        offset_start = (offset[0], offset[1])

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = (x - drag_start[0]) / scale
        dy = (y - drag_start[1]) / scale
        offset[0] = offset_start[0] - dx
        offset[1] = offset_start[1] - dy
        clamp_offset(img_w, img_h)

    elif event == cv2.EVENT_RBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        # zoom in/out centered on cursor position
        ox_before, oy_before = view_to_original(x, y)
        if flags > 0:
            scale = min(scale * 1.1, MAX_SCALE)   # finer zoom step
        else:
            scale = max(scale / 1.1, MIN_SCALE)
        # keep the point under the cursor fixed
        offset[0] = ox_before - x / scale
        offset[1] = oy_before - y / scale
        clamp_offset(img_w, img_h)

def compare_points(a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    distance = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(-dy, dx)) % 360
    return distance, angle

def angle_difference(a1, a2):
    return (a1 - a2 + 180) % 360 - 180

def make_vector(p_start, p_end):
    # image y grows downward, flip to standard math orientation
    dx = p_end[0] - p_start[0]
    dy = -(p_end[1] - p_start[1])
    return np.array([dx, dy], dtype=float)

def vector_angle_deg(v):
    # absolute orientation of a single vector, 0-360
    return math.degrees(math.atan2(v[1], v[0])) % 360

def angle_between_vectors(v1, v2):
    """
    Normalize both vectors and take the dot product to get the
    unsigned angle between them (0-180 deg).
    """
    n1 = v1 / np.linalg.norm(v1)
    n2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return math.degrees(math.acos(dot))

def render(img):
    img_h, img_w = img.shape[:2]
    clamp_offset(img_w, img_h)

    x0, y0 = offset
    view_w_orig = VIEW_W / scale
    view_h_orig = VIEW_H / scale
    x1 = min(img_w, x0 + view_w_orig)
    y1 = min(img_h, y0 + view_h_orig)

    crop = img[int(y0):int(y1), int(x0):int(x1)]
    disp_w = max(1, int((x1 - x0) * scale))
    disp_h = max(1, int((y1 - y0) * scale))

    # crisp pixels when zoomed in, smooth when zoomed out
    interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
    display = cv2.resize(crop, (disp_w, disp_h), interpolation=interp)

    canvas = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    canvas[:disp_h, :disp_w] = display

    # draw points that fall within the current view
    labels = ["A", "B", "C", "D"]
    colors = {
        "A": (0, 255, 0),     # green
        "B": (0, 200, 255),   # yellow-orange
        "C": (0, 0, 255),     # red
        "D": (255, 0, 255),   # magenta
    }
    for i, (px, py) in enumerate(points):
        role = labels[i % POINTS_PER_SET]
        set_num = i // POINTS_PER_SET + 1
        if x0 <= px <= x1 and y0 <= py <= y1:
            vx = int((px - x0) * scale)
            vy = int((py - y0) * scale)
            color = colors[role]
            cv2.circle(canvas, (vx, vy), 5, color, -1)
            cv2.putText(canvas, f"{role}{set_num}", (vx + 8, vy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # connect A->B and C->D within the same set
            if role in ("B", "D"):
                prev_px, prev_py = points[i - 1]
                if x0 <= prev_px <= x1 and y0 <= prev_py <= y1:
                    pvx = int((prev_px - x0) * scale)
                    pvy = int((prev_py - y0) * scale)
                    cv2.line(canvas, (pvx, pvy), (vx, vy), (255, 255, 0), 1)

    cv2.putText(canvas, f"zoom: {scale:.2f}x  points: {len(points)}/{NUM_SETS*POINTS_PER_SET}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
    img_h, img_w = img.shape[:2]

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, param={"w": img_w, "h": img_h})

    total_points = NUM_SETS * POINTS_PER_SET
    print("For each set, click 4 points in order: A, B, C, D")
    print("  -> A vs C = pixel translation error")
    print("  -> vector 1 = A to B, vector 2 = C to D -> angle between them")
    print(f"You need {NUM_SETS} sets ({total_points} points total).")
    print("Scroll = zoom (centered on cursor), right-click-drag = pan")
    print("Press 'r' to reset, 'q'/ESC when done.")

    while True:
        canvas = render(img)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            points.clear()
            print("Reset.")
        elif key in (ord('q'), 27):
            break
        elif len(points) >= total_points:
            break

    cv2.destroyAllWindows()

    if len(points) < total_points:
        print(f"Only {len(points)} points collected, need {total_points}. Exiting.")
        return

    print("\n--- Results ---")
    set_angles = []          # angle between v1 (A->B) and v2 (C->D), per set
    v1_orientations = []     # absolute orientation of A->B, per set
    v2_orientations = []     # absolute orientation of C->D, per set
    ac_distances = []        # pixel translation error: A to C, per set

    for s in range(NUM_SETS):
        base = s * POINTS_PER_SET
        A, B, C, D = points[base], points[base + 1], points[base + 2], points[base + 3]

        v1 = make_vector(A, B)
        v2 = make_vector(C, D)

        angle_between = angle_between_vectors(v1, v2)
        set_angles.append(angle_between)
        v1_orientations.append(vector_angle_deg(v1))
        v2_orientations.append(vector_angle_deg(v2))

        ac_dist, _ = compare_points(A, C)
        ac_distances.append(ac_dist)

        print(f"Set {s+1}: A=({A[0]:.1f},{A[1]:.1f}) B=({B[0]:.1f},{B[1]:.1f}) "
              f"C=({C[0]:.1f},{C[1]:.1f}) D=({D[0]:.1f},{D[1]:.1f})")
        print(f"  pixel translation error (A to C): {ac_dist:.2f}px")
        print(f"  vector AB orientation: {v1_orientations[-1]:.2f}°")
        print(f"  vector CD orientation: {v2_orientations[-1]:.2f}°")
        print(f"  angle between AB and CD (normalize + dot product): {angle_between:.2f}°")

    if NUM_SETS >= 2:
        print("\n--- Offsets between sets ---")
        for i in range(NUM_SETS):
            for j in range(i + 1, NUM_SETS):
                angle_offset = set_angles[j] - set_angles[i]
                dist_offset = ac_distances[j] - ac_distances[i]
                print(f"Set {i+1} vs Set {j+1}: "
                      f"angle-between differs by {angle_offset:.2f}° "
                      f"(Set {i+1}={set_angles[i]:.2f}°, Set {j+1}={set_angles[j]:.2f}°); "
                      f"A-C pixel distance differs by {dist_offset:.2f}px "
                      f"(Set {i+1}={ac_distances[i]:.2f}px, Set {j+1}={ac_distances[j]:.2f}px)")

    print("\n--- Summary ---")
    print(f"Mean angle-between (AB vs CD) across sets: {np.mean(set_angles):.2f}°")
    print(f"Std angle-between across sets:              {np.std(set_angles):.2f}°")
    print(f"Mean A-C pixel translation error:           {np.mean(ac_distances):.2f}px")
    print(f"Std A-C pixel translation error:             {np.std(ac_distances):.2f}px")

if __name__ == "__main__":
    main()