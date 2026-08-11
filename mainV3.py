import cv2
import math
import numpy as np

# ==================================================
# CONFIG
# ==================================================

IMAGE_PATHS = [
    "PrimaryHeadset/Primary_Div27_1.1.jpg",
    "SecondaryHeadset/Secondary_Div27_1.1.jpg"
]

IMAGE_NAMES = [
    "PRIMARY",
    "SECONDARY"
]

NUM_SETS = 2
POINTS_PER_SET = 4

CAL_POINTS = 2
TOTAL_POINTS = CAL_POINTS + NUM_SETS * POINTS_PER_SET

WINDOW_NAME = "Scroll=Zoom | RightDrag=Pan | LeftClick=Point | r=reset q=quit"

VIEW_W = 1000
VIEW_H = 800

MIN_SCALE = 0.1
MAX_SCALE = 60.0

CLICK_SEQUENCE = [
    "Nav Window Top-Left Corner",
    "Nav Window Top-Right Corner",

    "Set 1 - Left Needle Bottom (A1)",
    "Set 1 - Left Needle Top (B1)",
    "Set 1 - Left Trajectory Bottom (C1)",
    "Set 1 - Left Trajectory Top (D1)",

    "Set 2 - Right Needle Bottom (A2)",
    "Set 2 - Right Needle Top (B2)",
    "Set 2 - Right Trajectory Bottom (C2)",
    "Set 2 - Right Trajectory Top (D2)",
]

# ==================================================
# GLOBALS
# ==================================================

points = []

scale = 1.0
offset = [0.0, 0.0]

dragging = False
drag_start = None
offset_start = None

# ==================================================
# VIEW FUNCTIONS
# ==================================================

def view_to_original(vx, vy):
    return offset[0] + vx / scale, offset[1] + vy / scale


def clamp_offset(img_w, img_h):
    view_w_orig = VIEW_W / scale
    view_h_orig = VIEW_H / scale

    offset[0] = min(max(offset[0], 0), max(0, img_w - view_w_orig))
    offset[1] = min(max(offset[1], 0), max(0, img_h - view_h_orig))


# ==================================================
# MOUSE
# ==================================================

def mouse_event(event, x, y, flags, param):

    global scale
    global dragging
    global drag_start
    global offset_start

    img_w = param["w"]
    img_h = param["h"]

    if event == cv2.EVENT_LBUTTONDOWN:

        ox, oy = view_to_original(x, y)

        points.append((ox, oy))

        idx = len(points) - 1

        if idx < len(CLICK_SEQUENCE):
            print(f"{CLICK_SEQUENCE[idx]}: ({ox:.1f}, {oy:.1f})")

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

        ox_before, oy_before = view_to_original(x, y)

        if flags > 0:
            scale = min(scale * 1.1, MAX_SCALE)
        else:
            scale = max(scale / 1.1, MIN_SCALE)

        offset[0] = ox_before - x / scale
        offset[1] = oy_before - y / scale

        clamp_offset(img_w, img_h)

# ==================================================
# MATH
# ==================================================

def compare_points(a, b):

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    distance = math.hypot(dx, dy)

    return distance, dx, dy


def make_vector(start, end):

    dx = end[0] - start[0]
    dy = -(end[1] - start[1])

    return np.array([dx, dy], dtype=float)


def angle_between_vectors(v1, v2):

    n1 = v1 / np.linalg.norm(v1)
    n2 = v2 / np.linalg.norm(v2)

    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)

    return math.degrees(math.acos(dot))


def window_reference_size(points):

    top_left = points[0]
    top_right = points[1]

    width = math.hypot(
        top_right[0] - top_left[0],
        top_right[1] - top_left[1]
    )

    return width

# ==================================================
# DRAWING
# ==================================================

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

    interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA

    display = cv2.resize(
        crop,
        (disp_w, disp_h),
        interpolation=interp
    )

    canvas = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    canvas[:disp_h, :disp_w] = display

    colors = {
        "A": (0, 255, 0),
        "B": (0, 255, 255),
        "C": (0, 0, 255),
        "D": (255, 0, 255),
    }

    for i, (px, py) in enumerate(points):

        if i < CAL_POINTS:
            label = "TL" if i == 0 else "TR"
            color = (255, 255, 255)

        else:
            data_idx = i - CAL_POINTS
            role = "ABCD"[data_idx % POINTS_PER_SET]
            set_num = data_idx // POINTS_PER_SET + 1

            label = f"{role}{set_num}"
            color = colors[role]

        if x0 <= px <= x1 and y0 <= py <= y1:

            vx = int((px - x0) * scale)
            vy = int((py - y0) * scale)

            cv2.circle(canvas, (vx, vy), 5, color, -1)

            cv2.putText(
                canvas,
                label,
                (vx + 10, vy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    current_idx = len(points)

    if current_idx < len(CLICK_SEQUENCE):

        cv2.rectangle(
            canvas,
            (0, 0),
            (VIEW_W, 70),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            canvas,
            "CLICK:",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            CLICK_SEQUENCE[current_idx],
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

    cv2.putText(
        canvas,
        f"Points {len(points)}/{TOTAL_POINTS}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return canvas

# ==================================================
# ANALYSIS
# ==================================================

def analyze_points(points):

    window_width = window_reference_size(points)

    print("\nWindow Reference")
    print(f"Window Width: {window_width:.2f}px")

    translations = []
    normalized_translations = []
    angles = []

    for s in range(NUM_SETS):

        base = CAL_POINTS + s * POINTS_PER_SET

        A = points[base]
        B = points[base + 1]
        C = points[base + 2]
        D = points[base + 3]

        trans, dx, dy = compare_points(A, C)

        normalized_pct = (
            trans / window_width * 100.0
        )

        v1 = make_vector(A, B)
        v2 = make_vector(C, D)

        angle = angle_between_vectors(v1, v2)

        translations.append(trans)
        normalized_translations.append(normalized_pct)
        angles.append(angle)

        print(f"\nSet {s+1}")
        print(
            f"Translation = {trans:.2f}px "
            f"(dx={dx:.2f}, dy={dy:.2f})"
        )

        print(
            f"Translation (% Window Width) = "
            f"{normalized_pct:.2f}%"
        )

        print(
            f"Angular Error = {angle:.2f}°"
        )

    return {
        "translations": translations,
        "normalized_translations": normalized_translations,
        "angles": angles,
        "mean_translation": np.mean(translations),
        "mean_normalized_translation": np.mean(normalized_translations),
        "mean_angle": np.mean(angles)
    }

# ==================================================
# MAIN
# ==================================================

def main():

    global scale
    global offset

    all_results = []

    cv2.namedWindow(WINDOW_NAME)

    for image_index, image_path in enumerate(IMAGE_PATHS):

        print("\n" + "=" * 60)
        print(IMAGE_NAMES[image_index])
        print("=" * 60)

        points.clear()

        scale = 1.0
        offset = [0.0, 0.0]

        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(image_path)

        h, w = img.shape[:2]

        cv2.setMouseCallback(
            WINDOW_NAME,
            mouse_event,
            param={"w": w, "h": h}
        )

        print("Follow the instruction banner.")
        print(f"Total clicks required: {TOTAL_POINTS}")

        while True:

            canvas = render(img)

            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                points.clear()
                print("Reset.")

            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return

            elif len(points) >= TOTAL_POINTS:
                break

        results = analyze_points(points)
        all_results.append(results)

    cv2.destroyAllWindows()

    primary = all_results[0]
    secondary = all_results[1]

    print("\n")
    print("=" * 60)
    print("HEADSET COMPARISON")
    print("=" * 60)

    print("\nPRIMARY")
    print(f"Mean Translation: {primary['mean_translation']:.2f}px")
    print(f"Mean Translation (% Window Width): {primary['mean_normalized_translation']:.2f}%")
    print(f"Mean Angle: {primary['mean_angle']:.2f}°")

    print("\nSECONDARY")
    print(f"Mean Translation: {secondary['mean_translation']:.2f}px")
    print(f"Mean Translation (% Window Width): {secondary['mean_normalized_translation']:.2f}%")
    print(f"Mean Angle: {secondary['mean_angle']:.2f}°")

    translation_difference = abs(
        primary["mean_translation"]
        - secondary["mean_translation"]
    )

    normalized_difference = abs(
        primary["mean_normalized_translation"]
        - secondary["mean_normalized_translation"]
    )

    angle_difference = abs(
        primary["mean_angle"]
        - secondary["mean_angle"]
    )

    print("\nCOMPARISON")

    print("MeanTranslationDiffPx")
    print(f"{translation_difference:.2f}")
    print()

    print("MeanTranslationDiffPct")
    print(f"{normalized_difference:.2f}")
    print()

    print("MeanAngleDiffDeg")
    print(f"{angle_difference:.2f}")

    print("\nPer Set Comparison")

    for i in range(NUM_SETS):

        trans_diff = abs(
            primary["translations"][i]
            - secondary["translations"][i]
        )

        norm_diff = abs(
            primary["normalized_translations"][i]
            - secondary["normalized_translations"][i]
        )

        angle_diff = abs(
            primary["angles"][i]
            - secondary["angles"][i]
        )

        print(
            f"Set {i+1}: "
            f"Translation Diff={trans_diff:.2f}px, "
            f"Normalized Diff={norm_diff:.2f}%, "
            f"Angle Diff={angle_diff:.2f}°"
        )


if __name__ == "__main__":
    main()