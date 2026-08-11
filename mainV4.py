import cv2
import math
import numpy as np

# ==================================================
# CONFIG
# ==================================================

IMAGE_PATHS = [
    "PrimaryHeadsetPart2/P5.2.1.jpg",
    "SecondaryHeadsetPart2/S5.2.1.jpg"
]

IMAGE_NAMES = [
    "PRIMARY",
    "SECONDARY"
]

SCREW_LENGTH_MM = 10.0

WINDOW_NAME = "Scroll=Zoom | RightDrag=Pan | LeftClick=Point | r=reset q=quit"

VIEW_W = 1000
VIEW_H = 800

MIN_SCALE = 0.1
MAX_SCALE = 60.0

TOTAL_POINTS = 8

CLICK_SEQUENCE = [
    "Left Screw Bottom",
    "Left Screw Top",
    "Left Needle Bottom",
    "Left Needle Top",

    "Right Screw Bottom",
    "Right Screw Top",
    "Right Needle Bottom",
    "Right Needle Top",
]

POINT_LABELS = [
    "LSB", "LST",
    "LNB", "LNT",

    "RSB", "RST",
    "RNB", "RNT"
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

    offset[0] = min(
        max(offset[0], 0),
        max(0, img_w - view_w_orig)
    )

    offset[1] = min(
        max(offset[1], 0),
        max(0, img_h - view_h_orig)
    )

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
            print(
                f"{CLICK_SEQUENCE[idx]}: "
                f"({ox:.1f}, {oy:.1f})"
            )

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

def distance(a, b):

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    return math.hypot(dx, dy)


def make_vector(start, end):

    dx = end[0] - start[0]
    dy = -(end[1] - start[1])

    return np.array([dx, dy], dtype=float)


def angle_between_vectors(v1, v2):

    n1 = v1 / np.linalg.norm(v1)
    n2 = v2 / np.linalg.norm(v2)

    dot = np.clip(
        np.dot(n1, n2),
        -1.0,
        1.0
    )

    return math.degrees(math.acos(dot))

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

    interp = (
        cv2.INTER_NEAREST
        if scale >= 1.0
        else cv2.INTER_AREA
    )

    display = cv2.resize(
        crop,
        (disp_w, disp_h),
        interpolation=interp
    )

    canvas = np.zeros(
        (VIEW_H, VIEW_W, 3),
        dtype=np.uint8
    )

    canvas[:disp_h, :disp_w] = display

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

    for idx, (px, py) in enumerate(points):

        if x0 <= px <= x1 and y0 <= py <= y1:

            vx = int((px - x0) * scale)
            vy = int((py - y0) * scale)

            cv2.circle(
                canvas,
                (vx, vy),
                5,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                canvas,
                POINT_LABELS[idx],
                (vx + 8, vy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                2
            )

    return canvas

# ==================================================
# ANALYSIS
# ==================================================

def analyze_image(points):

    results = []

    # Left window starts at index 0
    # Right window starts at index 4
    for side in [0, 4]:

        screw_bottom = points[side + 0]
        screw_top = points[side + 1]

        needle_bottom = points[side + 2]
        needle_top = points[side + 3]

        screw_pixels = distance(
            screw_bottom,
            screw_top
        )

        px_per_mm = (
            screw_pixels
            / SCREW_LENGTH_MM
        )

        translation_pixels = distance(
            screw_bottom,
            needle_bottom
        )

        translation_mm = (
            translation_pixels
            / px_per_mm
        )

        trajectory_vector = make_vector(
            screw_bottom,
            screw_top
        )

        needle_vector = make_vector(
            needle_bottom,
            needle_top
        )

        angle_error = angle_between_vectors(
            trajectory_vector,
            needle_vector
        )

        results.append({
            "screw_pixels": screw_pixels,
            "px_per_mm": px_per_mm,
            "translation_mm": translation_mm,
            "angle_deg": angle_error
        })

    return results

# ==================================================
# MAIN
# ==================================================

def main():

    global scale
    global offset

    all_results = []

    cv2.namedWindow(WINDOW_NAME)

    for image_idx, image_path in enumerate(IMAGE_PATHS):

        print("\n" + "=" * 60)
        print(IMAGE_NAMES[image_idx])
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

        while True:

            canvas = render(img)

            cv2.imshow(
                WINDOW_NAME,
                canvas
            )

            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):

                points.clear()
                print("Reset.")

            elif key in [ord("q"), 27]:

                cv2.destroyAllWindows()
                return

            elif len(points) >= TOTAL_POINTS:

                break

        results = analyze_image(points)

        print("\nResults")

        print(
            f"Left View: "
            f"{results[0]['translation_mm']:.3f} mm, "
            f"{results[0]['angle_deg']:.3f} deg"
        )

        print(
            f"Right View: "
            f"{results[1]['translation_mm']:.3f} mm, "
            f"{results[1]['angle_deg']:.3f} deg"
        )

        all_results.append(results)

    cv2.destroyAllWindows()

    primary = all_results[0]
    secondary = all_results[1]

    left_translation_diff = abs(
        primary[0]["translation_mm"]
        - secondary[0]["translation_mm"]
    )

    right_translation_diff = abs(
        primary[1]["translation_mm"]
        - secondary[1]["translation_mm"]
    )

    left_angle_diff = abs(
        primary[0]["angle_deg"]
        - secondary[0]["angle_deg"]
    )

    right_angle_diff = abs(
        primary[1]["angle_deg"]
        - secondary[1]["angle_deg"]
    )

    mean_translation_diff = (
        left_translation_diff +
        right_translation_diff
    ) / 2.0

    mean_angle_diff = (
        left_angle_diff +
        right_angle_diff
    ) / 2.0

    print("\n")
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print("\nLeftTranslationDiffMm")
    print(f"{left_translation_diff:.3f}")

    print("\nLeftAngleDiffDeg")
    print(f"{left_angle_diff:.3f}")

    print("\nRightTranslationDiffMm")
    print(f"{right_translation_diff:.3f}")

    print("\nRightAngleDiffDeg")
    print(f"{right_angle_diff:.3f}")

    print("\nMeanTranslationDiffMm")
    print(f"{mean_translation_diff:.3f}")

    print("\nMeanAngleDiffDeg")
    print(f"{mean_angle_diff:.3f}")

    print("\nCSV_OUTPUT")
    print(
        f"{left_translation_diff:.3f},"
        f"{left_angle_diff:.3f},"
        f"{right_translation_diff:.3f},"
        f"{right_angle_diff:.3f},"
        f"{mean_translation_diff:.3f},"
        f"{mean_angle_diff:.3f}"
    )


if __name__ == "__main__":
    main()