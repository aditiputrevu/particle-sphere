import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 1280
HEIGHT = 720
NUM_PARTICLES = 3000

CAMERA_INDEX = 0

PINCH_THRESHOLD = 0.55
CHARGE_TIME = 1.4

BACKGROUND_DARKNESS = 0.65


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def hsv_to_bgr(hue):
    """
    Convert a hue value from 0-360 into an OpenCV BGR color.
    """
    hue = int(hue % 360)

    hsv = np.uint8([[
        [
            int(hue / 2),
            220,
            255
        ]
    ]])

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return tuple(int(x) for x in bgr[0][0])


# ============================================================
# PARTICLE SPHERE
# ============================================================

class ParticleSphere:

    def __init__(self, count):

        self.count = count

        self.points = []

        # Fibonacci sphere distribution
        golden_angle = 2.399963229728653

        for i in range(count):

            y = 1 - 2 * (i + 0.5) / count

            radius = math.sqrt(max(0, 1 - y * y))

            angle = i * golden_angle

            x = math.cos(angle) * radius
            z = math.sin(angle) * radius

            self.points.append({
                "x": x,
                "y": y,
                "z": z,

                "phase": random.random() * math.pi * 2,
                "grain": random.random(),

                "burst": 0,

                "vx": 0,
                "vy": 0,
                "vz": 0
            })

        # Sphere position
        self.center_x = WIDTH / 2
        self.center_y = HEIGHT / 2

        self.target_x = WIDTH / 2
        self.target_y = HEIGHT / 2

        # Rotation
        self.rotation_x = 0.2
        self.rotation_y = 0

        self.target_rotation_x = 0.2
        self.target_rotation_y = 0

        # Scale
        self.scale = 1
        self.target_scale = 1

        # Color
        self.hue = 185

        # Explosion
        self.explosion_time = 0
        self.exploding = False
    # --------------------------------------------------------

    def explode(self, power=1.0):

        print("💥 EXPLOSION TRIGGERED")

        self.explosion_time = time.time()
        self.exploding = True

        for p in self.points:

            length = math.sqrt(
                p["x"] ** 2 +
                p["y"] ** 2 +
                p["z"] ** 2
            )

            if length < 0.001:
                length = 1

            # Strong outward velocity
            speed = random.uniform(1.5, 2.5) * power

            p["vx"] = (p["x"] / length) * speed
            p["vy"] = (p["y"] / length) * speed
            p["vz"] = (p["z"] / length) * speed

            # Add randomness so explosion isn't perfectly uniform
            p["vx"] += random.uniform(-0.35, 0.35)
            p["vy"] += random.uniform(-0.35, 0.35)
            p["vz"] += random.uniform(-0.35, 0.35)

            p["burst"] = 1.0

    # --------------------------------------------------------

    def update(self):

        smoothing = 0.09

        self.center_x += (
            self.target_x - self.center_x
        ) * smoothing

        self.center_y += (
            self.target_y - self.center_y
        ) * smoothing

        self.rotation_x += (
            self.target_rotation_x - self.rotation_x
        ) * 0.07

        self.rotation_y += (
            self.target_rotation_y - self.rotation_y
        ) * 0.07

        self.scale += (
            self.target_scale - self.scale
        ) * 0.08

    # --------------------------------------------------------

    def draw(self, frame, charge_amount=0):

        self.update()

        radius = min(WIDTH, HEIGHT) * 0.29 * self.scale

        cos_x = math.cos(self.rotation_x)
        sin_x = math.sin(self.rotation_x)

        cos_y = math.cos(self.rotation_y)
        sin_y = math.sin(self.rotation_y)

        current_time = time.time()

        projected = []

        # ----------------------------------------------------
        # Calculate particle positions
        # ----------------------------------------------------

        for p in self.points:

            wobble = (
                1 +
                math.sin(
                    current_time * 1.1 + p["phase"]
                ) * 0.009
            )

            x = p["x"] * wobble
            y = p["y"] * wobble
            z = p["z"] * wobble

            # Explosion animation
            if p["burst"] > 0:

                age = current_time - self.explosion_time

                # -----------------------------
                # EXPLODE
                # -----------------------------
                if age < 0.8:

                    # Move rapidly outward
                    progress = age / 0.8

                    travel = progress * 3.5

                    x += p["vx"] * travel
                    y += p["vy"] * travel
                    z += p["vz"] * travel

                # -----------------------------
                # REFORM
                # -----------------------------
                elif age < 2.0:

                    reform_progress = (age - 0.8) / 1.2

                    # Smoothly return toward original position
                    return_amount = (1.0 - reform_progress) * 3.5

                    x += p["vx"] * return_amount
                    y += p["vy"] * return_amount
                    z += p["vz"] * return_amount

                else:

                    p["burst"] = 0

            # Y-axis rotation
            rx = x * cos_y + z * sin_y
            rz = z * cos_y - x * sin_y

            # X-axis rotation
            ry = y * cos_x - rz * sin_x
            zz = y * sin_x + rz * cos_x

            # Perspective
            perspective = 3.6 / (3.6 - zz * 0.75)

            px = int(
                self.center_x +
                rx * radius * perspective
            )

            py = int(
                self.center_y +
                ry * radius * perspective
            )

            if (
                px < 0 or px >= WIDTH or
                py < 0 or py >= HEIGHT
            ):
                continue

            size = clamp(
                (0.56 + p["grain"] * 0.85)
                * perspective,
                0.4,
                2.8
            )

            projected.append(
                (
                    zz,
                    px,
                    py,
                    size
                )
            )

        # Draw back particles first
        projected.sort(key=lambda particle: particle[0])

        # ----------------------------------------------------
        # Draw particles
        # ----------------------------------------------------

        for z, x, y, size in projected:

            particle_hue = self.hue

            if z < 0:
                particle_hue += 15
            else:
                particle_hue -= 8

            color = hsv_to_bgr(particle_hue)

            brightness = clamp(
                0.35 + (z + 1) * 0.32,
                0.25,
                1
            )

            color = tuple(
                int(c * brightness)
                for c in color
            )

            radius_px = max(
                1,
                int(size)
            )

            cv2.circle(
                frame,
                (x, y),
                radius_px,
                color,
                -1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Charge ring
        # ----------------------------------------------------

        if charge_amount > 0:

            ring_radius = int(radius * 1.16)

            color = hsv_to_bgr(self.hue)

            end_angle = int(
                -90 + 360 * charge_amount
            )

            cv2.ellipse(
                frame,
                (
                    int(self.center_x),
                    int(self.center_y)
                ),
                (
                    ring_radius,
                    ring_radius
                ),
                0,
                -90,
                end_angle,
                color,
                2,
                cv2.LINE_AA
            )


# ============================================================
# COUNTING
# ============================================================

def count_fingers(landmarks):

    count = 0

    base = landmarks[0]

    # Index, middle, ring, pinky
    finger_sets = [
        (8, 6, 5),
        (12, 10, 9),
        (16, 14, 13),
        (20, 18, 17)
    ]

    for tip, pip, mcp in finger_sets:

        if (
            distance(landmarks[tip], base)
            >
            distance(landmarks[pip], base) * 1.10
        ):

            if (
                distance(
                    landmarks[tip],
                    landmarks[mcp]
                )
                >
                distance(
                    landmarks[pip],
                    landmarks[mcp]
                ) * 1.23
            ):

                count += 1

    # Thumb
    if (
        distance(
            landmarks[4],
            landmarks[17]
        )
        >
        distance(
            landmarks[3],
            landmarks[17]
        ) * 1.12
    ):

        count += 1

    return count


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.66,
    min_tracking_confidence=0.62
)

mp_draw = mp.solutions.drawing_utils


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# CREATE SPHERE
# ============================================================

sphere = ParticleSphere(NUM_PARTICLES)


# ============================================================
# STATE VARIABLES
# ============================================================

previous_pinch = False

pinch_start = 0

charge_amount = 0

show_camera = True

last_hand_seen = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, camera_frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror camera
    camera_frame = cv2.flip(
        camera_frame,
        1
    )

    camera_frame = cv2.resize(
        camera_frame,
        (WIDTH, HEIGHT)
    )

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(
        camera_frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    if show_camera:

        frame = (
            camera_frame.astype(np.float32)
            * BACKGROUND_DARKNESS
        ).astype(np.uint8)

    else:

        frame = np.zeros(
            (HEIGHT, WIDTH, 3),
            dtype=np.uint8
        )

        frame[:] = (
            11,
            8,
            6
        )

    # --------------------------------------------------------
    # Hand tracking
    # --------------------------------------------------------

    detected_hands = []

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            detected_hands.append(
                hand_landmarks.landmark
            )

        last_hand_seen = time.time()

    hand_count = len(detected_hands)

    # ========================================================
    # NO HANDS
    # ========================================================

    if hand_count == 0:

        sphere.target_x = WIDTH / 2
        sphere.target_y = HEIGHT / 2

        sphere.target_scale = 1

        sphere.target_rotation_x = 0.17

        # Slowly rotate
        sphere.target_rotation_y += 0.005

        if previous_pinch and charge_amount > 0.16:

            sphere.explode(
                0.5 +
                charge_amount * 0.9
            )

        previous_pinch = False

        charge_amount = 0

    # ========================================================
    # ONE HAND
    # ========================================================

    elif hand_count == 1:

        hand = detected_hands[0]

        # Palm landmark
        palm = hand[9]

        # Since image is already mirrored,
        # normal coordinates work naturally.
        sphere.target_x = palm.x * WIDTH
        sphere.target_y = palm.y * HEIGHT

        sphere.target_rotation_y = (
            palm.x - 0.5
        ) * 2.8

        sphere.target_rotation_x = (
            palm.y - 0.5
        ) * 2.0

        sphere.target_scale = 1

        # ========================================================
        # PINCH DETECTION
        # ========================================================

        thumb = hand[4]
        index = hand[8]

        palm_width = distance(
            hand[5],
            hand[17]
        )

        pinch_distance = distance(
            thumb,
            index
        )

        pinch_ratio = pinch_distance / max(
            palm_width,
            0.001
        )

        # Different threshold for starting and releasing.
        # This prevents MediaPipe jitter from constantly
        # switching the pinch state.

        if previous_pinch:

            pinched = pinch_ratio < 0.75

        else:

            pinched = pinch_ratio < 0.55

        # ========================================================
        # PINCH START
        # ========================================================

        if pinched and not previous_pinch:
            pinch_start = time.time()

            charge_amount = 0

            print("🤏 PINCH START")

        # ========================================================
        # CHARGING
        # ========================================================

        if pinched:
            charge_amount = clamp(
                (time.time() - pinch_start)
                / CHARGE_TIME,
                0,
                1
            )

        # ========================================================
        # RELEASE
        # ========================================================

        if previous_pinch and not pinched:
            print(
                "✋ RELEASE",
                "charge =",
                round(charge_amount, 2)
            )

            # Always explode after a recognized pinch.
            # Charge determines strength.

            explosion_power = (
                    0.8 +
                    charge_amount * 1.5
            )

            sphere.explode(
                explosion_power
            )

            charge_amount = 0

        previous_pinch = pinched

    # ========================================================
    # TWO HANDS
    # ========================================================

    else:

        hand_a = detected_hands[0]
        hand_b = detected_hands[1]

        palm_a = hand_a[9]
        palm_b = hand_b[9]

        # Midpoint
        midpoint_x = (
            palm_a.x +
            palm_b.x
        ) / 2

        midpoint_y = (
            palm_a.y +
            palm_b.y
        ) / 2

        sphere.target_x = (
            midpoint_x * WIDTH
        )

        sphere.target_y = (
            midpoint_y * HEIGHT
        )

        # Distance between hands
        gap = distance(
            palm_a,
            palm_b
        )

        sphere.target_scale = clamp(
            0.56 +
            (gap - 0.1) * 2.7,
            0.58,
            2.05
        )

        previous_pinch = False
        charge_amount = 0

    # ========================================================
    # FINGER COUNT → COLOR
    # ========================================================

    total_fingers = 0

    for hand in detected_hands:

        total_fingers += count_fingers(
            hand
        )

    colors = [
        190,    # 0
        275,    # 1
        335,    # 2
        32,     # 3
        135,    # 4
        190,    # 5
        240,    # 6
        290,    # 7
        340,    # 8
        50,     # 9
        175     # 10
    ]

    if hand_count > 0:

        sphere.hue = colors[
            clamp(
                total_fingers,
                0,
                10
            )
        ]

    else:

        sphere.hue += (
            185 -
            sphere.hue
        ) * 0.025

    # ========================================================
    # DRAW SPHERE
    # ========================================================

    sphere.draw(
        frame,
        charge_amount
        if previous_pinch
        else 0
    )

    # ========================================================
    # UI
    # ========================================================

    cv2.putText(
        frame,
        "ORBIT.",
        (45, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (235, 235, 235),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "A universe at your fingertips.",
        (47, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
        cv2.LINE_AA
    )

    # Hand count
    cv2.putText(
        frame,
        f"HANDS  {hand_count:02}",
        (WIDTH - 230, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # Finger count
    finger_text = (
        str(total_fingers)
        if hand_count > 0
        else "-"
    )

    cv2.putText(
        frame,
        f"FINGERS  {finger_text}",
        (WIDTH - 230, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # Instructions
    instructions = [
        "1  Move one hand to orbit",
        "2  Pinch + hold to charge",
        "3  Release pinch to explode",
        "4  Two hands to resize",
        "5  Finger count changes color",
        "B  Toggle webcam background",
        "Q  Quit"
    ]

    start_y = HEIGHT - 175

    for i, text in enumerate(instructions):

        cv2.putText(
            frame,
            text,
            (
                45,
                start_y + i * 22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Charging UI
    # --------------------------------------------------------

    if previous_pinch:

        percentage = int(
            charge_amount * 100
        )

        cv2.putText(
            frame,
            f"CHARGING {percentage}%",
            (
                WIDTH // 2 - 75,
                HEIGHT - 50
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            hsv_to_bgr(sphere.hue),
            1,
            cv2.LINE_AA
        )

        # Progress bar
        bar_width = 180

        x1 = WIDTH // 2 - bar_width // 2
        y1 = HEIGHT - 35

        cv2.rectangle(
            frame,
            (x1, y1),
            (
                x1 + bar_width,
                y1 + 4
            ),
            (60, 60, 60),
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (
                x1 +
                int(
                    bar_width
                    * charge_amount
                ),
                y1 + 4
            ),
            hsv_to_bgr(
                sphere.hue
            ),
            -1
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Orbit - Gesture Particle Sphere",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord("q"):
        break

    # B = background
    elif key == ord("b"):

        show_camera = not show_camera

        # E = manually test explosion
    elif key == ord("e"):

        print("Manual explosion test")

        sphere.explode(1.5)


# ============================================================
# CLEANUP
# ============================================================

camera.release()

hands.close()

cv2.destroyAllWindows()
