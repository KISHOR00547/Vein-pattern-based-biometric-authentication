from picamera2 import Picamera2
import cv2
import time
import numpy as np


REFERENCE_IMAGE_PATH = "/home/ifp/sample.jpeg"


picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (1600, 1200),
        "format": "RGB888"
    }
)

picam2.configure(config)



picam2.set_controls({
    "AwbEnable": False,
    "ColourGains": (1.0, 2.0)
})

picam2.start()

time.sleep(2)


# CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# Bilateral filter
bilateral_d = 9
bilateral_sigma_color = 75
bilateral_sigma_space = 75


# Black-hat kernel
blackhat_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (21, 21)
)


# Small morphology kernel
small_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2, 2)
)


orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31
)


# ORB uses binary descriptors
bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

def process_image(frame_rgb):

    # --------------------------------------------------------
    # 1. GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2GRAY
    )


    clahe_img = clahe.apply(gray)

    filtered = cv2.bilateralFilter(
        clahe_img,
        bilateral_d,
        bilateral_sigma_color,
        bilateral_sigma_space
    )


    vein_enhanced = cv2.morphologyEx(
        filtered,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel
    )


    vein_enhanced = cv2.normalize(
        vein_enhanced,
        None,
        48,
        255,
        cv2.NORM_MINMAX
    )

    final_mask = cv2.convertScaleAbs(
        vein_enhanced,
        alpha=2,
        beta=0
    )


    # Slight smoothing

    final_mask = cv2.GaussianBlur(
        final_mask,
        (3, 3),
        0
    )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    _, final_mask = cv2.threshold(
        final_mask,
        100,
        255,
        cv2.THRESH_BINARY
    )

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        small_kernel
    )


    return (
        gray,
        clahe_img,
        filtered,
        vein_enhanced,
        final_mask
    )


def add_title(image, title):

    # Convert grayscale to BGR

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )


    # Resize each processing stage

    image = cv2.resize(
        image,
        (420, 300)
    )


    # Create title canvas

    canvas = np.zeros(
        (340, 420, 3),
        dtype=np.uint8
    )


    # Put image

    canvas[
        40:340,
        0:420
    ] = image


    # Add title

    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    return canvas

print()
print("Loading reference image...")

reference_gray = cv2.imread(
    REFERENCE_IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)


# ============================================================
# CHECK REFERENCE IMAGE
# ============================================================

if reference_gray is None:

    print()
    print("ERROR: Reference image not found!")
    print(
        "Path:",
        REFERENCE_IMAGE_PATH
    )

    picam2.stop()
    cv2.destroyAllWindows()

    exit()



print(
    "Reference image size:",
    reference_gray.shape[1],
    "x",
    reference_gray.shape[0]
)

print(
    "Extracting ORB features from original sample.jpeg..."
)


ref_keypoints, ref_descriptors = orb.detectAndCompute(
    reference_gray,
    None
)


if ref_descriptors is None:

    print(
        "ERROR: No ORB features found in sample.jpeg"
    )

    picam2.stop()
    cv2.destroyAllWindows()

    exit()


print(
    "Reference ORB features:",
    len(ref_keypoints)
)


cv2.namedWindow(
    "Palm Vein Processing",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Palm Vein Processing",
    1260,
    700
)


cv2.namedWindow(
    "ORB Vein Matching",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "ORB Vein Matching",
    700,
    450
)


while True:


    frame = picam2.capture_array()

    original = frame.copy()


    (
        gray,
        clahe_img,
        filtered,
        vein_enhanced,
        final_mask
    ) = process_image(
        frame
    )

    original_display = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )


    stage1 = add_title(
        original_display,
        "1. Original"
    )


    stage2 = add_title(
        gray,
        "2. Grayscale"
    )


    stage3 = add_title(
        clahe_img,
        "3. CLAHE"
    )


    stage4 = add_title(
        filtered,
        "4. Bilateral Filter"
    )


    stage5 = add_title(
        vein_enhanced,
        "5. Vein Enhanced"
    )


    stage6 = add_title(
        final_mask,
        "6. Final Clear Vein"
    )


    row1 = cv2.hconcat([
        stage1,
        stage2,
        stage3
    ])


    row2 = cv2.hconcat([
        stage4,
        stage5,
        stage6
    ])


    complete_view = cv2.vconcat([
        row1,
        row2
    ])


    cv2.imshow(
        "Palm Vein Processing",
        complete_view
    )


    live_for_orb = final_mask


    # --------------------------------------------------------
    # DETECT LIVE FEATURES
    # --------------------------------------------------------

    live_keypoints, live_descriptors = orb.detectAndCompute(
        live_for_orb,
        None
    )


    if (
        live_descriptors is not None
        and len(live_keypoints) > 0
    ):



        matches = bf.knnMatch(
            ref_descriptors,
            live_descriptors,
            k=2
        )



        good_matches = []


        for pair in matches:

            if len(pair) == 2:

                m, n = pair

                if m.distance < 0.75 * n.distance:

                    good_matches.append(m)


        good_matches = sorted(
            good_matches,
            key=lambda x: x.distance
        )


        match_count = len(
            good_matches
        )


        if match_count >= 20:

            status = "MATCH"

            status_color = (
                0,
                255,
                0
            )

        else:

            status = "NO MATCH"

            status_color = (
                0,
                0,
                255
            )


        orb_result = cv2.drawMatches(
            reference_gray,
            ref_keypoints,
            live_for_orb,
            live_keypoints,
            good_matches[:30],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )


        window_width = 700
        window_height = 450

        max_display_width = 650
        max_display_height = 300


        result_height, result_width = (
            orb_result.shape[:2]
        )


        scale = min(
            max_display_width / result_width,
            max_display_height / result_height
        )


        display_width = max(
            1,
            int(result_width * scale)
        )


        display_height = max(
            1,
            int(result_height * scale)
        )


        display_result = cv2.resize(
            orb_result,
            (
                display_width,
                display_height
            ),
            interpolation=cv2.INTER_AREA
        )

        orb_canvas = np.zeros(
            (
                window_height,
                window_width,
                3
            ),
            dtype=np.uint8
        )



        x = (
            window_width -
            display_width
        ) // 2


        y = 105


        orb_canvas[
            y:y + display_height,
            x:x + display_width
        ] = display_result


        cv2.putText(
            orb_canvas,
            f"Reference: {len(ref_keypoints)}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            orb_canvas,
            f"Live: {len(live_keypoints)}",
            (220, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            orb_canvas,
            f"Matches: {match_count}",
            (390, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            orb_canvas,
            status,
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            status_color,
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "ORB Vein Matching",
            orb_canvas
        )

    else:

        orb_canvas = np.zeros(
            (
                450,
                700,
                3
            ),
            dtype=np.uint8
        )


        cv2.putText(
            orb_canvas,
            "No ORB features detected",
            (180, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )


        cv2.imshow(
            "ORB Vein Matching",
            orb_canvas
        )


    key = cv2.waitKey(1) & 0xFF


    if key == ord('q'):

        break


picam2.stop()

cv2.destroyAllWindows()

print()
print("Program ended.")
