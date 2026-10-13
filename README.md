# Palm Vein Pattern-Based Biometric Authentication

## 1. Project Overview

Palm Vein Pattern-Based Biometric Authentication is a biometric recognition project that explores the use of vein patterns beneath the human palm for identity verification.

The system captures palm images using a Raspberry Pi camera with near-infrared (NIR) illumination. OpenCV image-processing techniques are used to enhance vein patterns, and the ORB (Oriented FAST and Rotated BRIEF) algorithm is used to extract and match distinctive image features.

The current implementation establishes a classical computer vision baseline for palm vein matching. Live camera images are processed continuously, and their ORB features are compared with features extracted from a reference image.

## 2. Objectives

- Capture palm images using a Raspberry Pi camera.
- Improve the visibility of palm vein patterns using image processing.
- Implement real-time image enhancement using Python and OpenCV.
- Extract distinctive image features using ORB.
- Match reference and live-image descriptors using Brute-Force Matching and KNN.
- Filter ambiguous matches using Lowe's ratio test.
- Display image-processing stages and feature matches in real time.
- Establish a baseline for future biometric authentication development.

## 3. Hardware Components

The current hardware setup includes:

- Raspberry Pi 5
- Raspberry Pi Camera Module 2 / IMX219 NoIR camera
- Near-infrared illumination source
- 850 nm infrared LED illumination
- Power supply for the Raspberry Pi
- Separate power supply for the IR illumination system

The camera and infrared illumination are used to capture images in which palm vein structures may be more visible.

**Note:** The camera's ability to capture vein patterns depends on illumination, exposure, camera response, hand position, and image quality.

## 4. Software and Technologies

| Technology | Purpose |
|---|---|
| Python | Camera capture and real-time processing |
| OpenCV | Image processing and feature matching |
| NumPy | Image-array operations |
| Picamera2 | Raspberry Pi camera interface |
| libcamera | Camera pipeline and hardware interface |
| ORB | Feature detection and descriptor extraction |
| BFMatcher | Descriptor matching |
| KNN | Finding the two nearest descriptor candidates |
| Lowe's ratio test | Filtering ambiguous feature matches |
| Linux | Operating system environment |

## 5. System Architecture

```text
              Raspberry Pi Camera
                       |
                       v
              Live Palm Image
                       |
                       v
                 Grayscale
                       |
                       v
                    CLAHE
                       |
                       v
               Bilateral Filter
                       |
                       v
                Black-Hat Filter
                       |
                       v
                 Normalization
                       |
                       v
              Enhanced Vein Image
                       |
                       v
                  ORB Features
                       |
                       v
              Binary Descriptors
                       |
                       v
             BFMatcher with KNN
                       |
                       v
               Lowe's Ratio Test
                       |
                       v
               Good Feature Matches
                       |
                       v
               Matching Information
```

A reference image is loaded separately. ORB features are extracted from that reference, and the resulting descriptors are compared with descriptors from each live camera frame.

## 6. Image Processing Pipeline

The current implementation processes live camera images through the following stages.

### Stage 1: Original Image

The Raspberry Pi camera captures the palm image.

The original image is retained for visual comparison with the processed stages.

### Stage 2: Grayscale Conversion

The RGB image is converted into a grayscale image.

```python
gray = cv2.cvtColor(
    frame_rgb,
    cv2.COLOR_RGB2GRAY
)
```

Grayscale conversion reduces the image to intensity information for subsequent processing.

### Stage 3: CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) improves local image contrast.

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_img = clahe.apply(gray)
```

This can help make faint vein-like structures more visible under suitable imaging conditions.

### Stage 4: Bilateral Filtering

Bilateral filtering reduces noise while attempting to preserve important image edges.

```python
filtered = cv2.bilateralFilter(
    clahe_img,
    9,
    75,
    75
)
```

This is a preprocessing operation before vein enhancement.

### Stage 5: Vein Enhancement

A morphological black-hat operation is applied to highlight dark structures against a brighter local background.

```python
blackhat_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (21, 21)
)

vein_enhanced = cv2.morphologyEx(
    filtered,
    cv2.MORPH_BLACKHAT,
    blackhat_kernel
)
```

The result is normalized for display and subsequent feature extraction.

```python
vein_enhanced = cv2.normalize(
    vein_enhanced,
    None,
    48,
    255,
    cv2.NORM_MINMAX
)
```

The enhanced image is displayed as Stage 5.

### Stage 6: Binary Vein Mask

An additional binary-mask stage has also been implemented experimentally.

The operations include:

- Contrast adjustment
- Gaussian smoothing
- Binary thresholding
- Morphological closing

This stage is intended to produce a more distinct representation of vein-like structures.

The ORB matching implementation can be configured to use either Stage 5 or Stage 6. The selected input should be kept consistent with the reference-image processing strategy.

## 7. ORB Feature Extraction

ORB stands for **Oriented FAST and Rotated BRIEF**.

It is a feature-detection and description algorithm used in computer vision.

ORB performs two main tasks:

1. Detects distinctive keypoints in an image.
2. Generates binary descriptors representing local image patterns around those keypoints.

The current implementation uses:

```python
orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    patchSize=31
)
```

The number of detected keypoints can vary depending on the image and its quality.

## 8. Reference Image

The current reference image is configured as:

```python
REFERENCE_IMAGE_PATH = "/home/ifp/sample.jpeg"
```

The reference image is loaded in grayscale:

```python
reference_gray = cv2.imread(
    REFERENCE_IMAGE_PATH,
    cv2.IMREAD_GRAYSCALE
)
```

ORB features are extracted from the reference image before the live-processing loop:

```python
ref_keypoints, ref_descriptors = orb.detectAndCompute(
    reference_gray,
    None
)
```

The reference descriptors are reused for matching against successive live frames.

**Important:** In the current implementation, the reference image is used directly in grayscale, without applying the CLAHE, bilateral filtering, black-hat enhancement, or normalization pipeline. The live image is processed before ORB extraction. This difference should be considered when evaluating matching performance.

## 9. Feature Matching

### Brute-Force Matcher

ORB produces binary descriptors. Therefore, Hamming distance is used to compare descriptors.

```python
bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)
```

A smaller Hamming distance means that two binary descriptors differ in fewer bit positions.

### K-Nearest Neighbours

KNN matching finds the two nearest descriptor candidates for each reference descriptor.

```python
matches = bf.knnMatch(
    ref_descriptors,
    live_descriptors,
    k=2
)
```

### Lowe's Ratio Test

The ratio test filters ambiguous matches.

```python
if m.distance < 0.75 * n.distance:
    good_matches.append(m)
```

Here:

- `m` is the best match.
- `n` is the second-best match.
- `0.75` is the ratio threshold used by the current implementation.

The ratio test helps reject cases in which the best candidate is not sufficiently better than the second-best candidate.

## 10. Continuous Matching

The program continuously performs the following operations:

1. Captures a live camera frame.
2. Processes the image through the selected image-processing pipeline.
3. Detects live ORB keypoints.
4. Computes live descriptors.
5. Matches the live descriptors with the reference descriptors.
6. Applies Lowe's ratio test.
7. Counts the accepted descriptor matches.
8. Displays the feature matches and matching information.

The matching display includes:

- Reference keypoint count
- Live keypoint count
- Number of good matches
- Match visualization
- Current matching status

The program runs continuously until the user presses `q`.

## 11. Current Matching Decision

The current experimental decision rule is:

```python
if match_count >= 20:
    status = "MATCH"
else:
    status = "NO MATCH"
```

The value of 20 is a preliminary threshold, not a validated biometric-authentication threshold.

The number of good matches alone does not prove that two images belong to the same person. A reliable system needs evaluation using genuine and impostor comparisons, and should consider match quality and spatial consistency.

## 12. Real-Time Visualization

The current implementation uses OpenCV windows to display:

### Window 1: Palm Vein Processing

The processing window displays:

1. Original image
2. Grayscale image
3. CLAHE output
4. Bilateral-filtered image
5. Vein-enhanced image
6. Experimental final binary mask

### Window 2: ORB Vein Matching

The matching window displays:

- Reference image features
- Live image features
- Visualized feature correspondences
- Reference and live keypoint counts
- Number of good matches
- Current matching status

The displays update while the camera is running.

## 13. Installation

Install the required system packages on a compatible Raspberry Pi OS installation.

```bash
sudo apt update

sudo apt install -y \
    python3-opencv \
    python3-numpy \
    python3-picamera2
```

Check that the camera is detected:

```bash
rpicam-hello --list-cameras
```

Test the camera:

```bash
rpicam-hello
```

Check Python and OpenCV:

```bash
python3 --version
```

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

```bash
python3 -c "from picamera2 import Picamera2; print('Picamera2 OK')"
```

## 14. Running the Project

Place the reference image in the configured location:

```text
/home/ifp/sample.jpeg
```

Save the Python program, for example, as:

```text
palm_vein_orb.py
```

Run:

```bash
python3 palm_vein_orb.py
```

Press `q` in an OpenCV window to stop the program.

If the reference image is stored elsewhere, update `REFERENCE_IMAGE_PATH` in the script.

## 15. Current Progress

### Completed

- Raspberry Pi camera setup and image capture
- Near-infrared imaging experiments
- Real-time camera streaming using Picamera2
- Grayscale conversion
- CLAHE-based contrast enhancement
- Bilateral filtering
- Morphological black-hat vein enhancement
- Image normalization
- Experimental binary-mask generation
- ORB feature extraction
- Binary descriptor matching using Hamming distance
- KNN matching with `k=2`
- Lowe's ratio test
- Continuous reference-to-live feature matching
- Real-time feature-match visualization
- Two-window OpenCV interface

### In Progress

- Improving the consistency and visibility of vein patterns
- Selecting the most suitable Stage 5 or Stage 6 representation for ORB
- Evaluating reference-to-live matching performance
- Calibrating the matching threshold using collected samples

## 16. Limitations

The current implementation is an experimental computer-vision baseline.

Potential limitations include:

- Changes in palm position and orientation
- Differences in camera distance
- Variations in infrared illumination
- Differences in image contrast and quality
- False matches caused by similar local image structures
- Differences between the reference-image processing and live-image processing pipelines
- Unvalidated match-count thresholds

A successful ORB match should not yet be treated as proof of identity.

## 17. Future Work

Planned development may include:

- Building a structured palm-vein image dataset
- Collecting multiple samples per participant
- Consistent region-of-interest selection and image alignment
- Evaluating ORB matching using genuine and impostor comparisons
- Geometric verification using RANSAC and homography
- Developing a CNN-based feature-learning model
- Comparing classical computer vision and deep-learning approaches
- Exploring TensorFlow Lite or ONNX for edge deployment
- Measuring false acceptance rate and false rejection rate
- Improving secure storage and handling of biometric data

These are future development goals and are not claimed as completed features.

## 18. Privacy and Security

Palm images and vein patterns are biometric information.

Use consented samples and avoid publishing identifiable biometric images or personal information in the public repository.

The current matching implementation is a research prototype and has not been validated as a secure production authentication system.

## 19. Technologies

Python | OpenCV | NumPy | Picamera2 | Raspberry Pi 5 | ORB | KNN | Computer Vision | Biometric Recognition

## 20. Project Status

**Status:** Classical computer vision baseline under development.

The current focus is real-time palm image acquisition, vein enhancement, and ORB-based feature matching. Deep-learning-based recognition and edge deployment remain future development stages.
