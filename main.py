import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Setup the segmentation function
# --- CHANGE 1: Use the more accurate landscape model ---
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load the background image
try:
    background_image = cv2.imread('background.png')
except:
    print("Error: 'background.png' not found. Make sure the image is in the same directory as the script.")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the frame dimensions from the webcam
ret, frame = cap.read()
if not ret:
    print("Error: Failed to grab a frame from the webcam.")
    cap.release()
    exit()

h, w, _ = frame.shape

# Resize the background image to match the webcam frame size
background_image = cv2.resize(background_image, (w, h))

print("Starting virtual background. Press 'ESC' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get the segmentation mask
    results = segmentation.process(rgb_frame)

    # The mask contains values from 0 (background) to 1 (person)
    mask = results.segmentation_mask

    # --- CHANGE 2: Blur the mask to create a feathered edge ---
    # The (51, 51) kernel size can be adjusted. Larger numbers mean more blur.
    # It must be an odd number.
    blurred_mask = cv2.GaussianBlur(mask, (51, 51), 0)

    # Stack the 2D blurred mask to 3 channels (for R, G, B)
    condition_3d = np.stack((blurred_mask,) * 3, axis=-1)

    # --- CHANGE 3: Combine images smoothly using the blurred mask ---
    # Convert images to float for smoother calculation
    frame_float = frame.astype(float)
    background_float = background_image.astype(float)

    # Perform the blend
    output_image = (condition_3d * frame_float + (1 - condition_3d) * background_float)

    # Convert back to uint8 for display
    output_image = output_image.astype(np.uint8)

    # Display the final image
    cv2.imshow('More Accurate Virtual Background - Press ESC to Exit', output_image)

    # Exit loop if 'ESC' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Application closed.")