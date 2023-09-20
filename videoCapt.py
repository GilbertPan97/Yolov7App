import cv2

# Get the list of available cameras
available_cameras = [f'Camera {i}' for i in range(10)]
for camera_index, camera_name in enumerate(available_cameras):
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is available
    if cap.isOpened():
        print(f"{camera_name}: Available")
        cap.release()
    else:
        print(f"{camera_name}: Not Available")

# Ask the user to select a camera to use
camera_index = int(input("Enter the camera index you want to use: "))

# Open the selected camera
cap = cv2.VideoCapture(camera_index)

# Check if the selected camera is successfully opened
if not cap.isOpened():
    print("Could not open the selected camera")
    exit()

while True:
    # Read the camera image
    ret, frame = cap.read()

    # Check if the image is successfully read
    if not ret:
        print("Could not read the image")
        break

    # Display the image in a window
    cv2.imshow('Camera', frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resources and close the window
cap.release()
cv2.destroyAllWindows()

