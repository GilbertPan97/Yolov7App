import cv2

# Open the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera, you can try other numbers if you have multiple cameras

# Check if the camera is successfully opened
if not cap.isOpened():
    print("Could not open the camera")
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

