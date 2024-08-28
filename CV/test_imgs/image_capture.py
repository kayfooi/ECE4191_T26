import cv2
import os
import time

def create_output_folder():
    folder_path = time.strftime("%Y_%m_%d")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path
    

def main():
    # Open the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    time.sleep(1) # wake up camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    
    folder_path = create_output_folder()
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    image_counter = len(files)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Display the frame
        cv2.imshow('SPACE to save, ESC to exit, any other key to skip frame', frame)

        # Wait for key press
        key = cv2.waitKey(50) & 0xFF

        # If spacebar is pressed, save the image
        if key == ord(' '):
            image_name = os.path.join(folder_path, f'image_{image_counter}.jpg')
            cv2.imwrite(image_name, frame)
            print(f"Image saved: {image_name}")
            image_counter += 1

        # If ESC key (27) is pressed, quit the program
        elif key == 27:
            print("ESC pressed. Exiting...")
            break



    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()