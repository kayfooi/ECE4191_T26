"""
Tool to help label images with ball coordinates
- Automatically iterates through all images in FOLDER_PATH
- Generates cases.json such that images are compatible with test_cv.py

This code was generated with the assistance of Claude, an AI assistant created by Anthropic.
"""

import cv2
import os
import json

# dummy camera state to populate cases.json
DEFAULT_CAM_POS = [0.0, 0.0, 0.0] 
DEFAULT_CAM_HEADING = 0.0

class ImageLabeler:
    def __init__(self):
        self.folder_path = input("Folder path/name: ")
        # self.image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.jpg')]
        self.image_files = ['calibration_img.jpg']
        self.current_image_index = 0
        self.points = []
        self.results = []

    def load_image(self):
        if self.current_image_index < len(self.image_files):
            self.points = []
            image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
            self.image = cv2.imread(image_path)
            self.refresh_display()
            return True
        return False

    def refresh_display(self):
        self.display_image = self.image.copy()
        for point in self.points:
            cv2.drawMarker(self.display_image, point, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.imshow("Image Labeler", self.display_image)

    def add_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"{x}, {y}")
            self.points.append((x, y))
            self.refresh_display()

    def run(self):
        cv2.namedWindow("Image Labeler")
        cv2.setMouseCallback("Image Labeler", self.add_point)

        while self.load_image():
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    self.save_current_image()
                    break
                elif key == 8 or key == 127:  # Backspace key
                    if self.points:
                        self.points.pop()
                        self.refresh_display()
                    
                elif key == 27:  # Esc key
                    self.save_results()
                    cv2.destroyAllWindows()
                    return

            self.current_image_index += 1

        self.save_results()
        cv2.destroyAllWindows()

    def save_current_image(self):
        image_name = self.image_files[self.current_image_index]
        case_number = f"{self.current_image_index:04d}"
        new_name = f"testing{case_number}.jpg"
        
        old_path = os.path.join(self.folder_path, image_name)
        new_path = os.path.join(self.folder_path, new_name)
        os.rename(old_path, new_path)

        # Match other cases.json
        result = {
            "caseID": case_number,
            "cam_heading": DEFAULT_CAM_HEADING,
            "cam_location": DEFAULT_CAM_POS,
            "balls": [{
                "ball_id": f"tennis-ball.{i:03d}",
                "world": [0, 0, 0],
                "image": point,
                "in_bounds": True
            } for i, point in enumerate(self.points)]
        }
        self.results.append(result)
        print(f"Saved {image_name} as {new_name}.")
        print(f"{len(self.points)} point(s) labelled at img coordinates: {self.points}\n")

    def save_results(self):
        with open(os.path.join(self.folder_path, "cases.json"), "w") as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    labeler = ImageLabeler()
    labeler.run()