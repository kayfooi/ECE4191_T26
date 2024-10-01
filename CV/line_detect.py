import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_adaptive_threshold(image):
    height, _, _ = image.shape
    bottom_half = image[height//2:, :, :]
    avg_color = np.mean(bottom_half, axis=(0, 1))
    avg_gray = np.mean(avg_color)
    threshold = min(int(avg_gray) + 70, 253)
    return threshold

def isolate_white_pixels(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(image, image, mask=white_mask)
    return result

def calculate_wcss(data, max_clusters):
    wcss = []
    for n in range(1, max_clusters + 1):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, n, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        wcss.append(np.sum((data - centers[labels.flatten()])**2))
    return wcss

def find_optimal_clusters(wcss):
    differences = np.diff(wcss)
    differences_2 = np.diff(differences)
    optimal_clusters = np.argmax(differences_2) + 2
    return optimal_clusters

def plot_results(original_img, white_only, hough_lines_img, clustered_lines_img, wcss, optimal_clusters):
    plt.figure(figsize=(20, 10))
    
    plt.subplot(231)
    plt.plot(range(1, len(wcss) + 1), wcss, 'bo-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.vlines(optimal_clusters, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashed', label=f'Optimal clusters: {optimal_clusters}')
    plt.legend()

    plt.subplot(232)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(cv2.cvtColor(white_only, cv2.COLOR_BGR2RGB))
    plt.title('White Pixels Isolated')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(cv2.cvtColor(hough_lines_img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Hough Lines')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(cv2.cvtColor(clustered_lines_img, cv2.COLOR_BGR2RGB))
    plt.title('Clustered Court Lines')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def detect_tennis_court_lines(image_path, max_clusters=15):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    adaptive_threshold = calculate_adaptive_threshold(img)
    print(f"Adaptive threshold: {adaptive_threshold}")
    
    white_only = isolate_white_pixels(img, adaptive_threshold)
    
    gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=100, maxLineGap=20)

    if lines is None:
        print("No lines detected. Try adjusting the HoughLinesP parameters.")
        return

    # Draw Hough lines
    hough_lines_img = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    line_desc_values = [] 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 < x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        
        size = (x2-x1)**2 + (y2-y1)**2

        # Normalise and weight heavily for kmeans
        x_comp = (x2-x1) / size * 1000
        y_comp = (y2-y1) / size * 1000

        y_int = np.inf
        x_int = np.inf
        if x_comp != 0:
            grad = y_comp / x_comp 
            y_int = y1 - x1*grad
            if grad != 0:
                x_int = -y_int / grad
        else:
            x_int = x1
        
        # also look at minimum intercept
        if y_int < 0:
            min_int = x_int
        elif x_int < 0:
            min_int = y_int
        else:
            min_int = min(x_int, y_int)
        line_desc_values.append([x_comp, y_comp, min_int, (height - ((y2-y1)/2 + y1))**2 / height])

    line_desc_values = np.array(line_desc_values).astype(np.float32)

    max_clusters = len(line_desc_values)

    wcss = calculate_wcss(line_desc_values, max_clusters)
    optimal_clusters = min(len(line_desc_values), find_optimal_clusters(wcss) + 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(line_desc_values, optimal_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Draw clustered lines
    clustered_lines_img = img.copy()
    
    for center in centers:
        xc, yc, min_int, yheight = center
        if xc == 0:
            x1, y1 = int(min_int), 0
            x2, y2 = int(min_int), int(height)
        elif yc == 0:
            x1, y1 = 0, int(min_int)
            x2, y2 = width, int(min_int)
        else:
            x1, y1 = 0, int(min_int)
            x2, y2 = width, int(width * (yc/xc) + min_int)
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(clustered_lines_img, (x1, y1), (x2, y2), color, 2)

    # Plot results
    plot_results(img, white_only, hough_lines_img, clustered_lines_img, wcss, optimal_clusters)


# Usage
for n in range(0, 50, 10):
    image_path = f'./test_imgs/test_images/testing{n:04g}.jpg'
    detect_tennis_court_lines(image_path)