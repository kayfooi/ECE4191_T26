
import time

s = time.time()
import numpy as np
import cv2
from scipy.signal import find_peaks
# from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
e = time.time()

GRADIENT_SIMILARITY_THRESH = 0.015

print(f"import taken {(e-s)*1e3} msec")

def speedy_ransac(points, num_iterations=100, threshold=10.0, min_inliers=3, estimate = None):
    """
    Implement a speedy RANSAC algorithm to find the line of best fit.
    
    Args:
    points (np.array): Nx2 array of (x, y) coordinates
    num_iterations (int): Number of iterations for RANSAC
    threshold (float): Maximum distance for a point to be considered an inlier
    min_inliers (int): Minimum number of inliers required for a good fit
    
    Returns:
    tuple: (slope, intercept) of the best fit line, or None if no good fit found
    """
    best_inliers = []
    best_slope = best_intercept = None
    
    for _ in range(num_iterations):
        # Randomly select two points
        sample = points[np.random.choice(points.shape[0], 2, replace=True)]
        
        # Calculate slope and intercept
        x1, y1 = sample[0]
        x2, y2 = sample[1]
        
        if x1 == x2:
            continue  # Skip vertical lines
        
        slope = (y2 - y1) / (x2 - x1)

        if estimate is not None and abs(estimate - slope) > GRADIENT_SIMILARITY_THRESH:
            continue # exit this iteration

        intercept = y1 - slope * x1
        
        # Calculate distances of all points to the line
        distances = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        
        # Find inliers
        inliers = np.where(distances < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_slope = slope
            best_intercept = intercept
        
        if len(best_inliers) > min_inliers:
            if estimate is None:
                break
    
    if len(best_inliers) > min_inliers:
        # Refine the fit using all inliers
        x = points[best_inliers, 0]
        y = points[best_inliers, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        best_slope, best_intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return best_slope, best_intercept, best_inliers
    else:
        return (None, None, best_inliers)


def detect_white_line(image, target_point, num_paths=10):
    height, width = image.shape[:2]
    mid_bottom = (width // 2, height - 1)
    
    def extract_path(start, end, offset):
        x1, y1 = start
        x2, y2 = end
        path_length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        x = np.linspace(x1, x2, path_length).astype(int)
        y = np.linspace(y1, y2, path_length).astype(int)
        
        # Apply horizontal offset
        x = np.clip(x + offset, 0, width - 1)
        
        return image[y, x], np.array(list(zip(x, y)))
    
    def process_path(path):
        # Apply Sobel Y filter
        sobel_y = cv2.Sobel(path, cv2.CV_64F, 0, 1, ksize=3)
        
        def detect_peaks(channel, take_max = 5):
            positive_peaks, _ = find_peaks(channel, distance=10)
            negative_peaks, _ = find_peaks(-channel, distance=10)
            
            # return top results
            return (
                positive_peaks[np.argsort(channel[positive_peaks])[- 1 : -1 - take_max: -1]], 
                negative_peaks[np.argsort(-channel[negative_peaks])[- 1 : -1 - take_max : -1]]
            )
        
        # Find peaks for R and G channels
        r_pos_peaks, r_neg_peaks = detect_peaks(sobel_y[:, 0])
        g_pos_peaks, g_neg_peaks = detect_peaks(sobel_y[:, 1])

        def blue_white_score(posp, negp):
            # plot_blue_white_score(path, posp, negp)
            # Compare line sample to before and after
            sample_size = 10
            padding = 5
            pre_sample = path[max(0, posp-padding-sample_size):posp-padding]
            line_sample = path[posp:negp]
            post_sample = path[negp+padding:min(negp+padding+sample_size, len(path)-5)]
            
            def get_blue_score(s):
                mean_rg = np.mean(s[:, [1,2]])
                mean_b = np.mean(s[:, 0])
                return ((mean_b / mean_rg) - 1) * 300, mean_rg

            pre_blue, pre_rg = get_blue_score(pre_sample)
            post_blue, post_rg = get_blue_score(post_sample)

            # Get white score
            # rgb values should be similar
            consistency = 1/(0.5+np.mean(np.var(line_sample, axis=1))) * 200

            white_rg = np.mean(line_sample[:, [1,2]])

            # both should be positive
            pre_change = white_rg - pre_rg
            post_change = white_rg - post_rg

            # Heavily penalise small jumps
            if pre_change < 10:
                pre_change = -100
            if post_change < 10:
                post_change = -100
            
            total_score = pre_blue + post_blue + consistency + pre_change + post_change
            # print(f"Total Blue-White Score: {total_score:.2f}")
            return total_score


        # Calculate confidence scores
        def calculate_best(pos_peaks, neg_peaks, channel):
            best_score = 0
            best_pair = None
            SPACE_BETWEEN_POS_NEG = 40 # roughly max line width
            DIFF_POS_NEG = 0 # pos_peaks[0] * 0.05 # allows slight differents between positive and negative

            for posp in pos_peaks:
                CLOSE_TO_CAM = 10 + posp # favour points closer to camera
                prominence = channel[posp]
                for negp in neg_peaks:
                    # positive edge is always leading
                    if negp > posp:
                        width = negp-posp
                        if width < SPACE_BETWEEN_POS_NEG * 1.1:
                            conf = prominence**2 / (CLOSE_TO_CAM + DIFF_POS_NEG + (width/SPACE_BETWEEN_POS_NEG) ** 8 + abs(prominence + channel[negp]))
                            conf += blue_white_score(posp, negp)
                            if conf > best_score:
                                best_pair = [posp, negp]
                                best_score = conf
            
            return np.array(best_pair), best_score
        
        r_pair, r_confidence = calculate_best(r_pos_peaks, r_neg_peaks, sobel_y[:, 0])
        g_pair, g_confidence = calculate_best(g_pos_peaks, g_neg_peaks, sobel_y[:, 1])
        
        # Combine r and g
        conf = 0
        pair = np.array([0, 0])
        if r_confidence != 0 and g_confidence != 0:
            distance = np.linalg.norm(r_pair - g_pair)
            if distance < 10:
                pair = (r_pair + g_pair) / 2 # average coordinates
                conf = (1 - distance/50) * (r_confidence + g_confidence)
            else:
                pair = r_pair
                conf = r_confidence
        elif r_confidence != 0:
            pair = r_pair
            conf = r_confidence
        elif g_confidence != 0:
            pair = g_pair
            conf = g_confidence

        return conf, sobel_y, (r_pos_peaks, r_neg_peaks, g_pos_peaks, g_neg_peaks), pair.astype(int)

    # Extract and process multiple paths
    offsets = np.linspace(-width // 10, width // 10, num_paths)
    confidences = []
    all_path_points = []
    all_sobel_data = []
    all_peaks = []
    leading_points = []
    trailing_points = []
    chosen_peaks = []
    for offset in offsets:
        path, path_points = extract_path(mid_bottom, target_point, int(offset))
        confidence, sobel_data, peaks, pair = process_path(path)

        leading_points.append(path_points[pair[0]])
        trailing_points.append(path_points[pair[1]])

        confidences.append(confidence)
        all_path_points.append(path_points)
        all_sobel_data.append(sobel_data)
        all_peaks.append(peaks)
        chosen_peaks.append(pair)
    
    # Find slope and intercept of leading and trailing edges
    lead_slope, lead_int, lead_inliers = speedy_ransac(np.array(leading_points), threshold=10)
    trail_slope, trail_int, trail_inliers = speedy_ransac(np.array(trailing_points), threshold=10, estimate=lead_slope)
    
    lines = [
        [lead_slope, lead_int],
        [trail_slope, trail_int]
    ]

    if lead_slope is not None and trail_slope is not None:
        # Check the two edges make a sensible line
        grad_diff = abs(lead_slope - trail_slope)
        int_diff = lead_int - trail_int
        # may depend on resolution (GRADIENT_SIMILARITY_THRESH works well for 640x480px)
        if not (grad_diff < 0.05 and 0 < int_diff < 55):
            lines = None # invalid detection
    else:
        # No lines found
        lines = None
    
    confidences = np.array(confidences)
    lead_confidence = np.sum(confidences[lead_inliers])
    trail_confidence = np.sum(confidences[trail_inliers])
    
    return leading_points, trailing_points, lead_confidence+trail_confidence, all_path_points, all_sobel_data, all_peaks, chosen_peaks, lines


def plot_peaks(sobel_data, peaks, path_index, chosen_peaks):
    r_pos_peaks, r_neg_peaks, g_pos_peaks, g_neg_peaks = peaks
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot R channel
    ax1.plot(sobel_data[:, 0], label='R channel')
    ax1.plot(r_pos_peaks, sobel_data[r_pos_peaks, 0], "ro", label='Positive peaks')
    ax1.plot(r_neg_peaks, sobel_data[r_neg_peaks, 0], "go", label='Negative peaks')
    ax1.plot(chosen_peaks, sobel_data[chosen_peaks, 0], "kx", label='Chosen Peaks')
    ax1.set_title(f'R Channel - Path {path_index}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot G channel
    ax2.plot(sobel_data[:, 1], label='G channel')
    ax2.plot(g_pos_peaks, sobel_data[g_pos_peaks, 1], "ro", label='Positive peaks')
    ax2.plot(g_neg_peaks, sobel_data[g_neg_peaks, 1], "go", label='Negative peaks')
    ax2.set_title(f'G Channel - Path {path_index}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# The visualize_results function and example usage remain the same as in the previous version

def visualize_results(image, target_point, leading_edge, trailing_edge, confidence, all_path_points, lines):
    height, width = image.shape[:2]
    mid_bottom = (width // 2, height - 1)
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw paths
    for path_points in all_path_points:
        for x, y in path_points:
            cv2.circle(vis_image, (x, y), 1, (0, 255, 255), -1)
    
    # Draw target point
    cv2.circle(vis_image, target_point, 5, (0, 0, 255), -1)
    
    # Draw mid-bottom point
    cv2.circle(vis_image, mid_bottom, 5, (255, 0, 0), -1)
    
    # Draw leading and trailing edges
    for p in leading_edge:
        cv2.drawMarker(vis_image, 
                tuple(p), 
                (0, 255, 0), cv2.MARKER_CROSS, 5)
    for p in trailing_edge:
        cv2.drawMarker(vis_image, 
                tuple(p), 
                (0, 0, 255), cv2.MARKER_CROSS, 5)
    
    if lines is not None:
        for l in lines:
            x1, x2 = 0, width
            y1, y2 = int(l[1]), int(l[0] * width + l[1])
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Add text with results
    # cv2.putText(vis_image, f"Leading edge: {leading_edge:.2f}", (10, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(vis_image, f"Trailing edge: {trailing_edge:.2f}", (10, 60), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image, f"Confidence: {confidence:.2f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_image


def plot_blue_white_score(path, posp, negp):
    sample_size = 10
    padding = 5
    pre_sample = path[max(0, posp-padding-sample_size):posp-padding]
    line_sample = path[posp:negp]
    post_sample = path[negp+padding:min(negp+padding+sample_size, len(path))]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Full path with samples highlighted
    x = np.arange(len(path))
    ax1.plot(x, path[:, 0], 'b-', label='Blue')
    ax1.plot(x, path[:, 1], 'g-', label='Green')
    ax1.plot(x, path[:, 2], 'r-', label='Red')
    ax1.axvspan(posp-padding-sample_size, posp-padding, color='yellow', alpha=0.3, label='Pre-sample')
    ax1.axvspan(posp, negp, color='gray', alpha=0.3, label='Line sample')
    ax1.axvspan(negp+padding, negp+padding+sample_size, color='orange', alpha=0.3, label='Post-sample')
    ax1.set_title('Full Path with Samples')
    ax1.legend()
    ax1.set_ylabel('Pixel Value')
    
    # Plot 2: Blue scores
    def plot_blue_score(ax, sample, label):
        mean_rg = np.mean(sample[:, [1,2]])
        mean_b = np.mean(sample[:, 0])
        blue_score = ((mean_b / mean_rg) - 1) * 100
        ax.bar(label, blue_score)
        ax.text(label, blue_score, f'{blue_score:.2f}', ha='center', va='bottom')
    
    plot_blue_score(ax2, pre_sample, 'Pre-sample')
    plot_blue_score(ax2, post_sample, 'Post-sample')
    ax2.set_title('Blue Scores')
    ax2.set_ylabel('Blue Score')
    
    # Plot 3: Consistency and RG changes
    consistency = 1/(1+np.var(line_sample, axis=1)) * 100
    white_rg = np.mean(line_sample[:, [1,2]])
    pre_rg = np.mean(pre_sample[:, [1,2]])
    post_rg = np.mean(post_sample[:, [1,2]])
    pre_change = white_rg - pre_rg
    post_change = white_rg - post_rg
    
    ax3.bar('Consistency', np.mean(consistency))
    ax3.bar('Pre RG Change', pre_change)
    ax3.bar('Post RG Change', post_change)
    ax3.set_title('Consistency and RG Changes')
    ax3.set_ylabel('Value')
    
    for i, v in enumerate([np.mean(consistency), pre_change, post_change]):
        ax3.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Example usage
for n in range(1):
    n = 180
    image_path = f'./test_imgs/test_images/testing{n:04g}.jpg'
    # image_path = f'./test_imgs/blender/oneball/normal{n:04g}.jpg'
    image = cv2.imread(image_path)
    target_points = [(640, 150), (100, 150), (800, 150)]  # Example target points

    for target_point in target_points:
        s = time.time()
        leading_edge, trailing_edge, confidence, all_path_points, all_sobel_data, all_peaks, chosen_peaks, lines = detect_white_line(image, target_point, 12)
        e = time.time()

        print(f"taken {(e-s)*1e3} msec")
        # Visualize results (previous visualization code here)
        # Plot peaks for each path
        # for i, (sobel_data, peaks, chosen) in enumerate(zip(all_sobel_data, all_peaks, chosen_peaks)):
        #    plot_peaks(sobel_data, peaks, i, chosen)

        # Visualize results
        result_image = visualize_results(image, target_point, leading_edge, trailing_edge, confidence, all_path_points, lines)

        # Display the result
        cv2.imshow(f'White Line Detection {n}', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the result
        cv2.imwrite('white_line_detection_result.jpg', result_image)