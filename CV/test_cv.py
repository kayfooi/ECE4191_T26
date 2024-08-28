"""
Used to evaluate ball detection algorithms
Specifically functions that take img array and return 2D image coordinates of detectected balls.
"""

import numpy as np
import cv2
import json
import time
import pandas as pd
import os
import glob

# import detection algorithms here
# expecting detection function to take img array and return 2D np array of detected ball image coordinates
from camera import detect_ball_circularity_no_colour
from cv_alg_test import detect_tennis_ball
from cv_alg_test import detect_ball_circularity_colour
from camera import detect_ball_circularity_no_blue, find_circles_template_match, YOLOv1, YOLOv2

CV_ALGS = [
    # detect_ball_circularity_no_colour, 
    # detect_tennis_ball, 
    # detect_ball_circularity_colour,
    # detect_ball_circularity_no_blue,
    # find_circles_template_match,
    YOLOv1,
    YOLOv2,
] # list of alg functions to test

show_each_result = True # display images
show_each_result_scores = True # annotate images that are shown
print_each_result = True # print all results to console

# Update path to relevant test set
IMGS_DIR = os.path.join('test_imgs', '2024_08_28')
# IMGS_DIR = os.path.join('test_imgs', 'blender', 'oneball')
CASE_DIR = os.path.join(IMGS_DIR, 'cases.json')
# All image files to test
TEST_FILES = sorted(glob.glob(os.path.join(IMGS_DIR, f'real*.jpg')))


def evaluate_detections(detected, true, distance_threshold=40.0):
    """
    Parameters
    ---
    detected : array like
        np array of detected ball 2D image coordinates. 
    true : array like
        np array of ball true position 2D image coordinates.
    distance_threshold : float
        maximum distance between detected and true ball to be considered a match (in pixels).

    Returns
    ---
    true_positives : array like
        Detected points that are within `distance_threshold` of a true point.
    false_positives: array like
        Detected points that are *not* within `distance_threshold` of a true point.
    false_negatives: array like
        list of true points that are not within `distance_threshold` of any detected points.
    """
    
    true_positives = []
    false_positives = []
    false_negatives = []

    # Create a copy of true coordinates to mark matched points
    true_copy = true.copy()

    for det_point in detected:
        if len(true_copy) > 0:
            # Calculate distances between the detected point and all true points
            distances = np.linalg.norm(true_copy - det_point, axis=1)
            
            # Find the closest true point
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

            if min_distance <= distance_threshold:
                # True positive
                true_positives.append(det_point)
                # Remove the matched true point to prevent multiple matches
                true_copy = np.delete(true_copy, min_distance_index, axis=0)
            else:
                # False positive
                false_positives.append(det_point)
        else:
            false_positives.append(det_point)

    # Remaining unmatched true points are false negatives
    false_negatives = true_copy.tolist()

    return true_positives, false_positives, false_negatives



if __name__ == "__main__":
    # Show each result with a cv2.imshow call
    
    results = {
        "algorithm":[],
        "avg. time (ms)":[],
        "balls detected (%)": [], # true positives / (true positives + false negatives) Proportion of all balls correctly detected
        "detections that are balls (%)": [], # true positives / (true positives + false positives) Proportion of all detections that are actual balls
        "errors": []
    }

    cases = {}
    with open(CASE_DIR, 'r') as file:
        cases = json.load(file)
    
    

    # Run each algorithm on all test styles
    for i, alg in enumerate(CV_ALGS):
        # show_each_result = i == 1
        print(f"\n---- Testing function: {alg.__name__} ----")
        results["algorithm"].append(alg.__name__)
        fns_alg = 0
        tps_alg = 0
        fps_alg = 0
        alg_time = 0.0
        alg_tests = len(TEST_FILES) # number of tests
        errors = 0
        for j, fname in enumerate(TEST_FILES):
            img = cv2.imread(fname)
            if img is None:
                continue

            # Run algorithm
            start_time = time.time()
            try:
                det_uv = alg(img)
            except Exception as e:
                print(f"Failed on {fname}")
                print(e)
                errors += 1
                
            end_time = time.time()
            true_uv = np.array(list(map(lambda b: b["image"], cases[j]["balls"])))
            tp, fp, fn = evaluate_detections(det_uv, true_uv)

            alg_time += end_time - start_time
            tps_alg += len(tp)
            fps_alg += len(fp)
            fns_alg += len(fn)

            # Displaying results
            if len(true_uv > 0):
                    tp_text = f"True positives:\t{len(tp)}/{len(true_uv)}\t{len(tp)/(len(true_uv)) * 100 : .2f}%"
                    fn_text = f"False negatives:\t{len(fn)}/{len(true_uv)}\t{len(fn)/(len(true_uv)) * 100 : .2f}%"
            else:
                tp_text = f"True positives:\t{len(tp)}/{len(true_uv)}\t"
                fn_text = f"False negatives:\t{len(fn)}/{len(true_uv)}\t"
                
            fp_text = f"False positives:\t{len(fp)}"

            if (print_each_result):     
                print(f"\n    ---- Case: {j:04d} ----")
                print(f"    {tp_text}")
                print(f"    {fn_text}")
                print(f"    {fp_text}")

            
            if show_each_result:
                if show_each_result_scores:
                    for point in tp:
                        cv2.drawMarker(img, (point[0], point[1]),(0,210,0), markerType=cv2.MARKER_SQUARE)
                    for point in fp:
                        cv2.drawMarker(img, (point[0], point[1]),(0,0,255), markerType=cv2.MARKER_TILTED_CROSS)
                    for point in fn:
                        cv2.drawMarker(img, (point[0], point[1]),(0,0,255), markerType=cv2.MARKER_TRIANGLE_DOWN)
                    
                    cv2.rectangle(img, (0, 0), (260, 37), (0, 0, 0), -1)

                    cv2.putText(img, tp_text.replace('\t', ' '), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 210, 0), 1)
                    cv2.putText(img, fp_text.replace('\t', ' '), (0, 22), cv2.FONT_HERSHEY_PLAIN, 1.0, (100, 100, 255), 1)
                    cv2.putText(img, fn_text.replace('\t', ' '), (0, 34), cv2.FONT_HERSHEY_PLAIN, 1.0, (100, 100, 255), 1)

                cv2.imshow('Test', img)
                cv2.setWindowTitle('Test', f"File: {fname}, Time: {(end_time - start_time )* 1e3 : .3f} msec")
                cv2.waitKey(0)
        
        print(f"\n{alg.__name__} Results:")
        alg_acc = tps_alg/(fns_alg+tps_alg) * 100
        alg_time = alg_time / (fns_alg + tps_alg)*1e3
        print(f"accuracy:\t{tps_alg}/{fns_alg+tps_alg}\t{alg_acc : .2f}%")
        print(f"avg. time:\t{alg_time:.3f} msec")
        results['balls detected (%)'].append(alg_acc)
        results['avg. time (ms)'].append(alg_time)
        try:
            results["detections that are balls (%)"].append(tps_alg / (tps_alg + fps_alg) * 100)
        except(ZeroDivisionError):
            # zero positive detections
            results["detections that are balls (%)"].append(0)
        results["errors"].append(errors)

    print("\n\n ---- RESULTS SUMMARY ----")
    results = pd.DataFrame(data=results)
    print(results)
