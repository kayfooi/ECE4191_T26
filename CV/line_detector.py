"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

# Convert image coords to world coords with homography H
def image_to_world(image_coords, H):
    # Convert to homogeneous coordinates
    homogeneous_coords = np.column_stack((image_coords, np.ones(len(image_coords))))
    
    # Apply the homography
    world_coords = np.dot(H, homogeneous_coords.T).T
    
    # Convert back from homogeneous coordinates
    world_coords = world_coords[:, :2] / world_coords[:, 2:]
    
    return world_coords

def find_intercept(line):
    point1, point2 = line

    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate slope
    if x2 - x1 == 0:
        # Vertical line
        x_intercept = x1
        return x_intercept
    else:
        slope = (y2 - y1) / (x2 - x1)
        b = y1 - slope * x1
        # Calculate y-intercept
        if abs(slope) < 0.5:
            return [b, slope*-100+b, 100]
        
        # Calculate x-intercept
        else:
            return [-b / slope, (-100 - b) / slope, -100]


def main(file):

    ''' 
    The calculated homography can be used to warp 
    the source image to destination. Size is the 
    size (width,height) of im_dst
    '''

    
    
    H = np.array([
        [-0.011753544099407346, 0.00011229854998494311, 3.8398253909871616],
        [-0.0006578853576128101, 0.005114152566914852, -7.511900143237492],
        [-0.0014712311304621235, -0.03164807845607367, 0.9999999999999999],
    ]) * 1000

    src_col = cv.imread(cv.samples.findFile(file), cv.IMREAD_COLOR)
    src = cv.cvtColor(src_col, cv.COLOR_BGR2GRAY)
    
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    xw, yw = 500, 500
    dst_h = np.zeros((xw, yw)).astype(float)
    h, w = src.shape

    # src_crop = cv.resize(src, (640, 360)) # src[h-480:, 000:000+640]
    # for i in range(xw):
    #     for j in range(yw):
    #         world = np.vstack([i/50 - 5, j/50 + 1, 1])
            
    #         img_hom = (np.linalg.inv(H) @ world).flatten()
    #         # print(img_hom.shape)
    #         img = (img_hom / img_hom[2]).astype(int)
    #         # print(img)
    #         if -1 < img[1] < 360 and -1 < img[0] < 640:
    #             # print(img[0], img[1])
    #             dst_h[xw-j-1, i] = src_crop[img[1], img[0]]/255
    #             # print(dst_h[i, j], src_crop[img[0], img[1]])

    # cv.imshow('sample', src)
    # cv.imshow('crop', src_crop)
    # cv.warpPerspective(src_crop, H, dst_h.shape, dst_h)
    
    dst = cv.Canny(src, 30, 150, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    pxls = src_col.reshape((-1, 3)).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    n_clusters = 5
    ret,label,center=cv.kmeans(pxls,n_clusters,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    
    col_variance = np.var(center, axis=1)
    col_avg = np.average(center, axis=1)
    col_avg_grays = (col_variance < 5.0) * col_avg
    whitest_gray_idx = np.argmax(col_avg_grays)
    if (col_avg_grays[whitest_gray_idx] > 200.0):
        whites = np.reshape((label == whitest_gray_idx)*255, src.shape).astype(np.uint8)
        
        
        edges = cv.Canny(whites, 50, 100)
        linesP = cv.HoughLinesP(edges, 1, np.pi / 720, 25, None, 200, 200)
        lines = linesP.reshape(-1, 2, 2)
        deltas = lines[:, 1, :] - lines[:, 0, :]
        linecenters = (lines[:, 1, :] + lines[:, 0, :]) /2

        maxK = 8
        angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])).astype(np.float32)
        print(lines.shape)
        intercepts = np.array(list(map(find_intercept, lines))).astype(np.float32)
        nlines = 4
        prev = 100
        
        ret,label,center=cv.kmeans(intercepts,nlines,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
        
        cols = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 255, 255), 
            (255, 0, 0),
            (255, 255, 0)
        ]

        line_cntrs = [[] for _ in range(nlines)]
        line_counts = np.zeros(nlines)
        colsp = ['r', 'g', 'k', 'b', 'm']
        # print(center)
        # print(np.degrees(np.arctan2()))
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                lab = label[i][0]
                image_coords = np.array([l[:2], l[2:]])
                world = image_to_world(image_coords, H)
                if np.sum(world) > 0:
                    # print(world)
                    plt.plot(world[:, 0], world[:, 1], color=colsp[lab])
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), cols[lab], 1, cv.LINE_AA)
                line_cntrs[lab].append(linecenters[i])
        
        avg_centers = []
        for lab, centers in enumerate(line_cntrs):
            centers = np.array(centers)
            # for c1 in centers:
            #     totalerror = 0
            #     for c2 in centers:
            #         delta = c1 - c2
            #         dist = np.linalg.norm(delta)
            #         if dist > 20:
            #             ang = np.degrees(np.arctan2(delta[1], delta[0]))
            #             angd =  ang-center[lab][0]
            #             error = np.min(np.abs([angd - 180, angd, angd + 180]))
            #             totalerror += error
            #     # if totalerror > 20:
            #     print(lab, c1, totalerror, totalerror/len(centers))
            
            avg_centers.append(centers.sum(axis=0)/len(centers))
            
        line_cntrs = np.array(avg_centers)
        # print(line_cntrs)
        for i, l in enumerate(line_cntrs):
            print(tuple(l))
            intercept = center[i][0] # assume y int
            # print(l)
            imgcs = np.array([l, [0, intercept]])
            # print(imgcs)
            wcs = image_to_world(imgcs, H)
            # print(wcs)
            cv.line(src_col, tuple(l.astype(int)), (0, intercept.astype(int)), (0, 0, 255), 2)
            plt.plot(wcs[:, 0], wcs[:, 1], f'{colsp[i]}--')
        line_cntrs = image_to_world(line_cntrs, H)

        
        plt.plot(line_cntrs[:, 0], line_cntrs[:, 1], 'rx')
        plt.title("Lines in World Coordinates")

    
        # cv.imshow("Source", dst)
        # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        # cv.imshow("mapped", dst_h)
        cv.imshow('whites', whites)
        cv.imshow("Detected Lines", src_col)
        cv.waitKey()

        plt.axis('equal')
        plt.xlim((-3, 7))
        plt.ylim((0, 10))
    
        # plt.show()

    return 0
    
if __name__ == "__main__":
    folder_path = os.path.join('test_imgs', 'test_images')
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.jpg')][:20]
    print(image_files)
    for f in image_files:
        main(f)
