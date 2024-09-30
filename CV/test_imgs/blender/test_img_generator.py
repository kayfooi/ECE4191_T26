import bpy # pip install fake-bpy-module-latest (to develop in VSCode)
import math
import random
import json
from mathutils import Vector, Euler
import os
from bpy_extras.object_utils import world_to_camera_view
import numpy as np
import sys

random.seed(42) # repeatable cases
# Get the directory of the current .blend file
blend_file_path = bpy.data.filepath
directory = os.path.dirname(blend_file_path)

def update_matrices(obj):
    if obj.parent is None:
        obj.matrix_world = obj.matrix_basis

    else:
        obj.matrix_world = obj.parent.matrix_world * \
                           obj.matrix_parent_inverse * \
                           obj.matrix_basis

def point_in_camera_view(scene, camera, point):
    update_matrices(camera)
    cp = world_to_camera_view(scene, camera, point) # camera point
    render = scene.render
    res_x, res_y = render.resolution_x, render.resolution_y
    if 0.0 < cp.x < 1.0 and 0.0 < cp.y < 1.0 and cp.z > 0:
        return (int(cp.x * res_x), int((1-cp.y) * res_y))
    return (None, None)

def obj_bounding_box(scene, camera, obj):
    depsgraph = bpy.context.evaluated_depsgraph_get() #Looks like this is only if modifiers and animations have been applied, it then updates this, based on the frame it is at. The scene will be updated based on the dependency graph 
    mesh_eval = obj.evaluated_get(depsgraph)

    mesh = mesh_eval.to_mesh() #Simply keep it like this, you do not need to specify arguments
    mesh.transform(obj.matrix_world)
    
    render = scene.render
    minx, maxx = render.resolution_x, 0
    miny, maxy = render.resolution_y, 0

    found = False
    update_matrices(camera)
    for v in mesh.vertices:
        c = world_to_camera_view(scene, camera, v.co)
        if c.z > 0:
            found = True
            if c.x < minx:
                minx = c.x
            if c.x > maxx:
                maxx = c.x
            if c.y < miny:
                miny = c.y
            if c.y > maxy:
                maxy = c.y
    
    if found:
        xs = np.clip([minx, maxx], 0, 1)
        ys = np.clip([1-maxy, 1-miny], 0, 1)

        bbox = [xs.sum()/2, ys.sum()/2]
        w = xs[1] - xs[0]
        h = ys[1] - ys[0]
        if w > 0 and h > 0:
            bbox.append(w)
            bbox.append(h)
            return bbox

    return None
    

COURT_LINES = []
def calc_court_lines():
    QUAD_WIDTH = 4.11 # width of all quadrant
    NETLINE = 6.40 # y distance from net to center line
    BASELINE = 5.48 # y distance from baseline to center line
    OUTSIDE_WIDTH = 5.44

    # Vertical lines
    for x in [-OUTSIDE_WIDTH ,-QUAD_WIDTH - 0.03, 0, QUAD_WIDTH + 0.03, OUTSIDE_WIDTH]:
        start_y, end_y = -BASELINE + 0.08, BASELINE
        COURT_LINES.append([[x, start_y], [x, end_y]])
    
    # Horizontal lines
    for y in [-BASELINE + 0.08, 0, BASELINE]:
        start_x, end_x = -OUTSIDE_WIDTH, OUTSIDE_WIDTH
        COURT_LINES.append([[start_x, y], [end_x, y]])

calc_court_lines()

def line_intersect(p1, q1, p2, q2):
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (p1[0] - q1[0], p2[0] - q2[0])
    ydiff = (p1[1] - q1[1], p2[1] - q2[1])

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines are parallel

    d = (det(p1, q1), det(p2, q2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Check if the intersection point is on both line segments
    if (min(p1[0], q1[0]) <= x <= max(p1[0], q1[0]) and
        min(p1[1], q1[1]) <= y <= max(p1[1], q1[1]) and
        min(p2[0], q2[0]) <= x <= max(p2[0], q2[0]) and
        min(p2[1], q2[1]) <= y <= max(p2[1], q2[1])):
        return (x, y)
    return None


def change_path(scene, path, IMGTYPES):
    path = os.path.join('YOLO_ball_box', path)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    for (i, t) in enumerate(IMGTYPES):
        scene.node_tree.nodes["File Output"].file_slots[i].path = os.path.join(path, 'images', t)
    print(f"Changed path to {path}")
    return os.path.join(path, 'labels')

def render_scene(case_id):
    # bpy.context.scene.render.filepath = os.path.join(directory, "test_imgs", "test_"+str(case_id)+".jpg")
    # print("path =", bpy.context.scene.render.filepath)
    bpy.ops.render.render()

def getRand(range):
    return np.random.uniform(range[0], range[1])

def getRandColour(cols):
    return tuple(np.append(cols[np.random.choice(len(cols))] + np.random.rand()/10, 1))

def generate_test_cases():
    CAMERA_HEIGHTS = [0.29, 0.34] # meters
    CAMERA_AODS = [21, 23, 25] # degreees (angle of depression)
    CAMERA_DISTORTIONS = [0.01, 0.04] # barrel distortion
    NUM_BALLSS = [0, 10]
    
    # RGB material colours (from real sample images)
    GREENS = np.array([
        [85,98,59],
        [178,190,150],
        [253,253,250],
        [210,216,194],
        [100, 200, 50],
        [223,255,79],
        [200,220,79]
    ])/255

    BLUES = np.array([
        [128, 119, 136],
        [93,90,143],
        [161,161,165],
        [78,69,92],
        [100, 100, 200],
        [10, 10, 250],
        [30, 135, 213]
    ])/255

    BROWNS =  np.array([
        [210, 180, 140],
        [250, 245, 230]
    ]) / 255

    CLOTHE_COLORS = np.array([
        [45, 45, 45],
        [233, 228, 223],
        [58, 58, 58],
        [226, 221, 216],
        [37, 37, 37],
        [229, 224, 219],
        [86, 66, 56],
        [73, 93, 103],
        [122, 82, 82],
        [68, 88, 98]
    ]) / 255

    LACES_COLOURS = np.array([
        [10, 10 ,10],
        [250, 250, 250]
    ]) / 255

    # BALL_BOUNDARY = 1 # allow balls this many meters outside test boundary
    NUM_SCENARIOS = 200
    MIN_FRAME = 200
    MIN_BALL_DIST = 0.4
    MAX_BALL_DIST = 4.0
    BALL_DIAMETER = 0.068
    IMGTYPES = ['normal', 'noise']

    objs = bpy.data.objects
    scene = bpy.data.scenes['Scene']
    mats = bpy.data.materials

    cases = []
    camera = objs['Camera']
    cam_fov = math.degrees(camera.data.angle) # type: ignore
    ball_template = objs['tennis-ball']
    boundary = objs['test-boundary']
    dims = boundary.dimensions
    center = boundary.matrix_world.to_translation()
    min_x = center.x - dims.x/2
    max_x = center.x + dims.x/2
    min_y = center.y - dims.y/2
    max_y = center.y + dims.y/2

    # Get focal lengths
    scale = scene.render.resolution_percentage / 100
    f_x = camera.data.lens * scale / camera.data.sensor_width #type:ignore * scene.render.resolution_x
    aspect_ratio = scene.render.resolution_x / scene.render.resolution_y

    current_label_path = change_path(scene, 'train', IMGTYPES)

    # remove all tennis balls (apart from template)
    for obj in objs:
        if 'tennis-ball.' in obj.name:
            objs.remove(objs[obj.name], do_unlink=True)

    # generate new tennis balls
    tennis_balls = []
    for _ in range(NUM_BALLSS[1]):
        ball = ball_template.copy()
        bpy.context.collection.objects.link(ball)
        tennis_balls.append(ball)
    
    cardboard_box = objs['CardboardBox']
    legs = objs['Legs']

    # Generate test images
    for case_id in range(MIN_FRAME, MIN_FRAME + NUM_SCENARIOS):
        CAMERA_HEIGHT = getRand(CAMERA_HEIGHTS)
        CAMERA_AOD = getRand(CAMERA_AODS) # degreees (angle of depression)
        CAMERA_DISTORTION = getRand(CAMERA_DISTORTIONS)
        CAMERA_DISPERSION = getRand(CAMERA_DISTORTIONS)
        NUM_BALLS = np.random.randint(NUM_BALLSS[0], NUM_BALLSS[1]+1)
        
        GREEN = getRandColour(GREENS)
        BLUE = getRandColour(BLUES)
        BROWN = getRandColour(BROWNS)
        SHOE_COL = getRandColour(CLOTHE_COLORS)
        PANTS_COL = getRandColour(CLOTHE_COLORS)
        LACES_COL = getRandColour(LACES_COLOURS)

        # frame number is linked to output file names and some randomisation
        scene.frame_set(case_id)

        # Set barrel distortion
        scene.node_tree.nodes["Lens Distortion"].inputs[1].default_value = CAMERA_DISTORTION # type: ignore
        # Set lens distortion
        scene.node_tree.nodes["Lens Distortion"].inputs[2].default_value = CAMERA_DISPERSION # type: ignore

        mats["court"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = BLUE # type: ignore
        mats["Base"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = GREEN # type: ignore
        mats["Cardboard"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = BROWN # type: ignore
        mats["Shoe"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = SHOE_COL # type: ignore
        mats["Pants"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = PANTS_COL # type: ignore
        mats["Laces"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = LACES_COL # type: ignore

        filenames = [imgtype+f'{case_id:04d}' for imgtype in IMGTYPES]
        if case_id == MIN_FRAME + int(NUM_SCENARIOS * 0.7):
            current_label_path = change_path(scene, 'valid', IMGTYPES)
        elif case_id == MIN_FRAME + int(NUM_SCENARIOS * 0.85):
            current_label_path = change_path(scene, 'test', IMGTYPES)

        # Random Lighting
        objs["Sun"].data.energy = random.uniform(2, 9) #type:ignore
        objs["Area"].data.energy = random.uniform(800, 2000) #type:ignore
        objs["Sun"].rotation_euler.x = np.radians(random.uniform(-60, 60))
        
        if case_id % 5 == 0:
            cam_location = Vector((
                random.choice([-1, 1]) * random.uniform(0.8, 2.5),
                random.choice([-1, 1]) * random.uniform(0.8, 2.5),
                CAMERA_HEIGHT
            ))
            cam_heading = random.uniform(-5, 5) + np.degrees(np.arctan2(-cam_location.y, -cam_location.x))
        else:
            cam_location = Vector((
                random.uniform(min_x, max_x),
                random.uniform(min_y, max_y),
                CAMERA_HEIGHT
            ))
            cam_heading = random.uniform(-110, 110) + np.degrees(np.arctan2(-cam_location.y, -cam_location.x))
        
        camera.location = cam_location
        camera.rotation_euler = Euler((math.radians(90 - CAMERA_AOD), 0, math.radians(cam_heading)), 'XYZ')
        # camera.keyframe_insert('rotation_euler')
        # camera.keyframe_insert('location')

        # Update legs location every 6 frames
        if case_id % 6 == 0:
            leg_angle = cam_heading + random.uniform(-cam_fov / 2 * 0.9, cam_fov / 2 * 0.9)
            leg_distance = random.uniform(MIN_BALL_DIST + 0.5, MAX_BALL_DIST)
            leg_heading = random.uniform(0, 359)

            leg_location = Vector((
                camera.location.x + leg_distance * math.cos(math.radians(leg_angle)),
                camera.location.y + leg_distance * math.sin(math.radians(leg_angle)),
                0
            ))

            legs.location = leg_location
            legs.rotation_euler = Euler((0, 0, math.radians(leg_heading)), 'XYZ')
        

        ball_info = []
        YOLO_objects = []

        for (ball_id, ball) in enumerate(tennis_balls):

            # give random location within FOV of camera
            # avoid borders of frame for now
            if ball_id >= NUM_BALLS:
                ball.location = Vector((0, 0, 0))
                # ball.keyframe_insert('location')
                ball.hide_render = True
                continue
            else:
                ball.hide_render = False
            
            ball_angle = cam_heading + random.uniform(-cam_fov / 2 * 0.9, cam_fov / 2 * 0.9)
            ball_distance = random.uniform(MIN_BALL_DIST, MAX_BALL_DIST)

            ball_location = Vector((
                camera.location.x + ball_distance * math.cos(math.radians(ball_angle)),
                camera.location.y + ball_distance * math.sin(math.radians(ball_angle)),
                ball_template.location.z
            ))
        
            ball.location = ball_location
            # ball.keyframe_insert('location')

            # Test if ball is visible
            x, y = point_in_camera_view(scene, camera, ball_location)

            if  x is not None:

                line_intersects = []
                for l in COURT_LINES:
                    intersection = line_intersect(
                        [camera.location.x, camera.location.y], # mid-point at bottom of frame
                        [ball_location.x, ball_location.y], # ball location in pixel coordinates
                        l[0], # Court line endpoints
                        l[1]
                    )
                    if intersection is not None:
                        xint, yint = point_in_camera_view(scene, camera, Vector((intersection[0], intersection[1], 0)))
                        if xint is not None:
                            line_intersects.append([xint, yint])

                ball_info.append({
                        "ball_id": ball.name,
                        "world": list(ball_location[:]),
                        "image": [x,y],
                        "in_bounds": (min_x <= ball_location.x <= max_x) and (min_y <= ball_location.y <= max_y),
                        "court_line_intersects": line_intersects
                    })
            
                # Get bounding box for YOLO training data
                cam_c = np.array(world_to_camera_view(scene, camera, ball.location))
                pixel_width = BALL_DIAMETER * f_x / cam_c[2]
                pixel_height = pixel_width * aspect_ratio

                # YOLO bounding box descriptor in form [class, x, y, width, height]
                YOLO_objects.append([0, cam_c[0], (1-cam_c[1])*0.961, pixel_width * 1.08, pixel_height * 1.14])
        
        # Add bounding boxes around carboard box
        cardboard_box_bbox = obj_bounding_box(scene, camera, cardboard_box)
        if cardboard_box_bbox is not None:
            YOLO_objects.append([1] + cardboard_box_bbox)

        # Add bounding boxes around legs box
        legs_bbox = obj_bounding_box(scene, camera, objs["LegsBbox"])
        if legs_bbox is not None:
            YOLO_objects.append([2] + legs_bbox)

        case = {
            "caseID": case_id,
            "cam_heading": cam_heading,
            "cam_aod": CAMERA_AOD,
            "cam_location": list(camera.location.to_tuple()),
            "balls": ball_info
        }

        # Add YOLO training data
        for fname in filenames:
            with open(os.path.join(current_label_path, fname+'.txt'), 'w') as f:
                lines = [' '.join([str(num) for num in d])+'\n' for d in YOLO_objects]
                f.writelines(lines)
        
        cases.append(case)
        render_scene(case_id)

    return cases

def save_cases_to_json(cases):
    with open("YOLO_ball_box/cases.json", "w") as f:
        json.dump(cases, f, indent=2)

# Generate and save test cases
test_cases = generate_test_cases()
save_cases_to_json(test_cases)

print("Test cases generated and saved to cases.json")