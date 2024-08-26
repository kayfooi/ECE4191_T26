import bpy # pip install fake-bpy-module-latest (to develop in VSCode)
import math
import random
import json
from mathutils import Vector, Euler
import os
from bpy_extras.object_utils import world_to_camera_view
import numpy as np

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
    return None

def change_path(scene, path, IMGTYPES):
    path = os.path.join('YOLO', path)
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
    CAMERA_AODS = [21, 23] # degreees (angle of depression)
    CAMERA_DISTORTIONS = [0.01, 0.04] # barrel distortion
    NUM_BALLSS = [0, 10]
    
    # RGB material colours (from real sample images)
    GREENS = np.array([
        [85,98,59],
        [178,190,150],
        [253,253,250],
        [210,216,194],
        [86,101,56],
        [77,95,52],
        [106,138,91],
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

    # BALL_BOUNDARY = 1 # allow balls this many meters outside test boundary
    NUM_SCENARIOS = 215
    MIN_CASE = 215*2
    MIN_BALL_DIST = 0.4
    MAX_BALL_DIST = 6.5
    BALL_DIAMETER = 0.068
    IMGTYPES = ['normal', 'hue', 'noise']

    np.random.seed(MIN_CASE) # repeatable cases
    random.seed(MIN_CASE)

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

    # Generate test images
    for case_id in range(MIN_CASE, NUM_SCENARIOS + MIN_CASE):
        CAMERA_HEIGHT = getRand(CAMERA_HEIGHTS)
        CAMERA_AOD = getRand(CAMERA_AODS) # degreees (angle of depression)
        CAMERA_DISTORTION = getRand(CAMERA_DISTORTIONS)
        CAMERA_DISPERSION = getRand(CAMERA_DISTORTIONS)
        np.random.exponential()
        NUM_BALLS = np.random.randint(NUM_BALLSS[0], NUM_BALLSS[1]+1)
        
        GREEN = getRandColour(GREENS)
        BLUE = getRandColour(BLUES)

        # frame number is linked to output file names and some randomisation
        scene.frame_set(case_id)

        # Set barrel distortion
        scene.node_tree.nodes["Lens Distortion"].inputs[1].default_value = CAMERA_DISTORTION # type: ignore
        # Set lens distortion
        scene.node_tree.nodes["Lens Distortion"].inputs[2].default_value = CAMERA_DISPERSION # type: ignore

        mats["court"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = BLUE # type: ignore
        mats["Base"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = GREEN # type: ignore

        

        filenames = [imgtype+f'{case_id:04d}' for imgtype in IMGTYPES]
        if case_id == int(NUM_SCENARIOS * 0.7):
            current_label_path = change_path(scene, 'valid', IMGTYPES)
        elif case_id == int(NUM_SCENARIOS * 0.85):
            current_label_path = change_path(scene, 'test', IMGTYPES)

        objs["Sun"].data.energy = random.uniform(2, 11) #type:ignore
        objs["Sun"].rotation_euler.x = np.radians(random.uniform(-50, 50))
        
        cam_heading = random.uniform(0, 360)
        cam_location = Vector((
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y),
            CAMERA_HEIGHT
        ))
        
        camera.location = cam_location
        camera.rotation_euler = Euler((math.radians(90 - CAMERA_AOD), 0, math.radians(cam_heading)), 'XYZ')
        # camera.keyframe_insert('rotation_euler')
        # camera.keyframe_insert('location')

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
            img_coord = point_in_camera_view(scene, camera, ball_location)
            if  img_coord is not None:
                ball_info.append({
                        "ball_id": ball.name,
                        "world": list(ball_location[:]),
                        "image": list(img_coord),
                        "in_bounds": (min_x <= ball_location.x <= max_x) and (min_y <= ball_location.y <= max_y)
                    })
            
                # Get bounding box for YOLO training data
                cam_c = np.array(world_to_camera_view(scene, camera, ball.location))
                pixel_width = BALL_DIAMETER * f_x / cam_c[2]
                pixel_height = pixel_width * aspect_ratio

                # YOLO bounding box descriptor in form [class, x, y, width, height]
                YOLO_objects.append([0, cam_c[0], (1-cam_c[1])*0.961, pixel_width * 1.08, pixel_height * 1.14])

        
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
    with open("YOLO/cases.json", "w") as f:
        json.dump(cases, f, indent=2)

# Generate and save test cases
test_cases = generate_test_cases()
save_cases_to_json(test_cases)

print("Test cases generated and saved to cases.json")