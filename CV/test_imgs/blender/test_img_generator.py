import bpy # pip install fake-bpy-module-latest (to develop in VSCode)
import math
import random
import json
from mathutils import Vector, Euler
import os
from bpy_extras.object_utils import world_to_camera_view

random.seed(42) # repeatable cases
# Get the directory of the current .blend file
blend_file_path = bpy.data.filepath
directory = os.path.dirname(blend_file_path)

def is_point_in_camera_view(camera, point):
    cp = world_to_camera_view(bpy.context.scene, camera, point) # camera point
    if 0.0 < cp.x < 1.0 and 0.0 < cp.y < 1.0 and cp.z > 0:
        return True
    return False

def render_scene(case_id):
    # bpy.context.scene.render.filepath = os.path.join(directory, "test_imgs", "test_"+str(case_id)+".jpg")
    # print("path =", bpy.context.scene.render.filepath)
    bpy.ops.render.render()

def generate_test_cases():
    CAMERA_HEIGHT = 0.32 # meters
    CAMERA_AOD = 24 # degreees (angle of depression)
    CAMERA_DISTORTION = 0.04 # barrel distortion
    BALL_BOUNDARY = 1 # allow balls this many meters outside test boundary
    NUM_SCENARIOS = 50

    objs = bpy.data.objects
    cases = []
    camera = objs['Camera']
    ball_template = objs['tennis-ball']
    boundary = objs['test-boundary']
    dims = boundary.dimensions
    center = boundary.location
    min_x = center.x - dims.x/2
    max_x = center.x + dims.x/2
    min_y = center.y - dims.y/2
    max_y = center.y + dims.y/2

    bpy.data.scenes["Scene"].node_tree.nodes["Lens Distortion"].inputs[1].default_value = CAMERA_DISTORTION # type: ignore

    n_balls = 20

    # remove all tennis balls (apart from template)
    for obj in objs:
        if 'tennis-ball.' in obj.name:
            objs.remove(objs[obj.name], do_unlink=True)

    # generate new tennis balls
    tennis_balls = []
    for _ in range(n_balls):
        ball = ball_template.copy()
        bpy.context.collection.objects.link(ball)
        tennis_balls.append(ball)
    
    # Generate test images
    for case_id in range(NUM_SCENARIOS):
        # frame number is linked to output file names and some randomisation
        bpy.data.scenes['Scene'].frame_set(case_id + 1)
        
        cam_heading = random.uniform(0, 360)
        cam_location = Vector((
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y),
            CAMERA_HEIGHT
        ))
        
        camera.location = cam_location
        camera.rotation_euler = Euler((math.radians(90 - CAMERA_AOD), 0, math.radians(cam_heading)), 'XYZ')

        balls_valid = {}
        balls_invalid = {}

        for ball in tennis_balls:

            # give random location
            
            ball_location = Vector((
                random.uniform(min_x - BALL_BOUNDARY, max_x + BALL_BOUNDARY),
                random.uniform(min_y - BALL_BOUNDARY, max_y + BALL_BOUNDARY),
                ball_template.location.z
            ))
        
            ball.location = ball_location
            
            # Test if ball is visible
            if is_point_in_camera_view(camera, ball_location):
                if (min_x <= ball_location.x <= max_x) and (min_y <= ball_location.y <= max_y):
                    balls_valid[ball.name] = list(ball_location[:])
                else:
                    balls_invalid[ball.name] = list(ball_location[:])
        
        case = {
            "caseID": case_id,
            "cam_heading": cam_heading,
            "cam_location": list(camera.location.to_tuple()),
            "balls_valid_location": balls_valid,
            "balls_invalid_location": balls_invalid
        }

        cases.append(case)
        render_scene(case_id)
    
    return cases

def save_cases_to_json(cases):
    with open("cases.json", "w") as f:
        json.dump(cases, f, indent=2)

# Generate and save test cases
test_cases = generate_test_cases()
save_cases_to_json(test_cases)

print("Test cases generated and saved to cases.json")