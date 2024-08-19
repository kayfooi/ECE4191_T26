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

def point_in_camera_view(camera, point):
    cp = world_to_camera_view(bpy.context.scene, camera, point) # camera point
    render = bpy.context.scene.render
    res_x, res_y = render.resolution_x, render.resolution_y
    if 0.0 < cp.x < 1.0 and 0.0 < cp.y < 1.0 and cp.z > 0:
        return (int(cp.x * res_x), int((1-cp.y) * res_y))
    return None

def render_scene(case_id):
    # bpy.context.scene.render.filepath = os.path.join(directory, "test_imgs", "test_"+str(case_id)+".jpg")
    # print("path =", bpy.context.scene.render.filepath)
    bpy.ops.render.render()

def generate_test_cases():
    CAMERA_HEIGHT = 0.32 # meters
    CAMERA_AOD = 24 # degreees (angle of depression)
    CAMERA_DISTORTION = 0.02 # barrel distortion
    # BALL_BOUNDARY = 1 # allow balls this many meters outside test boundary
    NUM_SCENARIOS = 20
    MIN_BALL_DIST = 0.5
    MAX_BALL_DIST = 3.0
    NUM_BALLS = 1

    objs = bpy.data.objects
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

    bpy.data.scenes["Scene"].node_tree.nodes["Lens Distortion"].inputs[1].default_value = CAMERA_DISTORTION # type: ignore

    # remove all tennis balls (apart from template)
    for obj in objs:
        if 'tennis-ball.' in obj.name:
            objs.remove(objs[obj.name], do_unlink=True)

    # generate new tennis balls
    tennis_balls = []
    for _ in range(NUM_BALLS):
        ball = ball_template.copy()
        bpy.context.collection.objects.link(ball)
        tennis_balls.append(ball)
    
    # Generate test images
    for case_id in range(NUM_SCENARIOS):
        # frame number is linked to output file names and some randomisation
        bpy.data.scenes['Scene'].frame_set(case_id)

        objs["Sun"].data.energy = random.uniform(1e4, 1e5) #type:ignore
        
        cam_heading = random.uniform(0, 360)
        cam_location = Vector((
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y),
            CAMERA_HEIGHT
        ))
        
        camera.location = cam_location
        camera.rotation_euler = Euler((math.radians(90 - CAMERA_AOD), 0, math.radians(cam_heading)), 'XYZ')
        camera.keyframe_insert('rotation_euler')
        camera.keyframe_insert('location')

        ball_info = []

        for ball in tennis_balls:

            # give random location within FOV of camera
            # avoid borders of frame for now
            ball_angle = cam_heading + random.uniform(-cam_fov / 2 * 0.9, cam_fov / 2 * 0.9)
            ball_distance = random.uniform(MIN_BALL_DIST, MAX_BALL_DIST)

            ball_location = Vector((
                camera.location.x + ball_distance * math.cos(math.radians(ball_angle)),
                camera.location.y + ball_distance * math.sin(math.radians(ball_angle)),
                ball_template.location.z
            ))
        
            ball.location = ball_location
            ball.keyframe_insert('location')

            # Test if ball is visible
            img_coord = point_in_camera_view(camera, ball_location)
            if  img_coord is not None:
                ball_info.append({
                        "ball_id": ball.name,
                        "world": list(ball_location[:]),
                        "image": list(img_coord),
                        "in_bounds": (min_x <= ball_location.x <= max_x) and (min_y <= ball_location.y <= max_y)
                    })
        
        case = {
            "caseID": case_id,
            "cam_heading": cam_heading,
            "cam_location": list(camera.location.to_tuple()),
            "balls": ball_info
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