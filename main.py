from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0.01, exposure=3)
scene.set_floor(-1, vec3(1))
scene.set_background_color(vec3(1))
scene.set_directional_light((-1, 0, -1), 0, (1, 1, 1))

@ti.func
def frame(position, length, width):
    for l in ti.ndrange(length):
        for a, b in ti.ndrange(width, width):
            scene.set_voxel(position + vec3(l, a, b), 1, vec3(0))
            scene.set_voxel(position + vec3(b, l, a), 1, vec3(0))
            scene.set_voxel(position + vec3(a, b, l), 1, vec3(0))

@ti.func
def cube(position, side_length):
    for i,j,k in ti.ndrange(side_length, side_length, side_length):
        scene.set_voxel(vec3(i, j, k) + position, 1, vec3(0))

@ti.kernel
def initialize_voxels():
    camera_positon = vec3(64) #approximately
    origin = vec3(-64)

    for i, j, k  in ti.ndrange(8, 8, 8):
        pos = vec3(i, j, k) * 15 + origin + vec3(1) #shift
        frame(pos, 15, 1)
        
    for i, j, k  in ti.ndrange(8, 8, 8):
        pos = vec3(i, j, k) * 15 + origin
        cube(pos, 3)
    
    for i, j ,k in ti.ndrange((-64,64), (-64,64), (-64,64)):
        mat, _ = scene.get_voxel(vec3(i,j,k))
        if mat == 1:
            a = distance(vec3(i, j, k), camera_positon)/128/ti.sqrt(3)
            scene.set_voxel(vec3(i, j, k), 1, mix(vec3(0), vec3(1), a)) #decay

initialize_voxels()

scene.finish()

