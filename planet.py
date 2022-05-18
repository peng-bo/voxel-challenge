from scene import Scene
import taichi as ti
from taichi.math import *

MANUAL_SEED = 0
LIGHT_DIRECTION = vec3(ti.cos(MANUAL_SEED), ti.sin(23.5*3.1416/180), ti.sin(MANUAL_SEED))

scene = Scene(voxel_edges=0.0, exposure=1)
scene.set_floor(-1, vec3(0))
scene.set_directional_light(LIGHT_DIRECTION, 1, vec3(1))
scene.set_background_color(vec3(0))

@ti.func
def rand(p):
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)), dot(p, vec3(113.5, 271.9, 124.6)))
    return -1.0 + 2.0*fract(ti.sin(p*1.0)*43758.5453123)

@ti.func
def grad_noise(p):
    i, f = ti.floor(p), fract(p)
    g = [rand(i + vec3((n >> 0) % 2, (n >> 1) % 2, (n >> 2) % 2)) for n in range(8)]
    v = [dot(g[n], f - vec3((n >> 0) % 2, (n >> 1) % 2, (n >> 2) % 2)) for n in range(8)]
    return mix(mix(mix(v[0], v[1], smoothstep(0, 1, f.x)),
                   mix(v[2], v[3], smoothstep(0, 1, f.x)), smoothstep(0, 1, f.y)),
               mix(mix(v[4], v[5], smoothstep(0, 1, f.x)),
                   mix(v[6], v[7], smoothstep(0, 1, f.x)), smoothstep(0, 1, f.y)), smoothstep(0, 1, f.z))

@ti.func
def fbm(p):
    value, amplitude = 0.0, 0.5
    for _ in range(8):
        value += amplitude * grad_noise(p)
        p *= 2
        amplitude *= 0.5
    return value

@ti.func
def land_color(latitude):
    n = latitude / 15
    return vec3(0.5-0.5*n+0.145*n*n-0.005*n*n*n, 0.4-0.055*n-0.04*n*n+0.015*n*n*n, 0.2+0.16*n-0.2*n*n+0.04*n*n*n)

@ti.func
def planet(r, sea_level):
    for I in ti.grouped(ti.ndrange((-r, r), (-r, r), (-r, r))):
        i = I / 64
        mat, clr = 0, vec3(0)
        if I.norm() < r:
            u, v = ti.atan2(I.z/r, I.x/r), ti.acos(I.y/r) 
            if (h := fbm(i + MANUAL_SEED)) < sea_level:
                mat, clr = 1, ti.max(vec3(0.02, 0.05, 0.1), vec3(0.12, 0.3, 0.6) * ti.exp(h * 32))
                if ti.abs(degrees(v)-90) + 10*grad_noise(vec3(u, v, 10*h)*4) > 70:
                    clr = vec3(1)
            elif h < 0.004 + sea_level and dot(LIGHT_DIRECTION, I) < 0:
                mat, clr = 2, vec3(1, 0.8+ 0.2*grad_noise(I*1.0), 0.4+0.1*grad_noise(I*1.0))
            else:
                mat, clr = 1, land_color(ti.abs(degrees(v)-90)+20*fbm(vec3(u, v, 10*h)*4 + fbm(vec3(u, v, 10*h)*4)))
        scene.set_voxel(I, mat, clr)

@ti.func
def rotate(v, k, theta):
    return v * ti.cos(theta) + cross(k, v) * ti.sin(theta) + k * dot(k, v) * (1 - ti.cos(theta))

@ti.func
def cyclone(i:ti.template() , center, radius):
    if (d := distance(i, center)) < radius:
        theta = 2 * ti.log(d * 2) * center.y / ti.abs(center.y)
        i = rotate(i, normalize(center), -theta)

@ti.func
def cloud(r, density):
    for I in ti.grouped(ti.ndrange((-r, r), (-r, r), (-r, r))):
        scale = 8
        i = I/scale
        [cyclone(i, normalize(rand(vec3((n>>0)%2, (n>>1)%2, (n>>2)%2)))*r/scale, 3+ti.randn()*2) for n in range(8)]
        if r-1 < I.norm() < r:
            if (d := fbm(fbm(i+MANUAL_SEED)+i)) > density:
                mat, clr = scene.get_voxel(I)
                if mat == 0:
                    scene.set_voxel(I, 1, vec3(0.8))
                elif mat == 1:
                    scene.set_voxel(I, 1, mix(clr, vec3(0.8), smoothstep(0, 1, 5*fbm(fbm(d+i)+i))))
                else:
                    scene.set_voxel(I, 2, mix(clr, 0.2 * vec3(0.8), smoothstep(0, 1, 5*fbm(fbm(d+i)+i))))

@ti.kernel
def initialize_voxels(radius: ti.i32):
    planet(radius-1, 0.01*(ti.random()*2 + 1))
    cloud(radius-1, 0.01*(ti.random()*2 + 1))
    cloud(radius, 0.15 + 0.01*(ti.random()*2 + 1))

initialize_voxels(64)
scene.finish()