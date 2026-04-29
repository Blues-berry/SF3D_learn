import argparse
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-path", required=True, type=str)
    parser.add_argument("--uv-albedo", required=True, type=str)
    parser.add_argument("--uv-normal", required=True, type=str)
    parser.add_argument("--uv-roughness", required=True, type=str)
    parser.add_argument("--uv-metallic", required=True, type=str)
    parser.add_argument("--hdri-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--elevation", required=True, type=float)
    parser.add_argument("--azimuth", required=True, type=float)
    parser.add_argument("--distance", required=True, type=float)
    parser.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"], type=str)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--cycles-samples", default=64, type=int)
    return parser.parse_args(argv)


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if obj.parent is None:
            yield obj


def reset_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image, do_unlink=True)
    camera_data = bpy.data.cameras.new(name="Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.lens = 35
    camera.data.sensor_width = 32
    return camera


def configure_render(engine: str, resolution: int, cycles_samples: int):
    scene = bpy.context.scene
    render = scene.render
    render.engine = engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100
    render.film_transparent = True
    scene.cycles.device = "GPU"
    scene.cycles.samples = cycles_samples
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transparent_max_bounces = 4
    scene.cycles.transmission_bounces = 3
    scene.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.get_devices()
    prefs.compute_device_type = "CUDA"
    requested_index = os.environ.get("BLENDER_CUDA_DEVICE_INDEX")
    for idx, device in enumerate(prefs.devices):
        device.use = requested_index is None or str(idx) == requested_index


def load_object(path: str):
    if path.endswith(".glb") or path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
        return
    if path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=path)
        return
    if path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=path)
        return
    raise ValueError(f"unsupported_object_path:{path}")


def scene_bbox():
    bbox_min = None
    bbox_max = None
    for obj in scene_meshes():
        for coord in obj.bound_box:
            world = obj.matrix_world @ Vector(coord)
            if bbox_min is None:
                bbox_min = world.copy()
                bbox_max = world.copy()
            else:
                bbox_min.x = min(bbox_min.x, world.x)
                bbox_min.y = min(bbox_min.y, world.y)
                bbox_min.z = min(bbox_min.z, world.z)
                bbox_max.x = max(bbox_max.x, world.x)
                bbox_max.y = max(bbox_max.y, world.y)
                bbox_max.z = max(bbox_max.z, world.z)
    if bbox_min is None or bbox_max is None:
        raise RuntimeError("no_mesh_objects_found")
    return bbox_min, bbox_max


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    span = bbox_max - bbox_min
    scale = 1.0 / max(span.x, span.y, span.z, 1e-6)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2.0
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.context.view_layer.update()


def set_world_hdri(hdri_path: str):
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path, check_existing=True)
    background.inputs["Strength"].default_value = 1.0
    links.new(env.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def set_camera(camera, elevation: float, azimuth: float, distance: float):
    import math
    from mathutils import Vector

    elevation_rad = math.radians(elevation)
    azimuth_rad = math.radians(azimuth)
    x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = distance * math.sin(elevation_rad)
    camera.location = (x, y, z)
    direction = -Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def load_image(path: str, *, colorspace: str) -> bpy.types.Image:
    image = bpy.data.images.load(path, check_existing=True)
    image.colorspace_settings.name = colorspace
    return image


def build_eval_material(
    *,
    uv_albedo: str,
    uv_normal: str,
    uv_roughness: str,
    uv_metallic: str,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name="EvalMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    albedo = nodes.new("ShaderNodeTexImage")
    roughness = nodes.new("ShaderNodeTexImage")
    metallic = nodes.new("ShaderNodeTexImage")
    normal = nodes.new("ShaderNodeTexImage")
    normal_map = nodes.new("ShaderNodeNormalMap")

    output.location = (900, 0)
    principled.location = (650, 0)
    texcoord.location = (-600, 0)
    albedo.location = (-350, 220)
    normal.location = (-350, 40)
    roughness.location = (-350, -140)
    metallic.location = (-350, -320)
    normal_map.location = (120, 40)

    albedo.image = load_image(uv_albedo, colorspace="sRGB")
    normal.image = load_image(uv_normal, colorspace="Non-Color")
    roughness.image = load_image(uv_roughness, colorspace="Non-Color")
    metallic.image = load_image(uv_metallic, colorspace="Non-Color")

    links.new(texcoord.outputs["UV"], albedo.inputs["Vector"])
    links.new(texcoord.outputs["UV"], normal.inputs["Vector"])
    links.new(texcoord.outputs["UV"], roughness.inputs["Vector"])
    links.new(texcoord.outputs["UV"], metallic.inputs["Vector"])
    links.new(albedo.outputs["Color"], principled.inputs["Base Color"])
    links.new(roughness.outputs["Color"], principled.inputs["Roughness"])
    links.new(metallic.outputs["Color"], principled.inputs["Metallic"])
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def assign_material(material: bpy.types.Material):
    for obj in scene_meshes():
        if not obj.data.materials:
            obj.data.materials.append(material)
            continue
        for index in range(len(obj.data.materials)):
            obj.data.materials[index] = material


def render_to(path: str):
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Filmic"
    try:
        scene.view_settings.look = "None"
    except TypeError:
        pass
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True, animation=False)


def main():
    args = parse_args()
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    camera = reset_scene()
    configure_render(args.engine, args.resolution, args.cycles_samples)
    load_object(args.object_path)
    normalize_scene()
    set_world_hdri(args.hdri_path)
    set_camera(camera, args.elevation, args.azimuth, args.distance)
    material = build_eval_material(
        uv_albedo=args.uv_albedo,
        uv_normal=args.uv_normal,
        uv_roughness=args.uv_roughness,
        uv_metallic=args.uv_metallic,
    )
    assign_material(material)
    render_to(args.output_path)


if __name__ == "__main__":
    main()
