import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-path", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--views-json", required=True, type=str)
    parser.add_argument(
        "--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"], type=str
    )
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--cycles-samples", default=32, type=int)
    return parser.parse_args(argv)


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if obj.parent is None:
            yield obj


def scene_bbox():
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in scene_meshes():
        found = True
        for coord in obj.bound_box:
            world = obj.matrix_world @ Vector(coord)
            bbox_min.x = min(bbox_min.x, world.x)
            bbox_min.y = min(bbox_min.y, world.y)
            bbox_min.z = min(bbox_min.z, world.z)
            bbox_max.x = max(bbox_max.x, world.x)
            bbox_max.y = max(bbox_max.y, world.y)
            bbox_max.z = max(bbox_max.z, world.z)
    if not found:
        raise RuntimeError("No mesh objects found in scene")
    return bbox_min, bbox_max


def reset_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in list(bpy.data.textures):
        bpy.data.textures.remove(texture, do_unlink=True)
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
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 4
    scene.cycles.transmission_bounces = 3
    scene.cycles.use_denoising = True

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.get_devices()
    prefs.compute_device_type = "CUDA"
    for device in prefs.devices:
        device.use = True


def load_object(path: str):
    if path.endswith(".glb") or path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    else:
        raise ValueError(f"Unsupported object path: {path}")


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1.0 / max(bbox_max - bbox_min)
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
    links.new(env.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    background.inputs["Strength"].default_value = 1.0


def set_camera(camera, elevation: float, azimuth: float, distance: float):
    elevation_rad = math.radians(elevation)
    azimuth_rad = math.radians(azimuth)
    x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = distance * math.sin(elevation_rad)
    camera.location = (x, y, z)
    direction = -camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_to(path: str, view_transform: str | None = None):
    scene = bpy.context.scene
    old_view_transform = scene.view_settings.view_transform
    old_look = scene.view_settings.look
    try:
        if view_transform is not None:
            scene.view_settings.view_transform = view_transform
            try:
                scene.view_settings.look = "None"
            except TypeError:
                pass
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True, animation=False)
    finally:
        scene.view_settings.view_transform = old_view_transform
        try:
            scene.view_settings.look = old_look
        except TypeError:
            pass


def find_output_node(node_tree):
    for node in node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL" and node.is_active_output:
            return node
    for node in node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            return node
    raise RuntimeError("No active material output node found")


def find_principled_node(node_tree):
    for node in node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def override_material_scalar(material, scalar_name: str):
    if not material.use_nodes:
        raise RuntimeError(f"Material {material.name} does not use nodes")

    node_tree = material.node_tree
    output = find_output_node(node_tree)
    principled = find_principled_node(node_tree)
    if principled is None:
        raise RuntimeError(f"Material {material.name} has no Principled BSDF node")

    surface_input = output.inputs["Surface"]
    original_from_socket = surface_input.links[0].from_socket if surface_input.links else None

    combine = node_tree.nodes.new("ShaderNodeCombineRGB")
    emission = node_tree.nodes.new("ShaderNodeEmission")
    combine.location = (principled.location.x + 450, principled.location.y + 120)
    emission.location = (principled.location.x + 700, principled.location.y + 120)

    scalar_input = principled.inputs[scalar_name]
    if scalar_input.is_linked:
        source_socket = scalar_input.links[0].from_socket
        node_tree.links.new(source_socket, combine.inputs["R"])
        node_tree.links.new(source_socket, combine.inputs["G"])
        node_tree.links.new(source_socket, combine.inputs["B"])
    else:
        value = float(scalar_input.default_value)
        combine.inputs["R"].default_value = value
        combine.inputs["G"].default_value = value
        combine.inputs["B"].default_value = value

    node_tree.links.new(combine.outputs["Image"], emission.inputs["Color"])
    while surface_input.links:
        node_tree.links.remove(surface_input.links[0])
    node_tree.links.new(emission.outputs["Emission"], surface_input)
    return {
        "material": material,
        "output": output,
        "original_from_socket": original_from_socket,
        "created_nodes": [combine, emission],
    }


def restore_material_override(state):
    material = state["material"]
    output = state["output"]
    node_tree = material.node_tree
    surface_input = output.inputs["Surface"]
    while surface_input.links:
        node_tree.links.remove(surface_input.links[0])
    if state["original_from_socket"] is not None:
        node_tree.links.new(state["original_from_socket"], surface_input)
    for node in state["created_nodes"]:
        if node.name in node_tree.nodes:
            node_tree.nodes.remove(node)


def render_scalar_pass(output_path: str, scalar_name: str):
    states = []
    try:
        seen = set()
        for obj in scene_meshes():
            for slot in obj.material_slots:
                material = slot.material
                if material is None:
                    continue
                if material.name in seen:
                    continue
                seen.add(material.name)
                states.append(override_material_scalar(material, scalar_name))
        render_to(output_path, view_transform="Raw")
    finally:
        for state in states:
            restore_material_override(state)


def main():
    args = parse_args()
    views = json.loads(Path(args.views_json).read_text())
    os.makedirs(args.output_dir, exist_ok=True)

    camera = reset_scene()
    configure_render(args.engine, args.resolution, args.cycles_samples)
    load_object(args.object_path)
    normalize_scene()

    for view in views:
        view_dir = Path(args.output_dir) / view["name"]
        view_dir.mkdir(parents=True, exist_ok=True)
        set_world_hdri(view["hdri"])
        set_camera(
            camera,
            elevation=float(view["elevation"]),
            azimuth=float(view["azimuth"]),
            distance=float(view["distance"]),
        )
        render_to(str(view_dir / "rgba.png"))
        render_scalar_pass(str(view_dir / "roughness.png"), "Roughness")
        render_scalar_pass(str(view_dir / "metallic.png"), "Metallic")
        (view_dir / "view.json").write_text(json.dumps(view, indent=2))


if __name__ == "__main__":
    main()
