import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-path", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_object(path: str):
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
        return
    if suffix == ".obj":
        bpy.ops.import_scene.obj(filepath=path)
        return
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
        return
    raise RuntimeError(f"unsupported_format:{suffix}")


def material_probe(material) -> dict:
    result = {
        "has_albedo": False,
        "has_normal": False,
        "has_roughness": False,
        "has_metallic": False,
        "texture_images": 0,
        "principled_nodes": 0,
    }
    if material is None:
        return result
    if not material.use_nodes or material.node_tree is None:
        return result
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            result["texture_images"] += 1
        if node.type != "BSDF_PRINCIPLED":
            continue
        result["principled_nodes"] += 1
        if node.inputs["Base Color"].is_linked or tuple(node.inputs["Base Color"].default_value[:3]) != (0.8, 0.8, 0.8):
            result["has_albedo"] = True
        if node.inputs["Normal"].is_linked:
            result["has_normal"] = True
        if node.inputs["Roughness"].is_linked or float(node.inputs["Roughness"].default_value) != 0.5:
            result["has_roughness"] = True
        if node.inputs["Metallic"].is_linked or float(node.inputs["Metallic"].default_value) != 0.0:
            result["has_metallic"] = True
    return result


def collect_scene_stats() -> dict:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    unique_materials = {}
    has_uv = False
    mesh_vertex_count = 0
    material_slot_count = 0
    texture_image_count = 0
    principled_nodes = 0
    has_albedo = False
    has_normal = False
    has_roughness = False
    has_metallic = False

    for obj in meshes:
        mesh_vertex_count += len(obj.data.vertices)
        if getattr(obj.data, "uv_layers", None) and len(obj.data.uv_layers) > 0:
            has_uv = True
        material_slot_count += len(obj.material_slots)
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.name in unique_materials:
                continue
            unique_materials[material.name] = material
            result = material_probe(material)
            texture_image_count += result["texture_images"]
            principled_nodes += result["principled_nodes"]
            has_albedo = has_albedo or result["has_albedo"]
            has_normal = has_normal or result["has_normal"]
            has_roughness = has_roughness or result["has_roughness"]
            has_metallic = has_metallic or result["has_metallic"]

    return {
        "mesh_object_count": len(meshes),
        "mesh_vertex_count": mesh_vertex_count,
        "has_mesh": len(meshes) > 0,
        "has_uv": has_uv,
        "material_slot_count": material_slot_count,
        "material_count_unique": len(unique_materials),
        "texture_image_count": texture_image_count,
        "principled_node_count": principled_nodes,
        "has_albedo": has_albedo,
        "has_normal": has_normal,
        "has_roughness": has_roughness,
        "has_metallic": has_metallic,
    }


def main():
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"object_path": args.object_path, "import_status": "ok"}
    try:
        reset_scene()
        import_object(args.object_path)
        result.update(collect_scene_stats())
    except Exception as exc:  # pragma: no cover - Blender runtime path
        result = {
            "object_path": args.object_path,
            "import_status": "error",
            "error": str(exc),
            "has_mesh": False,
            "has_uv": False,
            "material_slot_count": 0,
            "material_count_unique": 0,
            "texture_image_count": 0,
            "principled_node_count": 0,
            "has_albedo": False,
            "has_normal": False,
            "has_roughness": False,
            "has_metallic": False,
        }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
