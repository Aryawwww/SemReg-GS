"""Render synchronized HSSD RGB, semantics, depth, normals, and cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


SEMANTIC_CLASSES = {
    "wall": {"id": 0, "rgb": [217, 46, 46]},
    "floor": {"id": 1, "rgb": [46, 179, 64]},
    "ceiling": {"id": 2, "rgb": [51, 107, 230]},
    "door": {"id": 3, "rgb": [242, 148, 26]},
    "window": {"id": 4, "rgb": [26, 209, 224]},
    "other": {"id": 5, "rgb": [115, 115, 122]},
}


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--semantic-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args(argv)


def classify(object_name: str, material_name: str, normal: Vector, center: Vector) -> str:
    combined = f"{object_name} {material_name}".lower()
    if any(token in combined for token in ("fp_door", "doorframe", "doorhandle")):
        return "door"
    if any(token in combined for token in ("fp_glass", "glassborder", "window")):
        return "window"
    if "ceiling" in combined:
        return "ceiling"
    if "wall" in combined:
        return "wall"
    structural = object_name.lower().startswith("geometry_")
    if structural and abs(normal.z) >= 0.85 and center.z <= 0.08:
        return "floor"
    if structural and abs(normal.z) >= 0.85 and center.z >= 2.70:
        return "ceiling"
    if structural and abs(normal.z) <= 0.30:
        return "wall"
    return "other"


def point_inside_room(region: dict) -> Vector:
    # HSSD semantic config is (x, vertical-y, z); Blender import is (x, -z, y).
    points = region["poly_loop"]
    x = sum(point[0] for point in points) / len(points)
    y = -sum(point[2] for point in points) / len(points)
    return Vector((x, y, 1.50))


def camera_specifications(regions: list[dict]) -> tuple[list[dict], list[dict]]:
    by_name = {region["name"]: region for region in regions}
    donor_rooms = ["living room", "bedroom", "kitchen"]
    donor_yaws = [205.0, 35.0, 145.0]
    donors = [
        {"name": f"reference_{index:02d}", "room": room, "position": point_inside_room(by_name[room]), "yaw": yaw}
        for index, (room, yaw) in enumerate(zip(donor_rooms, donor_yaws))
    ]

    target_plan = [
        ("living room", 25.0),
        ("living room", 205.0),
        ("bedroom", 35.0),
        ("kitchen", 145.0),
        ("hallway", 90.0),
        ("hallway", 270.0),
        ("bathroom", 135.0),
        ("utilityroom", 315.0),
    ]
    targets = [
        {"name": f"view_{index:02d}", "room": room, "position": point_inside_room(by_name[room]), "yaw": yaw}
        for index, (room, yaw) in enumerate(target_plan)
    ]
    return donors, targets


def configure_camera(camera: bpy.types.Object, spec: dict) -> None:
    camera.location = spec["position"]
    yaw = math.radians(spec["yaw"])
    target = camera.location + Vector((math.cos(yaw), math.sin(yaw), -0.08))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def camera_metadata(camera: bpy.types.Object, width: int, height: int, spec: dict) -> dict:
    sensor_width = camera.data.sensor_width
    fx = camera.data.lens / sensor_width * width
    fy = fx
    return {
        "name": spec["name"],
        "room": spec["room"],
        "width": width,
        "height": height,
        "intrinsics": {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0},
        "camera_to_world_blender": [list(row) for row in camera.matrix_world],
        "world_to_camera_blender": [list(row) for row in camera.matrix_world.inverted()],
        "coordinate_notes": "Blender world: +Z up; camera looks along local -Z with local +Y up.",
    }


def configure_compositor(scene: bpy.types.Scene, view_directory: Path) -> None:
    scene.use_nodes = True
    if scene.compositing_node_group is None:
        bpy.ops.node.new_compositing_node_group()
        scene.compositing_node_group = bpy.data.node_groups[-1]
    tree = scene.compositing_node_group
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    for pass_name, file_name, socket_type in (
        ("Depth", "depth", "FLOAT"),
        ("Normal", "normal", "VECTOR"),
    ):
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.directory = str(view_directory)
        output.file_name = file_name
        output.use_file_extension = True
        output.format.color_depth = "32"
        output.file_output_items.new(socket_type, pass_name)
        tree.links.new(layers.outputs[pass_name], output.inputs[pass_name])


def render_rgb_passes(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    groups: list[tuple[str, list[dict]]],
    output_root: Path,
    width: int,
    height: int,
) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.use_nodes = False
    scene.compositing_node_group = None
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.frame_set(1)
    for group, specs in groups:
        for spec in specs:
            view_directory = output_root / group / spec["name"]
            view_directory.mkdir(parents=True, exist_ok=True)
            configure_camera(camera, spec)
            bpy.ops.render.render(write_still=False)
            bpy.data.images["Render Result"].save_render(filepath=str(view_directory / "rgb.png"), scene=scene)
            (view_directory / "camera.json").write_text(
                json.dumps(camera_metadata(camera, width, height, spec), indent=2) + "\n",
                encoding="utf-8",
            )


def render_geometry_passes(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    groups: list[tuple[str, list[dict]]],
    output_root: Path,
) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_normal = True
    for group, specs in groups:
        for spec in specs:
            view_directory = output_root / group / spec["name"]
            configure_camera(camera, spec)
            configure_compositor(scene, view_directory)
            bpy.ops.render.render(write_still=False)


def validate_rgb_outputs(groups: list[tuple[str, list[dict]]], output_root: Path) -> dict:
    records = []
    hashes = set()
    errors = []
    for group, specs in groups:
        for spec in specs:
            path = output_root / group / spec["name"] / "rgb.png"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes.add(digest)
            image = bpy.data.images.load(str(path), check_existing=False)
            pixels = image.pixels[:]
            channels = image.channels
            rgb = pixels if channels == 3 else [value for index, value in enumerate(pixels) if index % channels < 3]
            quantized = {max(0, min(255, round(value * 255.0))) for value in rgb}
            record = {
                "group": group,
                "view": spec["name"],
                "sha256": digest,
                "minimum": min(rgb),
                "maximum": max(rgb),
                "unique_8bit_values": len(quantized),
            }
            records.append(record)
            if record["maximum"] <= 0.01 and record["unique_8bit_values"] <= 4:
                errors.append(f"{group}/{spec['name']}: RGB is effectively a 0/1 image")
            bpy.data.images.remove(image)
    if len(hashes) < 2:
        errors.append("all RGB views have identical file hashes")
    if max(record["unique_8bit_values"] for record in records) < 16:
        errors.append("the RGB view set has insufficient color variation")
    report = {
        "status": "passed" if not errors else "failed",
        "view_count": len(records),
        "distinct_file_hashes": len(hashes),
        "views": records,
        "errors": errors,
    }
    (output_root / "rgb_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("RGB validation failed: " + "; ".join(errors))
    return report


def apply_semantic_materials(meshes: list[bpy.types.Object]) -> None:
    def srgb_to_linear(value: int) -> float:
        channel = value / 255.0
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    semantic_materials = {}
    for label, spec in SEMANTIC_CLASSES.items():
        material = bpy.data.materials.new(f"SEM_{label}")
        material.diffuse_color = tuple(srgb_to_linear(channel) for channel in spec["rgb"]) + (1.0,)
        semantic_materials[label] = material

    for obj in meshes:
        original = [slot.material.name if slot.material else "" for slot in obj.material_slots]
        base_index = len(obj.data.materials)
        for label in SEMANTIC_CLASSES:
            obj.data.materials.append(semantic_materials[label])
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        for polygon in obj.data.polygons:
            material_name = original[polygon.material_index] if polygon.material_index < len(original) else ""
            label = classify(
                obj.name,
                material_name,
                (normal_matrix @ polygon.normal).normalized(),
                obj.matrix_world @ polygon.center,
            )
            polygon.material_index = base_index + SEMANTIC_CLASSES[label]["id"]


def render_semantic_passes(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    groups: list[tuple[str, list[dict]]],
    output_root: Path,
) -> None:
    scene.use_nodes = False
    scene.compositing_node_group = None
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = False
    scene.render.image_settings.file_format = "PNG"
    for group, specs in groups:
        for spec in specs:
            configure_camera(camera, spec)
            scene.render.filepath = str(output_root / group / spec["name"] / "semantic.png")
            bpy.ops.render.render(write_still=True)


def main() -> None:
    args = arguments()
    output_root = args.output.resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    semantic_config = json.loads(args.semantic_config.read_text(encoding="utf-8"))
    donors, targets = camera_specifications(semantic_config["region_annotations"])
    groups = [("donor_reference", donors), ("target_views", targets)]

    camera_data = bpy.data.cameras.new("RenderCamera")
    camera_data.lens = 28.0
    camera_data.sensor_width = 36.0
    camera_data.clip_start = 0.05
    camera = bpy.data.objects.new("RenderCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    for region in semantic_config["region_annotations"]:
        position = point_inside_room(region)
        light_data = bpy.data.lights.new(f"Fill_{region['name']}", "AREA")
        light_data.energy = 220.0
        light_data.shape = "DISK"
        light_data.size = 2.0
        light = bpy.data.objects.new(f"Fill_{region['name']}", light_data)
        bpy.context.collection.objects.link(light)
        light.location = Vector((position.x, position.y, 2.55))
        light.rotation_euler = (0.0, 0.0, 0.0)

    scene = bpy.context.scene
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    if scene.world is None:
        scene.world = bpy.data.worlds.new("RenderWorld")
    scene.world.color = (0.08, 0.08, 0.08)
    render_rgb_passes(scene, camera, groups, output_root, args.width, args.height)
    rgb_validation = validate_rgb_outputs(groups, output_root)
    render_geometry_passes(scene, camera, groups, output_root)
    apply_semantic_materials(meshes)
    render_semantic_passes(scene, camera, groups, output_root)

    manifest = {
        "source_scene": str(args.input.resolve()),
        "resolution": [args.width, args.height],
        "semantic_classes": SEMANTIC_CLASSES,
        "donor_reference_count": len(donors),
        "target_view_count": len(targets),
        "rgb_validation": rgb_validation,
        "donor_references": [{**spec, "position": list(spec["position"])} for spec in donors],
        "target_views": [{**spec, "position": list(spec["position"])} for spec in targets],
        "modalities": [
            "rgb.png",
            "semantic.png",
            "depth.exr",
            "normal.exr",
            "camera.json",
        ],
    }
    (output_root / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"donor_references": len(donors), "target_views": len(targets)}))


if __name__ == "__main__":
    main()
