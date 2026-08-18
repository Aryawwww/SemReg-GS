"""Create the six-class SemReg-GS mapping and a colored HSSD preview."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Vector


CLASSES = {
    "wall": {"id": 0, "color": [0.85, 0.18, 0.18, 1.0]},
    "floor": {"id": 1, "color": [0.18, 0.70, 0.25, 1.0]},
    "ceiling": {"id": 2, "color": [0.20, 0.42, 0.90, 1.0]},
    "door": {"id": 3, "color": [0.95, 0.58, 0.10, 1.0]},
    "window": {"id": 4, "color": [0.10, 0.82, 0.88, 1.0]},
    "other": {"id": 5, "color": [0.45, 0.45, 0.48, 1.0]},
}


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    return parser.parse_args(argv)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def classify(object_name: str, material_name: str, normal: Vector, center: Vector) -> tuple[str, str]:
    obj = object_name.lower()
    mat = material_name.lower()
    combined = f"{obj} {mat}"

    if any(token in combined for token in ("fp_door", "doorframe", "doorhandle")):
        return "door", "official_glb_name"
    if any(token in combined for token in ("fp_glass", "glassborder", "window")):
        return "window", "official_glb_name"
    if "ceiling" in combined:
        return "ceiling", "official_glb_name"
    if "wall" in combined:
        return "wall", "official_glb_name"

    structural = obj.startswith("geometry_")
    if structural and abs(normal.z) >= 0.85 and center.z <= 0.08:
        return "floor", "structural_height_normal"
    if structural and abs(normal.z) >= 0.85 and center.z >= 2.70:
        return "ceiling", "structural_height_normal"
    if structural and abs(normal.z) <= 0.30:
        return "wall", "structural_vertical_normal"
    return "other", "fallback"


def main() -> None:
    args = arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    semantic_materials = {}
    for label, spec in CLASSES.items():
        material = bpy.data.materials.new(f"SEM_{label}")
        material.diffuse_color = spec["color"]
        semantic_materials[label] = material

    polygon_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    object_labels: dict[str, Counter[str]] = {}
    for obj in meshes:
        original_materials = [slot.material.name if slot.material else "" for slot in obj.material_slots]
        base_index = len(obj.data.materials)
        for label in CLASSES:
            obj.data.materials.append(semantic_materials[label])
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        labels = Counter()
        for polygon in obj.data.polygons:
            material_name = (
                original_materials[polygon.material_index]
                if polygon.material_index < len(original_materials)
                else ""
            )
            normal = (normal_matrix @ polygon.normal).normalized()
            center = obj.matrix_world @ polygon.center
            label, rule = classify(obj.name, material_name, normal, center)
            polygon.material_index = base_index + CLASSES[label]["id"]
            polygon_counts[label] += 1
            rule_counts[rule] += 1
            labels[label] += 1
        object_labels[obj.name] = labels

    # Keep ceiling labels in the mapping but hide ceiling-only meshes in the
    # diagnostic cutaway so interior classes remain visible from above.
    for obj in meshes:
        labels = object_labels[obj.name]
        if labels and labels.most_common(1)[0][0] == "ceiling":
            obj.hide_render = True

    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(map(min, zip(*corners)))
    maximum = Vector(map(max, zip(*corners)))
    dimensions = maximum - minimum
    center = (minimum + maximum) * 0.5

    mapping = {
        "schema_version": 1,
        "source_scene": str(args.input.resolve()),
        "classes": CLASSES,
        "precedence": ["door", "window", "ceiling", "wall", "floor", "other"],
        "rules": {
            "official_glb_name": "FP_DOOR/DOORFRAME/DOORHANDLE, FP_GLASS/GLASSBORDER/WINDOW, CEILING, WALL",
            "structural_height_normal": "geometry_* horizontal faces near z=0 or z=2.8 m",
            "structural_vertical_normal": "remaining vertical geometry_* faces",
            "fallback": "all remaining faces, including furniture",
        },
        "polygon_counts": dict(polygon_counts),
        "rule_counts": dict(rule_counts),
        "object_majority_labels": {
            name: labels.most_common(1)[0][0] if labels else "other"
            for name, labels in object_labels.items()
        },
        "limitations": [
            "FP_GLASS is mapped to window; glazed doors may require manual review.",
            "Structural floor/ceiling fallback uses HSSD's meter-scale z heights.",
            "Furniture and unresolved object template 224-132 are mapped to other.",
        ],
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    camera_data = bpy.data.cameras.new("SemanticCamera")
    camera = bpy.data.objects.new("SemanticCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    span = max(dimensions)
    camera.location = center + Vector((1.15 * span, -1.15 * span, 0.85 * span))
    camera.data.lens = 48
    look_at(camera, center)
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    preview_path = args.preview.resolve()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(preview_path)
    bpy.ops.render.render(write_still=True)
    print(json.dumps({"polygon_counts": dict(polygon_counts), "rule_counts": dict(rule_counts)}))


if __name__ == "__main__":
    main()
