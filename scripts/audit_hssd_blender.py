"""Import an HSSD GLB in Blender and write a machine-readable asset audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    return parser.parse_args(argv)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input))

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    vertices = sum(len(obj.data.vertices) for obj in meshes)
    polygons = sum(len(obj.data.polygons) for obj in meshes)

    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    minimum = Vector(map(min, zip(*corners)))
    maximum = Vector(map(max, zip(*corners)))
    dimensions = maximum - minimum
    center = (minimum + maximum) * 0.5

    materials = list(bpy.data.materials)
    material_records = []
    image_paths = set()
    for material in materials:
        node_types = []
        images = []
        if material.use_nodes and material.node_tree:
            for node in material.node_tree.nodes:
                node_types.append(node.bl_idname)
                if node.bl_idname == "ShaderNodeTexImage" and node.image:
                    image_name = node.image.filepath or node.image.name
                    images.append(image_name)
                    image_paths.add(image_name)
        material_records.append(
            {"name": material.name, "node_types": sorted(set(node_types)), "images": images}
        )

    audit = {
        "input": str(args.input.resolve()),
        "blender_version": bpy.app.version_string,
        "object_count": len(bpy.context.scene.objects),
        "mesh_count": len(meshes),
        "vertex_count": vertices,
        "polygon_count": polygons,
        "material_count": len(materials),
        "image_count": len(bpy.data.images),
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "dimensions": list(dimensions),
        "mesh_objects": [
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "polygons": len(obj.data.polygons),
                "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
            }
            for obj in meshes
        ],
        "materials": material_records,
        "image_paths": sorted(image_paths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    camera_data = bpy.data.cameras.new("AuditCamera")
    camera = bpy.data.objects.new("AuditCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    span = max(dimensions)
    camera.location = center + Vector((1.15 * span, -1.15 * span, 0.85 * span))
    camera.data.lens = 48
    look_at(camera, center)
    bpy.context.scene.camera = camera

    sun_data = bpy.data.lights.new("AuditSun", "SUN")
    sun_data.energy = 2.0
    sun = bpy.data.objects.new("AuditSun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(35), 0, math.radians(35))

    world = bpy.context.scene.world or bpy.data.worlds.new("AuditWorld")
    bpy.context.scene.world = world
    world.color = (0.08, 0.08, 0.08)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    preview_path = args.preview.resolve()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(preview_path)
    scene.render.film_transparent = False
    bpy.ops.render.render(write_still=True)
    print(json.dumps({key: audit[key] for key in (
        "blender_version", "mesh_count", "vertex_count", "polygon_count",
        "material_count", "image_count", "dimensions"
    )}))


if __name__ == "__main__":
    main()
