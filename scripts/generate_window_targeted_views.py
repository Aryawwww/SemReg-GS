"""Append donor/train/held-out cameras that explicitly observe window geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from render_hssd_multiview import (
    apply_semantic_materials,
    camera_metadata,
    classify,
    point_inside_room,
    render_geometry_passes,
    render_rgb_passes,
    render_semantic_passes,
    validate_rgb_outputs,
)


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--semantic-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--camera-distance", type=float, default=1.25)
    parser.add_argument("--lateral-offset", type=float, default=0.30)
    parser.add_argument("--window-candidates", type=int, default=3)
    parser.add_argument("--maximum-window-area", type=float, default=1.0)
    parser.add_argument("--maximum-proxy-distance", type=float, default=0.50)
    return parser.parse_args(argv)


def find_window_targets(meshes: list[bpy.types.Object], count: int, maximum_area: float) -> tuple[list[Vector], dict]:
    candidates = []
    for obj in meshes:
        material_names = [slot.material.name if slot.material else "" for slot in obj.material_slots]
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        weighted_center = Vector((0.0, 0.0, 0.0))
        total_area = 0.0
        face_count = 0
        matched_materials = set()
        for polygon in obj.data.polygons:
            material_name = material_names[polygon.material_index] if polygon.material_index < len(material_names) else ""
            world_normal = (normal_matrix @ polygon.normal).normalized()
            world_center = obj.matrix_world @ polygon.center
            if classify(obj.name, material_name, world_normal, world_center) != "window":
                continue
            matched_materials.add(material_name)
            area = max(float(polygon.area), 1e-8)
            weighted_center += world_center * area
            total_area += area
            face_count += 1
        if face_count:
            candidates.append({
                "object": obj.name,
                "center": weighted_center / total_area,
                "area": total_area,
                "face_count": face_count,
                "materials": sorted(matched_materials),
            })
    if not candidates:
        raise RuntimeError("No faces classified as window were found")
    plausible = [item for item in candidates if item["area"] <= maximum_area]
    if not plausible:
        plausible = candidates
    plausible.sort(key=lambda item: item["area"], reverse=True)
    selected = []
    for candidate in plausible:
        if any((candidate["center"] - existing["center"]).length < 0.50 for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    if not selected:
        raise RuntimeError("No spatially distinct window candidates were found")
    metadata = {
        "selection_rule": "largest spatially distinct classified-window objects below maximum_window_area",
        "maximum_window_area": maximum_area,
        "selected_objects": [
            {
                "object": item["object"],
                "face_count": item["face_count"],
                "local_area": item["area"],
                "center": list(item["center"]),
                "materials": item["materials"],
            }
            for item in selected
        ],
        "candidate_objects": [
            {"object": item["object"], "face_count": item["face_count"], "local_area": item["area"], "materials": item["materials"]}
            for item in sorted(candidates, key=lambda item: item["area"], reverse=True)
        ],
    }
    return [item["center"] for item in selected], metadata


def targeted_specs(window_center: Vector, regions: list[dict], distance: float, lateral: float, index: int) -> tuple[list[dict], list[dict], dict]:
    room_centers = [(region["name"], point_inside_room(region)) for region in regions]
    room_name, room_center = min(room_centers, key=lambda item: (item[1].xy - window_center.xy).length)
    inward = Vector((room_center.x - window_center.x, room_center.y - window_center.y, 0.0))
    if inward.length < 1e-6:
        raise RuntimeError("Could not determine an inward direction for the selected window")
    inward.normalize()
    lateral_axis = Vector((-inward.y, inward.x, 0.0))
    camera_height = float(room_center.z)
    base = Vector((window_center.x, window_center.y, camera_height)) + inward * distance
    target = window_center.copy()

    def make(name: str, offset: float, role: str) -> dict:
        position = base + lateral_axis * offset
        return {
            "name": name,
            "room": room_name,
            "role": role,
            "position": position,
            "target": target,
        }

    suffix = f"{index:02d}"
    donors = [make(f"reference_window_{suffix}", 0.0, "donor")]
    targets = [
        make(f"view_window_train_{suffix}", -lateral, "train_candidate"),
        make(f"view_window_heldout_{suffix}", lateral, "heldout_candidate"),
    ]
    placement = {
        "nearest_room": room_name,
        "window_center": list(window_center),
        "camera_height": camera_height,
        "target_height": float(window_center.z),
        "inward_direction": list(inward),
        "camera_distance": distance,
        "lateral_offset": lateral,
    }
    return donors, targets, placement


def serializable_spec(spec: dict) -> dict:
    return {
        key: list(value) if isinstance(value, Vector) else value
        for key, value in spec.items()
    }


def visible_window_proxy_faces(scene: bpy.types.Scene, specs: list[dict], maximum_distance: float) -> tuple[dict[str, set[int]], list[dict]]:
    """Find visible faces in front of known FP_GLASS targets for label propagation."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    forced: dict[str, set[int]] = {}
    records = []
    for spec in specs:
        origin = Vector(spec["position"])
        target = Vector(spec["target"])
        direction = target - origin
        distance = direction.length
        direction.normalize()
        hit, location, normal, polygon_index, obj, _ = scene.ray_cast(
            depsgraph, origin, direction, distance=distance + 0.25
        )
        record = {"view": spec["name"], "hit": bool(hit)}
        if hit and obj is not None and obj.type == "MESH" and polygon_index >= 0:
            target_distance = float((location - target).length)
            polygon = obj.data.polygons[polygon_index]
            original_material_index = polygon.material_index
            related = {
                candidate.index
                for candidate in obj.data.polygons
                if candidate.material_index == original_material_index
                and ((obj.matrix_world @ candidate.center) - target).length <= 2.0
            }
            accepted = target_distance <= maximum_distance
            if accepted:
                forced.setdefault(obj.name, set()).update(related or {polygon_index})
            material_name = ""
            if original_material_index < len(obj.material_slots) and obj.material_slots[original_material_index].material:
                material_name = obj.material_slots[original_material_index].material.name
            record.update({
                "object": obj.name,
                "hit_polygon": polygon_index,
                "propagated_polygon_count": len(related or {polygon_index}),
                "original_material": material_name,
                "hit_location": list(location),
                "distance_to_fp_glass_target": float((location - target).length),
                "accepted_as_window_proxy": accepted,
            })
        records.append(record)
    return forced, records


def apply_window_proxy_labels(meshes: list[bpy.types.Object], forced: dict[str, set[int]]) -> None:
    by_name = {obj.name: obj for obj in meshes}
    for object_name, polygon_indices in forced.items():
        obj = by_name.get(object_name)
        if obj is None:
            continue
        window_index = next(
            (index for index, material in enumerate(obj.data.materials) if material and material.name == "SEM_window"),
            None,
        )
        if window_index is None:
            raise RuntimeError(f"SEM_window material is missing from {object_name}")
        for polygon_index in polygon_indices:
            if polygon_index < len(obj.data.polygons):
                obj.data.polygons[polygon_index].material_index = window_index


def main() -> None:
    args = arguments()
    if args.camera_distance <= 0 or args.lateral_offset < 0:
        raise ValueError("camera-distance must be positive and lateral-offset must be non-negative")
    output_root = args.output.resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    scene = bpy.context.scene
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    semantic_config = json.loads(args.semantic_config.resolve().read_text(encoding="utf-8"))
    window_centers, window_metadata = find_window_targets(
        meshes, args.window_candidates, args.maximum_window_area
    )
    donors = []
    targets = []
    placements = []
    for index, window_center in enumerate(window_centers):
        candidate_donors, candidate_targets, placement = targeted_specs(
            window_center,
            semantic_config["region_annotations"],
            args.camera_distance,
            args.lateral_offset,
            index,
        )
        donors.extend(candidate_donors)
        targets.extend(candidate_targets)
        placements.append(placement)
    groups = [("donor_reference", donors), ("target_views", targets)]

    camera_data = bpy.data.cameras.new("WindowRenderCamera")
    camera_data.lens = 35.0
    camera_data.sensor_width = 36.0
    camera_data.clip_start = 0.05
    camera = bpy.data.objects.new("WindowRenderCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    if scene.world is None:
        scene.world = bpy.data.worlds.new("WindowRenderWorld")
    scene.world.color = (0.08, 0.08, 0.08)

    first_window_center = window_centers[0]
    room_center = point_inside_room(min(
        semantic_config["region_annotations"],
        key=lambda region: (point_inside_room(region).xy - first_window_center.xy).length,
    ))
    light_data = bpy.data.lights.new("WindowCameraFill", "AREA")
    light_data.energy = 300.0
    light_data.shape = "DISK"
    light_data.size = 2.0
    light = bpy.data.objects.new("WindowCameraFill", light_data)
    bpy.context.collection.objects.link(light)
    light.location = Vector((room_center.x, room_center.y, 2.40))

    render_rgb_passes(scene, camera, groups, output_root, args.width, args.height)
    rgb_validation = validate_rgb_outputs(groups, output_root)
    render_geometry_passes(scene, camera, groups, output_root)
    forced_window_faces, proxy_records = visible_window_proxy_faces(
        scene, donors + targets, args.maximum_proxy_distance
    )
    apply_semantic_materials(meshes)
    apply_window_proxy_labels(meshes, forced_window_faces)
    render_semantic_passes(scene, camera, groups, output_root)

    manifest = {
        "status": "rendered",
        "source_scene": str(args.input.resolve()),
        "resolution": [args.width, args.height],
        "window_detection": window_metadata,
        "placements": placements,
        "visible_window_proxy_labeling": proxy_records,
        "donor_views": [serializable_spec(spec) for spec in donors],
        "target_views": [serializable_spec(spec) for spec in targets],
        "rgb_validation": rgb_validation,
        "note": "Views are appended to the existing multiview directory; coverage audit decides the final split.",
    }
    (output_root / "window_view_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"window_objects": [item["object"] for item in window_metadata["selected_objects"]], "donor_views": len(donors), "target_views": len(targets)}))


if __name__ == "__main__":
    main()
