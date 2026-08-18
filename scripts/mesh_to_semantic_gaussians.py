"""Sample CAD-anchored semantic Gaussians from a GLB mesh using Blender.

Run with:
    blender --background --python scripts/mesh_to_semantic_gaussians.py -- ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gaussians", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-position", action="store_true")
    parser.add_argument("--freeze-rotation", action="store_true")
    parser.add_argument("--freeze-scale", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def collect_triangles(class_ids: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles: list[np.ndarray] = []
    semantic_ids: list[int] = []
    areas: list[float] = []
    for obj in (item for item in bpy.context.scene.objects if item.type == "MESH"):
        mesh = obj.data
        mesh.calc_loop_triangles()
        original_materials = [slot.material.name if slot.material else "" for slot in obj.material_slots]
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        for triangle in mesh.loop_triangles:
            polygon = mesh.polygons[triangle.polygon_index]
            vertices = np.asarray(
                [obj.matrix_world @ mesh.vertices[index].co for index in triangle.vertices],
                dtype=np.float64,
            )
            cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            area = 0.5 * float(np.linalg.norm(cross))
            if not math.isfinite(area) or area <= 1e-12:
                continue
            material_name = (
                original_materials[polygon.material_index]
                if polygon.material_index < len(original_materials)
                else ""
            )
            normal = (normal_matrix @ polygon.normal).normalized()
            center = obj.matrix_world @ polygon.center
            label = classify(obj.name, material_name, normal, center)
            triangles.append(vertices)
            semantic_ids.append(class_ids[label])
            areas.append(area)
    if not triangles:
        raise RuntimeError("No non-degenerate triangles were found in the mesh")
    return (
        np.asarray(triangles, dtype=np.float64),
        np.asarray(semantic_ids, dtype=np.int16),
        np.asarray(areas, dtype=np.float64),
    )


def rotations_from_normals(normals: np.ndarray) -> np.ndarray:
    """Return wxyz quaternions rotating local +Z onto each normal."""
    rotations = np.zeros((len(normals), 4), dtype=np.float64)
    regular = normals[:, 2] > -0.999999
    rotations[regular, 0] = np.sqrt((1.0 + normals[regular, 2]) * 0.5)
    denominator = 2.0 * rotations[regular, 0]
    rotations[regular, 1] = -normals[regular, 1] / denominator
    rotations[regular, 2] = normals[regular, 0] / denominator
    rotations[~regular] = np.asarray([0.0, 1.0, 0.0, 0.0])
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    return rotations


def write_binary_ply(path: Path, arrays: dict[str, np.ndarray], properties: list[tuple[str, str]]) -> None:
    type_map = {
        "float": ("<f4", "float"),
        "uchar": ("u1", "uchar"),
        "int": ("<i4", "int"),
    }
    count = len(next(iter(arrays.values())))
    dtype = np.dtype([(name, type_map[kind][0]) for name, kind in properties])
    records = np.empty(count, dtype=dtype)
    for name, _ in properties:
        records[name] = arrays[name]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header.extend(f"property {type_map[kind][1]} {name}" for name, kind in properties)
    header.extend(["end_header", ""])
    with path.open("wb") as stream:
        stream.write("\n".join(header).encode("ascii"))
        stream.write(records.tobytes())


def main() -> None:
    args = arguments()
    if args.max_gaussians <= 0:
        raise ValueError("--max-gaussians must be positive")
    mesh_path = args.mesh.resolve()
    mapping_path = args.semantic_mapping.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_ids = {name: int(spec["id"]) for name, spec in mapping["classes"].items()}
    class_colors = {
        int(spec["id"]): np.rint(np.asarray(spec["color"][:3]) * 255.0).astype(np.uint8)
        for spec in mapping["classes"].values()
    }

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    triangles, triangle_semantics, triangle_areas = collect_triangles(class_ids)
    total_area = float(triangle_areas.sum())
    rng = np.random.default_rng(args.seed)
    source_face_id = rng.choice(
        len(triangles), size=args.max_gaussians, replace=True, p=triangle_areas / total_area
    ).astype(np.int32)
    selected = triangles[source_face_id]
    r1 = np.sqrt(rng.random(args.max_gaussians))
    r2 = rng.random(args.max_gaussians)
    barycentric = np.column_stack((1.0 - r1, r1 * (1.0 - r2), r1 * r2))
    xyz = np.einsum("ni,nij->nj", barycentric, selected)
    cross = np.cross(selected[:, 1] - selected[:, 0], selected[:, 2] - selected[:, 0])
    normals = cross / np.linalg.norm(cross, axis=1, keepdims=True)
    rotation = rotations_from_normals(normals)
    tangent_scale = math.sqrt(total_area / args.max_gaussians) * 0.8
    scale = np.tile(
        np.asarray([tangent_scale, tangent_scale, tangent_scale * 0.1]),
        (args.max_gaussians, 1),
    )
    opacity = np.full(args.max_gaussians, 0.1, dtype=np.float64)
    semantic_id = triangle_semantics[source_face_id].astype(np.int16)
    colors = np.stack([class_colors[int(value)] for value in semantic_id])

    np.savez_compressed(
        output / "gaussians.npz",
        xyz=xyz.astype(np.float32),
        normal=normals.astype(np.float32),
        rotation_wxyz=rotation.astype(np.float32),
        scale=scale.astype(np.float32),
        opacity=opacity.astype(np.float32),
        semantic_id=semantic_id,
        source_face_id=source_face_id,
        source_triangle=selected.astype(np.float32),
        barycentric=barycentric.astype(np.float32),
    )

    common = {
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
        "scale_0": scale[:, 0], "scale_1": scale[:, 1], "scale_2": scale[:, 2],
        "rot_0": rotation[:, 0], "rot_1": rotation[:, 1],
        "rot_2": rotation[:, 2], "rot_3": rotation[:, 3],
        "opacity": opacity, "semantic_id": semantic_id, "source_face_id": source_face_id,
    }
    gaussian_properties = [
        ("x", "float"), ("y", "float"), ("z", "float"),
        ("nx", "float"), ("ny", "float"), ("nz", "float"),
        ("scale_0", "float"), ("scale_1", "float"), ("scale_2", "float"),
        ("rot_0", "float"), ("rot_1", "float"), ("rot_2", "float"), ("rot_3", "float"),
        ("opacity", "float"), ("semantic_id", "int"), ("source_face_id", "int"),
    ]
    write_binary_ply(output / "gaussians.ply", common, gaussian_properties)
    color_arrays = {
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "red": colors[:, 0], "green": colors[:, 1], "blue": colors[:, 2],
        "semantic_id": semantic_id,
    }
    write_binary_ply(
        output / "semantic_colors.ply",
        color_arrays,
        [("x", "float"), ("y", "float"), ("z", "float"),
         ("red", "uchar"), ("green", "uchar"), ("blue", "uchar"),
         ("semantic_id", "int")],
    )

    counts = Counter(int(value) for value in semantic_id)
    id_to_name = {value: name for name, value in class_ids.items()}
    metadata = {
        "schema_version": 1,
        "source_mesh": str(mesh_path),
        "source_mesh_sha256": sha256(mesh_path),
        "semantic_mapping": str(mapping_path),
        "gaussian_count": args.max_gaussians,
        "source_triangle_count": len(triangles),
        "source_surface_area_m2": total_area,
        "seed": args.seed,
        "rotation_convention": "wxyz; rotates local +Z to surface normal",
        "scale_convention": "linear xyz; z is surface-normal axis",
        "initial_opacity": 0.1,
        "freeze": {
            "position": args.freeze_position,
            "rotation": args.freeze_rotation,
            "scale": args.freeze_scale,
        },
        "class_ids": class_ids,
        "semantic_counts": {id_to_name[key]: counts.get(key, 0) for key in sorted(id_to_name)},
        "files": ["gaussians.npz", "gaussians.ply", "semantic_colors.ply", "metadata.json"],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gaussian_count": args.max_gaussians, "semantic_counts": metadata["semantic_counts"]}))


if __name__ == "__main__":
    main()
