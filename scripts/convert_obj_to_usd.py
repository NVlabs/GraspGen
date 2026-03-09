# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Convert mesh files (OBJ, STL, PLY) to USD format using scene_synthesizer."""

import argparse
import os
import sys

import scene_synthesizer as synth

# Light blue (RGB 0-1) for --light-blue option.
LIGHT_BLUE_RGB = (0.68, 0.85, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a mesh file to USD format using scene_synthesizer"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input mesh file (obj, stl, or ply)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to the output USD file. If empty, replaces the input extension with .usd",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the mesh",
    )
    parser.add_argument(
        "--up-axis",
        type=str,
        default="Z",
        choices=["Y", "Z"],
        help="Up axis for the USD stage",
    )
    parser.add_argument(
        "--light-blue",
        action="store_true",
        help="Set the mesh to a light blue display color in the exported USD",
    )
    parser.add_argument(
        "--color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help="Set mesh display color in the exported USD (RGB 0-1, e.g. --color 0.68 0.85 1.0)",
    )
    return parser.parse_args()


def set_usd_mesh_display_color(usd_path: str, color_rgb: tuple) -> None:
    """Set displayColor on all Mesh prims in a USD file.

    Args:
        usd_path: Path to the USD file (modified in place).
        color_rgb: (r, g, b) in 0-1 range.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.Open(usd_path)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            primvars_api = UsdGeom.PrimvarsAPI(prim)
            display_color = primvars_api.CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3f
            )
            display_color.Set(Vt.Vec3fArray([Gf.Vec3f(*color_rgb)]))
    stage.GetRootLayer().Save()


def convert_mesh_to_usd(
    input_path: str,
    output_path: str,
    scale: float = 1.0,
    up_axis: str = "Z",
    display_color_rgb: tuple = None,
):
    """Convert a mesh file to USD format.

    Args:
        input_path: Path to the input mesh file.
        output_path: Path to the output USD file.
        scale: Scale factor to apply.
        up_axis: Up axis for the USD stage ('Y' or 'Z').
        display_color_rgb: Optional (r, g, b) in 0-1 to set on all mesh prims.

    Returns:
        The output USD file path.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    asset = synth.Asset(input_path, scale=scale)
    scene = asset.scene()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scene.export(output_path, file_type=output_path.rsplit(".", 1)[-1], up_axis=up_axis)

    if display_color_rgb is not None:
        set_usd_mesh_display_color(output_path, display_color_rgb)
        print(f"Set display color to RGB{display_color_rgb}")

    print(f"Converted {input_path} -> {output_path}")
    return output_path


if __name__ == "__main__":
    args = parse_args()

    valid_input_extensions = (".obj", ".stl", ".ply")
    if not args.input.endswith(valid_input_extensions):
        print(f"Error: Input must be one of {valid_input_extensions}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path == "":
        output_path = os.path.splitext(args.input)[0] + ".usd"

    valid_output_extensions = (".usd", ".usda", ".usdc")
    if not output_path.endswith(valid_output_extensions):
        print(f"Error: Output must be one of {valid_output_extensions}", file=sys.stderr)
        sys.exit(1)

    display_color = None
    if args.light_blue:
        display_color = LIGHT_BLUE_RGB
    elif args.color is not None:
        display_color = tuple(args.color)

    convert_mesh_to_usd(
        args.input,
        output_path,
        scale=args.scale,
        up_axis=args.up_axis,
        display_color_rgb=display_color,
    )
