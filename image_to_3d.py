#!/usr/bin/env python
"""CLI demo: generate a textured 3D mesh (GLB) from a single input image using Hunyuan3D-2.1."""

import argparse
import gc
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Hunyuan3D-2.1 image-to-3D CLI demo")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--output", default=None,
                         help="Output .glb path (default: <output_dir>/<run_name>/<run_name>.glb). "
                              "An explicit path here is used as-is, bypassing the per-run folder.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs (default: outputs/)")
    parser.add_argument("--run-name", default=None,
                         help="Subfolder (under output-dir) all of this run's files are saved into "
                              "(default: the image filename stem). All outputs for one run -- shape "
                              ".glb, textured .obj/.mtl/.jpg, final .glb -- land in this one folder.")
    parser.add_argument("--gpu", default="1", help="CUDA device index to use (default: 1)")
    parser.add_argument("--no-texture", action="store_true", help="Skip PBR texture generation; export shape only")
    parser.add_argument("--max-num-view", type=int, default=6, help="Number of views for texture synthesis (6-9)")
    parser.add_argument("--resolution", type=int, default=512, help="Texture view resolution (512 or 768)")
    parser.add_argument("--seed", type=int, default=1234, help="Shape generation RNG seed")
    parser.add_argument("--octree-resolution", type=int, default=384, help="Marching-cubes octree resolution for shape extraction (higher preserves thin struts better)")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Diffusion steps for shape generation")
    parser.add_argument("--no-plane-fix", action="store_true",
                         help="Skip the hard-alpha-threshold + tight-recrop preprocessing (default: on). "
                              "That preprocessing fixed 13/13 hallucinated-background-plane cases in the "
                              "large-scale consistency test; disable only to reproduce old/raw behavior.")
    parser.add_argument("--rembg-model", default=None,
                         help="Override rembg model name. Default (None) uses BackgroundRemover's own "
                              "default, isnet-general-use -- swapped in from u2net after u2net was found to "
                              "drop entire thin/bright foreground regions outright (e.g. a jewelry holder's "
                              "metal tree) and to fuse a hallucinated base plane tightly enough that "
                              "per-component debris checks missed it. Pass e.g. 'u2net' here to reproduce the "
                              "old behavior for comparison.")
    return parser.parse_args()


def harden_alpha_and_recrop(image, threshold: int = 250, margin_frac: float = 0.08):
    """Snap alpha to fully opaque/transparent, then crop tight to the opaque
    bounding box with a small margin. Removes feathered-edge gradients and
    surrounding negative-space canvas -- empirically, this is what stopped
    the shape model from hallucinating a flat background/floor plane under
    the object (see images/large scale test/ consistency run: 13/13 fixed).
    Mirrors preprocess_hardalpha.py at the project root, duplicated here
    (rather than imported) so generation has no cross-directory dependency.
    """
    import numpy as np
    from PIL import Image

    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    arr[:, :, 3] = np.where(alpha >= threshold, 255, 0).astype(np.uint8)

    ys, xs = np.where(arr[:, :, 3] > 0)
    if len(ys) == 0:
        return Image.fromarray(arr, mode="RGBA")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    my, mx = int(h * margin_frac), int(w * margin_frac)
    y0, y1 = max(0, y0 - my), min(arr.shape[0], y1 + my)
    x0, x1 = max(0, x0 - mx), min(arr.shape[1], x1 + mx)
    return Image.fromarray(arr[y0:y1, x0:x1], mode="RGBA")


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_root, "hy3dshape"))
    sys.path.insert(0, os.path.join(repo_root, "hy3dpaint"))
    os.chdir(repo_root)

    import torch
    from PIL import Image

    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except Exception as e:
        print(f"Warning: torchvision_fix not applied: {e}")

    image_stem = os.path.splitext(os.path.basename(args.image))[0]
    run_name = args.run_name or image_stem
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    output_path = args.output or os.path.join(run_dir, f"{run_name}.glb")
    shape_glb_path = os.path.join(run_dir, f"{run_name}_shape.glb")

    image = Image.open(args.image).convert("RGBA")
    if image.getchannel("A").getextrema() == (255, 255):
        # No real transparency in the source image, so isolate the subject first.
        print(f"Removing background (model={args.rembg_model or 'isnet-general-use (default)'})...")
        if args.rembg_model:
            from rembg import remove, new_session
            image = remove(image, session=new_session(args.rembg_model), bgcolor=[255, 255, 255, 0])
        else:
            from hy3dshape.rembg import BackgroundRemover
            image = BackgroundRemover()(image)

    image_path_for_paint = args.image
    if not args.no_plane_fix:
        print("Applying hard-alpha-threshold + tight-recrop (hallucinated-plane fix)...")
        image = harden_alpha_and_recrop(image)
        preprocessed_path = os.path.join(run_dir, f"{run_name}_preprocessed.png")
        image.save(preprocessed_path)
        image_path_for_paint = preprocessed_path

    print("Loading shape generation pipeline...")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")

    print(f"Generating mesh from image (seed={args.seed}, octree_resolution={args.octree_resolution}, "
          f"num_inference_steps={args.num_inference_steps})...")
    generator = torch.Generator().manual_seed(args.seed)
    mesh = shape_pipeline(
        image=image,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_inference_steps=args.num_inference_steps,
    )[0]
    mesh.export(shape_glb_path)
    print(f"Untextured shape saved to {shape_glb_path}")

    del shape_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    if args.no_texture:
        os.replace(shape_glb_path, output_path)
        print(f"Done (shape only): {output_path}")
        return

    print("Loading texture generation pipeline...")
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    print("Generating PBR texture...")
    # Must be a .obj path: the pipeline writes OBJ content to this path verbatim,
    # then separately derives a sibling .glb next to it via string replacement.
    textured_obj_path = os.path.join(run_dir, f"{run_name}_textured.obj")
    paint_pipeline(
        mesh_path=shape_glb_path,
        image_path=image_path_for_paint,
        output_mesh_path=textured_obj_path,
    )

    del paint_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    textured_glb_path = textured_obj_path.replace(".obj", ".glb")
    os.replace(textured_glb_path, output_path)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
