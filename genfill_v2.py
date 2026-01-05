import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)

        manager_path = os.path.join(
            comfyui_path, "custom_nodes", "ComfyUI-Manager", "glob"
        )

        if os.path.isdir(manager_path) and os.listdir(manager_path):
            sys.path.append(manager_path)
            global has_manager
            has_manager = True

        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from comfy.options import enable_args_parsing

    enable_args_parsing()
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    if has_manager:
        try:
            import manager_core as manager
        except ImportError:
            print("Could not import manager_core, proceeding without it.")
            return
        else:
            if hasattr(manager, "get_config"):
                print("Patching manager_core.get_config to enforce offline mode.")
                try:
                    get_config = manager.get_config

                    def _get_config(*args, **kwargs):
                        config = get_config(*args, **kwargs)
                        config["network_mode"] = "offline"
                        return config

                    manager.get_config = _get_config
                except Exception as e:
                    print("Failed to patch manager_core.get_config:", e)

    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def inner():
        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        await init_extra_nodes(init_custom_nodes=True)

    loop.run_until_complete(inner())


def save_image_wrapper(context, cls):
    if args.output is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output == "-":
                    raise ValueError("Cannot save multiple images to stdout")
                filename_prefix += self.prefix_append

                results = list()
                for batch_number, image in enumerate(images):
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    if args.output == "-":
                        # Hack to briefly restore stdout
                        if context is not None:
                            context.__exit__(None, None, None)
                        try:
                            img.save(
                                sys.stdout.buffer,
                                format="png",
                                pnginfo=metadata,
                                compress_level=self.compress_level,
                            )
                        finally:
                            if context is not None:
                                context.__enter__()
                    else:
                        subfolder = ""
                        if len(images) == 1:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output)

                            if subfolder == "":
                                subfolder = os.getcwd()

                            files = os.listdir(subfolder)
                            file_pattern = file
                            while True:
                                filename_with_batch_num = file_pattern.replace(
                                    "%batch_num%", str(batch_number)
                                )
                                file = (
                                    f"{filename_with_batch_num}_{self.counter:05}.png"
                                )
                                self.counter += 1

                                if file not in files:
                                    break

                        img.save(
                            os.path.join(subfolder, file),
                            pnginfo=metadata,
                            compress_level=self.compress_level,
                        )
                        print("Saved image to", os.path.join(subfolder, file))
                        results.append(
                            {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type,
                            }
                        )

                return {"ui": {"images": results}}

    return WrappedSaveImage


def parse_arg(s: Any, default: Any = None) -> Any:
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Node inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "--image1",
    default="replicate-prediction-21yq6xyvwhrmw0cv7bpaxt8d3w.webp",
    help='Argument 0, input `image` for node "Load Image" id 17 (autogenerated)',
)

parser.add_argument(
    "--vae_name2",
    default="ae.safetensors",
    help='Argument 0, input `vae_name` for node "Load VAE" id 134 (autogenerated)',
)

parser.add_argument(
    "--clip_name13",
    default="clip_l.safetensors",
    help='Argument 0, input `clip_name1` for node "DualCLIPLoader" id 145 (autogenerated)',
)

parser.add_argument(
    "--clip_name24",
    default="t5xxl_fp16.safetensors",
    help='Argument 1, input `clip_name2` for node "DualCLIPLoader" id 145 (autogenerated)',
)

parser.add_argument(
    "--type5",
    default="flux",
    help='Argument 2, input `type` for node "DualCLIPLoader" id 145 (autogenerated)',
)

parser.add_argument(
    "--text6",
    default="high-definition, perfect composition",
    help='Argument 0, input `text` for node "CLIP Text Encode (Positive Prompt)" id 143 (autogenerated)',
)

parser.add_argument(
    "--guidance7",
    default=30,
    help='Argument 1, input `guidance` for node "FluxGuidance" id 135 (autogenerated)',
)

parser.add_argument(
    "--value8",
    default=10,
    help='Argument 0, input `value` for node "INT Constant" id 175 (autogenerated)',
)

parser.add_argument(
    "--value9",
    default="a/b",
    help='Argument 0, input `value` for node "🔧 Simple Math" id 176 (autogenerated)',
)

parser.add_argument(
    "--left10",
    default=208,
    help='Argument 1, input `left` for node "Pad Image for Outpainting" id 142 (autogenerated)',
)

parser.add_argument(
    "--top11",
    default=0,
    help='Argument 2, input `top` for node "Pad Image for Outpainting" id 142 (autogenerated)',
)

parser.add_argument(
    "--right12",
    default=208,
    help='Argument 3, input `right` for node "Pad Image for Outpainting" id 142 (autogenerated)',
)

parser.add_argument(
    "--bottom13",
    default=0,
    help='Argument 4, input `bottom` for node "Pad Image for Outpainting" id 142 (autogenerated)',
)

parser.add_argument(
    "--width14",
    default=1536,
    help='Argument 1, input `width` for node "Resize Image v2" id 158 (autogenerated)',
)

parser.add_argument(
    "--height15",
    default=1536,
    help='Argument 2, input `height` for node "Resize Image v2" id 158 (autogenerated)',
)

parser.add_argument(
    "--upscale_method16",
    default="bilinear",
    help='Argument 3, input `upscale_method` for node "Resize Image v2" id 158 (autogenerated)',
)

parser.add_argument(
    "--keep_proportion17",
    default="resize",
    help='Argument 4, input `keep_proportion` for node "Resize Image v2" id 158 (autogenerated)',
)

parser.add_argument(
    "--pad_color18",
    default="0, 0, 0",
    help='Argument 5, input `pad_color` for node "Resize Image v2" id 158 (autogenerated)',
)

parser.add_argument(
    "--crop_position19",
    default="center",
    help='Argument 6, input `crop_position` for node "Resize Image v2" id 158 (autogenerated)',
)

parser.add_argument(
    "--divisible_by20",
    default=2,
    help='Argument 7, input `divisible_by` for node "Resize Image v2" id 158 (autogenerated)',
)

parser.add_argument(
    "--noise_mask21",
    default=False,
    help='Argument 5, input `noise_mask` for node "InpaintModelConditioning" id 139 (autogenerated)',
)

parser.add_argument(
    "--model_name22",
    default="flux.1-fill-dev-OneReward-transformer_fp8.safetensors",
    help='Argument 0, input `model_name` for node "Diffusion Model Loader KJ" id 148 (autogenerated)',
)

parser.add_argument(
    "--weight_dtype23",
    default="fp8_e4m3fn",
    help='Argument 1, input `weight_dtype` for node "Diffusion Model Loader KJ" id 148 (autogenerated)',
)

parser.add_argument(
    "--compute_dtype24",
    default="default",
    help='Argument 2, input `compute_dtype` for node "Diffusion Model Loader KJ" id 148 (autogenerated)',
)

parser.add_argument(
    "--patch_cublaslinear25",
    default=True,
    help='Argument 3, input `patch_cublaslinear` for node "Diffusion Model Loader KJ" id 148 (autogenerated)',
)

parser.add_argument(
    "--sage_attention26",
    default="sageattn_qk_int8_pv_fp16_triton",
    help='Argument 4, input `sage_attention` for node "Diffusion Model Loader KJ" id 148 (autogenerated)',
)

parser.add_argument(
    "--enable_fp16_accumulation27",
    default=True,
    help='Argument 5, input `enable_fp16_accumulation` for node "Diffusion Model Loader KJ" id 148 (autogenerated)',
)

parser.add_argument(
    "--lora_name28",
    default="FLUX.1-Turbo-Alpha.safetensors",
    help='Argument 1, input `lora_name` for node "LoraLoaderModelOnly" id 152 (autogenerated)',
)

parser.add_argument(
    "--strength_model29",
    default=1,
    help='Argument 2, input `strength_model` for node "LoraLoaderModelOnly" id 152 (autogenerated)',
)

parser.add_argument(
    "--seed30",
    default=684500389013406,
    help='Argument 1, input `seed` for node "KSampler" id 140 (autogenerated)',
)

parser.add_argument(
    "--steps31",
    default=8,
    help='Argument 2, input `steps` for node "KSampler" id 140 (autogenerated)',
)

parser.add_argument(
    "--cfg32",
    default=1,
    help='Argument 3, input `cfg` for node "KSampler" id 140 (autogenerated)',
)

parser.add_argument(
    "--sampler_name33",
    default="euler",
    help='Argument 4, input `sampler_name` for node "KSampler" id 140 (autogenerated)',
)

parser.add_argument(
    "--scheduler34",
    default="normal",
    help='Argument 5, input `scheduler` for node "KSampler" id 140 (autogenerated)',
)

parser.add_argument(
    "--denoise35",
    default=1,
    help='Argument 9, input `denoise` for node "KSampler" id 140 (autogenerated)',
)

parser.add_argument(
    "--upscale_method36",
    default="bilinear",
    help='Argument 3, input `upscale_method` for node "Resize Image v2" id 161 (autogenerated)',
)

parser.add_argument(
    "--keep_proportion37",
    default="resize",
    help='Argument 4, input `keep_proportion` for node "Resize Image v2" id 161 (autogenerated)',
)

parser.add_argument(
    "--pad_color38",
    default="0, 0, 0",
    help='Argument 5, input `pad_color` for node "Resize Image v2" id 161 (autogenerated)',
)

parser.add_argument(
    "--crop_position39",
    default="center",
    help='Argument 6, input `crop_position` for node "Resize Image v2" id 161 (autogenerated)',
)

parser.add_argument(
    "--divisible_by40",
    default=2,
    help='Argument 7, input `divisible_by` for node "Resize Image v2" id 161 (autogenerated)',
)

parser.add_argument(
    "--x41",
    default=0,
    help='Argument 2, input `x` for node "ImageCompositeMasked" id 164 (autogenerated)',
)

parser.add_argument(
    "--y42",
    default=0,
    help='Argument 3, input `y` for node "ImageCompositeMasked" id 164 (autogenerated)',
)

parser.add_argument(
    "--resize_source43",
    default=False,
    help='Argument 4, input `resize_source` for node "ImageCompositeMasked" id 164 (autogenerated)',
)

parser.add_argument(
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)


comfy_args = [sys.argv[0]]
if __name__ == "__main__" and "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output is not None and args.output == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    if has_manager:
        try:
            import manager_core as manager
        except ImportError:
            print("Could not import manager_core, proceeding without it.")
            return
        else:
            if hasattr(manager, "get_config"):
                print("Patching manager_core.get_config to enforce offline mode.")
                try:
                    get_config = manager.get_config

                    def _get_config(*args, **kwargs):
                        config = get_config(*args, **kwargs)
                        config["network_mode"] = "offline"
                        return config

                    manager.get_config = _get_config
                except Exception as e:
                    print("Failed to patch manager_core.get_config:", e)

    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def inner():
        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        await init_extra_nodes(init_custom_nodes=True)

    loop.run_until_complete(inner())


_custom_nodes_imported = False
_custom_path_added = False


def main(*func_args, **func_kwargs):
    global args, _custom_nodes_imported, _custom_path_added
    if __name__ == "__main__":
        if args is None:
            args = parser.parse_args()
    else:
        defaults = dict(
            (arg, parser.get_default(arg))
            for arg in ["queue_size", "comfyui_directory", "output", "disable_metadata"]
            + [
                "image1",
                "vae_name2",
                "clip_name13",
                "clip_name24",
                "type5",
                "text6",
                "guidance7",
                "value8",
                "value9",
                "left10",
                "top11",
                "right12",
                "bottom13",
                "width14",
                "height15",
                "upscale_method16",
                "keep_proportion17",
                "pad_color18",
                "crop_position19",
                "divisible_by20",
                "noise_mask21",
                "model_name22",
                "weight_dtype23",
                "compute_dtype24",
                "patch_cublaslinear25",
                "sage_attention26",
                "enable_fp16_accumulation27",
                "lora_name28",
                "strength_model29",
                "seed30",
                "steps31",
                "cfg32",
                "sampler_name33",
                "scheduler34",
                "denoise35",
                "upscale_method36",
                "keep_proportion37",
                "pad_color38",
                "crop_position39",
                "divisible_by40",
                "x41",
                "y42",
                "resize_source43",
            ]
        )

        all_args = dict()
        all_args.update(defaults)
        all_args.update(func_kwargs)

        args = argparse.Namespace(**all_args)

    with ctx:
        if not _custom_path_added:
            add_comfyui_directory_to_sys_path()
            add_extra_model_paths()

            _custom_path_added = True

        if not _custom_nodes_imported:
            import_custom_nodes()

            _custom_nodes_imported = True

        from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode(), ctx:
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_17 = loadimage.load_image(image=parse_arg(args.image1))

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_134 = vaeloader.load_vae(vae_name=parse_arg(args.vae_name2))

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_145 = dualcliploader.load_clip(
            clip_name1=parse_arg(args.clip_name13),
            clip_name2=parse_arg(args.clip_name24),
            type=parse_arg(args.type5),
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_143 = cliptextencode.encode(
            text=parse_arg(args.text6), clip=get_value_at_index(dualcliploader_145, 0)
        )

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_135 = fluxguidance.EXECUTE_NORMALIZED(
            guidance=parse_arg(args.guidance7),
            conditioning=get_value_at_index(cliptextencode_143, 0),
        )

        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        conditioningzeroout_136 = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cliptextencode_143, 0)
        )

        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()
        getimagesize_177 = getimagesize.get_size(
            image=get_value_at_index(loadimage_17, 0), unique_id=5344843070174262326
        )

        intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
        intconstant_175 = intconstant.get_value(value=parse_arg(args.value8))

        simplemath = NODE_CLASS_MAPPINGS["SimpleMath+"]()
        simplemath_176 = simplemath.execute(
            value=parse_arg(args.value9),
            a=get_value_at_index(getimagesize_177, 0),
            b=get_value_at_index(intconstant_175, 0),
        )

        imagepadforoutpaint = NODE_CLASS_MAPPINGS["ImagePadForOutpaint"]()
        imagepadforoutpaint_142 = imagepadforoutpaint.expand_image(
            left=parse_arg(args.left10),
            top=parse_arg(args.top11),
            right=parse_arg(args.right12),
            bottom=parse_arg(args.bottom13),
            feathering=get_value_at_index(simplemath_176, 0),
            image=get_value_at_index(loadimage_17, 0),
        )

        imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        imageresizekjv2_158 = imageresizekjv2.resize(
            width=parse_arg(args.width14),
            height=parse_arg(args.height15),
            upscale_method=parse_arg(args.upscale_method16),
            keep_proportion=parse_arg(args.keep_proportion17),
            pad_color=parse_arg(args.pad_color18),
            crop_position=parse_arg(args.crop_position19),
            divisible_by=parse_arg(args.divisible_by20),
            device="cpu",
            image=get_value_at_index(imagepadforoutpaint_142, 0),
            mask=get_value_at_index(imagepadforoutpaint_142, 1),
            unique_id=12529069524583121912,
        )

        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_139 = inpaintmodelconditioning.encode(
            noise_mask=parse_arg(args.noise_mask21),
            positive=get_value_at_index(fluxguidance_135, 0),
            negative=get_value_at_index(conditioningzeroout_136, 0),
            vae=get_value_at_index(vaeloader_134, 0),
            pixels=get_value_at_index(imageresizekjv2_158, 0),
            mask=get_value_at_index(imageresizekjv2_158, 3),
        )

        diffusionmodelloaderkj = NODE_CLASS_MAPPINGS["DiffusionModelLoaderKJ"]()
        diffusionmodelloaderkj_148 = diffusionmodelloaderkj.patch_and_load(
            model_name=parse_arg(args.model_name22),
            weight_dtype=parse_arg(args.weight_dtype23),
            compute_dtype=parse_arg(args.compute_dtype24),
            patch_cublaslinear=parse_arg(args.patch_cublaslinear25),
            sage_attention=parse_arg(args.sage_attention26),
            enable_fp16_accumulation=parse_arg(args.enable_fp16_accumulation27),
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_152 = loraloadermodelonly.load_lora_model_only(
            lora_name=parse_arg(args.lora_name28),
            strength_model=parse_arg(args.strength_model29),
            model=get_value_at_index(diffusionmodelloaderkj_148, 0),
        )

        differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        setlatentnoisemask = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        for q in range(args.queue_size):
            differentialdiffusion_137 = differentialdiffusion.EXECUTE_NORMALIZED(
                strength=1, model=get_value_at_index(loraloadermodelonly_152, 0)
            )

            setlatentnoisemask_141 = setlatentnoisemask.set_mask(
                samples=get_value_at_index(inpaintmodelconditioning_139, 2),
                mask=get_value_at_index(imageresizekjv2_158, 3),
            )

            ksampler_140 = ksampler.sample(
                seed=parse_arg(args.seed30),
                steps=parse_arg(args.steps31),
                cfg=parse_arg(args.cfg32),
                sampler_name=parse_arg(args.sampler_name33),
                scheduler=parse_arg(args.scheduler34),
                denoise=parse_arg(args.denoise35),
                model=get_value_at_index(differentialdiffusion_137, 0),
                positive=get_value_at_index(inpaintmodelconditioning_139, 0),
                negative=get_value_at_index(inpaintmodelconditioning_139, 1),
                latent_image=get_value_at_index(setlatentnoisemask_141, 0),
            )

            vaedecode_138 = vaedecode.decode(
                samples=get_value_at_index(ksampler_140, 0),
                vae=get_value_at_index(vaeloader_134, 0),
            )

            getimagesize_163 = getimagesize.get_size(
                image=get_value_at_index(imagepadforoutpaint_142, 0),
                unique_id=4182233309666817756,
            )

            imageresizekjv2_161 = imageresizekjv2.resize(
                width=get_value_at_index(getimagesize_163, 0),
                height=get_value_at_index(getimagesize_163, 1),
                upscale_method=parse_arg(args.upscale_method36),
                keep_proportion=parse_arg(args.keep_proportion37),
                pad_color=parse_arg(args.pad_color38),
                crop_position=parse_arg(args.crop_position39),
                divisible_by=parse_arg(args.divisible_by40),
                device="cpu",
                image=get_value_at_index(vaedecode_138, 0),
                unique_id=16343936511075396008,
            )

            imagecompositemasked_164 = imagecompositemasked.EXECUTE_NORMALIZED(
                x=parse_arg(args.x41),
                y=parse_arg(args.y42),
                resize_source=parse_arg(args.resize_source43),
                destination=get_value_at_index(imagepadforoutpaint_142, 0),
                source=get_value_at_index(imageresizekjv2_161, 0),
                mask=get_value_at_index(imagepadforoutpaint_142, 1),
            )


if __name__ == "__main__":
    main()
