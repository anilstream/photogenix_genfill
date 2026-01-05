# Standard library
import os
import sys

# Third-party
import torch

# Local application
from genfill_utils import (
    add_comfyui_directory_to_sys_path,
    add_extra_model_paths,
    get_value_at_index,
    import_custom_nodes,
    output_to_bytes,
)

# ---- ComfyUI bootstrap ----
add_comfyui_directory_to_sys_path()
add_extra_model_paths()
import_custom_nodes()

from nodes import NODE_CLASS_MAPPINGS


class FluxOneRewardOutpainter(object):
    def __init__(self):

        # image
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()

        # models
        self.vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.vaeloader_134 = self.vaeloader.load_vae(
            vae_name="ae.safetensors"
        )

        self.dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        self.dualcliploader_145 = self.dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp16.safetensors",
            type="flux",
            device="default",
        )

        self.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        self.cliptextencode_143 = self.cliptextencode.encode(
            text="high-definition, perfect composition",
            clip=get_value_at_index(self.dualcliploader_145, 0),
        )

        self.diffusionmodelloaderkj = NODE_CLASS_MAPPINGS["DiffusionModelLoaderKJ"]()
        self.diffusionmodelloaderkj_148 = self.diffusionmodelloaderkj.patch_and_load(
            model_name="flux.1-fill-dev-OneReward-transformer_fp8.safetensors",
            weight_dtype="fp8_e4m3fn",
            compute_dtype="default",
            patch_cublaslinear=True,
            sage_attention="sageattn_qk_int8_pv_fp16_triton",
            enable_fp16_accumulation=True,
        )

        self.loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        self.loraloadermodelonly_152 = self.loraloadermodelonly.load_lora_model_only(
            lora_name="FLUX.1-Turbo-Alpha.safetensors",
            strength_model=1,
            model=get_value_at_index(self.diffusionmodelloaderkj_148, 0),
        )

        # node initialization
        self.intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
        self.intconstant_175 = self.intconstant.get_value(value=10)

        self.simplemath = NODE_CLASS_MAPPINGS["SimpleMath+"]()

        self.differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        self.setlatentnoisemask = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
        self.ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self.imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        self.conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        self.getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()
        self.imagepadforoutpaint = NODE_CLASS_MAPPINGS["ImagePadForOutpaint"]()
        self.imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        self.inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()

    def run(self, image, top, bottom, left, right):

        with torch.inference_mode():

            loadimage_17 = self.loadimage.load_image(
                image="replicate-prediction-21yq6xyvwhrmw0cv7bpaxt8d3w.webp"
            )

            fluxguidance_135 = self.fluxguidance.EXECUTE_NORMALIZED(
                guidance=30,
                conditioning=get_value_at_index(self.cliptextencode_143, 0),
            )

            conditioningzeroout_136 = self.conditioningzeroout.zero_out(
                conditioning=get_value_at_index(self.cliptextencode_143, 0)
            )


            getimagesize_177 = self.getimagesize.get_size(
                image=get_value_at_index(loadimage_17, 0),
                unique_id=5344843070174262326,
            )

            simplemath_176 = self.simplemath.execute(
                value="a/b",
                a=get_value_at_index(getimagesize_177, 0),
                b=get_value_at_index(self.intconstant_175, 0),
            )

            imagepadforoutpaint_142 = self.imagepadforoutpaint.expand_image(
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                feathering=get_value_at_index(simplemath_176, 0),
                image=get_value_at_index(loadimage_17, 0),
            )

            imageresizekjv2_158 = self.imageresizekjv2.resize(
                width=1536,
                height=1536,
                upscale_method="bilinear",
                keep_proportion="resize",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=2,
                device="cpu",
                image=get_value_at_index(imagepadforoutpaint_142, 0),
                mask=get_value_at_index(imagepadforoutpaint_142, 1),
                unique_id=12529069524583121912,
            )

            inpaintmodelconditioning_139 = self.inpaintmodelconditioning.encode(
                noise_mask=False,
                positive=get_value_at_index(fluxguidance_135, 0),
                negative=get_value_at_index(conditioningzeroout_136, 0),
                vae=get_value_at_index(self.vaeloader_134, 0),
                pixels=get_value_at_index(imageresizekjv2_158, 0),
                mask=get_value_at_index(imageresizekjv2_158, 3),
            )

            differentialdiffusion_137 = self.differentialdiffusion.EXECUTE_NORMALIZED(
                strength=1,
                model=get_value_at_index(self.loraloadermodelonly_152, 0),
            )

            setlatentnoisemask_141 = self.setlatentnoisemask.set_mask(
                samples=get_value_at_index(inpaintmodelconditioning_139, 2),
                mask=get_value_at_index(imageresizekjv2_158, 3),
            )

            ksampler_140 = self.ksampler.sample(
                seed=684500389013406,
                steps=8,
                cfg=1,
                sampler_name="euler",
                scheduler="normal",
                denoise=1,
                model=get_value_at_index(differentialdiffusion_137, 0),
                positive=get_value_at_index(inpaintmodelconditioning_139, 0),
                negative=get_value_at_index(inpaintmodelconditioning_139, 1),
                latent_image=get_value_at_index(setlatentnoisemask_141, 0),
            )

            vaedecode_138 = self.vaedecode.decode(
                samples=get_value_at_index(ksampler_140, 0),
                vae=get_value_at_index(self.vaeloader_134, 0),
            )

            getimagesize_163 = self.getimagesize.get_size(
                image=get_value_at_index(imagepadforoutpaint_142, 0),
                unique_id=4182233309666817756,
            )

            imageresizekjv2_161 = self.imageresizekjv2.resize(
                width=get_value_at_index(getimagesize_163, 0),
                height=get_value_at_index(getimagesize_163, 1),
                upscale_method="bilinear",
                keep_proportion="resize",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=2,
                device="cpu",
                image=get_value_at_index(vaedecode_138, 0),
                unique_id=16343936511075396008,
            )

            imagecompositemasked_164 = self.imagecompositemasked.EXECUTE_NORMALIZED(
                x=0,
                y=0,
                resize_source=False,
                destination=get_value_at_index(imagepadforoutpaint_142, 0),
                source=get_value_at_index(imageresizekjv2_161, 0),
                mask=get_value_at_index(imagepadforoutpaint_142, 1),
            )

            output_image = get_value_at_index(imagecompositemasked_164, 0)
            image = output_to_bytes(output_image)

        return image



if __name__ == "__main__":
    image="/home/anil/DEV/photogenix_genfill/test/image.png"
    top, bottom, left, right = 0, 0, 208, 208
    flux_outpainter = FluxOneRewardOutpainter()
    output = flux_outpainter.run(image, top=top, bottom=bottom, left=left, right=right)
    with open("output1.png", "wb") as f:
        f.write(output)
