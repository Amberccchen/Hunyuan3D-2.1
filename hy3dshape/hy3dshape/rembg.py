# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover():
    # isnet-general-use, not rembg's own default (u2net): validated across 6
    # objects (agentic system/images/large scale test/) after u2net was found
    # to drop entire thin/bright foreground regions outright (e.g. a jewelry
    # holder's metal tree, alpha=0 -- not a soft/partial edge, real content
    # loss) and to fuse a hallucinated base plane onto at least one other
    # object (game controller holder) tightly enough that per-component
    # debris checks didn't catch it. isnet-general-use fixed both failure
    # modes with no regressions on the 4 already-clean control cases tested
    # alongside them, at no measurable runtime cost.
    #
    # alpha_matting=True is a second, separate fix on top of that: found on
    # 3 of the Gemini-holders batch's own images (tablet/laptop/umbrella
    # holders -- all light-gray/silver/white objects with large flat
    # low-saturation surfaces) where isnet-general-use's raw alpha output
    # itself was near-zero across whole interior regions, not just soft at
    # the edges -- e.g. the entire tablet backplate came back alpha=~0,
    # confirmed by inspecting the raw pre-threshold alpha channel directly.
    # rembg's alpha_matting refinement (a trimap + closed-form matting
    # solve on top of the same base network output) recovered all 3 cleanly
    # with no thresholds changed downstream. Re-checked against the 7
    # already-clean holders in that same batch (opaque-pixel-count delta
    # under 2% in every case, confirmed visually on two) -- no regression
    # found, so this is the new default rather than a per-object override.
    def __init__(self, model_name: str = "isnet-general-use", alpha_matting: bool = True):
        self.session = new_session(model_name)
        self.alpha_matting = alpha_matting

    def __call__(self, image: Image.Image):
        if self.alpha_matting:
            output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0],
                             alpha_matting=True, alpha_matting_foreground_threshold=240,
                             alpha_matting_background_threshold=10)
        else:
            output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output
