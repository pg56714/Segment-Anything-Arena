from typing import Tuple

import gradio as gr
import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from utils.efficient_sam import load, inference_with_box

# from utils.efficient_sam import load, inference_with_box, inference_with_point
# from utils.draw import draw_circle, calculate_dynamic_circle_radius

MARKDOWN = """
# EfficientSAM vs SAM

Paper source：
[EfficientSAM](https://arxiv.org/abs/2312.00863) and 
[SAM](https://arxiv.org/abs/2304.02643)
"""

BOX_EXAMPLES = [
    ["https://media.roboflow.com/efficient-sam/corgi.jpg", 801, 510, 1782, 993],
]

POINT_EXAMPLES = [
    ["https://media.roboflow.com/efficient-sam/corgi.jpg", 1291, 751],
]

PROMPT_COLOR = sv.Color.from_hex("#D3D3D3")
MASK_COLOR = sv.Color.from_hex("#FF0000")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
SAM_PROCESSOR = SamProcessor.from_pretrained("facebook/sam-vit-huge")
EFFICIENT_SAM_MODEL = load(device=DEVICE)
MASK_ANNOTATOR = sv.MaskAnnotator(color=MASK_COLOR, color_lookup=sv.ColorLookup.INDEX)


def annotate_image_with_box_prompt_result(
    image: np.ndarray,
    detections: sv.Detections,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray:
    h, w, _ = image.shape
    bgr_image = image[:, :, ::-1]
    annotated_bgr_image = MASK_ANNOTATOR.annotate(
        scene=bgr_image, detections=detections
    )
    annotated_bgr_image = sv.draw_rectangle(
        scene=annotated_bgr_image,
        rect=sv.Rect(
            x=x_min,
            y=y_min,
            width=int(x_max - x_min),
            height=int(y_max - y_min),
        ),
        color=PROMPT_COLOR,
        thickness=sv.calculate_dynamic_line_thickness(resolution_wh=(w, h)),
    )
    return annotated_bgr_image[:, :, ::-1]


# def annotate_image_with_point_prompt_result(
#     image: np.ndarray, detections: sv.Detections, x: int, y: int
# ) -> np.ndarray:
#     h, w, _ = image.shape
#     bgr_image = image[:, :, ::-1]
#     annotated_bgr_image = MASK_ANNOTATOR.annotate(
#         scene=bgr_image, detections=detections
#     )
#     annotated_bgr_image = draw_circle(
#         scene=annotated_bgr_image,
#         center=sv.Point(x=x, y=y),
#         radius=calculate_dynamic_circle_radius(resolution_wh=(w, h)),
#         color=PROMPT_COLOR,
#     )
#     return annotated_bgr_image[:, :, ::-1]


def efficient_sam_box_inference(
    image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
) -> np.ndarray:
    box = np.array([[x_min, y_min], [x_max, y_max]])
    mask = inference_with_box(image, box, EFFICIENT_SAM_MODEL, DEVICE)
    mask = mask[np.newaxis, ...]
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)
    return annotate_image_with_box_prompt_result(
        image=image,
        detections=detections,
        x_max=x_max,
        x_min=x_min,
        y_max=y_max,
        y_min=y_min,
    )


def sam_box_inference(
    image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
) -> np.ndarray:
    input_boxes = [[[x_min, y_min, x_max, y_max]]]
    inputs = SAM_PROCESSOR(
        Image.fromarray(image), input_boxes=[input_boxes], return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = SAM_MODEL(**inputs)

    mask = SAM_PROCESSOR.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0][0].numpy()
    mask = mask[np.newaxis, ...]
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)
    return annotate_image_with_box_prompt_result(
        image=image,
        detections=detections,
        x_max=x_max,
        x_min=x_min,
        y_max=y_max,
        y_min=y_min,
    )


def box_inference(
    image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        efficient_sam_box_inference(image, x_min, y_min, x_max, y_max),
        sam_box_inference(image, x_min, y_min, x_max, y_max),
    )


# def efficient_sam_point_inference(image: np.ndarray, x: int, y: int) -> np.ndarray:
#     point = np.array([[x, y]])
#     mask = inference_with_point(image, point, EFFICIENT_SAM_MODEL, DEVICE)
#     mask = mask[np.newaxis, ...]
#     detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)
#     return annotate_image_with_point_prompt_result(
#         image=image, detections=detections, x=x, y=y
#     )


# def sam_point_inference(image: np.ndarray, x: int, y: int) -> np.ndarray:
#     input_points = [[[x, y]]]
#     inputs = SAM_PROCESSOR(
#         Image.fromarray(image), input_points=[input_points], return_tensors="pt"
#     ).to(DEVICE)

#     with torch.no_grad():
#         outputs = SAM_MODEL(**inputs)

#     mask = SAM_PROCESSOR.image_processor.post_process_masks(
#         outputs.pred_masks.cpu(),
#         inputs["original_sizes"].cpu(),
#         inputs["reshaped_input_sizes"].cpu(),
#     )[0][0][0].numpy()
#     mask = mask[np.newaxis, ...]
#     detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)
#     return annotate_image_with_point_prompt_result(
#         image=image, detections=detections, x=x, y=y
#     )


# def point_inference(image: np.ndarray, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
#     return (
#         efficient_sam_point_inference(image, x, y),
#         sam_point_inference(image, x, y),
#     )


def clear(_: np.ndarray) -> Tuple[None, None]:
    return None, None


box_input_image = gr.Image()
x_min_number = gr.Number(label="x_min")
y_min_number = gr.Number(label="y_min")
x_max_number = gr.Number(label="x_max")
y_max_number = gr.Number(label="y_max")
box_inputs = [box_input_image, x_min_number, y_min_number, x_max_number, y_max_number]

# point_input_image = gr.Image()
# x_number = gr.Number(label="x")
# y_number = gr.Number(label="y")
# point_inputs = [point_input_image, x_number, y_number]


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Tab(label="Box prompt"):
        with gr.Row():
            with gr.Column():
                box_input_image.render()
                with gr.Accordion(label="Box", open=False):
                    with gr.Row():
                        x_min_number.render()
                        y_min_number.render()
                        x_max_number.render()
                        y_max_number.render()
            efficient_sam_box_output_image = gr.Image(label="EfficientSAM")
            sam_box_output_image = gr.Image(label="SAM")
        with gr.Row():
            submit_box_inference_button = gr.Button("Submit")
        gr.Examples(
            fn=box_inference,
            examples=BOX_EXAMPLES,
            inputs=box_inputs,
            outputs=[efficient_sam_box_output_image, sam_box_output_image],
        )
    # with gr.Tab(label="Point prompt"):
    #     with gr.Row():
    #         with gr.Column():
    #             point_input_image.render()
    #             with gr.Accordion(label="Point", open=False):
    #                 with gr.Row():
    #                     x_number.render()
    #                     y_number.render()
    #         efficient_sam_point_output_image = gr.Image(label="EfficientSAM")
    #         sam_point_output_image = gr.Image(label="SAM")
    #     with gr.Row():
    #         submit_point_inference_button = gr.Button("Submit")
    #     gr.Examples(
    #         fn=point_inference,
    #         examples=POINT_EXAMPLES,
    #         inputs=point_inputs,
    #         outputs=[efficient_sam_point_output_image, sam_point_output_image],
    #     )

    submit_box_inference_button.click(
        efficient_sam_box_inference,
        inputs=box_inputs,
        outputs=efficient_sam_box_output_image,
    )
    submit_box_inference_button.click(
        sam_box_inference, inputs=box_inputs, outputs=sam_box_output_image
    )

    # submit_point_inference_button.click(
    #     efficient_sam_point_inference,
    #     inputs=point_inputs,
    #     outputs=efficient_sam_point_output_image,
    # )
    # submit_point_inference_button.click(
    #     sam_point_inference, inputs=point_inputs, outputs=sam_point_output_image
    # )

    box_input_image.change(
        clear,
        inputs=box_input_image,
        outputs=[efficient_sam_box_output_image, sam_box_output_image],
    )

    # point_input_image.change(
    #     clear,
    #     inputs=point_input_image,
    #     outputs=[efficient_sam_point_output_image, sam_point_output_image],
    # )

demo.launch(debug=False, show_error=True)
