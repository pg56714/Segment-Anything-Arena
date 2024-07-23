from typing import Tuple

import gradio as gr
import numpy as np
import supervision as sv
import torch
import time
from PIL import Image

from torchvision.transforms import ToTensor

# from transformers import SamModel, SamProcessor

from efficient_sam.build_efficient_sam import build_efficient_sam_vits

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model

MARKDOWN = """
# EfficientViT-SAM vs EfficientSAM vs SAM

Paper source：
[EfficientViT-SAM](https://arxiv.org/abs/2402.05008) and [EfficientSAM](https://arxiv.org/abs/2312.00863) and 
[SAM](https://arxiv.org/abs/2304.02643)
\n
Github Source Code: [Link](https://github.com/pg56714/Segment-Anything-Arena)
\n
The SAM model takes one minute to run to completion, which slow down other models. Currently, EfficientViT-SAM and EfficientSAM are displayed first.
The source code for all three models is available, but the SAM is commented out.
"""

BOX_EXAMPLES = [
    ["https://media.roboflow.com/efficient-sam/corgi.jpg", 801, 510, 1782, 993],
]

PROMPT_COLOR = sv.Color.from_hex("#D3D3D3")
MASK_COLOR = sv.Color.from_hex("#FF0000")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE).eval()
# SAM_PROCESSOR = SamProcessor.from_pretrained("facebook/sam-vit-huge")

EFFICIENT_SAM_MODEL = build_efficient_sam_vits().to(DEVICE).eval()

MASK_ANNOTATOR = sv.MaskAnnotator(color=MASK_COLOR, color_lookup=sv.ColorLookup.INDEX)

EFFICIENTVITSAM = EfficientViTSamPredictor(
    create_sam_model(name="xl1", weight_url="./weights/xl1.pt").to(DEVICE).eval()
)


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
        scene=bgr_image.copy(), detections=detections
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
        thickness=sv.calculate_optimal_line_thickness(resolution_wh=(w, h)),
    )

    return annotated_bgr_image[:, :, ::-1]


def efficientvit_sam_box_inference(
    image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
) -> np.ndarray:
    t1 = time.time()

    box = np.array([[x_min, y_min, x_max, y_max]])
    EFFICIENTVITSAM.set_image(image)
    mask = EFFICIENTVITSAM.predict(box=box, multimask_output=False)
    mask = mask[0]
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)
    result = annotate_image_with_box_prompt_result(
        image=image,
        detections=detections,
        x_max=x_max,
        x_min=x_min,
        y_max=y_max,
        y_min=y_min,
    )
    t2 = time.time()

    print(f"timecost: {t2-t1}")

    return result


def inference_with_box(
    image: np.ndarray,
    box: np.ndarray,
    model: torch.jit.ScriptModule,
    device: torch.device,
) -> np.ndarray:
    bbox = torch.reshape(torch.tensor(box), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        bbox.to(device),
        bbox_labels.to(device),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou


def efficient_sam_box_inference(
    image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
) -> np.ndarray:
    t1 = time.time()

    box = np.array([[x_min, y_min], [x_max, y_max]])
    mask = inference_with_box(image, box, EFFICIENT_SAM_MODEL, DEVICE)
    mask = mask[np.newaxis, ...]
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)

    result = annotate_image_with_box_prompt_result(
        image=image,
        detections=detections,
        x_max=x_max,
        x_min=x_min,
        y_max=y_max,
        y_min=y_min,
    )
    t2 = time.time()

    print(f"timecost: {t2-t1}")

    return result


# def sam_box_inference(
#     image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
# ) -> np.ndarray:
#     t1 = time.time()

#     input_boxes = [[[x_min, y_min, x_max, y_max]]]
#     inputs = SAM_PROCESSOR(
#         Image.fromarray(image), input_boxes=[input_boxes], return_tensors="pt"
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

#     result = annotate_image_with_box_prompt_result(
#         image=image,
#         detections=detections,
#         x_max=x_max,
#         x_min=x_min,
#         y_max=y_max,
#         y_min=y_min,
#     )
#     t2 = time.time()

#     print(f"timecost: {t2-t1}")

#     return result


def box_inference(
    image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        efficientvit_sam_box_inference(image, x_min, y_min, x_max, y_max),
        efficient_sam_box_inference(image, x_min, y_min, x_max, y_max),
        # sam_box_inference(image, x_min, y_min, x_max, y_max),
    )


# def clear(_: np.ndarray) -> Tuple[None, None, None]:
#     return None, None, None


def clear(_: np.ndarray) -> Tuple[None, None]:
    return None, None


box_input_image = gr.Image()
x_min_number = gr.Number(label="x_min")
y_min_number = gr.Number(label="y_min")
x_max_number = gr.Number(label="x_max")
y_max_number = gr.Number(label="y_max")
box_inputs = [box_input_image, x_min_number, y_min_number, x_max_number, y_max_number]

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        box_input_image.render()
        efficientvit_sam_box_output_image = gr.Image(label="EfficientVit-SAM")
        efficient_sam_box_output_image = gr.Image(label="EfficientSAM")
        # sam_box_output_image = gr.Image(label="SAM")

    with gr.Row():
        x_min_number.render()
        y_min_number.render()
        x_max_number.render()
        y_max_number.render()
        submit_box_inference_button = gr.Button(
            value="Submit", scale=1, variant="primary"
        )
    gr.Examples(
        # fn=box_inference,
        examples=BOX_EXAMPLES,
        inputs=box_inputs,
        outputs=[
            efficientvit_sam_box_output_image,
            efficient_sam_box_output_image,
            # sam_box_output_image,
        ],
    )

    submit_box_inference_button.click(
        efficientvit_sam_box_inference,
        inputs=box_inputs,
        outputs=efficientvit_sam_box_output_image,
    )
    submit_box_inference_button.click(
        efficient_sam_box_inference,
        inputs=box_inputs,
        outputs=efficient_sam_box_output_image,
    )
    # submit_box_inference_button.click(
    #     sam_box_inference, inputs=box_inputs, outputs=sam_box_output_image
    # )

    box_input_image.change(
        clear,
        inputs=box_input_image,
        outputs=[
            efficientvit_sam_box_output_image,
            efficient_sam_box_output_image,
            # sam_box_output_image,
        ],
    )


demo.launch(debug=False, show_error=True)
