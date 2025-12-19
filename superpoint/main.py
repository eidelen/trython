from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")

# LightGlue requires pairs of images
image_paths = ["rsc/capitol1.jpg", "rsc/capitol2.jpg"]
images = [Image.open(path) for path in image_paths]
inputs = processor(images, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

# Extract matching information
batch_keypoints = outputs.keypoints[0]  # [2, num_keypoints, 2] for the pair
keypoints0, keypoints1 = batch_keypoints[0], batch_keypoints[1]
matches = outputs.matches[0]  # Matching indices per image
matching_scores = outputs.matching_scores[0]  # Confidence scores per image

# Process outputs for visualization
image_sizes = [[(image.height, image.width) for image in images]]
processed_outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

for i, output in enumerate(processed_outputs):
    print(f"For the image pair {i}")
    for keypoint0, keypoint1, matching_score in zip(
            output["keypoints0"], output["keypoints1"], output["matching_scores"]
    ):
        print(f"Keypoint at {keypoint0.numpy()} matches with keypoint at {keypoint1.numpy()} with score {matching_score}")

list_match_pil_images = processor.visualize_keypoint_matching(images, processed_outputs)
for idx, match_img in enumerate(list_match_pil_images):
    out_path = f"rsc/match_viz_{idx}.png"
    match_img.save(out_path)
    print(f"Saved visualization to {out_path}")
