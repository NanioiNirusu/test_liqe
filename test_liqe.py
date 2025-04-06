import os
import torch
import clip
from itertools import product
from PIL import Image, ImageFile
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Normalize, transforms
import csv
import time

#############################################
# CONFIGURATION AND SETUP
#############################################

# Allow PIL to load truncated images if needed.
ImageFile.LOAD_TRUNCATED_IMAGES = True

#############################################
# CONSTANTS AND DATA DEFINITIONS
#############################################

# Define constants used for constructing text prompts.
dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast',
             'overexposure', 'underexposure', 'spatial', 'quantization', 'other']
scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

#############################################
# LIQE MODEL DEFINITION
#############################################

class LIQE(nn.Module):
    def __init__(self, ckpt, device):
        super(LIQE, self).__init__()
        # Load CLIP model (ViT-B/32)
        self.model, _ = clip.load("ViT-B/32", device=device, jit=False)

        # Load the LIQE checkpoint containing fine-tuned weights.
        checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint)

        # Create a set of text prompts for every combination of quality, scene, and distortion.
        joint_texts = torch.cat(
            [clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality")
             for q, c, d in product(qualitys, scenes, dists_map)]
        ).to(device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(joint_texts)
            self.text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)

        self.step = 32
        self.num_patch = 15
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073),
                                   (0.26862954, 0.26130258, 0.27577711))
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.size(0)
        x = self.normalize(x)
        # Unfold image into patches of size 224x224 with stride = self.step.
        x = x.unfold(2, 224, self.step).unfold(3, 224, self.step)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, 3, 224, 224)

        # Select a subset of patches to reduce computation.
        sel_step = x.size(1) // self.num_patch
        sel = torch.arange(0, self.num_patch, device=self.device) * sel_step
        x = x[:, sel, ...]
        x = x.reshape(batch_size * self.num_patch, 3, 224, 224)

        # Encode image patches using CLIP's image encoder.
        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity between image features and text features.
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ self.text_features.t()
        logits_per_image = logits_per_image.view(batch_size, self.num_patch, -1)
        logits_per_image = logits_per_image.mean(1)
        logits_per_image = F.softmax(logits_per_image, dim=1)

        # Reshape logits to separate quality, scene, and distortion dimensions.
        logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
        logits_quality = logits_per_image.sum(3).sum(2)

        similarity_scene = logits_per_image.sum(3).sum(1)
        similarity_distortion = logits_per_image.sum(1).sum(1)
        distortion_index = similarity_distortion.argmax(dim=1)
        scene_index = similarity_scene.argmax(dim=1)

        # Map predicted indices to labels.
        scene = [scenes[i] for i in scene_index.tolist()]
        distortion = [dists_map[i] for i in distortion_index.tolist()]

        # Compute weighted quality score.
        quality = (1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] +
                   3 * logits_quality[:, 2] + 4 * logits_quality[:, 3] +
                   5 * logits_quality[:, 4])

        return quality, scene, distortion

#############################################
# UTILITY FUNCTIONS
#############################################

def preprocess_image(image_path):
    """Load and preprocess an image given its path."""
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return preprocess(img)


def get_unique_filename(base_output_file):
    """
    Generate a unique output filename by incrementing the distortion number

    Args:
        base_output_file (str): Base filename to check and potentially modify

    Returns:
        str: A unique filename with incremented distortion number
    """

    # Use regex to find and increment the distortion number
    def increment_distortion_number(filename):
        # Match pattern like D2 or D3 at the start of the filename
        import re
        match = re.match(r'(D)(\d+)(_liqe_metrics\.csv)', filename)
        if match:
            prefix, number, suffix = match.groups()
            return f"{prefix}{int(number) + 1}{suffix}"
        return filename

    current_filename = base_output_file

    # Check if file exists, increment distortion number if it does
    import os
    while os.path.exists(current_filename):
        current_filename = increment_distortion_number(current_filename)

    return current_filename

#############################################
# MAIN SCRIPT EXECUTION
#############################################

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt = './LIQE.pt'  # Path to your LIQE checkpoint (trained model weights)
    liqe = LIQE(ckpt, device)

    # Specify your dataset folder containing image files.
    dataset_folder = "J:/Masters/Datasets/AGIQA-1k-Database/file" # Change this to the path of your image folder
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # Output CSV file
    base_output_file = "D1_liqe_metrics.csv"
    output_file = get_unique_filename(base_output_file)

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image", "LIQE", "Scene Prediction", "Distortion Prediction", "Evaluation Time (s)"
        ])

        # Process each image in the folder.
        for image_path in image_files:
            try:
                img_tensor = preprocess_image(image_path).unsqueeze(0)  # Add batch dimension
                start_time = time.time()
                quality, scene, distortion = liqe(img_tensor)
                eval_time = time.time() - start_time

                # Write results to CSV
                writer.writerow([
                    image_path, quality.item(), scene[0], distortion[0], eval_time
                ])

                print(f"Image: {image_path}")
                print(f"  LIQE: {quality.item()}")
                print(f"  Scene Prediction: {scene[0]}")
                print(f"  Distortion Prediction: {distortion[0]}\n")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    print(f"LIQE metrics evaluation completed. Results saved to {output_file}")
