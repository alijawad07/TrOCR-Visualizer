import argparse
import random
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm.auto import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="OCR Evaluation")
    parser.add_argument("--data_path", type=str, default="output/*",
                        help="Directory path for the images")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples to evaluate")
    args = parser.parse_args()
    return args

def read_image(image_path: str) -> Image.Image:
    """Reads an image from a file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Image.Image: PIL Image object.
    """
    return Image.open(image_path).convert('RGB')

def ocr(image: Image.Image, processor, model: torch.nn.Module) -> str:
    """Performs OCR on an image.

    Args:
        image (Image.Image): PIL Image object.
        processor: OCR processor.
        model (torch.nn.Module): OCR model.

    Returns:
        str: Recognized text.
    """
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def eval_new_data(data_path: str, num_samples: int, model: torch.nn.Module):
    """Evaluates OCR on new data and saves the output as a video.

    Args:
        data_path (str): Directory path for the images.
        num_samples (int): Number of samples to evaluate.
        model (torch.nn.Module): OCR model.
    """
    data_path=os.path.join(data_path, '*')
    image_paths = glob.glob(data_path)
    random.shuffle(image_paths)

    first_image = read_image(image_paths[0])
    plt.figure()
    plt.axis('off')
    plt.imshow(first_image)
    plt.draw()
    canvas = plt.gca().figure.canvas
    canvas.draw()
    img_array = np.array(canvas.renderer.buffer_rgba())
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    height, width, _ = img_bgr.shape
    plt.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('ocr_output.mp4', fourcc, 1.0, (width, height))

    for i, image_path in tqdm(enumerate(image_paths), total=min(len(image_paths), num_samples)):
        if i >= num_samples:
            break
        image = read_image(image_path)
        text = ocr(image, processor, model)
        plt.figure()
        plt.axis('off')
        plt.imshow(image)
        plt.title(text)
        plt.draw()
        canvas = plt.gca().figure.canvas
        canvas.draw()
        img_array = np.array(canvas.renderer.buffer_rgba())
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        out.write(img_bgr)
        plt.close()

    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)
    eval_new_data(args.data_path, args.num_samples, model)
