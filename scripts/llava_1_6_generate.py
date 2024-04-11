from argparse import ArgumentParser
from collections.abc import Iterator
import torch
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import requests
from PIL import Image
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from typing import Dict, Optional, Sequence, List
from llava.train.train import preprocess, preprocess_llama_2


input_prompt_with_images = """
<image> <image> <image> Please describe the three images.
"""

input_prompt = """
Please descript this image: <image>
"""


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file)
    return image

# def decode_text():
def preprocess_multimodal(
    sources: Sequence[str]
) -> Dict:

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                replace_token = DEFAULT_IMAGE_TOKEN + '\n'
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, replace_token).strip()
                # sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
    return sources

def llava_1_5_generate(prompt, images, image_processor, model, tokenizer):
    image_tensors = None
    if len(images) != 0:
        image_data = [load_image(image) for image in images]
        image_tensors = process_images(image_data, image_processor, model.config)
        print("image tensor shape: ", image_tensors.shape)
        # image_tensors = image_tensors.cuda().type(torch.bfloat16)
        image_tensors = [image_tensor.type(torch.bfloat16) for image_tensor in image_tensors]
        # print("image_tensor device: {}".format(image_tensors.device))
    

    # just one prompt
    sources = [[{"from": "human", "value": prompt}]]
    sources = preprocess_multimodal(sources)
    print("sources: ", sources)
    input_ids = preprocess_llama_2(sources, tokenizer, True)['input_ids']
    input_ids = input_ids[0].unsqueeze(0).cuda()
    print(input_ids)
    output_ids = model.generate(
            inputs=input_ids,
            images=image_tensors,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=4096,
        )
    res_decode = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
    return res_decode
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--pretrained_path',
        type=str,
        default='')
    parser.add_argument("--max_length", type=int, default=4096)
    args, cfg_overrided = parser.parse_known_args()
    

    model_name = get_model_name_from_path(args.pretrained_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.pretrained_path, None, model_name, max_length=args.max_length,
            device_map="cuda", torch_dtype=torch.bfloat16, use_flash_attn=False)
    print(model)
    res = llava_1_5_generate(input_prompt_with_images, \
        ["images/llava_example_cmp.png", "images/llava_logo.png", "images/llava_v1_5_radar.jpg"], \
            image_processor, model, tokenizer)
    print(res)
    
