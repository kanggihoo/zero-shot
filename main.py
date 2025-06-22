import glob
from pathlib import Path
from natsort import natsorted
import json
import torch 

from model_init import ZeroShotCLS
from factory import CLIPImageDataProcessor , show_tensor
from template import LABEL_TEMPLATES
from utils import copy_files , move_files

BASE_DIR = Path("__file__").parent.parent


#REVIEW : 여러 제품에 대한 splited image를 만들고 이를 분류하는 코드 테스트 필요
def classify_images(data, visualize=False, verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zero_shot = ZeroShotCLS(model_name="MobileCLIP-S2" , pretrained_name="datacompdr" , device=device, is_inference=True)
    processor = CLIPImageDataProcessor(processor=zero_shot.preprocess, tokenizer=zero_shot.tokenizer, device=device)

    #TODO : 색상 분류 코드 추가해야함(단 주어진 데이터로부터 color_size_info의 색상정보 LLM 이용해서 전처리 한 뒤)
    
        
    product_id = data["product_id"]
    category_main = data["category_main"]
    category_sub = data["category_sub"]
    color = "" if data["color_size_info"]["color"]=="one_color" else data["color_size_info"]["color"]
    
    image_dir = BASE_DIR / category_main / str(category_sub) / str(product_id)
    summary_images_dir = image_dir / "summary"
    segmented_images_dir = image_dir / "segment"

    template_variable = {
        "category_main" : category_main,
        "front_view" : "from the front view", # or "Front view of a model ...",
        "back_view" : "from the back view",
        "body crop" :  "" if category_main.lower() == "top" else "below the face",
        "color" : color 
    }

    # 이미지 경로 합치기 
    images_path = glob.glob( str( segmented_images_dir / "*.jpg")) + glob.glob( str( summary_images_dir / "*.jpg"))
    assert len(images_path) > 0 , f"No images found in the directory , {str(segmented_images_dir) , str(summary_images_dir)}" 

    clip_data = processor(images_path)
    
    print(len(clip_data))

    # 이미지 콘텐츠 스타일 분류
    image_content_style_label = LABEL_TEMPLATES["image_content_style_label"]
    result = zero_shot.predict(clip_data , image_content_style_label, verbose=verbose)
    pred_index1 , pred_probs = result.get("pred_index") , result.get("pred_probs")

    mask1 = (pred_index1 == 0) | (pred_index1 == 1) | (pred_index1 == 2)
    mask1_text = pred_index1 == 4
    exclude1_mask = ~(mask1 | mask1_text)
    clip_data.update_prob(pred_probs)

    exclude1_label = [image_content_style_label[pred_index1[idx]] for idx, m in enumerate(exclude1_mask) if m ]
    accep1 , exclude1 = clip_data.filter([mask1 , mask1_text])

    clothes,text = accep1
    if verbose:
        print(image_content_style_label)
        print(len(clothes) , len(text) , len(exclude1))
    
    if visualize:
        show_tensor(clothes , title = "is related clothes")
        show_tensor(text , title = "is text")
        show_tensor(exclude1 , exclude1_label, title = "not related clothes")

    single_model_or_item_label = [l.format(**template_variable) for l in LABEL_TEMPLATES["single_model_or_item_label"]]

    if len(clothes):
        result = zero_shot.predict(clothes , single_model_or_item_label, verbose=verbose)
        pred_index2 , pred_probs = result.get("pred_index") , result.get("pred_probs")

        mask2_model , mask2_clothes = pred_index2 == 0 , pred_index2 == 1
        exclude2_mask = ~(mask2_model | mask2_clothes)
        clothes.update_prob(pred_probs)

        exclude2_label = [single_model_or_item_label[pred_index2[idx]] for idx, m in enumerate(exclude2_mask) if m ]
        accep2 , exclude2 = clip_data.filter([mask2_model , mask2_clothes])

        clothes_model , clothes_one  = accep2
        if verbose:
            print(single_model_or_item_label)
            print(len(clothes_model) , len(clothes_one) , len(exclude2))
        
        if visualize:
            show_tensor(clothes_model , title = "there is one model")
            show_tensor(clothes_one , title = "there is one clothes")
            show_tensor(exclude2, exclude2_label , title = "multi model or multi clothes")

    model_view_label = [l.format(**template_variable) for l in LABEL_TEMPLATES["model_view_label"]]
    if len(clothes_model):
        result3_1 = zero_shot.predict(clothes_model , model_view_label, verbose=verbose)
        pred_index3_1 , pred_probs3_1 = result3_1.get("pred_index") , result3_1.get("pred_probs")
        mask3_1_front , mask3_1_back = pred_index3_1 == 0 , pred_index3_1 == 1
        exclude_mask3_1 = ~(mask3_1_front | mask3_1_back)

        clothes_model.update_prob(pred_probs3_1)
        exclude3_1_label = [model_view_label[pred_index3_1[idx]] for idx, m in enumerate(exclude_mask3_1) if m ]

        accep3_1 , exclude3_1 = clothes_model.filter([mask3_1_front , mask3_1_back])
        clothes_model_front , clothes_model_back = accep3_1
        if verbose:
            print(model_view_label)
            print(f"model_front : {len(clothes_model_front)} , model_back : {len(clothes_model_back)} , 제외 : {len(exclude3_1)}")
        
        if visualize:
            show_tensor(clothes_model_front)
            show_tensor(clothes_model_back)
            show_tensor(exclude3_1, exclude3_1_label)
        
    if len(clothes_one):
        clothing_view_label = [l.format(**template_variable) for l in LABEL_TEMPLATES["clothing_view_label"]]
        result3_2 = zero_shot.predict(clothes_one , clothing_view_label, verbose=verbose)
        pred_index3_2 , pred_probs3_2 = result3_2.get("pred_index") , result3_2.get("pred_probs")
        mask3_2 = pred_index3_2 == 0
        exclude_mask3_2 = ~mask3_2
        exclude3_2_label = [clothing_view_label[pred_index3_2[idx]] for idx, m in enumerate(exclude_mask3_2) if m ]
        clothes_one.update_prob(pred_probs3_2)
        accep3_2 , exclude3_2 = clothes_one.filter(mask3_2)
        clothes_one_center = accep3_2[0]

        if verbose:
            print(clothing_view_label)
            print(f"clothes_one_center : {len(clothes_one_center)} , 제외 : {len(exclude3_2)}")
        
        if visualize:
            show_tensor(clothes_one_center , title="A centered single piece of clothing")
            show_tensor(exclude3_2,  exclude3_2_label , title = "not centered and zoomed-in or other")

    ## file move
    # src_dir = (BASE_DIR / category_main / category_sub / product_id).absolute()
    # if len(text):
    #     copy_files(text.abs_url , dst_dir = src_dir / "texts" , suffix="moved")
    # if len(clothes) and len(clothes_one) and len(clothes_one_center):
    #     copy_files(clothes_one_center.abs_url , dst_dir = src_dir / "images" / "clothes")
    # if len(clothes_model) and len(clothes_model_front):
    #     copy_files(clothes_model_front.abs_url , dst_dir = src_dir / "images" / "model" / "front")
    # if len(clothes_model) and len(clothes_model_back):
    #     copy_files(clothes_model_back.abs_url , dst_dir = src_dir / "images" / "model" / "back")

if __name__ == "__main__":
    file_name = "musinsa_product_detail_상의_후드티셔츠_.json"
    file_path = BASE_DIR / "data" / file_name
    with open(file_path , encoding="utf-8") as f:
        data = json.load(f)
    
    test_data = data[0]

    # Set visualization and verbosity flags
    classify_images(test_data, visualize=True, verbose=True)


