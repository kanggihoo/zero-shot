import glob
from pathlib import Path
from natsort import natsorted
import json
import torch 
import argparse

from model_init import ZeroShotCLS
from factory import CLIPImageDataProcessor , show_tensor
from template import LABEL_TEMPLATES
from utils import copy_files , move_files




#REVIEW : 여러 제품에 대한 splited image를 만들고 이를 분류하는 코드 테스트 필요
def classify_images(data, data_dir, visualize=False, verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zero_shot = ZeroShotCLS(model_name="MobileCLIP-S2" , pretrained_name="datacompdr" , device=device, is_inference=True)
    processor = CLIPImageDataProcessor(processor=zero_shot.preprocess, tokenizer=zero_shot.tokenizer, device=device)

    #TODO : 색상 분류 코드 추가해야함(단 주어진 데이터로부터 color_size_info의 색상정보 LLM 이용해서 전처리 한 뒤)
    
        
    product_id = data["product_id"]
    category_main = data["category_main"]
    category_sub = data["category_sub"]
    colors = "" if data["color_size_info"]["color"]=="one_color" else data["color_size_info"]["color"]
    
    image_dir = data_dir / category_main / str(category_sub) / str(product_id)
    summary_images_dir = image_dir / "summary"
    segmented_images_dir = image_dir / "segment"

    template_variable = {
        "category_main" : category_main,
        "front_view" : "from the front view", # or "Front view of a model ...",
        "back_view" : "from the back view",
        "body crop" :  "" if category_main.lower() == "top" else "below the face",
        "color" : colors
    }

    # 이미지 경로 합치기 
    images_path = glob.glob( str( segmented_images_dir / "*.jpg")) + glob.glob( str( summary_images_dir / "*.jpg"))
    assert len(images_path) > 0 , f"No images found in the directory , {str(segmented_images_dir) , str(summary_images_dir)}" 

    clip_data = processor(images_path)
    
    print(len(clip_data))

    # STEP1 : 이미지 콘텐츠 스타일 분류
    image_content_style_label = LABEL_TEMPLATES["image_content_style_label"]
    result = zero_shot.predict(clip_data , image_content_style_label, verbose=False)
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

    # STEP2 : 단일 모델 또는 단일 의류 분류
    single_model_or_item_label = [l.format(**template_variable) for l in LABEL_TEMPLATES["single_model_or_item_label"]]

    if len(clothes):
        result = zero_shot.predict(clothes , single_model_or_item_label, verbose=False)
        pred_index2 , pred_probs = result.get("pred_index") , result.get("pred_probs")

        mask2_model , mask2_clothes = pred_index2 == 0 , pred_index2 == 1
        exclude2_mask = ~(mask2_model | mask2_clothes)
        clothes.update_prob(pred_probs)

        exclude2_label = [single_model_or_item_label[pred_index2[idx]] for idx, m in enumerate(exclude2_mask) if m ]
        accep2 , exclude2 = clothes.filter([mask2_model , mask2_clothes])

        clothes_model , clothes_one  = accep2
        if verbose:
            print(single_model_or_item_label)
            print(len(clothes_model) , len(clothes_one) , len(exclude2))
        
        if visualize:
            show_tensor(clothes_model , title = "there is one model")
            show_tensor(clothes_one , title = "there is one clothes")
            show_tensor(exclude2, exclude2_label , title = "multi model or multi clothes")

    # STEP3 : 모델 뷰 분류 (앞 or 뒤)
    # model_view_label = [l.format(**template_variable) for l in LABEL_TEMPLATES["model_view_label"]]
    # if len(clothes_model):
    #     result3_1 = zero_shot.predict(clothes_model , model_view_label, verbose=verbose)
    #     pred_index3_1 , pred_probs3_1 = result3_1.get("pred_index") , result3_1.get("pred_probs")
    #     mask3_1_front , mask3_1_back = pred_index3_1 == 0 , pred_index3_1 == 1
    #     exclude_mask3_1 = ~(mask3_1_front | mask3_1_back)

    #     clothes_model.update_prob(pred_probs3_1)
    #     exclude3_1_label = [model_view_label[pred_index3_1[idx]] for idx, m in enumerate(exclude_mask3_1) if m ]

    #     accep3_1 , exclude3_1 = clothes_model.filter([mask3_1_front , mask3_1_back])
    #     clothes_model_front , clothes_model_back = accep3_1
    #     if verbose:
    #         print(model_view_label)
    #         print(f"model_front : {len(clothes_model_front)} , model_back : {len(clothes_model_back)} , 제외 : {len(exclude3_1)}")
        
    #     if visualize:
    #         show_tensor(clothes_model_front)
    #         show_tensor(clothes_model_back)
    #         show_tensor(exclude3_1, exclude3_1_label)
    
    # STEP3 : 단일 의류의 뷰 분류 (중앙 or 줌인 or 그외)
    if len(clothes_one):
        clothing_view_label = [l.format(**template_variable) for l in LABEL_TEMPLATES["clothing_view_label"]]
        result3_2 = zero_shot.predict(clothes_one , clothing_view_label, verbose=False)
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
            
    # STEP4 : 색상 분류 
    if len(clothes_one_center) and len(colors) >=2:
        color_templates = LABEL_TEMPLATES["create_color_templates"](colors, category_main)
        
        result4 = zero_shot.predict(clothes_one_center, color_templates, verbose=verbose)
        pred_index4, pred_probs4 = result4.get("pred_index"), result4.get("pred_probs")
        
        # 각 색상별로 마스크 생성 (마지막 인덱스는 others)
        color_masks = [pred_index4 == i for i in range(len(colors))]
        exclude_mask4 = pred_index4 == len(colors)  # others에 대한 마스크
        
        clothes_one_center.update_prob(pred_probs4)
        accep4, exclude4 = clothes_one_center.filter(color_masks)
        
        if verbose:
            print("\nColor classification results:")
            print(color_templates)
            for i, color_group in enumerate(accep4):
                print(f"{colors[i]}: {len(color_group)} images")
            print(f"Others: {len(exclude4)} images")
        
        if visualize:
            for i, color_group in enumerate(accep4):
                show_tensor(color_group, title=f"Color: {colors[i]}")
            show_tensor(exclude4, title="Others")
    
    # LLM을 활용한 대표 이미지 선정
    '''
    0. 색상 정보가 여러개 라면 일단 전처리 작업 필요 (노이즈 전처리) => 이거는 여기서 딱 한번만 적용해서 이후에 끝까지 들고 가야하는데 
    1. 모델이 1명 존재하는 이미지는 가장 옷의 느낌을 가장 잘 살리는 대표 이미지 선정 
    2. 중앙에 위치한 의류 이미지에 대해서는 색상 별로 가장 옷의 느낌을 잘 살리는 앞면, 뒷면 이미지 선정 
    '''
            
    #STEP5 : file move
    # src_dir = (BASE_DIR / category_main / str(category_sub) / str(product_id)).absolute()
    # if len(text):
    #     copy_files(text.abs_url , dst_dir = src_dir / "text" , suffix="moved")
    # if len(clothes) and len(clothes_one) and len(clothes_one_center):
    #     copy_files(clothes_one_center.abs_url , dst_dir = src_dir / "images" / "clothes")
    
    # if len(clothes_model) and len(clothes_model_front):
    #     copy_files(clothes_model_front.abs_url , dst_dir = src_dir / "images" / "model" / "front")
    # if len(clothes_model) and len(clothes_model_back):
    #     copy_files(clothes_model_back.abs_url , dst_dir = src_dir / "images" / "model" / "back")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification with CLIP')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Base directory path containing the image data')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization of results')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    file_name = "musinsa_product_detail_상의_후드티셔츠_.json"
    file_path = Path("__file__").parent / file_name
    
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    
    test_data = data[1]
    print(test_data["product_id"])
    
    # Run classification with command line arguments
    classify_images(test_data, 
                   data_dir=data_dir,
                   visualize=args.visualize, 
                   verbose=args.verbose)


