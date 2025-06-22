# label_templates.py 파일 내용

# 이전 답변에서 수정된 형태를 적용한 템플릿 리스트들
LABEL1_TEMPLATE = [
    "a flat lay photo of clothing",
    "a product photo of clothing",
    "a person wearing clothing",
    "a photo of an illustration",
    "a photo containing only a text description",
    "a photo of a diagram",
    "others"
]

LABEL2_TEMPLATE = [
    "a product photo of only a single model wearing a {category_main} clothing",
    "a product photo of only a single piece of {category_main} clothing",
    # "a photo with multiple models wearing clothing",
    "a product photo showing multiple models or multiple views of a model wearing clothing"
    "a photo with multiple product pieces of clothing",
    "others"
]

LABEL3_1_TEMPLATE = [
    "a photo of a model wearing a {category_main} {front_view}",
    "a photo of a model wearing a {category_main} {back_view}",
    "others"
]

LABEL3_2_TEMPLATE = [
    "a full view of a {category_main} on a plain background without a person",
    "a zoomed-in photo of a {category_main} without a person" ,
    "a cropped photo of a {category_main} without a person",
    "a schematic diagram of a {category_main}",
    "others"
]

def create_color_templates(colors, category_main):
    """
    색상별 분류를 위한 템플릿을 동적으로 생성하는 함수
    
    Args:
        colors (str or list): 콤마로 구분된 색상 문자열 또는 색상 리스트
        category_main (str): 주 카테고리 (예: "상의")
    Returns:
        list: 생성된 템플릿 리스트
    """
    
    templates = [
        f"a photo of a {color} {category_main}"
        for color in colors
    ] + ["others"]
    
    return templates


# 필요하다면 딕셔너리 형태로도 함께 제공 가능
LABEL_TEMPLATES = {
    "image_content_style_label": LABEL1_TEMPLATE,
    "single_model_or_item_label": LABEL2_TEMPLATE,
    "model_view_label": LABEL3_1_TEMPLATE,
    "clothing_view_label": LABEL3_2_TEMPLATE,
    "create_color_templates": create_color_templates  # 함수도 함께 export
}