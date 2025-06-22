import open_clip
from constant import IMAGE_MEAN, IMAGE_STD
from typing import List
from factory import CLIPImageDataset
import torch


class ZeroShotCLS:
    def __init__(self ,
                 model_name="MobileCLIP-S2" ,
                 pretrained_name="datacompdr" ,
                 device="cpu",
                 is_inference:bool=True,
                 image_mean:tuple = IMAGE_MEAN ,
                 image_std:tuple = IMAGE_STD,
                 image_resize_mode:str = "longest"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name, image_mean=image_mean, image_std=image_std , image_resize_mode=image_resize_mode)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self.model.to(self.device)
        print(f"Successfully loaded {model_name} model from {pretrained_name}")
        if is_inference:
            self.model.eval()


# For inference/model exporting purposes, please reparameterize first
    def predict(self , data:list[CLIPImageDataset], label:List[str] , verbose:bool = False):        
        tokenizer = data.tokenizer
        
        images = data.tensor.to(self.device)
        text = tokenizer(label).to(self.device)
            
        with torch.no_grad(), torch.amp.autocast(device_type=self.device):
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            pred_probs = (100*image_features @ text_features.T).softmax(dim=-1)
            pred_index = pred_probs.argmax(-1)
            
        if verbose:
            for row_index in range(len(images)):
                print(f"predicted : {data.data_item[row_index].absolute_path} => ({pred_index[row_index].item()} , {label[pred_index[row_index].item()]})")
                print("   ".join([f"{label[col_index]}:{pred_probs[row_index][col_index].item()*100:.2f}%" for col_index in range(len(label))])) 
        
            
        return {"pred_index":pred_index , "pred_probs": pred_probs.max(-1)[0].float().cpu().numpy().round(2) }