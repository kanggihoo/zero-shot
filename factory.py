import torch
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import torch 
import matplotlib.pyplot as plt
import math
from constant import IMAGE_MEAN, IMAGE_STD

@dataclass
class DataItem:
    pil: Image.Image
    absolute_path: str
    relative_path: str
    tensor: Optional[torch.Tensor] = None
    label : Optional[str] = None
    prob : Optional[float] = None

    def __post_init__(self):
        """Validate image data after initialization"""
        if not isinstance(self.pil, Image.Image):
            raise ValueError("pil must be a PIL Image")
        if not self.absolute_path or not self.relative_path:
            raise ValueError("Path strings cannot be empty")
        

class CLIPImageDataset:
    def __init__(self, data_item: List[DataItem], tokenizer: Callable, device: torch.device ="cpu" , mean = IMAGE_MEAN , std = IMAGE_STD):
        self.data_item = data_item
        self.device = device
        self.tokenizer = tokenizer
        # Lazy loading of tensor to save memory
        self._tensor: Optional[torch.Tensor] = None
        self.mean = mean
        self.std = std

        
    @property
    def tensor(self) -> torch.Tensor:
        """Lazy load tensor only when needed"""
        if self._tensor is None:
            self._tensor = self.to_tensor(self.data_item)
        return self._tensor

    @property
    def abs_url(self):
        return [d.absolute_path for d in self.data_item]

    @property
    def rel_url(self):
        return [d.relative_path for d in self.data_item]
        
    @property
    def prob(self):
        return [d.prob for d in self.data_item]
    
    @property
    def denormalize(self):
        # Clone the tensor to avoid modifying the original one
        tensor = self.tensor.clone().detach()

        mean_tensor = torch.tensor(self.mean, device=tensor.device).view(1, -1, 1, 1)
        std_tensor = torch.tensor(self.std, device=tensor.device).view(1, -1, 1, 1)

        # Broadcasting을 이용한 역정규화: (normalized * std) + mean
        tensor = tensor.mul(std_tensor).add(mean_tensor) # 인플레이스 연산(mul_ 등)은 broadcasting 시 예상치 못한 동작을 할 수 있어 피하는 것이 좋음

        # 값을 0에서 1 사이로 클램핑
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor
        
    
        
    def to_tensor(self, data_item: List[DataItem]) -> torch.Tensor:
        """Convert data items to tensor efficiently"""
        if not data_item:
            return torch.empty(0, device=self.device)
        tensors = [d.tensor for d in data_item]
        return torch.stack(tensors, dim=0).to(self.device)
    
    def filter(self, mask: Union[torch.Tensor, List[torch.Tensor], np.ndarray]) -> Tuple[List['CLIPImageDataset'], 'CLIPImageDataset']:
        """Filter dataset based on mask(s)"""
        if not self.data_item:
            return [], self.__class__([], self.tokenizer, self.device)

        # Convert mask to numpy array for consistent processing
        mask_array = self._prepare_mask(mask)
        
        try:
            # Process each mask to create accepted datasets
            accepted = []
            excluded_mask = np.ones(len(self.data_item), dtype=bool)
            
            for mask_row in mask_array:
                if len(mask_row) != len(self.data_item):
                    raise ValueError(f"Mask length {len(mask_row)} doesn't match data length {len(self.data_item)}")
                
                accepted_items = [d for idx, d in enumerate(self.data_item) if mask_row[idx]]
                accepted.append(self.__class__(accepted_items, self.tokenizer, self.device))
                excluded_mask &= ~mask_row
            
            # Create excluded dataset
            excluded_items = [d for idx, d in enumerate(self.data_item) if excluded_mask[idx]]
            excluded = self.__class__(excluded_items, self.tokenizer, self.device)
            
            
            return accepted, excluded
            
        except Exception as e:
            raise
    
    def update_prob(self , prob:np.ndarray):
        for idx , p in enumerate(prob):
            self.data_item[idx].prob = p 
    
        

    def _prepare_mask(self, mask: Union[torch.Tensor, List[torch.Tensor], np.ndarray]) -> np.ndarray:
        """Prepare mask for filtering"""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy() if mask.dim() == 2 else mask.unsqueeze(0).cpu().numpy()
        elif isinstance(mask, list):
            mask = np.stack([m.cpu().numpy() if isinstance(m, torch.Tensor) else m for m in mask])
        elif isinstance(mask, np.ndarray):
            mask = mask if mask.ndim == 2 else mask[np.newaxis, :]
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")
        return mask.astype(bool)

    def __getitem__(self, index: Union[int, slice]) -> Union[DataItem, List[DataItem]]:
        if isinstance(index, int):
            if not (0 <= index < len(self.data_item)):
                raise IndexError(f"Index {index} out of range [0, {len(self.data_item)})")
            return self.data_item[index]
        if isinstance(index, slice):
            return self.data_item[index]
        raise TypeError(f"Invalid index type: {type(index)}")
        
    def __len__(self) -> int:
        return len(self.data_item)
        
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}, count='{len(self.data_item)}', device='{self.device}'>"

class CLIPImageDataProcessor:
    def __init__(self, processor: Callable, tokenizer: Callable, device: torch.device):
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        
    def preprocess(self, imgs_url: List[str], num_workers: int = 4) -> 'CLIPImageDataset':
        """Preprocess images in parallel"""
        if not imgs_url:
            return CLIPImageDataset([], self.tokenizer, self.device)
            
        process_func = partial(self._process_single_image, processor=self.processor)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            data_items = list(filter(None, executor.map(process_func, imgs_url)))
            
        return CLIPImageDataset(data_items, self.tokenizer, device=self.device)

    @staticmethod
    def _process_single_image(img_url: str, processor: Callable) -> Optional[DataItem]:
        """Process a single image with error handling"""
        try:
            path = Path(img_url)
            absolute_path = str(path.absolute())
            
            # Validate image file
                
            with Image.open(absolute_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Create a copy to avoid file handle issues
                pil_img = img.copy()
                
            # Process image
            tensor_img = processor(pil_img)
            
            return DataItem(
                pil=pil_img,
                absolute_path=absolute_path,
                relative_path=str(path),
                tensor=tensor_img
            )
            
        except Exception as e:
            return None

    def __call__(self, imgs_url: Union[str, List[str]], num_workers: int = 4) -> 'CLIPImageDataset':
        """Process single image or list of images"""
        if isinstance(imgs_url, str):
            imgs_url = [imgs_url]
        return self.preprocess(imgs_url)



def show_tensor(d:CLIPImageDataset, labels:List[str]=None,max_cols: int = 4 , title:str = None):
    if not len(d):
        print("No data for showing!")
        return 
    if not isinstance(d, CLIPImageDataset):
        print(type(d))
        raise ValueError("d must be a CLIPIMageDataset")
    if d.prob and len(d.prob) != d.tensor.size(0):
        raise ValueError("prob_info must have the same length as the number of images")
    
    normalized_tensor = d.denormalize  
    prob_info = d.prob
    
    if normalized_tensor.dim() == 3:
        # Single image case
        plt.figure(figsize=(6, 6))
        numpy_img = normalized_tensor.cpu().numpy().transpose((1,2,0))
        plt.imshow(numpy_img)
        if d.prob_info and len(d.prob_info) > 0:
            if labels is not None:
                plt.title(f"{labels[idx]}:{prob_info[idx]:.2f}", pad=10, wrap=True)
            else:
                plt.title(str(prob_info[0]), pad=10, wrap=True)
        plt.axis("off")
    elif normalized_tensor.dim() == 4:
        # Multiple images case
        num_images = normalized_tensor.size(0)
        num_cols = min(num_images, max_cols)
        num_rows = math.ceil(num_images / num_cols)
        
        # Calculate figure size (width, height)
        fig_width = num_cols * 4
        fig_height = num_rows * 4
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height) , constrained_layout=True)
        # fig.tight_layout(pad=3.0)
        
        if title is not None:
            fig.suptitle(title , fontsize = 16)
        
        if num_rows == 1 and num_cols == 1:
            axes = np.array([axes])
        elif num_rows == 1 or num_cols == 1:
            axes = axes.ravel()
        
        numpy_imags = normalized_tensor.cpu().numpy()
        for idx in range(num_images):
            ax = axes[idx // num_cols, idx % num_cols] if num_rows > 1 else axes[idx]
            numpy_img = numpy_imags[idx].transpose(1,2,0)
            ax.imshow(numpy_img)
            if prob_info and idx < len(prob_info):
                if labels is not None:
                    ax.set_title(f"{labels[idx]}:{prob_info[idx]:.2f}", pad=10, wrap=True)
                else:
                    ax.set_title(f"{prob_info[idx]:.2f}", pad=10, wrap=True)
            ax.axis("off")
        
        # Hide empty subplots
        for idx in range(num_images, num_rows * num_cols):
            ax = axes[idx // num_cols, idx % num_cols] if num_rows > 1 else axes[idx]
            ax.axis("off")
            
    else:
        raise ValueError(f"Unsupported tensor dimension: {normalized_tensor.ndim}, expected 3 or 4.")
    
    plt.show()
    
def search_product_id(product_id:str):
    import json
    with open("./data/musinsa_product_detail_info.json" , encoding="utf-8") as f:
        data = json.load(f)
    try:
        test_json = next((item for item in data if item.get("product_id" , None) == product_id))
        # product_id를 찾지 못하면 generator에는 아무것도 없는데 next() 호출해버려서 StopIteration 발생
    except StopIteration:
        raise ValueError(f"Product_id : {product_id} not found")