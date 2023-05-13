import torch
import numpy as np
import torchvision.transforms as T
from models.clip_relevancy import ClipRelevancy
from util.util import get_screen_termplate, get_text_criterion, get_augmentations_template
class Last_Layer(torch.nn.Module):
  def __init__(self, cfg, clip_extractor):
    self.cfg = cfg
    template = get_augmentations_template()
    self.target_comp_e = clip_extractor.get_text_embedding(cfg["comp_text"], template)
    self.clip_extractor = clip_extractor
    self.text_criterion = get_text_criterion(cfg)
  
  def forward(self,inputs,outputs):
      for img in outputs:  # avoid memory limitations
          img_e = self.clip_extractor.get_image_embedding(img.unsqueeze(0))
         
  def func(self,outputs):
    model=nn.Sequential(
            nn.ReLU(),
      
    )
    
      
  
      
     
  
