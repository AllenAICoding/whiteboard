from mmseg.apis import init_model #,inference_model, show_result_pyplot
import mmcv
# from mmengine.runner import load_checkpoint,save_checkpoint
import cv2
import torch
from mmengine.dataset import Compose, default_collate
from mmengine.utils import mkdir_or_exist
from collections import defaultdict
import numpy as np
from mmseg.visualization import SegLocalVisualizer
from typing import Union,Sequence, List,Callable
from .moai_util import mask_to_moai_label,timing_decorator
from mmengine.structures import PixelData
from mmengine import Config
import enum
from mmpretrain.apis.base import InputType
from mmcv.image import imread
from mmpretrain.structures import DataSample
from mmpretrain.registry import TRANSFORMS
from mmpretrain import ImageClassificationInferencer
import matplotlib.pyplot as plt

from anomalib.deploy import OpenVINOInferencer
from anomalib.data.utils import read_image
from anomalib.utils.visualization.image import ImageVisualizer, VisualizationMode
from typing import Any
from PIL import Image
from anomalib import TaskType as AnomalibTaskType
import os.path as osp
from  os.path import join as opj
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.utils.normalization.min_max import normalize as normalize_min_max
from enum import Enum
from anomalib.utils.visualization import ImageResult
import time
from torchvision.transforms.v2.functional import to_dtype, to_image
import json
from anomalib.utils.post_processing import add_anomalous_label, add_normal_label, draw_boxes, superimpose_anomaly_map
from skimage.segmentation import mark_boundaries

class TaskType(enum.Enum):
    Segmentation = enum.auto()
    Object_Detection = enum.auto()
    Classification = enum.auto()
    One_Class_Classification = enum.auto()

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


class Inference_Cls(ImageClassificationInferencer):
        def __init__(self,weight:str,device='cuda',inputAspath=False):
            # self.checkpoint = '/data/moai/project/cityscape_mmsegmentation/Version5/working/logs/iter_160000.pth' # for degug
           
            weight = str(weight)
            self.inputAspath = inputAspath
            self.device = device  # FIX: Set device attribute
            
            # load the checkpoint
            chk_dict = torch.load(weight, map_location='cpu')
            
            # load the model config
            self.config = chk_dict['meta']['moai_model_config']            
            self.isMultilabels = chk_dict['meta']['moai_isMultilabels']            
            
            super().__init__(model=self.config,pretrained=weight,device=device)
            
            # load the class color mapping
            # self.class_color_map = chk_dict['meta'].get('moai_name_mapping',{})
            # print(f'=== {self.class_color_map=}')
            
            print(f'=== {self.config=}')
            
            # self.model.dataset_meta['palette'][0] = [0,0,0]
            
            self.classes_numpy = np.array(self.classes)
            self.id2clsid = {i:clsid for i,clsid in enumerate(self.classes)}
            
            self.init_pipeline()
        
        def get_last_checkpoint(self,chk_txt):
            '''
            get the last checkpoint from a txt file
            '''
            with open(chk_txt,'r') as f:
                chk_path = f.readline()
                chk_path = chk_path.strip()
            return chk_path
        
        
        def init_pipeline(self) -> Callable:
            # test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline            

            # # Image loading is finished in `self.preprocess`.
            # test_pipeline_cfg = remove_transform(test_pipeline_cfg,
            #                                     'LoadImageFromFile')
            # test_pipeline = Compose(
            #     [TRANSFORMS.build(t) for t in test_pipeline_cfg])
            
            # def load_image(input_):
            #     img = imread(input_) # input_ can be a path or a np.ndarray
            #     if img is None:
            #         raise ValueError(f'Failed to read image {input_}.')
            #     return dict(
            #         img=img,
            #         img_shape=img.shape[:2],
            #         ori_shape=img.shape[:2],
            #     )
                
            # test_pipeline = Compose([load_image, test_pipeline])
            # return test_pipeline
            def load_image(img):                              
                return dict(
                    img=img,
                    img_shape=img.shape[:2],
                    ori_shape=img.shape[:2],
                )            
            '''
            ResizeEdge(scale=256, edge=short, backend=cv2, interpolation=bilinear)
            CenterCrop(crop_size = (224, 224), auto_pad=False, pad_cfg={'type': 'Pad'},clip_object_border = True)
            PackInputs(...)
            '''            
            self.pipeline = Compose([load_image, self.pipeline])
            
    
        def preprocess(self, inputs: List[InputType], batch_size: int = 1):            
            # def load_image(input_):
            #     img = imread(input_)
            #     if img is None:
            #         raise ValueError(f'Failed to read image {input_}.')
            #     return dict(
            #         img=img,
            #         img_shape=img.shape[:2],
            #         ori_shape=img.shape[:2],
            #     )
            # if self.inputAspath:
            #     pipeline = Compose([load_image, self.pipeline])
            # else:
            #     pipeline = Compose([dict(type='LoadImageFromNDArray'), self.pipeline])
            chunked_data = self._get_chunk_data(map(self.pipeline, inputs), batch_size)
            yield from map(default_collate, chunked_data)
        
        @timing_decorator
        def predict(self,img:ImageType,conf=None,batch_size=1):
            '''

            '''
            ori_inputs = self._inputs_to_list(img)      
            inputs = self.preprocess(ori_inputs, batch_size=batch_size)
            
            results = []
            
            for data in inputs:
                if self.isMultilabels:
                    # Use improved multi-label flow
                    predictions, method_used = self.get_logits_safely(data) # head_logits is used
                    if method_used in ['direct_logits', 'head_logits']:
                        results_batch = self._process_logits(predictions, conf)
                    else:
                        results_batch = self.postprocess(predictions, return_datasamples=False, conf=conf)
                else:                    
                    preds = self.forward(data)
                    results_batch = self.postprocess(preds, return_datasamples=False, conf=conf)
                
                results.extend(results_batch)
            
            return results
        
        def postprocess(self,
                    preds: List[DataSample],
                    visualization: List[np.ndarray]=None,
                    return_datasamples=False,
                    conf=None) -> dict:
            if return_datasamples:
                return preds

            results = []
            if self.isMultilabels:
                conf = conf if conf is not None else 0.5
                for data_sample in preds:
                    pred_scores = data_sample.pred_score
                    # filter out the low score predictions                    
                    positive_scores = pred_scores[pred_scores > conf].detach().cpu().numpy()
                    pred_label = (pred_scores > conf).detach().cpu().numpy()
                    result = {
                        # 'pred_scores': pred_scores.detach().cpu().numpy(),
                        # 'pred_label': pred_label.nonzero()[0],
                        'pred_score': positive_scores,
                    }
                    if self.classes is not None:
                        result['pred_class'] = self.classes_numpy[pred_label]
                    results.append(result)
            else:
                # Set default confidence threshold for single-label classification
                conf = conf if conf is not None else 0.5
                for data_sample in preds:
                    pred_scores = data_sample.pred_score
                    pred_score = float(torch.max(pred_scores).item())
                    pred_label = torch.argmax(pred_scores).item()
                    
                    # Check if the highest confidence is above threshold
                    if pred_score >= conf:
                        result = {
                            'pred_score': pred_score,
                        }
                        if self.classes is not None:
                            result['pred_class'] = self.classes[pred_label]
                    else:
                        # Return "no object" or "background" result when confidence is too low
                        result = {
                            'pred_score': pred_score,
                            'pred_class': 'background'  # Indicates no confident prediction
                        }
                    results.append(result)

            return results
        
        def show_result_pyplot(self,img: Union[str, np.ndarray],
                            result,
                            opacity: float = 0.5,
                            title: str = '',
                            draw_gt = True,
                            draw_pred = True,
                            wait_time: float = 0,
                            show = True,
                            with_labels = True,
                            save_dir=None,
                            out_file=None,
                            gt_mask = None):
            """Visualize the segmentation results on the image.

            Args:
                model (nn.Module): The loaded segmentor.
                img (str or np.ndarray): Image filename or loaded image.
                result (SegDataSample): The prediction SegDataSample result.
                opacity(float): Opacity of painted segmentation map.
                    Default 0.5. Must be in (0, 1] range.
                title (str): The title of pyplot figure.
                    Default is ''.
                draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
                draw_pred (bool): Whether to draw Prediction SegDataSample.
                    Defaults to True.
                wait_time (float): The interval of show (s). 0 is the special value
                    that means "forever". Defaults to 0.
                show (bool): Whether to display the drawn image.
                    Default to True.
                with_labels(bool, optional): Add semantic labels in visualization
                    result, Default to True.
                save_dir (str, optional): Save file dir for all storage backends.
                    If it is None, the backend storage will not save any data.
                out_file (str, optional): Path to output file. Default to None.



            Returns:
                np.ndarray: the drawn image which channel is RGB.
            """                       
            model = self.model
            if hasattr(self.model, 'module'):
                model = self.model.module
            if isinstance(img, str):
                image = mmcv.imread(img, channel_order='rgb')
            else:
                image = img
            if save_dir is not None:
                mkdir_or_exist(save_dir)
            # init visualizer
            visualizer = SegLocalVisualizer(
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=save_dir,
                alpha=opacity)
            
            visualizer.dataset_meta = dict(
                classes=model.dataset_meta['classes'],
                palette=model.dataset_meta['palette'])
            
            if gt_mask is not None:                
                sem_seg = torch.from_numpy(gt_mask)
                result.gt_sem_seg = PixelData(data=sem_seg)
            
            visualizer.add_datasample(
                name=title,
                image=image,
                data_sample=result,
                draw_gt=draw_gt,
                draw_pred=draw_pred,
                wait_time=wait_time,
                out_file=out_file,
                show=show,
                with_labels=with_labels)
            
            vis_img = visualizer.get_image()
           
            return vis_img

        def get_logits_from_head(self, data):
            """
            Extract logits directly from model head before activation            
            """
            try:
                img_tensor = data['inputs']
                
                # Ensure tensor is properly formatted
                if img_tensor.dtype == torch.uint8:
                    img_tensor = img_tensor.float() / 255.0
                img_tensor = img_tensor.to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    # Extract features using the backbone
                    if hasattr(self.model, 'extract_feat'):
                        features = self.model.extract_feat(img_tensor)
                        
                        # Handle tuple output from extract_feat
                        if isinstance(features, tuple):
                            features = features[-1]  # Use the last feature map
                            print(f"=== Using last feature from tuple, shape: {features.shape}")
                        else:
                            print(f"=== Extracted features shape: {features.shape}")
                        
                        # Get logits from head without activation
                        if hasattr(self.model, 'head'):
                            head = self.model.head
                            
                            # Try different head structures
                            if hasattr(head, 'pre_logits'):
                                pre_logits = head.pre_logits(features)
                                
                                # Get final layer (usually fc or classifier)
                                if hasattr(head, 'fc'):
                                    raw_logits = head.fc(pre_logits)
                                elif hasattr(head, 'classifier'):
                                    raw_logits = head.classifier(pre_logits)
                                elif hasattr(head, 'layers') and hasattr(head.layers, 'head'):
                                    raw_logits = head.layers.head(pre_logits)
                                else:
                                    # Try calling head directly on features
                                    raw_logits = head(features)
                            else:
                                # Direct head call
                                raw_logits = head(features)
                            
                            print(f"=== Got logits from head: {raw_logits.shape}")
                            return raw_logits, 'head_logits'
                        
                return None, 'failed'
                
            except Exception as e:
                print(f"❌ Head logits method failed: {e}")
                return None, 'failed'

        def get_logits_safely(self, data):
            """
            Safely get logits by trying multiple methods in order of preference
            """
            # Method 1: Try direct head access (most reliable)
            raw_logits, method = self.get_logits_from_head(data)
            if method == 'head_logits':
                return raw_logits, method
            
            # Method 2: Try to get raw logits using mode='tensor'
            try:
                img_tensor = data['inputs']
                
                # CRITICAL FIX: Ensure tensor is Float type and properly normalized
                if img_tensor.dtype == torch.uint8:
                    print("Converting uint8 to float32 and normalizing")
                    img_tensor = img_tensor.float() / 255.0
                
                # Ensure the tensor is on the correct device
                img_tensor = img_tensor.to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    raw_logits = self.model(img_tensor, mode='tensor')
                    return raw_logits, 'direct_logits'
                    
            except Exception as e:
                print(f"⚠️ Direct logits method failed: {e}")
                
            # Method 3: Fallback to using forward method
            try:
                preds = self.forward(data)
                return preds, 'datasamples'
            except Exception as e2:
                print(f"❌ Fallback method also failed: {e2}")
                raise e2

        def _process_logits(self, raw_logits, conf=None):
            """Process raw logits for multi-label classification only"""
            results = []
            conf = conf if conf is not None else 0.5
            
            # Ensure raw_logits has batch dimension
            if raw_logits.dim() == 1:
                raw_logits = raw_logits.unsqueeze(0)  # Add batch dimension: [3] -> [1, 3]
            
            # Process each sample in the batch
            for i in range(raw_logits.shape[0]):
                sample_logits = raw_logits[i]  # Shape: [num_classes]
                
                # Apply sigmoid for multilabel (only path that will be used)
                pred_scores = torch.sigmoid(sample_logits)
                print(f"Multilabel sigmoid scores: {pred_scores}")
                
                # Apply confidence threshold
                positive_mask = pred_scores > conf
                positive_scores = pred_scores[positive_mask].detach().cpu().numpy()
                
                result = {
                    'pred_score': positive_scores,
                    'raw_logits': sample_logits.detach().cpu().numpy(),
                    'all_scores': pred_scores.detach().cpu().numpy(),
                }
                
                if self.classes is not None:
                    positive_indices = positive_mask.detach().cpu().numpy()
                    result['pred_class'] = self.classes_numpy[positive_indices].tolist()
                    
                results.append(result)
            
            return results

        def save_result(self,results,out_file):
            '''
            save the predicted masks to .png format
            '''         
            out_file = str(out_file)   
            array = results
            if array.ndim == 3 and array.shape[0] == 1:
                array = array.squeeze(0)  # Remove the single channel dimension
            cv2.imwrite(out_file,array)
 

class Inference_Seg:
        def __init__(self,weight:str,device='cuda'):
            # self.checkpoint = '/data/moai/project/cityscape_mmsegmentation/Version5/working/logs/iter_160000.pth' # for degug
            # for conf in args.model_configs:
            #     if (args.work_dir/conf).is_file():
            #         self.config = args.work_dir/conf
            #         break
            weight = str(weight)                        
            
            self.device = device
            
            # load the checkpoint
            chk_dict = torch.load(weight, map_location='cpu')
            
            # load the class color mapping
            self.class_color_map = chk_dict['meta'].get('moai_name_mapping',{})
            
            # load the model config
            self.config = chk_dict['meta']['moai_model_config']
            
            print(f'=== {self.config=}')
            print(f'=== {self.class_color_map=}')
            self.model = init_model(config = self.config, checkpoint = weight, device=device)
            self.model.dataset_meta['palette'][0] = [0,0,0]
            
            self.id2clsid = {i:clsid for i,clsid in enumerate(self.model.dataset_meta['classes'])}
            print(f'{self.id2clsid=}')
        
        def get_last_checkpoint(self,chk_txt):
            '''
            get the last checkpoint from a txt file
            '''
            with open(chk_txt,'r') as f:
                chk_path = f.readline()
                chk_path = chk_path.strip()
            return chk_path
        
        def predict(self,img:ImageType,conf=None,plotMask=False):
            '''
            Args:
                img: the input image
                conf: the confidence threshold for the prediction             
            '''                        
            # prepare data
            data, is_batch = self._preprare_data(img)
            
            # forward the model
            with torch.no_grad():
                    results = self.model.test_step(data)
            
            
            if conf is not None: # filter out the low confidence predictions
                logits = results[0].seg_logits.data # shape: (Num_class, H, W)
                                    
                probabilities = torch.softmax(logits, dim=0)
                max_probabilities,pred_classes = probabilities.max(dim=0)
                
                pred_mask = (max_probabilities<=conf).to(self.device)
                pred_mask = pred_mask.unsqueeze(0)                
                
                results[0].pred_sem_seg.data[pred_mask] = 0
            
            if not is_batch:                    
                moai_format = mask_to_moai_label(results[0].pred_sem_seg.data.cpu().numpy().squeeze(0),id2clsid=self.id2clsid,cls2color=self.class_color_map)                      
            else: 
                moai_format = [mask_to_moai_label(res.pred_sem_seg.data.cpu().numpy().squeeze(0),id2clsid=self.id2clsid,cls2color=self.class_color_map) for res in results]
            
            if plotMask:
                img = self.show_result_pyplot(img,results[0],show=False)
                
                return moai_format,img
            else:
                return moai_format
        
        
        def __call__(self,img:ImageType,conf=None,forPlot=False):
            '''

            '''
            # prepare data
            data, is_batch = self._preprare_data(img)

            # forward the model
            with torch.no_grad():
                    results = self.model.test_step(data)
            
            if forPlot:
                return results if is_batch else results[0]
            else:
                return results[0].pred_sem_seg.data.cpu().numpy() if not is_batch else [res.pred_sem_seg.data.cpu().numpy() for res in results]
                
        def _preprare_data(self,imgs):
            cfg = self.model.cfg
            for t in cfg.test_pipeline:
                if t.get('type') == 'LoadAnnotations':
                    cfg.test_pipeline.remove(t)

            is_batch = True
            if not isinstance(imgs, (list, tuple)):
                imgs = [imgs]
                is_batch = False

            if isinstance(imgs[0], np.ndarray):
                cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

            # TODO: Consider using the singleton pattern to avoid building
            # a pipeline for each inference
            pipeline = Compose(cfg.test_pipeline)

            data = defaultdict(list)
            for img in imgs:
                if isinstance(img, np.ndarray):
                    data_ = dict(img=img)
                else:
                    data_ = dict(img_path=img)

                data_ = pipeline(data_)
                data['inputs'].append(data_['inputs'])
                data['data_samples'].append(data_['data_samples'])

            return data, is_batch
        
        def show_result_pyplot(self,img: Union[str, np.ndarray],
                            result,
                            opacity: float = 0.5,
                            title: str = '',
                            draw_gt = True,
                            draw_pred = True,
                            wait_time: float = 0,
                            show = True,
                            with_labels = True,
                            save_dir=None,
                            out_file=None,
                            gt_mask = None,
                            ignore_BG = True):
            """Visualize the segmentation results on the image.

            Args:
                model (nn.Module): The loaded segmentor.
                img (str or np.ndarray): Image filename or loaded image.
                result (SegDataSample): The prediction SegDataSample result.
                opacity(float): Opacity of painted segmentation map.
                    Default 0.5. Must be in (0, 1] range.
                title (str): The title of pyplot figure.
                    Default is ''.
                draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
                draw_pred (bool): Whether to draw Prediction SegDataSample.
                    Defaults to True.
                wait_time (float): The interval of show (s). 0 is the special value
                    that means "forever". Defaults to 0.
                show (bool): Whether to display the drawn image.
                    Default to True.
                with_labels(bool, optional): Add semantic labels in visualization
                    result, Default to True.
                save_dir (str, optional): Save file dir for all storage backends.
                    If it is None, the backend storage will not save any data.
                out_file (str, optional): Path to output file. Default to None.



            Returns:
                np.ndarray: the drawn image which channel is RGB.
            """                       
            model = self.model
            if hasattr(self.model, 'module'):
                model = self.model.module
            if isinstance(img, str):
                image = mmcv.imread(img, channel_order='rgb')
            else:
                image = img
            if save_dir is not None:
                mkdir_or_exist(save_dir)
            # init visualizer
            visualizer = SegLocalVisualizer(
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=save_dir,
                alpha=opacity)
            
            visualizer.dataset_meta = dict(
                classes=model.dataset_meta['classes'],
                palette=model.dataset_meta['palette'])


            if gt_mask is not None:                
                sem_seg = torch.from_numpy(gt_mask)
                result.gt_sem_seg = PixelData(data=sem_seg)
            
            if ignore_BG:
                result.pred_sem_seg.data[result.pred_sem_seg.data==0] = 255
                print(f"ignore background labels")
            
            visualizer.add_datasample(
                name=title,
                image=image,
                data_sample=result,
                draw_gt=draw_gt,
                draw_pred=draw_pred,
                wait_time=wait_time,
                out_file=out_file,
                show=show,
                with_labels=with_labels)
            
            vis_img = visualizer.get_image()
           
            return vis_img

        def save_result(self,results,out_file):
            '''
            save the predicted masks to .png format
            '''         
            out_file = str(out_file)   
            array = results
            if array.ndim == 3 and array.shape[0] == 1:
                array = array.squeeze(0)  # Remove the single channel dimension
            cv2.imwrite(out_file,array)
        
          
        def draw_mask_on_image(self,image, results, borders=True):
            '''
            Draw a mask on an image with optional borders
            obj_id: int, optional: used to select a color from the tab10 colormap
            
            '''
            obj_labels = results['label']
            # if random_color:
            #     color = np.random.randint(0, 255, size=3).tolist()
            # else:
            #     # color = [30, 144, 255]
            #     cmap = plt.get_cmap("tab10")
            #     cmap_idx = 0
            #     color = np.array([*cmap(cmap_idx)[:3], 0.6])

            # Convert PIL image to NumPy array
            if isinstance(image, Image.Image):
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Ensure the image and mask are in the same format
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            overlay = np.zeros_like(image)
            
            for obj in obj_labels: # obj keys = ['shape', 'position', 'item_id', 'attributes', 'color']
                print(f"{obj['item_id']} {obj['color']=}")
                contours = obj['position']
                if isinstance(obj['color'],dict):
                    color = obj['color']['color'] # {"name": "apple", "color": "#ef2929"}
                else:
                    color = obj['color']
                rgb_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                
                colored_mask = self.draw_contour(overlay=overlay,
                                                 contours=contours,
                                                 color=rgb_color,
                                                 borders=borders,
                                                 contour_thickness=4,
                                                 className=obj['color']['name'])
            
            # Alpha (transparency factor)
            alpha = 0.4  # 0.0 is fully transparent, 1.0 is fully opaque            
            # Overlay the colored mask on the image
            result_image = cv2.addWeighted(image, alpha, colored_mask, 1-alpha, 0)                        

            
            return result_image

        def draw_contour(self,overlay,contours,color,borders=True,contour_thickness=2,className=None):  
            h,w,_ = overlay.shape
            contour = np.array([[d['x']*w/100, d['y']*h/100] for d in contours], dtype=np.float32)

            # if not np.array_equal(contour[0], contour[-1]):
            #     contour = np.vstack([contour, contour[0]])
            
            # Convert to integer pixel values (since OpenCV works with pixel indices)
            contour = contour.astype(int)

            # Reshape to match OpenCV's expected shape (N, 1, 2)
            contours = contour.reshape((-1, 1, 2))
            
            overlay = cv2.fillPoly(overlay, [contours], color=color)  # Green polygon
            
            # Add the class name
            if className is not None:
                # Calculate the centroid of the polygon
                centroid_x = np.mean(contours[:, 0, 0]).astype(int)
                centroid_y = np.mean(contours[:, 0, 1]).astype(int)
                centroid = (centroid_x, centroid_y)
                
                # Get the text size to determine rectangle size
                (text_width, text_height), baseline = cv2.getTextSize(className, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                position = (centroid[0]- text_width // 2 , centroid[1] + text_height // 2 ) 
                                               
                # Define the rectangle coordinates (centered on the centroid)
                rect_start = (centroid[0] - text_width // 2 - 5, centroid[1] - text_height // 2 - 5)
                rect_end = (centroid[0] + text_width // 2 + 5, centroid[1] + text_height // 2 + baseline)

                # Draw the purple rectangle (BGR format for purple: (255, 0, 255))
                overlay = cv2.rectangle(overlay, rect_start, rect_end, (255, 0, 255), thickness=-1)  # Filled rectangle

                # Add the class name text on top of the rectangle
                cv2.putText(overlay, className, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.5
                # font_thickness = 1
                # text = "classname"
                # text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                # text_x = int(contour[0, 0, 0] + 5)
                # text_y = int(contour[0, 0, 1] + 5)
                # overlay = cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                
            if borders:
                # Draw contours on the result image
                overlay = cv2.drawContours(overlay, contours, -1, (255, 255, 255), thickness=contour_thickness)
                
            return overlay
    

class LabelName(int, Enum):
    """Name of label."""

    NORMAL = 0
    ABNORMAL = 1

class MyImageResult(ImageResult):    
    def __init__(
        self,
        image: np.ndarray,
        pred_score: float,
        pred_label: str,
        anomaly_map: np.ndarray | None = None,
        gt_mask: np.ndarray | None = None,
        pred_mask: np.ndarray | None = None,
        gt_boxes: np.ndarray | None = None,
        pred_boxes: np.ndarray | None = None,
        box_labels: np.ndarray | None = None,
        normalize: bool = False,
        alpha: float = 0.8,
    ) -> None:
        self.anomaly_map = anomaly_map
        self.box_labels = box_labels
        self.gt_boxes = gt_boxes
        self.gt_mask = gt_mask
        self.image = image
        self.pred_score = pred_score
        self.pred_label = pred_label
        self.pred_boxes = pred_boxes
        self.heat_map: np.ndarray | None = None
        self.segmentations: np.ndarray | None = None
        self.normal_boxes: np.ndarray | None = None
        self.anomalous_boxes: np.ndarray | None = None

        if anomaly_map is not None:
            self.heat_map = superimpose_anomaly_map(self.anomaly_map, self.image, normalize=normalize,alpha=alpha)

        if self.gt_mask is not None and self.gt_mask.max() <= 1.0:
            self.gt_mask *= 255

        self.pred_mask = pred_mask
        if self.pred_mask is not None and self.pred_mask.max() <= 1.0:
            self.pred_mask *= 255
            self.segmentations = mark_boundaries(self.image, self.pred_mask, color=(1, 0, 0), mode="thick")
            if self.segmentations.max() <= 1.0:
                self.segmentations = (self.segmentations * 255).astype(np.uint8)

        if self.pred_boxes is not None:
            if self.box_labels is None:
                msg = "Box labels must be provided when box locations are provided."
                raise ValueError(msg)

            self.normal_boxes = self.pred_boxes[~self.box_labels.astype(bool)]
            self.anomalous_boxes = self.pred_boxes[self.box_labels.astype(bool)]
        
    
class Inference_VAD:
    def __init__(self,weight:str,device='cuda'):        
        self.weight = weight
        self.meta_file = Path(self.weight).parent/'metadata.json'
        
        # self.inferencer = OpenVINOInferencer(
        #     path=weight,
        #     metadata=meta_data,
        #     device=device, #"CPU",
        # )

        self.device = device
        
        self.model = Patchcore.load_from_checkpoint(weight)
        
        # Ensure model is in evaluation mode
        self.model.eval()
        self.model = self.model.to(self.device)        
        
        self.init_metadata()
        

    def init_metadata(self):        
        self.metadata = dict()
        if hasattr(self.model, "normalization_metrics") and self.model.normalization_metrics.state_dict() is not None:
            for key, value in self.model.normalization_metrics.state_dict().items():
                self.metadata[key] = value.cpu()     
                print(f"===== {key=}: {self.metadata[key]}") 
        
        # load json file
        with open(self.meta_file) as f:          
            self.metadata.update(json.load(f))
        print(f"{self.metadata=}")        
        self.image_size = self.metadata["img_size"]
        if "image_threshold_train" in self.metadata:
            self.image_threshold = self.metadata["image_threshold_train"]            
        
        self.pixel_threshold_normalized = (self.anomaly_maps_max - self.anomaly_maps_min) * self.pixel_threshold        
        
        # Load statistical normalization parameters if available
        self.normal_score_mean = self.metadata.get("normal_score_mean", None)
        self.normal_score_std = self.metadata.get("normal_score_std", None)
        
        if self.normal_score_mean is None or self.normal_score_std is None:
            print("Warning: Statistical normalization parameters not found in metadata")
        
    
    @property
    def anomaly_maps_min(self):
        return self.metadata["anomaly_maps.min"].item()
    
    @property
    def anomaly_maps_max(self):
        return self.metadata["anomaly_maps.max"].item()
    
    @property
    def pred_scores_min(self):
        return self.metadata["pred_scores.min"].item()
    
    @property
    def pred_scores_max(self):
        return self.metadata["pred_scores.max"].item()
    
    @property
    def pixel_threshold(self):
        return self.model.pixel_threshold.value.item()

    @pixel_threshold.setter
    def pixel_threshold(self, value):
        self.model.pixel_threshold.value = torch.tensor(value,dtype=torch.float32)
    
    @property
    def image_threshold(self):
        return self.model.image_threshold.value.item()
    
    @image_threshold.setter
    def image_threshold(self, value):
        self.model.image_threshold.value = torch.tensor(value,dtype=torch.float32)
    
    def apply_statistical_normalization(self, raw_score):
        """Apply statistical normalization (z-score + sigmoid) to raw anomaly score.
        
        Args:
            raw_score: Raw anomaly score from the model
            
        Returns:
            Normalized score in range [0, 1]
        """
        if self.normal_score_mean is None or self.normal_score_std is None:
            print("Warning: Statistical normalization parameters not available, using raw score")
            return raw_score
        
        # Apply z-score transformation
        z_score = (raw_score - self.normal_score_mean) / self.normal_score_std
        
        # Apply sigmoid to map to [0, 1] range
        normalized_score = 1 / (1 + np.exp(-z_score))
        
        return normalized_score
    
    @staticmethod
    def read_image(path: str | Path, as_tensor: bool = False,image_size = None) -> torch.Tensor | np.ndarray:
        """Read image from disk in RGB format.

        Args:
            path (str, Path): path to the image file
            as_tensor (bool, optional): If True, returns the image as a tensor. Defaults to False.

        Example:
            >>> image = read_image("test_image.jpg")
            >>> type(image)
            <class 'numpy.ndarray'>
            >>>
            >>> image = read_image("test_image.jpg", as_tensor=True)
            >>> type(image)
            <class 'torch.Tensor'>

        Returns:
            image as numpy array
        """
        image = Image.open(path).convert("RGB")
        if image_size is not None:
            # resize the image to the model's input size
            image = image.resize(image_size)
            
        return to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0

    def path2tensor(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        return image_tensor         
    
    def predict(self, img_path, conf=0.5, pixel_threshold=None, 
               use_dynamic_threshold=True, user_threshold=None):
        """Predict anomalies in an image.
        
        Args:
            img_path: Input image path or numpy array
            conf: Confidence threshold for final prediction (default: 0.5)
            pixel_threshold: Optional pixel-level threshold override
            use_dynamic_threshold: If True, uses dynamic threshold-responsive heatmap (default: True)
            user_threshold: User-adjustable threshold (0.0-1.0). If None, uses conf parameter.
            
        Returns:
            If use_dynamic_threshold=True: Returns backward-compatible tuple but with dynamic heatmap
            If use_dynamic_threshold=False: tuple (is_anomalous, img_result, raw_score)
        """
        if pixel_threshold is not None:
            self.pixel_threshold = (self.anomaly_maps_max - self.anomaly_maps_min) * pixel_threshold
            
        # Only use dynamic threshold when explicitly requested
        # Default to False to maintain training evaluation consistency
        if use_dynamic_threshold:
            threshold_to_use = user_threshold if user_threshold is not None else conf
            dynamic_result = self.generate_dynamic_heatmap(img_path, threshold_to_use)
            
            # Create backward-compatible MyImageResult object
            img_result = MyImageResult(
                image=dynamic_result['original_image'],
                pred_score=dynamic_result['pred_score'],
                pred_label='ABNORMAL' if dynamic_result['is_anomalous'] else 'NORMAL',
                anomaly_map=dynamic_result['raw_anomaly_map'],  # Keep raw for compatibility
                normalize=False  # Don't auto-generate heat_map
            )
            
            # Use the overlapped heatmap (threshold-adjusted heatmap overlaid on original image)
            img_result.heat_map = dynamic_result['overlapped_heatmap']
            
            return dynamic_result['is_anomalous'], img_result, dynamic_result['pred_score']
        
        with torch.no_grad():
            # read image and translate to tensor
            # new_img_tensor = read_image(img_path, as_tensor=True) # [3,h,w]
            if isinstance(img_path, str):
                # read image from path and translate to tensor
                new_img_tensor = self.read_image(img_path, as_tensor=True)  # [3,h,w]                                
                original_image = (self.read_image(img_path) * 255).astype(np.uint8)
            elif isinstance(img_path, np.ndarray):
                # convert numpy array to tensor
                temp_img = cv2.cvtColor(img_path,cv2.COLOR_BGR2RGB)
                nd_array = Image.fromarray(temp_img)
                new_img_tensor = to_dtype(to_image(nd_array), torch.float32, scale=True)
                new_img_tensor = new_img_tensor
                original_image = temp_img
            elif isinstance(img_path, Image.Image):
                new_img_tensor = to_dtype(to_image(img_path), torch.float32, scale=True)
                
                original_image = np.array(img_path).astype(np.uint8)
            else:
                raise ValueError("Input must be a file path or a numpy array")

            # resize the image tensor to the model's input size
            image_tensor = torch.nn.functional.interpolate(new_img_tensor.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
            image_tensor = image_tensor.unsqueeze(0).to(self.device) # [1,3,h,w]
            
            # Pass the image tensor to the model
            output = self.model(image_tensor)            
            
            # resize the anomaly map to the original image size
            output['anomaly_map'] = cv2.resize(output['anomaly_map'].squeeze().cpu().numpy(), (new_img_tensor.shape[2], new_img_tensor.shape[1]))
            
        
        # post_results = self.post_process(predictions=output['anomaly_map'].cpu().numpy()) # for debug
        post_results = self.post_process(predictions=output['anomaly_map']) # for debug
        
        # Get raw score and apply statistical normalization
        raw_score = output['pred_score'].item()
        normalized_score = self.apply_statistical_normalization(raw_score)
        
        # Update the post_results with normalized score
        post_results["pred_score"] = normalized_score
        
        # Update prediction label based on normalized score and threshold
        post_results["pred_label"] = "ABNORMAL" if normalized_score >= conf else "NORMAL"
        
        img_result = MyImageResult(
            # image=(self.read_image(img_path) * 255).astype(np.uint8),
            image=original_image,
            pred_score=post_results["pred_score"],
            pred_label=post_results["pred_label"],
            anomaly_map=post_results["anomaly_map"],
            pred_mask=post_results["pred_mask"],
            pred_boxes=post_results["pred_boxes"],
            box_labels=post_results["box_labels"],
        )
    
        print(f"Raw score: {raw_score:.6f}, Normalized score: {normalized_score:.6f}, Threshold: {conf}")
        return img_result.pred_score>conf, img_result, raw_score
    
    def post_process(self, predictions: np.ndarray) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (np.ndarray): Raw output predicted by the model.
            metadata (Dict, optional): Metadata. Post-processing step sometimes requires
                additional metadata such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, Any]: Post processed prediction results.
        """
        # Initialize the result variables.
        anomaly_map: np.ndarray | None = None
        pred_label: LabelName | None = None
        pred_mask: float | None = None

        # If predictions returns a single value, this means that the task is
        # classification, and the value is the classification prediction score.
        if len(predictions.shape) == 1:
            task = AnomalibTaskType.CLASSIFICATION
            pred_score = predictions
        else:
            task = AnomalibTaskType.SEGMENTATION
            anomaly_map = predictions.squeeze()
            pred_score = anomaly_map.reshape(-1).max()
        
        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        pred_idx = pred_score >= self.image_threshold
        pred_label = LabelName.ABNORMAL if pred_idx else LabelName.NORMAL
        
        pred_mask = (anomaly_map >= self.pixel_threshold).astype(np.uint8)
        anomaly_map, pred_score = self._normalize(
            pred_scores=pred_score,
            anomaly_maps=anomaly_map,            
        )
        
        # if "image_shape" in metadata and anomaly_map.shape != metadata["image_shape"]:
        #     image_height = metadata["image_shape"][0]
        #     image_width = metadata["image_shape"][1]
        #     anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

        #     if pred_mask is not None:
        #         pred_mask = cv2.resize(pred_mask, (image_width, image_height))
      

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
            "pred_boxes": None,
            "box_labels": None,
        }

    
    def _normalize(
        self,
        pred_scores: torch.Tensor | np.float32,
        # metadata: dict,
        anomaly_maps: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[np.ndarray | torch.Tensor | None, float]:
        """Apply normalization and resizes the image.

        Args:
            pred_scores (Tensor | np.float32): Predicted anomaly score
            metadata (dict | DictConfig): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
            anomaly_maps (Tensor | np.ndarray | None): Predicted raw anomaly map.

        Returns:
            tuple[np.ndarray | torch.Tensor | None, float]: Post processed predictions that are ready to be
                visualized and predicted scores.
        """
        # min max normalization
        # if "pred_scores.min" in metadata and "pred_scores.max" in metadata:
            # if anomaly_maps is not None and "anomaly_maps.max" in metadata:
        
        anomaly_maps = normalize_min_max(
            anomaly_maps,
            self.pixel_threshold,
            self.anomaly_maps_min, # metadata["anomaly_maps.min"],
            self.anomaly_maps_max,# metadata["anomaly_maps.max"],
        )
        pred_scores = normalize_min_max(
            pred_scores,
            self.image_threshold,
            self.pred_scores_min, # metadata["pred_scores.min"],
            self.pred_scores_max, # metadata["pred_scores.max"],
        )

        return anomaly_maps, float(pred_scores)

    @staticmethod
    def normalize_min_max(
        targets: np.ndarray | np.float32 | torch.Tensor,
        threshold: float | np.ndarray | torch.Tensor,
        min_val: float | np.ndarray | torch.Tensor,
        max_val: float | np.ndarray | torch.Tensor,
        ) -> np.ndarray | torch.Tensor:
        """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
        normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
        
        if isinstance(targets, np.ndarray | np.float32 | np.float64):
            normalized = np.minimum(normalized, 1)
            normalized = np.maximum(normalized, 0)
        elif isinstance(targets, torch.Tensor):
            normalized = torch.minimum(normalized, torch.tensor(1))  # pylint: disable=not-callable
            normalized = torch.maximum(normalized, torch.tensor(0))  # pylint: disable=not-callable
        else:
            msg = f"Targets must be either Tensor or Numpy array. Received {type(targets)}"
            raise TypeError(msg)
        return normalized

    
    # def predict(self,image_path: str):
    #     image = read_image(path=image_path)
    #     predictions = self.inferencer.predict(image=image)
    #     return predictions


    def visualize_anomaly(self,predictions: Any) -> Image:
        visualizer = ImageVisualizer(
            mode=VisualizationMode.FULL, task=AnomalibTaskType.SEGMENTATION
        )
        output_image = visualizer.visualize_image(predictions)
        return Image.fromarray(output_image)

    def generate_dynamic_heatmap(self, img_path, user_threshold: float = 0.5):
        """Generate heatmap with dynamic user threshold for interactive threshold adjustment.
        
        Args:
            img_path: Input image path or numpy array
            user_threshold: User-adjustable threshold (0.0 to 1.0)
            
        Returns:
            dict: Contains heatmap, original image, prediction info
        """
        with torch.no_grad():
            # Get raw anomaly map (before normalization)
            if isinstance(img_path, str):
                new_img_tensor = self.read_image(img_path, as_tensor=True)
                original_image = (self.read_image(img_path) * 255).astype(np.uint8)
            elif isinstance(img_path, np.ndarray):
                temp_img = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
                nd_array = Image.fromarray(temp_img)
                new_img_tensor = to_dtype(to_image(nd_array), torch.float32, scale=True)
                original_image = temp_img
            elif isinstance(img_path, Image.Image):
                new_img_tensor = to_dtype(to_image(img_path), torch.float32, scale=True)
                original_image = np.array(img_path).astype(np.uint8)
            else:
                raise ValueError("Input must be a file path or a numpy array")

            # Resize and forward through model
            image_tensor = torch.nn.functional.interpolate(
                new_img_tensor.unsqueeze(0), size=self.image_size, 
                mode="bilinear", align_corners=False
            ).squeeze(0)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            output = self.model(image_tensor)
            
            # Resize anomaly map to original image size
            raw_anomaly_map = cv2.resize(
                output['anomaly_map'].squeeze().cpu().numpy(), 
                (new_img_tensor.shape[2], new_img_tensor.shape[1])
            )
            
        # Generate threshold-responsive heatmap using the dedicated method
        threshold_adjusted_heatmap_rgb = self.update_heatmap_threshold(raw_anomaly_map, user_threshold)
        
        # Create overlapped heatmap by manually overlaying the threshold-adjusted heatmap
        # Keep everything in RGB format for consistency
        alpha = 0.6  # More transparent heatmap overlay
        
        # Convert original image to RGB if it's BGR
        if original_image.shape[-1] == 3:
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image_rgb = original_image
            
        # Blend in RGB space
        overlapped_heatmap_rgb = cv2.addWeighted(
            original_image_rgb.astype(np.uint8), 1 - alpha,
            threshold_adjusted_heatmap_rgb.astype(np.uint8), alpha,
            0
        )
        
        # Convert back to BGR for consistency with MyImageResult.heat_map format
        overlapped_heatmap = cv2.cvtColor(overlapped_heatmap_rgb, cv2.COLOR_RGB2BGR)
        
        # Apply statistical normalization (same as predict method)
        raw_pred_score = raw_anomaly_map.max()
        normalized_pred_score = self.apply_statistical_normalization(raw_pred_score)
        
        # user_threshold is directly the threshold in normalized space (0-1)
        is_anomalous = normalized_pred_score >= user_threshold
        pred_score = normalized_pred_score
        
        # Calculate min/max for return data
        anomaly_min = raw_anomaly_map.min()
        anomaly_max = raw_anomaly_map.max()
        
        
        return {
            'heatmap': threshold_adjusted_heatmap_rgb,  # Standalone threshold-responsive heatmap (RGB format)
            'overlapped_heatmap': overlapped_heatmap,  # Heatmap overlaid on original image (BGR format)
            'original_image': original_image,
            'raw_anomaly_map': raw_anomaly_map,
            'pred_score': pred_score,
            'threshold_used': self.image_threshold,  # The actual threshold used
            'user_threshold': user_threshold,
            'is_anomalous': is_anomalous,
            'anomaly_score_range': (anomaly_min, anomaly_max)
        }

    def update_heatmap_threshold(self, raw_anomaly_map: np.ndarray, user_threshold: float):
        """Quick method to regenerate heatmap with different threshold without re-inference.
        
        Args:
            raw_anomaly_map: Previously computed raw anomaly map
            user_threshold: New user threshold (0.0 to 1.0)
            
        Returns:
            np.ndarray: Updated heatmap with new threshold
        """
        from .Trainer import Trainer_VAD
        trainer = Trainer_VAD.__new__(Trainer_VAD)
        
        return trainer.anomaly_map_to_color_map(
            raw_anomaly_map, 
            user_threshold=user_threshold,
            normalize=True,
            metadata=self.metadata
        )



def get_inference_model(task,weight:str,device='cuda'):
    '''
    get the inference model
    '''
    if task == TaskType.Segmentation:
        return Inference_Seg(weight,device=device)
    elif task == TaskType.Classification:
        return Inference_Cls(weight,device=device)
    elif task == TaskType.One_Class_Classification:        
        return Inference_VAD(weight,device=device)
        
        
if __name__=='__main__':
    args=dict(
            checkpoint = 'work_dirs/configModel_pspnet/iter_20000.pth',
            config = 'configModel_pspnet.py',
            show_dir = 'psp_cityscape_results')

    args=dict(
            checkpoint = 'work_dirs/configModel_segFormer/iter_160000.pth',
            config = 'configModel_segFormer.py',
            show_dir = 'segFormer_cityscape_results')


    config_file = args['config']
    checkpoint_file = args['checkpoint']

    # ================================================================================================
    updated_cfg = dict(model=dict(decode_head=dict(num_classes=17)))

    # build the model from a config file and a checkpoint file

    # ================================================================================================

    # test a video and show the results
    # video = mmcv.VideoReader('video.mp4')
    # for frame in video:
    #    result = inference_model(model, frame)
    #    show_result_pyplot(model, frame, result, wait_time=1)


