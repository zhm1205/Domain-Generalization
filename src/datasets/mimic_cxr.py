import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional, Callable, Literal, Any
from .transforms import _to_tensor

from torch.utils.data import Dataset
from ..utils.logger import get_logger


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR Dataset Base Class"""
    
    # MIMIC-CXR
    DISEASE_LABELS = [
        # 'No Finding',
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    def __init__(self, 
                 csv_path: str,
                 split: Literal['train','validate','val','test']='train',
                 transform: Optional[Callable[..., Any]] = None,
                 task_type: Literal['classification','reid','ue']='classification',
                 **kwargs):
        """
        Args:
            csv_path: Path to the CSV file
            split: 'train', 'validate', or 'test'
            transform: Image transformations
            task_type: 'classification' or 'reid'
        """
        super().__init__(**kwargs)
        self.logger = get_logger()
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        self.task_type = task_type
        
        # Load and filter data
        self.df = pd.read_csv(csv_path)
        
        self.logger.info(f"Loaded {split} split: {len(self.df)} images, {self.df['subject_id'].nunique()} patients")
        
        # Setup task-specific attributes
        if task_type == 'reid':
            self._setup_reid()
        elif task_type == 'classification':
            self._setup_classification()
        elif task_type == 'ue':
            self._setup_ue()   # <--- 新增
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    def _setup_reid(self):
        """Setup for ReID task"""
        # Create subject_id to label mapping (continuous integers starting from 0)
        unique_subjects = sorted(self.df['subject_id'].unique())
        self.subject_to_label = {subject_id: idx for idx, subject_id in enumerate(unique_subjects)}
        self.label_to_subject = {idx: subject_id for subject_id, idx in self.subject_to_label.items()}
        
        # Add reid_label column
        self.df['reid_label'] = self.df['subject_id'].map(self.subject_to_label)
        
        self.num_classes = len(unique_subjects)
        self.logger.info(f"ReID setup: {self.num_classes} unique subjects/classes")
        
        # Statistics for ReID
        images_per_subject = self.df['reid_label'].value_counts()
        self.logger.info(f"Images per subject - Min: {images_per_subject.min()}, "
              f"Max: {images_per_subject.max()}, "
              f"Mean: {images_per_subject.mean():.2f}")
    
    def _setup_classification(self):
        """Setup for multi-label classification task"""
        # Check which disease labels are available in the CSV
        available_labels = [label for label in self.DISEASE_LABELS if label in self.df.columns]
        self.disease_labels = available_labels
        self.num_classes = len(available_labels)
        
        self.logger.info(f"Classification setup: {self.num_classes} disease labels")
        self.logger.info(f"Available labels: {available_labels}")
        
        # Convert disease labels to binary format (handle NaN as 0, positive as 1, negative as 0)
        self.disease_matrix = []
        for _, row in self.df.iterrows():
            label_vector = []
            for label in self.disease_labels:
                value = row[label]
                if pd.isna(value):
                    label_vector.append(0)  # NaN -> 0
                else:
                    label_vector.append(1 if value == 1.0 else 0)  # 1.0 -> 1, others -> 0
            self.disease_matrix.append(label_vector)
        
        self.disease_matrix = np.array(self.disease_matrix, dtype=np.float32)
        
        # Print label statistics
        positive_counts = self.disease_matrix.sum(axis=0)
        for i, label in enumerate(self.disease_labels):
            pos_count = int(positive_counts[i])
            pos_rate = pos_count / len(self.df) * 100
            self.logger.info(f"  {label}: {pos_count}/{len(self.df)} ({pos_rate:.1f}%)")

    def _setup_ue(self):
        """
        # Prepare a unified view for UE training:
        # - If the CSV contains disease columns -> compute multi-label matrix (self.disease_matrix / self.disease_labels)
        # - Always construct reid labels based on subject_id -> df['reid_label']
        """
        # 1) classification side (optional: if CSV has disease columns)
        available_labels = [label for label in self.DISEASE_LABELS if label in self.df.columns]
        self.disease_labels = available_labels
        self._has_cls = len(available_labels) > 0

        if self._has_cls:
            mat = []
            for _, r in self.df.iterrows():
                vec = []
                for lab in available_labels:
                    v = r[lab]
                    if pd.isna(v):
                        vec.append(0)
                    else:
                        vec.append(1 if v == 1.0 else 0)
                mat.append(vec)
            self.disease_matrix = np.array(mat, dtype=np.float32)
            self.num_classes_cls = len(available_labels)
        else:
            self.disease_matrix = None
            self.num_classes_cls = 0

        # 2) ReID side (always available, as long as subject_id exists)
        if 'subject_id' not in self.df.columns:
            raise ValueError("UE view requires 'subject_id' column for reid labeling.")
        uniq = sorted(self.df['subject_id'].unique())
        self.ue_subject_to_label = {sid: i for i, sid in enumerate(uniq)}
        self.df['reid_label'] = self.df['subject_id'].map(self.ue_subject_to_label)
        self.num_classes_reid = len(uniq)
        self._has_reid = True


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx, do_transform=True):
        row = self.df.iloc[idx]

        image_path = row['image_path']
        img_np = self._load_image(image_path)  # numpy HWC, uint8

        # Albumentations A.Compose
        if self.transform and do_transform:
            image = self.transform(img_np)
        else:
            image = _to_tensor(img_np)

        if self.task_type == 'reid':
            return self._get_reid_item(row, image)
        elif self.task_type == 'classification':
            return self._get_classification_item(row, image)
        elif self.task_type == 'ue':
            return self._get_ue_item(row, image)

    def _load_image(self, image_path):
        """Load image as uint8 numpy array (H,W,C, RGB)."""
        try:
            img = Image.open(image_path).convert('RGB')
            return np.array(img)  # uint8, HWC
        except Exception as e:
            self.logger.info(f"Error loading image {image_path}: {e}")
            raise ValueError(f"Failed to load image: {image_path}")

    def _get_reid_item(self, row, image):
        """Return data for ReID task"""
        return {
            'image': image,
            'label': int(row['reid_label']),  # Subject ID as class label
            'subject_id': int(row['subject_id']),  # Original subject ID
            'dicom_id': row['dicom_id'],
            'study_id': row['study_id'],
            'view_position': row['ViewPosition'],
        }

    def _get_classification_item(self, row, image):
        """Return data for multi-label classification task"""
        return {
            'image': image,
            'label': torch.tensor(self.disease_matrix[row.name], dtype=torch.float32),  # Multi-label vector
            'subject_id': int(row['subject_id']),
            'dicom_id': row['dicom_id'],
            'study_id': row['study_id'],
            'view_position': row['ViewPosition'],
        }

    def _get_ue_item(self, row, image):
        """
        Return a unified UE sample structure, making it easy for UE algorithms to take all information at once.
        - Currently: provide image / targets.cls / targets.reid / meta
        - Reserve seg/det positions (None) for future use
        """
        # cls targets (if available)
        cls_target = None
        if self._has_cls and self.disease_matrix is not None:
            cls_target = torch.tensor(self.disease_matrix[row.name], dtype=torch.float32)

        # reid target (always available)
        reid_target = int(row['reid_label'])

        sample = {
            'image': image,
            'targets': {
                'cls': cls_target,     # Tensor[num_cls] or None
                'reid': reid_target,   # int
                # 'seg_mask': None,      # Reserved: future use HxW mask
                # 'det': None,           # Reserved: future use {boxes, labels}
            },
            'meta': {
                'subject_id': int(row['subject_id']),
                'dicom_id': row['dicom_id'],
                'study_id': row['study_id'],
                'view_position': row['ViewPosition'],
                'image_path': row['image_path'],
            }
        }
        return sample

    
    def get_class_info(self):
        """Get class information for the current task"""
        if self.task_type == 'reid':
            return {
                'num_classes': self.num_classes,
                'class_names': [f"Subject_{self.label_to_subject[i]}" for i in range(self.num_classes)],
                'task_type': 'reid'
            }
        elif self.task_type == 'classification':
            return {
                'num_classes': self.num_classes,
                'class_names': self.disease_labels,
                'task_type': 'multi_label_classification'
            }


class MIMICCXRReIDDataset(MIMICCXRDataset):
    """MIMIC-CXR Dataset for ReID Task"""
    
    def __init__(self, *args, **kwargs):
        kwargs['task_type'] = 'reid'
        super().__init__(*args, **kwargs)
    
    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        assert kind == "reid"
        return torch.as_tensor(self.df["reid_label"].values, dtype=torch.long)



class MIMICCXRClassificationDataset(MIMICCXRDataset):
    """MIMIC-CXR Dataset for Multi-label Classification Task"""
    
    def __init__(self, *args, **kwargs):
        kwargs['task_type'] = 'classification'
        super().__init__(*args, **kwargs)

    def labels_for_sampling(self, kind: str = "cls") -> torch.Tensor:
        assert kind == "cls"
        raise NotImplementedError("Multi-label classification labels for sampling not implemented.")

class MIMICCXRUEDataset(MIMICCXRDataset):
    """MIMIC-CXR Dataset for UE Task"""
    
    def __init__(self, *args, **kwargs):
        kwargs['task_type'] = 'ue'
        super().__init__(*args, **kwargs)
    
    def labels_for_sampling(self, kind: str = "reid") -> torch.Tensor:
        assert kind == "reid"
        return torch.as_tensor(self.df["reid_label"].values, dtype=torch.long)