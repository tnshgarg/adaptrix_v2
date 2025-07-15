"""
MoE Task Classifier for Adaptrix.

This module implements a task classifier that automatically selects the most
relevant LoRA adapter for each input query, enabling sparse MoE activation.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class TaskClassifier:
    """
    Task classifier for automatic adapter selection.
    
    Uses sentence-transformers for encoding and a lightweight classifier
    for predicting the most appropriate adapter for each input.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        classifier_type: str = "logistic",
        model_path: Optional[str] = None
    ):
        """
        Initialize the task classifier.
        
        Args:
            embedding_model: Sentence transformer model name
            classifier_type: Type of classifier ('logistic', 'neural')
            model_path: Path to saved model (if loading existing)
        """
        self.embedding_model_name = embedding_model
        self.classifier_type = classifier_type
        self.model_path = model_path
        
        # Components
        self.encoder = None
        self.classifier = None
        self.label_encoder = None
        self.adapter_mapping = {}
        
        # Training data
        self.training_data = []
        self.training_labels = []
        
        # Model state
        self.is_trained = False
        self.is_loaded = False
        
    def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            logger.info(f"🚀 Initializing task classifier with {self.embedding_model_name}")
            
            # Load sentence transformer
            self.encoder = SentenceTransformer(self.embedding_model_name)
            
            # Initialize label encoder
            self.label_encoder = LabelEncoder()
            
            logger.info("✅ Task classifier initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize task classifier: {e}")
            return False
    
    def add_training_data(self, texts: List[str], adapter_names: List[str]):
        """
        Add training data for the classifier.
        
        Args:
            texts: List of input texts
            adapter_names: Corresponding adapter names for each text
        """
        if len(texts) != len(adapter_names):
            raise ValueError("Texts and adapter names must have the same length")
        
        self.training_data.extend(texts)
        self.training_labels.extend(adapter_names)
        
        # Update adapter mapping
        for adapter_name in adapter_names:
            if adapter_name not in self.adapter_mapping:
                self.adapter_mapping[adapter_name] = len(self.adapter_mapping)
        
        logger.info(f"Added {len(texts)} training samples. Total: {len(self.training_data)}")
    
    def train(self, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the task classifier.
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training metrics and results
        """
        if not self.encoder:
            raise RuntimeError("Classifier not initialized. Call initialize() first.")
        
        if len(self.training_data) == 0:
            raise RuntimeError("No training data available. Call add_training_data() first.")
        
        logger.info(f"🏋️ Training task classifier on {len(self.training_data)} samples")
        
        try:
            # Encode training texts
            logger.info("   📝 Encoding training texts...")
            embeddings = self.encoder.encode(self.training_data, show_progress_bar=True)
            
            # Encode labels
            encoded_labels = self.label_encoder.fit_transform(self.training_labels)
            
            # Split data
            split_idx = int(len(embeddings) * (1 - validation_split))
            
            X_train = embeddings[:split_idx]
            y_train = encoded_labels[:split_idx]
            X_val = embeddings[split_idx:]
            y_val = encoded_labels[split_idx:]
            
            # Train classifier
            logger.info(f"   🧠 Training {self.classifier_type} classifier...")
            
            if self.classifier_type == "logistic":
                self.classifier = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    multi_class='ovr'
                )
            else:
                raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
            
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            train_pred = self.classifier.predict(X_train)
            val_pred = self.classifier.predict(X_val)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            # Get class names for report
            class_names = self.label_encoder.classes_
            val_report = classification_report(
                y_val, val_pred, 
                target_names=class_names,
                output_dict=True
            )
            
            self.is_trained = True
            
            results = {
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "num_classes": len(class_names),
                "class_names": class_names.tolist(),
                "classification_report": val_report,
                "training_samples": len(X_train),
                "validation_samples": len(X_val)
            }
            
            logger.info(f"✅ Training completed!")
            logger.info(f"   📊 Train Accuracy: {train_acc:.3f}")
            logger.info(f"   📊 Validation Accuracy: {val_acc:.3f}")
            logger.info(f"   📊 Classes: {class_names.tolist()}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def predict(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Predict the best adapter for a given text.
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results with adapter name and confidence
        """
        if not self.is_trained and not self.is_loaded:
            raise RuntimeError("Classifier not trained or loaded. Train or load a model first.")
        
        try:
            # Encode input text
            embedding = self.encoder.encode([text])
            
            # Predict
            prediction = self.classifier.predict(embedding)[0]
            probabilities = self.classifier.predict_proba(embedding)[0]
            
            # Decode prediction
            adapter_name = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(np.max(probabilities))
            
            result = {
                "adapter_name": adapter_name,
                "confidence": confidence,
                "prediction_class": int(prediction)
            }
            
            if return_probabilities:
                class_names = self.label_encoder.classes_
                result["probabilities"] = {
                    class_name: float(prob) 
                    for class_name, prob in zip(class_names, probabilities)
                }
            
            logger.debug(f"Predicted adapter '{adapter_name}' with confidence {confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def save(self, save_path: str):
        """Save the trained classifier to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained classifier")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save classifier
            classifier_path = save_path / "classifier.pkl"
            with open(classifier_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            
            # Save label encoder
            encoder_path = save_path / "label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Save metadata
            metadata = {
                "embedding_model": self.embedding_model_name,
                "classifier_type": self.classifier_type,
                "adapter_mapping": self.adapter_mapping,
                "class_names": self.label_encoder.classes_.tolist(),
                "num_classes": len(self.label_encoder.classes_),
                "is_trained": self.is_trained
            }
            
            metadata_path = save_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Classifier saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save classifier: {e}")
            raise
    
    def load(self, load_path: str) -> bool:
        """Load a trained classifier from disk."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            logger.error(f"Model path does not exist: {load_path}")
            return False
        
        try:
            # Load metadata
            metadata_path = load_path / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Initialize encoder if needed
            if not self.encoder:
                self.embedding_model_name = metadata["embedding_model"]
                if not self.initialize():
                    return False
            
            # Load classifier
            classifier_path = load_path / "classifier.pkl"
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            
            # Load label encoder
            encoder_path = load_path / "label_encoder.pkl"
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Restore metadata
            self.classifier_type = metadata["classifier_type"]
            self.adapter_mapping = metadata["adapter_mapping"]
            self.is_loaded = True
            
            logger.info(f"✅ Classifier loaded from {load_path}")
            logger.info(f"   📊 Classes: {metadata['class_names']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get classifier status information."""
        return {
            "initialized": self.encoder is not None,
            "trained": self.is_trained,
            "loaded": self.is_loaded,
            "embedding_model": self.embedding_model_name,
            "classifier_type": self.classifier_type,
            "num_classes": len(self.adapter_mapping) if self.adapter_mapping else 0,
            "adapter_mapping": self.adapter_mapping,
            "training_samples": len(self.training_data)
        }
