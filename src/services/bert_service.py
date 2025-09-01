import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import re
import unicodedata
from functools import lru_cache
import hashlib
import pickle
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError, AIServiceError
from src.services.portuguese_preprocessor import PortugueseTextPreprocessor

logger = get_logger(__name__)


class BERTTextClassifier:
    """
    BERT-based text classifier for Portuguese financial transaction categorization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize BERT classifier with configuration

        Args:
            config: Configuration dictionary for BERT model
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.preprocessor = PortugueseTextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model state
        self.is_trained = False
        self.model_path = self.config['model_path']
        self.cache_dir = self.config['cache_dir']

        # Performance tracking
        self.training_history = []
        self.best_score = 0.0

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)

        self.logger.info(f"BERT classifier initialized. Device: {self.device}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for BERT classifier"""
        return {
            'model_name': 'neuralmind/bert-base-portuguese-cased',
            'max_length': 128,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 5,
            'weight_decay': 0.01,
            'warmup_steps': 500,
            'save_steps': 500,
            'eval_steps': 500,
            'early_stopping_patience': 3,
            'model_path': 'models/bert_classifier',
            'cache_dir': 'cache/bert_cache',
            'use_cache': True,
            'cache_max_size': 1000,
            'preprocessing': {
                'lowercase': True,
                'remove_accents': True,
                'remove_numbers': False,
                'remove_punctuation': True,
                'stemming': False,
                'stopwords': True
            }
        }

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load pre-trained BERT model and tokenizer

        Args:
            model_path: Path to load the model from (optional)

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            load_path = model_path or self.model_path

            if os.path.exists(os.path.join(load_path, 'pytorch_model.bin')) or \
               os.path.exists(os.path.join(load_path, 'model.safetensors')):

                self.logger.info(f"Loading fine-tuned model from {load_path}")
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(load_path)

                    # Load label encoder
                    label_encoder_path = os.path.join(load_path, 'label_encoder.pkl')
                    if os.path.exists(label_encoder_path):
                        with open(label_encoder_path, 'rb') as f:
                            self.label_encoder = pickle.load(f)

                    self.is_trained = True
                    self.logger.info("Fine-tuned model loaded successfully")

                except Exception as load_error:
                    self.logger.warning(f"Error loading saved model: {str(load_error)}. Loading base model instead.")
                    # Clear the saved model and load base model
                    import shutil
                    if os.path.exists(load_path):
                        shutil.rmtree(load_path)
                    self._load_base_model()
            else:
                self._load_base_model()

            self.model.to(self.device)
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

    def _load_base_model(self):
        """Load the base BERT model"""
        self.logger.info(f"Loading base model: {self.config['model_name']}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=12  # Default number of categories
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Base model loaded successfully")

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts using Portuguese text preprocessor

        Args:
            texts: List of raw text strings

        Returns:
            List of preprocessed text strings
        """
        try:
            preprocessed_texts = []

            for text in texts:
                # Use Portuguese preprocessor
                processed = self.preprocessor.preprocess(text, self.config['preprocessing'])
                preprocessed_texts.append(processed)

            return preprocessed_texts

        except Exception as e:
            self.logger.error(f"Error preprocessing texts: {str(e)}")
            return texts  # Return original texts as fallback

    def prepare_dataset(self, texts: List[str], labels: Optional[List[str]] = None) -> Dataset:
        """
        Prepare dataset for training/inference

        Args:
            texts: List of text strings
            labels: List of label strings (optional for inference)

        Returns:
            HuggingFace Dataset object
        """
        try:
            # Preprocess texts
            processed_texts = self.preprocess_texts(texts)

            # Create dataset dictionary
            dataset_dict = {'text': processed_texts}

            if labels is not None:
                # Ensure labels is a list
                if isinstance(labels, (list, tuple)):
                    labels_list = list(labels)
                else:
                    labels_list = [labels]

                # Encode labels if label encoder exists
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                    encoded_labels = self.label_encoder.fit_transform(labels_list)
                else:
                    encoded_labels = self.label_encoder.transform(labels_list)

                # Ensure labels are 1D
                if len(encoded_labels.shape) > 1:
                    encoded_labels = encoded_labels.flatten()

                dataset_dict['label'] = encoded_labels

            # Create HuggingFace dataset
            dataset = Dataset.from_dict(dataset_dict)

            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['max_length']
                )

            tokenized_dataset = dataset.map(tokenize_function, batched=True)

            # Remove text column and set format for PyTorch
            tokenized_dataset = tokenized_dataset.remove_columns(['text'])
            tokenized_dataset.set_format('torch')

            return tokenized_dataset

        except Exception as e:
            self.logger.error(f"Error preparing dataset: {str(e)}")
            raise ValidationError(f"Failed to prepare dataset: {str(e)}")

    def train(self, train_texts: List[str], train_labels: List[str],
              val_texts: Optional[List[str]] = None, val_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train BERT model on the provided data

        Args:
            train_texts: Training text data
            train_labels: Training labels
            val_texts: Validation text data (optional)
            val_labels: Validation labels (optional)

        Returns:
            Dictionary with training results
        """
        try:
            self.logger.info("Starting BERT model training")

            # Load base model if not already loaded
            if self.model is None:
                if not self.load_model():
                    raise AIServiceError("Failed to load base model")

            # Prepare training dataset
            # Fit label encoder on all available labels to handle unseen labels in validation
            all_labels = train_labels.copy()
            if val_labels:
                all_labels.extend(val_labels)

            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(all_labels)

            num_labels = len(self.label_encoder.classes_)
            self.model.config.num_labels = num_labels

            train_dataset = self.prepare_dataset(train_texts, train_labels)

            # Prepare validation dataset if provided
            val_dataset = None
            if val_texts and val_labels:
                val_dataset = self.prepare_dataset(val_texts, val_labels)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.model_path,
                num_train_epochs=self.config['num_epochs'],
                per_device_train_batch_size=self.config['batch_size'],
                per_device_eval_batch_size=self.config['batch_size'],
                learning_rate=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                warmup_steps=self.config['warmup_steps'],
                logging_dir=os.path.join(self.model_path, 'logs'),
                logging_steps=100,
                save_steps=self.config['save_steps'],
                eval_steps=self.config['eval_steps'],
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                greater_is_better=True,
                eval_strategy='steps' if val_dataset else 'no',
                save_strategy='steps' if val_dataset else 'no',
                report_to=[]  # Disable wandb/tensorboard logging
            )

            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # Define compute metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)

                accuracy = accuracy_score(labels, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='weighted', zero_division=0
                )

                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # Create trainer
            callbacks = []
            if val_dataset:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.config['early_stopping_patience']))

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks
            )

            # Train the model
            self.logger.info("Starting model training...")
            train_result = trainer.train()

            # Save the model
            trainer.save_model(self.model_path)

            # Save label encoder
            label_encoder_path = os.path.join(self.model_path, 'label_encoder.pkl')
            with open(label_encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)

            # Get training metrics
            metrics = train_result.metrics

            # Evaluate on validation set if available
            eval_metrics = {}
            if val_dataset:
                eval_results = trainer.evaluate()
                eval_metrics = eval_results

            # Update model state
            self.is_trained = True
            self.best_score = eval_metrics.get('eval_f1', metrics.get('train_loss', 0))

            # Store training history
            training_info = {
                'timestamp': datetime.now().isoformat(),
                'train_samples': len(train_texts),
                'val_samples': len(val_texts) if val_texts else 0,
                'num_labels': num_labels,
                'training_metrics': metrics,
                'evaluation_metrics': eval_metrics,
                'best_score': self.best_score
            }
            self.training_history.append(training_info)

            result = {
                'success': True,
                'message': 'BERT model trained successfully',
                'training_samples': len(train_texts),
                'validation_samples': len(val_texts) if val_texts else 0,
                'num_labels': num_labels,
                'training_metrics': metrics,
                'evaluation_metrics': eval_metrics,
                'best_score': self.best_score
            }

            self.logger.info(f"BERT training completed. Best F1: {self.best_score:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"Error training BERT model: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'BERT model training failed'
            }

    @lru_cache(maxsize=1000)
    def _cached_predict(self, text_hash: str, text: str) -> str:
        """
        Cached prediction for efficient inference

        Args:
            text_hash: Hash of the text for caching
            text: Original text

        Returns:
            Predicted category
        """
        return self._predict_single(text)

    def _predict_single(self, text: str) -> str:
        """
        Predict category for a single text

        Args:
            text: Input text

        Returns:
            Predicted category
        """
        try:
            if not self.is_trained or self.model is None:
                raise AIServiceError("Model not trained or loaded")

            # Prepare input
            dataset = self.prepare_dataset([text])

            # Create data loader
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer)
            )

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                for batch in data_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    predicted_class = predictions.item()
                    break

            # Decode prediction
            if self.label_encoder:
                predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            else:
                predicted_label = f"class_{predicted_class}"

            return predicted_label

        except Exception as e:
            self.logger.error(f"Error in single prediction: {str(e)}")
            return 'outros'  # Fallback category

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict categories for a list of texts

        Args:
            texts: List of input texts

        Returns:
            List of predicted categories
        """
        try:
            if not self.is_trained or self.model is None:
                raise AIServiceError("Model not trained or loaded")

            predictions = []

            for text in texts:
                if self.config['use_cache']:
                    # Create hash for caching
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    prediction = self._cached_predict(text_hash, text)
                else:
                    prediction = self._predict_single(text)

                predictions.append(prediction)

            return predictions

        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            return ['outros'] * len(texts)  # Fallback predictions

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for all classes

        Args:
            texts: List of input texts

        Returns:
            Array of prediction probabilities
        """
        try:
            if not self.is_trained or self.model is None:
                raise AIServiceError("Model not trained or loaded")

            # Prepare input
            dataset = self.prepare_dataset(texts)

            # Create data loader
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer)
            )

            # Make predictions
            self.model.eval()
            all_probabilities = []

            with torch.no_grad():
                for batch in data_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    all_probabilities.extend(probabilities.cpu().numpy())

            return np.array(all_probabilities)

        except Exception as e:
            self.logger.error(f"Error in probability prediction: {str(e)}")
            # Return uniform probabilities as fallback
            num_classes = len(self.label_encoder.classes_) if self.label_encoder else 12
            return np.full((len(texts), num_classes), 1.0 / num_classes)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        info = {
            'is_trained': self.is_trained,
            'model_name': self.config['model_name'],
            'device': str(self.device),
            'max_length': self.config['max_length'],
            'cache_enabled': self.config['use_cache'],
            'best_score': self.best_score
        }

        if self.label_encoder:
            info['num_classes'] = len(self.label_encoder.classes_)
            info['class_names'] = list(self.label_encoder.classes_)

        if self.training_history:
            info['last_training'] = self.training_history[-1]

        return info

    def save_model(self, save_path: Optional[str] = None) -> bool:
        """
        Save the trained model and components

        Args:
            save_path: Path to save the model (optional)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            save_path = save_path or self.model_path

            if self.model and self.is_trained:
                # Save model and tokenizer
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)

                # Save label encoder
                if self.label_encoder:
                    label_encoder_path = os.path.join(save_path, 'label_encoder.pkl')
                    with open(label_encoder_path, 'wb') as f:
                        pickle.dump(self.label_encoder, f)

                # Save configuration and training history
                config_path = os.path.join(save_path, 'bert_config.json')
                with open(config_path, 'w') as f:
                    json.dump({
                        'config': self.config,
                        'training_history': self.training_history,
                        'best_score': self.best_score
                    }, f, indent=2)

                self.logger.info(f"Model saved to {save_path}")
                return True
            else:
                self.logger.warning("No trained model to save")
                return False

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def clear_cache(self):
        """Clear prediction cache"""
        self._cached_predict.cache_clear()
        self.logger.info("Prediction cache cleared")