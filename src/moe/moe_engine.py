"""
MoE-Enhanced Adaptrix Engine.

This module extends the base Adaptrix engine with automatic adapter selection
using the trained task classifier.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..core.modular_engine import ModularAdaptrixEngine
from .classifier import TaskClassifier
from ..rag.vector_store import FAISSVectorStore
from ..rag.document_processor import DocumentProcessor
from ..rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


class MoEAdaptrixEngine(ModularAdaptrixEngine):
    """
    MoE-enhanced Adaptrix Engine with automatic adapter selection.
    
    This engine automatically selects the most appropriate LoRA adapter
    for each input using a trained task classifier.
    """
    
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        adapters_dir: str = "adapters",
        classifier_path: str = "models/classifier",
        enable_auto_selection: bool = True,
        rag_vector_store_path: Optional[str] = None,
        enable_rag: bool = False
    ):
        """
        Initialize MoE Adaptrix Engine.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on
            adapters_dir: Directory containing LoRA adapters
            classifier_path: Path to trained task classifier
            enable_auto_selection: Whether to enable automatic adapter selection
            rag_vector_store_path: Path to RAG vector store
            enable_rag: Whether to enable RAG functionality
        """
        super().__init__(model_id, device, adapters_dir)
        
        self.classifier_path = classifier_path
        self.enable_auto_selection = enable_auto_selection
        self.rag_vector_store_path = rag_vector_store_path
        self.enable_rag = enable_rag

        # MoE components
        self.task_classifier: Optional[TaskClassifier] = None
        self.classifier_initialized = False

        # RAG components
        self.vector_store: Optional[FAISSVectorStore] = None
        self.retriever: Optional[DocumentRetriever] = None
        self.rag_initialized = False
        
        # Statistics
        self.selection_stats = {
            "total_selections": 0,
            "adapter_usage": {},
            "confidence_scores": []
        }
        
        logger.info(f"Initialized MoE Adaptrix Engine with auto-selection: {enable_auto_selection}, RAG: {enable_rag}")
    
    def initialize(self) -> bool:
        """Initialize the MoE engine with base model and classifier."""
        try:
            # Initialize base engine first
            if not super().initialize():
                logger.error("Failed to initialize base engine")
                return False
            
            # Initialize task classifier if auto-selection is enabled
            if self.enable_auto_selection:
                if not self._initialize_classifier():
                    logger.warning("Failed to initialize classifier, auto-selection disabled")
                    self.enable_auto_selection = False

            # Initialize RAG system if enabled
            if self.enable_rag:
                if not self._initialize_rag():
                    logger.warning("Failed to initialize RAG system, RAG disabled")
                    self.enable_rag = False
            
            logger.info("✅ MoE Adaptrix Engine initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"MoE engine initialization failed: {e}")
            return False
    
    def _initialize_classifier(self) -> bool:
        """Initialize the task classifier."""
        try:
            classifier_path = Path(self.classifier_path)
            
            if not classifier_path.exists():
                logger.warning(f"Classifier not found at {classifier_path}")
                return False
            
            logger.info(f"🧠 Loading task classifier from {classifier_path}")
            
            self.task_classifier = TaskClassifier()
            
            if not self.task_classifier.load(str(classifier_path)):
                logger.error("Failed to load task classifier")
                return False
            
            self.classifier_initialized = True
            logger.info("✅ Task classifier loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Classifier initialization failed: {e}")
            return False

    def _initialize_rag(self) -> bool:
        """Initialize the RAG system."""
        try:
            if not self.rag_vector_store_path:
                logger.warning("No RAG vector store path provided")
                return False

            vector_store_path = Path(self.rag_vector_store_path)

            if not vector_store_path.exists():
                logger.warning(f"RAG vector store not found at {vector_store_path}")
                return False

            logger.info(f"🔍 Loading RAG vector store from {vector_store_path}")

            # Initialize vector store
            self.vector_store = FAISSVectorStore()

            if not self.vector_store.load(str(vector_store_path)):
                logger.error("Failed to load RAG vector store")
                return False

            # Initialize retriever
            self.retriever = DocumentRetriever(
                vector_store=self.vector_store,
                top_k=5,
                score_threshold=0.0,
                rerank=True
            )

            self.rag_initialized = True
            logger.info("✅ RAG system loaded successfully")

            return True

        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_length: int = None,
        task_type: str = "auto",
        use_context: bool = None,
        adapter_name: str = None,
        use_rag: bool = None,
        rag_top_k: int = 3,
        **kwargs
    ) -> str:
        """
        Generate text with automatic adapter selection and RAG.

        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            task_type: Task type ('auto' for automatic selection)
            use_context: Whether to use conversation context
            adapter_name: Specific adapter to use (overrides auto-selection)
            use_rag: Whether to use RAG (overrides default)
            rag_top_k: Number of documents to retrieve for RAG
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._initialized:
            return "Error: Engine not initialized"
        
        try:
            # Determine RAG usage
            should_use_rag = use_rag if use_rag is not None else self.enable_rag

            # Retrieve relevant documents if RAG is enabled
            rag_context = ""
            rag_info = {}

            if should_use_rag and self.rag_initialized:
                try:
                    retrieval_results = self.retriever.retrieve(
                        prompt,
                        top_k=rag_top_k,
                        score_threshold=0.0
                    )

                    if retrieval_results:
                        rag_context = self.retriever.create_context(
                            retrieval_results,
                            max_context_length=1500,
                            include_metadata=False
                        )

                        rag_info = {
                            "enabled": True,
                            "documents_retrieved": len(retrieval_results),
                            "context_length": len(rag_context),
                            "top_scores": [r.score for r in retrieval_results[:3]]
                        }

                        logger.debug(f"Retrieved {len(retrieval_results)} documents for RAG")
                    else:
                        rag_info = {"enabled": True, "documents_retrieved": 0}

                except Exception as e:
                    logger.error(f"RAG retrieval failed: {e}")
                    rag_info = {"enabled": True, "error": str(e)}
            else:
                rag_info = {"enabled": False}

            # Determine which adapter to use
            selected_adapter = None
            selection_info = {}

            if adapter_name:
                # Use explicitly specified adapter
                selected_adapter = adapter_name
                selection_info = {
                    "method": "explicit",
                    "adapter": selected_adapter,
                    "confidence": 1.0
                }
            elif task_type == "auto" and self.enable_auto_selection and self.classifier_initialized:
                # Use automatic selection
                selection_result = self._select_adapter(prompt)
                selected_adapter = selection_result["adapter_name"]
                selection_info = {
                    "method": "automatic",
                    "adapter": selected_adapter,
                    "confidence": selection_result["confidence"],
                    "all_probabilities": selection_result.get("probabilities", {})
                }
            
            # Load the selected adapter if available
            if selected_adapter and selected_adapter in self.list_adapters():
                if not self.switch_adapter(selected_adapter):
                    logger.warning(f"Failed to load adapter {selected_adapter}, using base model")
                    selected_adapter = None
            elif selected_adapter:
                logger.warning(f"Adapter {selected_adapter} not available, using base model")
                selected_adapter = None
            
            # Update statistics
            self._update_selection_stats(selection_info)

            # Construct final prompt with RAG context
            final_prompt = prompt
            if rag_context:
                final_prompt = f"Context:\n{rag_context}\n\nQuestion: {prompt}"

            # Generate response using base engine
            response = super().generate(
                prompt=final_prompt,
                max_length=max_length,
                task_type=task_type if task_type != "auto" else "general",
                use_context=use_context,
                **kwargs
            )
            
            # Add selection and RAG info to response metadata (for debugging)
            if hasattr(self, '_last_generation_info'):
                self._last_generation_info = {
                    "selection": selection_info,
                    "rag": rag_info,
                    "final_prompt_length": len(final_prompt)
                }

            logger.debug(f"Generated response using adapter: {selected_adapter or 'base_model'}, RAG: {rag_info.get('enabled', False)}")

            return response
            
        except Exception as e:
            logger.error(f"MoE generation failed: {e}")
            return f"Error: Generation failed - {str(e)}"
    
    def _select_adapter(self, prompt: str) -> Dict[str, Any]:
        """Select the best adapter for the given prompt."""
        try:
            if not self.task_classifier:
                return {"adapter_name": None, "confidence": 0.0}
            
            # Get prediction from classifier
            prediction = self.task_classifier.predict(
                prompt, 
                return_probabilities=True
            )
            
            logger.debug(f"Adapter selection: {prediction['adapter_name']} (confidence: {prediction['confidence']:.3f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Adapter selection failed: {e}")
            return {"adapter_name": None, "confidence": 0.0}
    
    def _update_selection_stats(self, selection_info: Dict[str, Any]):
        """Update selection statistics."""
        try:
            self.selection_stats["total_selections"] += 1
            
            adapter = selection_info.get("adapter")
            if adapter:
                if adapter not in self.selection_stats["adapter_usage"]:
                    self.selection_stats["adapter_usage"][adapter] = 0
                self.selection_stats["adapter_usage"][adapter] += 1
            
            confidence = selection_info.get("confidence", 0.0)
            self.selection_stats["confidence_scores"].append(confidence)
            
        except Exception as e:
            logger.error(f"Failed to update selection stats: {e}")
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get adapter selection statistics."""
        stats = self.selection_stats.copy()
        
        # Calculate average confidence
        if stats["confidence_scores"]:
            stats["average_confidence"] = sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
        else:
            stats["average_confidence"] = 0.0
        
        # Calculate usage percentages
        total = stats["total_selections"]
        if total > 0:
            stats["adapter_usage_percent"] = {
                adapter: (count / total) * 100
                for adapter, count in stats["adapter_usage"].items()
            }
        else:
            stats["adapter_usage_percent"] = {}
        
        return stats
    
    def get_moe_status(self) -> Dict[str, Any]:
        """Get comprehensive MoE system status."""
        base_status = self.get_system_status()
        
        moe_status = {
            "classifier_initialized": self.classifier_initialized,
            "auto_selection_enabled": self.enable_auto_selection,
            "classifier_path": self.classifier_path,
            "selection_stats": self.get_selection_stats(),
            "rag_initialized": self.rag_initialized,
            "rag_enabled": self.enable_rag,
            "rag_vector_store_path": self.rag_vector_store_path
        }

        if self.task_classifier:
            moe_status["classifier_status"] = self.task_classifier.get_status()

        if self.vector_store:
            moe_status["vector_store_stats"] = self.vector_store.get_stats()

        if self.retriever:
            moe_status["retrieval_stats"] = self.retriever.get_retrieval_stats()

        base_status["moe"] = moe_status
        return base_status
    
    def predict_adapter(self, prompt: str) -> Dict[str, Any]:
        """
        Predict the best adapter for a prompt without generating text.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Prediction results
        """
        if not self.classifier_initialized:
            return {"error": "Classifier not initialized"}
        
        return self._select_adapter(prompt)

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query without generating text.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieval results
        """
        if not self.rag_initialized:
            return []

        try:
            results = self.retriever.retrieve(query, top_k=top_k)
            return [result.to_dict() for result in results]
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def enable_auto_selection(self, enable: bool = True):
        """Enable or disable automatic adapter selection."""
        if enable and not self.classifier_initialized:
            logger.warning("Cannot enable auto-selection: classifier not initialized")
            return False
        
        self.enable_auto_selection = enable
        logger.info(f"Auto-selection {'enabled' if enable else 'disabled'}")
        return True
    
    def reset_selection_stats(self):
        """Reset adapter selection statistics."""
        self.selection_stats = {
            "total_selections": 0,
            "adapter_usage": {},
            "confidence_scores": []
        }
        logger.info("Selection statistics reset")

    def enable_rag_system(self, enable: bool = True):
        """Enable or disable RAG system."""
        if enable and not self.rag_initialized:
            logger.warning("Cannot enable RAG: system not initialized")
            return False

        self.enable_rag = enable
        logger.info(f"RAG system {'enabled' if enable else 'disabled'}")
        return True
    
    def cleanup(self):
        """Clean up MoE engine resources."""
        try:
            if self.task_classifier:
                # Task classifier doesn't need explicit cleanup
                self.task_classifier = None

            # Clean up RAG components
            if self.retriever:
                self.retriever = None

            if self.vector_store:
                self.vector_store = None

            # Clean up base engine
            super().cleanup()

            logger.info("✅ MoE engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"MoE engine cleanup failed: {e}")
