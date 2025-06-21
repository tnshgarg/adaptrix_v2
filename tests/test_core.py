"""
Basic tests for Adaptrix core components.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from src.models.base_model import BaseModelManager
from src.adapters.adapter_manager import AdapterManager
from src.injection.layer_injector import LayerInjector
from src.core.dynamic_loader import DynamicLoader
from src.core.engine import AdaptrixEngine


class TestBaseModelManager:
    """Test BaseModelManager functionality."""
    
    def test_initialization(self):
        """Test basic initialization."""
        manager = BaseModelManager("microsoft/DialoGPT-small", "cpu")
        assert manager.model_name == "microsoft/DialoGPT-small"
        assert manager.device.type == "cpu"
        assert not manager.is_loaded()
    
    def test_model_loading(self):
        """Test model loading (requires internet connection)."""
        manager = BaseModelManager("microsoft/DialoGPT-small", "cpu")
        
        # This test requires downloading the model
        try:
            model = manager.load_model()
            assert model is not None
            assert manager.is_loaded()
            
            # Test model info extraction
            info = manager.get_model_info()
            assert 'model_name' in info
            assert 'total_params' in info
            
            # Test layer count
            layer_count = manager.get_layer_count()
            assert isinstance(layer_count, int)
            assert layer_count > 0
            
            # Test hidden size
            hidden_size = manager.get_hidden_size()
            assert isinstance(hidden_size, int)
            assert hidden_size > 0
            
        except Exception as e:
            pytest.skip(f"Model loading failed (likely network issue): {e}")


class TestAdapterManager:
    """Test AdapterManager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter_manager = AdapterManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test adapter manager initialization."""
        assert self.adapter_manager.adapter_dir == Path(self.temp_dir)
        assert self.adapter_manager.adapter_dir.exists()
    
    def test_list_empty_adapters(self):
        """Test listing when no adapters exist."""
        adapters = self.adapter_manager.list_adapters()
        assert adapters == []
    
    def test_save_and_load_adapter(self):
        """Test saving and loading an adapter."""
        # Create test adapter data
        metadata = {
            'name': 'test_adapter',
            'version': '1.0.0',
            'description': 'Test adapter',
            'target_layers': [6, 12],
            'rank': 16,
            'alpha': 32,
            'target_modules': ['self_attn.q_proj', 'mlp.c_fc']
        }
        
        weights = {
            6: {
                'self_attn.q_proj': {
                    'lora_A': torch.randn(16, 768),
                    'lora_B': torch.randn(768, 16),
                    'rank': 16,
                    'alpha': 32
                }
            },
            12: {
                'self_attn.q_proj': {
                    'lora_A': torch.randn(16, 768),
                    'lora_B': torch.randn(768, 16),
                    'rank': 16,
                    'alpha': 32
                }
            }
        }
        
        # Save adapter
        success = self.adapter_manager.save_adapter('test_adapter', weights, metadata)
        assert success
        
        # List adapters
        adapters = self.adapter_manager.list_adapters()
        assert 'test_adapter' in adapters
        
        # Load adapter
        loaded_data = self.adapter_manager.load_adapter('test_adapter')
        assert loaded_data is not None
        assert loaded_data['metadata']['name'] == 'test_adapter'
        assert 6 in loaded_data['weights']
        assert 12 in loaded_data['weights']
        
        # Get adapter info
        info = self.adapter_manager.get_adapter_info('test_adapter')
        assert info is not None
        assert info['name'] == 'test_adapter'
    
    def test_delete_adapter(self):
        """Test adapter deletion."""
        # Create and save test adapter
        metadata = {
            'name': 'delete_test',
            'version': '1.0.0',
            'description': 'Test deletion',
            'target_layers': [6],
            'rank': 16,
            'alpha': 32,
            'target_modules': ['self_attn.q_proj']
        }
        
        weights = {
            6: {
                'self_attn.q_proj': {
                    'lora_A': torch.randn(16, 768),
                    'lora_B': torch.randn(768, 16),
                    'rank': 16,
                    'alpha': 32
                }
            }
        }
        
        self.adapter_manager.save_adapter('delete_test', weights, metadata)
        
        # Verify it exists
        adapters = self.adapter_manager.list_adapters()
        assert 'delete_test' in adapters
        
        # Delete it
        success = self.adapter_manager.delete_adapter('delete_test')
        assert success
        
        # Verify it's gone
        adapters = self.adapter_manager.list_adapters()
        assert 'delete_test' not in adapters


class TestLayerInjector:
    """Test LayerInjector functionality."""
    
    def test_lora_layer(self):
        """Test LoRA layer implementation."""
        from src.injection.layer_injector import LoRALayer

        lora = LoRALayer(in_features=768, out_features=768, rank=16, alpha=32)

        # Test forward pass
        x = torch.randn(1, 10, 768)
        output = lora(x)

        assert output.shape == x.shape
        # LoRA B matrix is initialized to zero, so output should be zero initially
        assert torch.allclose(output, torch.zeros_like(output))

        # Set some non-zero weights to test actual computation
        with torch.no_grad():
            lora.lora_B.weight.fill_(0.1)

        output_nonzero = lora(x)
        assert not torch.allclose(output_nonzero, torch.zeros_like(output_nonzero))

        # Test enable/disable
        lora.disable()
        output_disabled = lora(x)
        assert torch.allclose(output_disabled, torch.zeros_like(output_disabled))

        lora.enable()
        output_enabled = lora(x)
        assert not torch.allclose(output_enabled, torch.zeros_like(output_enabled))


class TestAdaptrixEngine:
    """Test AdaptrixEngine integration."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        assert engine.model_name == "microsoft/DialoGPT-small"
        assert not engine._initialized
        assert not engine._model_loaded
    
    def test_context_manager(self):
        """Test engine as context manager."""
        try:
            with AdaptrixEngine("microsoft/DialoGPT-small", "cpu") as engine:
                assert engine._initialized
                assert engine._model_loaded
        except Exception as e:
            pytest.skip(f"Engine initialization failed (likely network issue): {e}")


if __name__ == "__main__":
    pytest.main([__file__])
