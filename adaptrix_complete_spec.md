# Adaptrix: Middle-Layer LoRA Injection System - Complete Technical Specification

## Project Overview

Adaptrix is a revolutionary AI system that enhances small language models (3B-13B parameters) by dynamically injecting specialized LoRA adapters into middle transformer layers rather than just the output layer. This creates a composable intelligence system where different reasoning capabilities can be loaded on-demand.

### Core Innovation

- **Middle-Layer Injection**: LoRA adapters inject into layers 6, 12, 18 (not just final layer)
- **Dynamic Switching**: Hot-swap adapters during inference without model reload
- **Composable Intelligence**: Multiple adapters can work together
- **Continuous Learning**: System improves through distillation and meta-learning

---

# WEEK 1 MVP SPECIFICATION

## Day 1-2: Core Infrastructure

### 1. Base Model Setup

**File**: `src/models/base_model.py`

**Requirements**:

- Load base model: `microsoft/DialoGPT-small` (for speed) or `microsoft/DialoGPT-medium`
- Initialize tokenizer with proper padding token
- Set model to evaluation mode by default
- Implement device management (CPU/GPU auto-detection)
- Add model metadata extraction (layer count, hidden size, etc.)

**Class Structure**:

```python
class BaseModelManager:
    def __init__(self, model_name: str, device: str = "auto")
    def load_model(self) -> transformers.PreTrainedModel
    def get_model_info(self) -> Dict[str, Any]
    def get_layer_count(self) -> int
    def get_hidden_size(self) -> int
```

### 2. Middle-Layer Injection Engine

**File**: `src/injection/layer_injector.py`

**Technical Details**:

- Use PyTorch forward hooks (`register_forward_hook`)
- Target specific transformer layers: `model.transformer.h[layer_idx]`
- Inject LoRA computation into attention and MLP layers
- Support multiple injection points per layer:
  - `self_attn.q_proj` (query projection)
  - `self_attn.k_proj` (key projection)
  - `self_attn.v_proj` (value projection)
  - `mlp.c_fc` (MLP input)
  - `mlp.c_proj` (MLP output)

**LoRA Mathematics**:

- Standard LoRA: `h = h + (W_A @ W_B) @ x * (alpha/rank)`
- Where W_A: (rank, input_dim), W_B: (output_dim, rank)
- Default rank=16, alpha=32 for balance of quality/speed

**Hook Implementation**:

```python
def create_injection_hook(layer_idx: int, adapter_name: str):
    def hook_fn(module, input, output):
        if adapter_name in active_adapters[layer_idx]:
            # Apply LoRA transformation
            pass
    return hook_fn
```

**Class Structure**:

```python
class LayerInjector:
    def __init__(self, base_model: nn.Module)
    def register_injection_point(self, layer_idx: int, module_name: str)
    def inject_adapter(self, layer_idx: int, adapter_name: str, adapter_weights: Dict)
    def remove_adapter(self, layer_idx: int, adapter_name: str)
    def clear_all_adapters(self)
    def get_active_adapters(self) -> Dict[int, List[str]]
```

### 3. Adapter Management System

**File**: `src/adapters/adapter_manager.py`

**Adapter Storage Structure**:

```
adapters/
├── math_reasoning/
│   ├── metadata.json
│   ├── layer_6.pt
│   ├── layer_12.pt
│   └── layer_18.pt
├── code_generation/
│   ├── metadata.json
│   ├── layer_6.pt
│   └── layer_18.pt
```

**Metadata Schema**:

```json
{
  "name": "math_reasoning",
  "version": "1.0.0",
  "description": "Mathematical problem solving",
  "target_layers": [6, 12, 18],
  "rank": 16,
  "alpha": 32,
  "target_modules": ["self_attn.q_proj", "mlp.c_fc"],
  "created_date": "2025-01-01",
  "performance_metrics": {
    "accuracy": 0.85,
    "latency_ms": 120
  }
}
```

**Class Structure**:

```python
class AdapterManager:
    def __init__(self, adapter_dir: str = "./adapters")
    def load_adapter(self, adapter_name: str) -> Dict
    def save_adapter(self, adapter_name: str, weights: Dict, metadata: Dict)
    def list_adapters(self) -> List[str]
    def get_adapter_metadata(self, adapter_name: str) -> Dict
    def delete_adapter(self, adapter_name: str)
    def validate_adapter(self, adapter_data: Dict) -> bool
```

### 4. Dynamic Loading System

**File**: `src/core/dynamic_loader.py`

**Memory Management**:

- LRU cache for adapter weights (max 3 adapters in memory)
- Lazy loading: only load when needed
- Background preloading based on usage patterns
- Memory monitoring and cleanup

**Hot-Swapping Logic**:

1. Detect adapter change request
2. Load new adapter weights (if not cached)
3. Remove old hooks
4. Install new hooks
5. Update active adapter registry
6. Verify injection success

**Class Structure**:

```python
class DynamicLoader:
    def __init__(self, injector: LayerInjector, adapter_manager: AdapterManager)
    def load_adapter(self, adapter_name: str, layer_indices: List[int] = None)
    def unload_adapter(self, adapter_name: str)
    def switch_adapter(self, old_name: str, new_name: str)
    def get_memory_usage(self) -> Dict
    def cleanup_unused_adapters(self)
    def preload_adapters(self, adapter_names: List[str])
```

### 5. CLI Interface

**File**: `src/cli/main.py`

**Commands**:

- `adaptrix load <adapter_name>` - Load an adapter
- `adaptrix unload <adapter_name>` - Unload an adapter
- `adaptrix list` - List all available adapters
- `adaptrix active` - Show currently active adapters
- `adaptrix query "text"` - Run inference with current adapters
- `adaptrix benchmark <adapter_name>` - Run performance tests
- `adaptrix status` - Show system status

**Implementation**:

- Use `click` library for CLI
- Support for configuration files
- Verbose/quiet modes
- JSON output option for programmatic use

## Day 3-4: Basic Adapters

### 6. Adapter Training Pipeline

**File**: `src/training/adapter_trainer.py`

**Training Process**:

1. Load base model
2. Add LoRA layers to target modules
3. Freeze base model weights
4. Train only LoRA parameters
5. Extract and save LoRA weights per layer
6. Generate metadata with performance metrics

**Dataset Integration**:

- **Math**: GSM8K dataset (grade school math problems)
- **Code**: Code-Alpaca dataset (programming instructions)
- **Reasoning**: ReClor dataset (logical reasoning)

**Training Configuration**:

```python
training_config = {
    "batch_size": 8,
    "learning_rate": 3e-4,
    "epochs": 3,
    "warmup_steps": 100,
    "max_length": 512,
    "lora_rank": 16,
    "lora_alpha": 32,
    "target_modules": ["self_attn.q_proj", "mlp.c_fc"]
}
```

### 7. Pre-trained Adapter Integration

**File**: `src/adapters/pretrained_loader.py`

**Supported Sources**:

- HuggingFace Hub adapters
- Local adapter files
- Alpaca-LoRA variants
- Custom trained adapters

**Conversion Logic**:

- Extract LoRA weights from standard PEFT format
- Redistribute weights to target layers
- Validate compatibility with base model
- Generate Adaptrix-compatible metadata

## Day 5-7: MVP Polish

### 8. Basic Routing System

**File**: `src/routing/keyword_router.py`

**Routing Logic**:

```python
routing_rules = {
    "math": ["calculate", "solve", "equation", "number", "arithmetic"],
    "code": ["python", "function", "algorithm", "programming", "debug"],
    "reasoning": ["analyze", "logic", "argument", "conclude", "reasoning"]
}
```

**Implementation**:

- Keyword matching with TF-IDF scoring
- Confidence thresholds for adapter selection
- Fallback to base model if no match
- Multi-adapter selection for complex queries

### 9. Performance Monitoring

**File**: `src/monitoring/performance_tracker.py`

**Metrics to Track**:

- Inference latency (per adapter)
- Memory usage (per adapter)
- GPU utilization
- Adapter loading/unloading time
- Query success rate
- User satisfaction scores

**Storage**:

- SQLite database for lightweight deployment
- JSON logs for detailed debugging
- CSV exports for analysis

### 10. Web Interface (Gradio)

**File**: `src/web/gradio_app.py`

**Interface Components**:

- Query input textbox
- Adapter selection dropdown
- Real-time response streaming
- Performance metrics display
- Adapter management panel
- System status dashboard

**Features**:

- Live adapter switching
- Side-by-side comparison mode
- Performance visualization
- Adapter marketplace preview

---

# COMPLETE PROJECT SPECIFICATION

## Phase 2: Intelligence Layer (Weeks 2-4)

### 11. Advanced Routing System

**File**: `src/routing/semantic_router.py`

**Semantic Similarity Matching**:

- Use sentence-transformers for query embeddings
- Maintain adapter embeddings database
- Cosine similarity for matching
- Ensemble voting for multi-adapter selection

**Query Classification**:

- Train lightweight classifier on query types
- Support for multi-label classification
- Confidence-based adapter selection
- Dynamic threshold adjustment

### 12. Multi-Adapter Composition

**File**: `src/composition/adapter_composer.py`

**Composition Strategies**:

- **Sequential**: Chain adapters in pipeline
- **Parallel**: Run multiple adapters simultaneously
- **Hierarchical**: Early/mid/late stage specialization
- **Conditional**: Adapter selection based on intermediate results

**Attention Mechanisms**:

- Weighted adapter outputs
- Learned attention over adapter responses
- Dynamic weight adjustment
- Conflict resolution between adapters

### 13. Confidence Scoring System

**File**: `src/confidence/confidence_estimator.py`

**Confidence Metrics**:

- Perplexity-based confidence
- Attention entropy analysis
- Response consistency across adapters
- Uncertainty quantification

**Threshold Management**:

- Adaptive thresholds per adapter
- User feedback integration
- Performance-based adjustments
- Fallback strategies

## Phase 3: Optimization (Weeks 5-8)

### 14. Memory Optimization

**File**: `src/optimization/memory_optimizer.py`

**Optimization Techniques**:

- Gradient checkpointing
- Mixed precision inference
- Adapter weight quantization
- Memory pooling for adapters

**Caching Strategy**:

- Intelligent adapter caching
- Predictive loading
- Memory pressure monitoring
- Automatic cleanup

### 15. Inference Speed Optimization

**File**: `src/optimization/speed_optimizer.py`

**Speed Improvements**:

- KV-cache optimization
- Batch processing
- Parallel adapter loading
- JIT compilation for critical paths

**Hardware Acceleration**:

- CUDA kernel optimization
- TensorRT integration
- Apple Metal support
- CPU-specific optimizations

### 16. Quality Assurance Pipeline

**File**: `src/quality/qa_pipeline.py`

**Automated Testing**:

- Adapter performance benchmarks
- Regression testing
- A/B testing framework
- Continuous integration

**Quality Metrics**:

- Response accuracy
- Latency consistency
- Memory stability
- Error rate tracking

## Phase 4: Advanced Features (Weeks 9-16)

### 17. Distillation System

**File**: `src/distillation/knowledge_distiller.py`

**Distillation Process**:

1. Collect high-performing adapter responses
2. Create distillation dataset
3. Fine-tune base model on successful patterns
4. Evaluate improvement
5. Selectively retire adapters

**Knowledge Transfer**:

- Teacher-student framework
- Attention transfer
- Feature matching
- Progressive distillation

### 18. Meta-Learning Framework

**File**: `src/meta_learning/meta_learner.py`

**Meta-Learning Capabilities**:

- Learn adapter selection patterns
- Optimize routing decisions
- Adapt to user preferences
- Improve from failure cases

**Few-Shot Adaptation**:

- Rapid adapter creation
- Transfer learning from existing adapters
- Minimal data requirements
- Quick deployment

### 19. Advanced Adapter Types

**File**: `src/adapters/advanced_adapters.py`

**Specialized Adapters**:

- **Temporal Adapters**: Multi-stage reasoning
- **Hierarchical Adapters**: Nested decision making
- **Ensemble Adapters**: Multiple expert combination
- **Meta Adapters**: Learning to learn

### 20. Distributed System

**File**: `src/distributed/distributed_manager.py`

**Distributed Features**:

- Multi-GPU adapter loading
- Remote adapter serving
- Federated learning support
- Adapter marketplace

## Phase 5: Production Ready (Weeks 17-24)

### 21. Scalability Infrastructure

**File**: `src/scalability/scale_manager.py`

**Scalability Features**:

- Horizontal scaling
- Load balancing
- Auto-scaling based on demand
- Resource monitoring

### 22. Security & Privacy

**File**: `src/security/security_manager.py`

**Security Measures**:

- Adapter signing and verification
- Sandboxed execution
- Privacy-preserving inference
- Audit logging

### 23. Enterprise Features

**File**: `src/enterprise/enterprise_manager.py`

**Enterprise Capabilities**:

- Multi-tenant support
- Role-based access control
- Usage analytics
- SLA monitoring

### 24. Monitoring & Observability

**File**: `src/observability/monitoring.py`

**Monitoring Stack**:

- Metrics collection (Prometheus)
- Distributed tracing (Jaeger)
- Log aggregation (ELK Stack)
- Real-time dashboards

## Technical Architecture

### Core Components Interaction

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI/Web UI    │    │   API Gateway   │    │  Adapter Store  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │              ┌───────▼───────┐              │
          │              │ Query Router  │              │
          │              └───────┬───────┘              │
          │                      │                      │
          │              ┌───────▼───────┐              │
          └─────────────▶│ Core Engine   │◀─────────────┘
                         └───────┬───────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Layer Injector        │
                    │   ┌───────────────────┐ │
                    │   │ Base Model        │ │
                    │   │ + LoRA Adapters  │ │
                    │   └───────────────────┘ │
                    └─────────────────────────┘
```

### File Structure

```
adaptrix/
├── src/
│   ├── models/
│   │   ├── base_model.py
│   │   └── model_utils.py
│   ├── injection/
│   │   ├── layer_injector.py
│   │   └── hook_manager.py
│   ├── adapters/
│   │   ├── adapter_manager.py
│   │   ├── pretrained_loader.py
│   │   └── advanced_adapters.py
│   ├── core/
│   │   ├── dynamic_loader.py
│   │   └── engine.py
│   ├── routing/
│   │   ├── keyword_router.py
│   │   └── semantic_router.py
│   ├── training/
│   │   ├── adapter_trainer.py
│   │   └── data_processor.py
│   ├── monitoring/
│   │   ├── performance_tracker.py
│   │   └── metrics.py
│   ├── cli/
│   │   ├── main.py
│   │   └── commands.py
│   ├── web/
│   │   ├── gradio_app.py
│   │   └── api.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── adapters/
├── configs/
├── tests/
├── docs/
└── requirements.txt
```

### Dependencies

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
gradio>=4.0.0
click>=8.0.0
sentence-transformers>=2.2.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
sqlite3 (built-in)
```

### Configuration Management

**File**: `configs/default.yaml`

```yaml
model:
  name: "microsoft/DialoGPT-medium"
  device: "auto"
  precision: "fp16"

injection:
  target_layers: [6, 12, 18]
  target_modules: ["self_attn.q_proj", "mlp.c_fc"]

adapters:
  cache_size: 3
  storage_path: "./adapters"

performance:
  max_memory_gb: 8
  batch_size: 1
  max_length: 512
```

### Error Handling Strategy

- Graceful degradation to base model
- Comprehensive logging
- Retry mechanisms
- User-friendly error messages
- Automatic recovery procedures

### Testing Strategy

- Unit tests for each component
- Integration tests for full pipeline
- Performance benchmarks
- Regression testing
- User acceptance testing

This specification provides complete technical details for implementing Adaptrix from MVP to production. Each component is designed to be modular, testable, and scalable.

Critical Implementation Details for Cursor AI
Forward Hook Implementation Details
The middle-layer injection uses PyTorch's forward hook mechanism. The hook function intercepts the output of transformer layers and applies LoRA transformations:

Hook Registration: Register hooks on model.transformer.h[layer_idx] for each target layer
LoRA Computation: Inside hook, compute output = output + (lora_A @ lora_B @ input) \* scaling
Memory Management: Store hooks in a registry for proper cleanup
Error Handling: Wrap hook execution in try-catch to prevent model breaking

Adapter Weight Format
Each adapter file contains:

lora_A: Low-rank matrix A (rank × input_dim)
lora_B: Low-rank matrix B (output_dim × rank)
scaling: Alpha/rank scaling factor
target_module: Which module this applies to ("self_attn.q_proj", etc.)

Dynamic Loading Memory Management

Use weakref to track loaded adapters
Implement LRU eviction with collections.OrderedDict
Monitor GPU memory with torch.cuda.memory_allocated()
Preload next likely adapter based on usage patterns

CLI Command Processing

Parse commands with click.command() decorators
Use rich library for beautiful console output
Implement progress bars for long operations
Support both interactive and batch modes

Performance Monitoring Integration

Use time.perf_counter() for high-precision timing
Track GPU memory with torch.cuda.memory_stats()
Log metrics to SQLite with sqlite3 module
Implement circular buffers for real-time metrics

Error Recovery Mechanisms

If adapter loading fails, fall back to base model
If memory runs out, automatically unload least-used adapters
If hook registration fails, log error but continue
Implement circuit breaker pattern for failing adapters

Threading Considerations

Use threading.Lock() for adapter loading/unloading
Implement async loading with asyncio for web interface
Queue adapter switches to prevent race conditions
Use thread-safe collections for adapter registry

The system is designed to be fault-tolerant, efficient, and developer-friendly. Each component has clear interfaces and can be developed/tested independently.RetryClaude can make mistakes. Please double-check responses.

Context Maintenance Throughout the Flow
This is a crucial concern! Here's how we maintain context integrity:
Context Preservation Strategy
pythonclass ContextPreservingInjector:
def **init**(self):
self.context_cache = {}
self.attention_states = {}

    def inject_with_context(self, layer_idx, input_hidden_states, attention_mask):
        # 1. Preserve original context
        original_context = input_hidden_states.clone()

        # 2. Apply LoRA transformation
        adapter_output = self.apply_lora(input_hidden_states)

        # 3. Context-aware blending
        blended_output = self.blend_with_context(
            original_context,
            adapter_output,
            attention_mask
        )

        return blended_output

Multi-Layer Context Flow
Input Query: "Solve this calculus problem: ∫x²dx"

Layer 6 Injection (Math Adapter):
├── Input: [general language representation]
├── Adapter: [mathematical reasoning injection]  
├── Output: [math-enhanced representation]
└── Context: Mathematical symbols recognized

Layer 12 Injection (Math Adapter):  
├── Input: [math-enhanced from layer 6] ← CONTEXT PRESERVED
├── Adapter: [advanced mathematical operations]
├── Output: [integration-specific representation]
└── Context: Integration patterns activated

Layer 18 Injection (Math Adapter):
├── Input: [integration-specific from layer 12] ← CONTEXT PRESERVED
├── Adapter: [solution formatting]
├── Output: [final mathematical solution]
└── Context: Complete solution pathway
Technical Context Preservation Methods

Residual Connections:

pythondef apply_adapter_with_residual(self, hidden_states, adapter_weights): # Standard LoRA with residual preservation
adapter_output = self.lora_forward(hidden_states, adapter_weights)
return hidden_states + adapter_output # Residual connection

Attention Mask Propagation:

pythondef maintain_attention_context(self, attention_mask, layer_idx): # Ensure attention patterns are preserved across injections
self.attention_states[layer_idx] = attention_mask
return self.propagate_attention(attention_mask)

Key-Value Cache Management:

pythondef preserve_kv_cache(self, past_key_values, layer_idx): # Maintain conversation context in chat scenarios
if past_key_values is not None:
self.update_cache_with_adapter(past_key_values, layer_idx)
Context Validation System
pythonclass ContextValidator:
def validate_context_integrity(self, pre_injection, post_injection): # 1. Semantic similarity check
similarity = cosine_similarity(pre_injection, post_injection)

        # 2. Attention pattern consistency
        attention_drift = self.measure_attention_drift()

        # 3. Context coherence score
        coherence = self.calculate_coherence_score()

        return {
            'similarity': similarity,
            'attention_drift': attention_drift,
            'coherence': coherence,
            'context_preserved': similarity > 0.8
        }

Conversation Context (Chat Scenarios)
For multi-turn conversations:
pythonclass ConversationContextManager:
def **init**(self):
self.conversation_history = []
self.context_embeddings = []

    def maintain_chat_context(self, new_query, adapter_response):
        # 1. Update conversation history
        self.conversation_history.append({
            'query': new_query,
            'response': adapter_response,
            'adapters_used': self.active_adapters,
            'timestamp': time.now()
        })

        # 2. Update context embeddings
        context_embedding = self.encode_conversation_context()
        self.context_embeddings.append(context_embedding)

        # 3. Prune old context if needed
        if len(self.conversation_history) > 50:
            self.prune_old_context()

Potential Context Issues & Solutions
Issue 1: Adapter conflicts between layers
python# Solution: Conflict detection
def detect_adapter_conflicts(self, layer_adapters):
conflict_score = self.calculate_semantic_conflict(layer_adapters)
if conflict_score > 0.7:
self.resolve_conflict_with_attention_weighting()
Issue 2: Context drift over multiple injections
python# Solution: Context anchoring
def anchor_context(self, original_query_embedding):
for layer_idx in self.injection_layers:
current_context = self.get_layer_context(layer_idx)
drift = cosine_distance(original_query_embedding, current_context)
if drift > self.drift_threshold:
self.apply_context_correction(layer_idx)
The key insight is that context preservation is as important as capability injection.
