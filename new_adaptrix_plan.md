# Adaptrix MVP Development Plan

This document provides a comprehensive and detailed development plan for building the Minimum Viable Product (MVP) of Adaptrix, a platform designed to democratize access to advanced Large Language Model (LLM) capabilities. The plan addresses the user’s request to include the "middle layer injection system" and "MoE-LoRA" components, interpreted as the integration of Low-Rank Adaptation (LoRA) adapters into specific transformer layers and the implementation of a Mixture of LoRA Experts (MoE-LoRA) mechanism for dynamic adapter selection, respectively. The focus is on creating a robust, modular, and efficient backend, optimized for AI-assisted coding tools like Cursor and Claude Code, with comprehensive testing and documentation to support development and investor presentation.

## Project Overview

Adaptrix leverages Alibaba’s Qwen-3 1.7B dense model as its backbone, enhanced with LoRA adapters for task-specific capabilities, a Mixture of Experts (MoE) mechanism for dynamic adapter selection, Retrieval-Augmented Generation (RAG) for incorporating user-provided data, and optimizations for speed and efficiency. The MVP prioritizes backend development, ensuring modularity, scalability, and compatibility with modest hardware (e.g., laptops with 16GB RAM). Each feature is accompanied by tests and documented in the `README.md` file, with a file structure designed for clarity and scalability.

### Key Features for MVP

1. **Base Model**: Integration of Qwen-3 1.7B for general conversational capabilities.
2. **LoRA Adapters (Middle Layer Injection System)**: Modular adapters injected into specific transformer layers (e.g., attention layers) for task-specific performance (e.g., code generation, legal text summarization).
3. **MoE-LoRA**: A task classifier to dynamically select the most relevant LoRA adapter(s) for each input, enhancing efficiency and reasoning capabilities.
4. **RAG Integration**: Basic system to index and retrieve user-uploaded data for enhanced responses.
5. **Inference Optimizations**: Quantization and caching, leveraging vLLM for efficient inference.
6. **API**: A REST API to interact with the model, select adapters, and upload data.

### Objectives

- Deliver a functional MVP demonstrating core LLM capabilities with modular enhancements.
- Ensure the system is robust, with comprehensive tests for each component.
- Provide detailed documentation to support development with AI tools and investor presentations.
- Optimize for performance on consumer hardware, targeting 50-100 tokens/sec on devices like a MacBook Air M1.

## Development Phases

The development is structured into six phases, each with detailed tasks, expected outcomes, potential challenges, and mitigation strategies. The plan is designed to be actionable for AI-assisted coding tools, with clear instructions and references to relevant documentation.

### Phase 1: Base Model Integration

#### Task 1.1: Project Setup

- **Objective**: Establish a clean, organized project structure to facilitate development.
- **Steps**:
  1. Create a Git repository named `adaptrix` on a platform like GitHub.
  2. Set up the following directory structure to ensure modularity and clarity:
     ```
     adaptrix/
     ├── src/
     │   ├── models/
     │   ├── moe/
     │   ├── rag/
     │   ├── inference/
     │   ├── api/
     │   └── utils/
     ├── tests/
     ├── docs/
     ├── adapters/
     ├── requirements.txt
     ├── README.md
     └── setup.py
     ```
  3. Initialize `README.md` with:
     - Project overview and goals.
     - Setup instructions (e.g., Python version, virtual environment setup).
     - Placeholder sections for features (base model, LoRA, MoE, RAG, API).
  4. Create a `.gitignore` file excluding common Python artifacts (e.g., `__pycache__`, `.venv`, `*.pyc`).
  5. Initialize a virtual environment using Python 3.10.
- **Expected Outcome**: A Git repository with a clear, modular structure, ready for development.
- **Potential Issues**: Inconsistent directory naming, missing files, or incorrect `.gitignore` entries.
- **Mitigation**: Follow standard Python project conventions (e.g., PEP 8), verify setup with a linter (e.g., `flake8`), and test repository cloning in a clean environment.
- **Documentation**: Update `README.md` with setup instructions and project structure overview.

#### Task 1.2: Dependency Installation

- **Objective**: Install necessary libraries to support development and inference.
- **Steps**:
  1. Specify Python 3.10 in `requirements.txt` to ensure compatibility.
  2. Include the following dependencies:
     - `torch>=2.0.0` (for model operations)
     - `transformers>=4.30.0` (for Qwen-3 model and tokenizer)
     - `peft>=0.5.0` (for LoRA adapter integration)
     - `vllm>=0.8.4` (for efficient inference with multiple LoRA adapters)
     - `faiss-cpu>=1.7.0` (for RAG vector store; consider `faiss-gpu` for NVIDIA GPUs)
     - `fastapi>=0.95.0` (for API development)
     - `uvicorn>=0.20.0` (for running the FastAPI server)
     - `pytest>=7.0.0` (for unit and integration testing)
     - `sentence-transformers>=2.2.0` (for embedding generation and task classifier)
  3. Install dependencies in the virtual environment using `pip install -r requirements.txt`.
  4. Verify installation by running a simple script to import all libraries.
- **Expected Outcome**: A fully configured development environment with all required libraries.
- **Potential Issues**: Version conflicts, missing dependencies, or platform-specific installation issues.
- **Mitigation**: Use a dependency manager like `poetry` for conflict resolution, test installation in a clean environment, and consult library documentation for platform-specific instructions.
- **Documentation**: Document dependency installation steps and troubleshooting in `README.md`.

#### Task 1.3: Load Qwen-3 1.7B Model

- **Objective**: Integrate the Qwen-3 1.7B model for basic inference.
- **Steps**:
  1. Identify the model source on Hugging Face (e.g., `Qwen/Qwen3-1.7B`).
  2. Write `src/models/base_model.py` to load the model and tokenizer using `transformers`:
     - Use `AutoModelForCausalLM` and `AutoTokenizer` for loading.
     - Enable GPU support if available with `torch.cuda.is_available()`.
  3. Implement a function to generate text from a prompt (e.g., `generate_text(prompt: str) -> str`).
  4. Test with sample inputs (e.g., “Hello, how are you?”) to verify functionality.
- **Expected Outcome**: A script that successfully loads the Qwen-3 1.7B model and generates coherent text.
- **Potential Issues**: Model not found, memory errors, or slow loading times.
- **Mitigation**: Verify model availability on Hugging Face, use `torch.float16` for reduced memory usage, and consider initial quantization if memory is limited.
- **Documentation**: Add usage examples and model loading instructions to `README.md`.
- **Reference**: [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

#### Task 1.4: Verify Model Performance

- **Objective**: Ensure the base model performs as expected on target hardware.
- **Steps**:
  1. Run benchmark queries covering general Q&A, simple reasoning, and text generation tasks.
  2. Measure performance metrics:
     - Memory usage (target: <16GB RAM).
     - Inference speed (target: 50-100 tokens/sec on a MacBook Air M1).
  3. Write unit tests in `tests/test_models.py` to verify output quality (e.g., check for coherent responses).
  4. Log performance metrics using a logging utility in `src/utils/logging.py`.
- **Expected Outcome**: Confirmed model functionality with documented performance metrics.
- **Potential Issues**: Poor performance, high resource usage, or inconsistent outputs.
- **Mitigation**: Profile with `torch.profiler`, optimize model loading (e.g., use `torch.no_grad()`), and adjust prompts for better output quality.
- **Documentation**: Document benchmark results and test setup in `README.md`.

### Phase 2: LoRA Adapters and MoE Mechanism

#### Task 2.1: Integrate PEFT with Base Model (Middle Layer Injection System)

- **Objective**: Enable LoRA adapter support by injecting adapters into specific transformer layers.
- **Steps**:
  1. Study LoRA and PEFT documentation to understand adapter integration ([LoRA Paper](https://arxiv.org/abs/2106.09685), [PEFT Documentation](https://huggingface.co/docs/peft/main/en/index)).
  2. Modify `src/models/base_model.py` to support LoRA using `peft`:
     - Configure `LoraConfig` with parameters:
       - `r=8` (rank of LoRA matrices).
       - `lora_alpha=16` (scaling factor).
       - `target_modules=['q_proj', 'v_proj']` (attention layers for injection).
       - `lora_dropout=0.1` (dropout for regularization).
     - Use `get_peft_model` to wrap the base model with LoRA support.
  3. Test adapter integration with a dummy adapter to ensure no errors during loading.
  4. Verify that the model can switch between base model inference and LoRA-enhanced inference.
- **Expected Outcome**: The Qwen-3 model supports LoRA adapters injected into attention layers, enabling task-specific fine-tuning.
- **Potential Issues**: Incompatible layers, configuration errors, or increased memory usage.
- **Mitigation**: Consult Qwen-3 model architecture documentation to confirm target modules, test incrementally, and monitor memory usage.
- **Documentation**: Document LoRA configuration and integration steps in `README.md`.

#### Task 2.2: Create Example Adapters

- **Objective**: Develop LoRA adapters for at least two tasks to demonstrate modularity.
- **Steps**:
  1. Select datasets for two tasks:
     - **Code Generation**: Use CodeSearchNet or similar open-source coding dataset.
     - **Legal Text Summarization**: Use open legal datasets (e.g., from Hugging Face or Kaggle).
  2. Preprocess datasets to match the Qwen-3 input format (e.g., tokenized prompts and responses).
  3. Fine-tune LoRA adapters using `peft` in `src/models/adapters.py`:
     - Use `SFTTrainer` from `trl` for supervised fine-tuning.
     - Save adapters to `adapters/task1` and `adapters/task2`.
  4. Test each adapter with sample inputs to verify task-specific performance.
  5. Store a portion of the training data for use in training the MoE task classifier.
- **Expected Outcome**: Two functional LoRA adapters for code generation and legal text summarization.
- **Potential Issues**: Lack of suitable datasets, overfitting, or compute resource constraints.
- **Mitigation**: Use public datasets or synthetic data, apply regularization (e.g., dropout), and leverage cloud resources (e.g., AWS, GCP) if local compute is insufficient.
- **Documentation**: Document dataset sources, preprocessing steps, and fine-tuning process in `README.md`.

#### Task 2.3: Implement Task Classifier for Adapter Selection (MoE-LoRA)

- **Objective**: Develop a task classifier to dynamically select the most relevant LoRA adapter for each input, implementing the MoE mechanism.
- **Steps**:
  1. **Data Collection**:
     - Use a subset of the training data from each adapter’s dataset (e.g., 1000 samples per task).
     - Label each sample with the corresponding adapter name or ID (e.g., `code`, `legal`).
  2. **Model Selection**:
     - Option 1: Use `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to encode inputs and train a classifier head.
     - Option 2: Implement a small neural network (e.g., 2-layer MLP) using PyTorch to classify inputs based on embeddings.
  3. **Training**:
     - Encode input texts using the chosen model.
     - Train the classifier to predict the adapter ID using a cross-entropy loss.
     - Save the trained classifier to `models/classifier`.
  4. **Integration**:
     - Write `src/moe/classifier.py` to load the classifier and predict the adapter for a given input.
     - Implement a function `select_adapter(input_text: str) -> str` that returns the adapter ID.
  5. Test the classifier with sample inputs to ensure accurate adapter selection.
- **Expected Outcome**: A functional task classifier that predicts the appropriate LoRA adapter for each input, enabling sparse MoE activation.
- **Potential Issues**: Insufficient or overlapping training data, leading to misclassification; high computational cost for encoding.
- **Mitigation**: Ensure diverse and representative data for each task, use a lightweight embedding model, and validate classifier accuracy with a test set.
- **Documentation**: Document classifier training and integration in `README.md`.
- **Reference**: [Sentence Transformers Documentation](https://www.sbert.net/)

#### Task 2.4: Set Up vLLM for Inference with Multiple Adapters

- **Objective**: Configure vLLM to serve the Qwen-3 model with multiple LoRA adapters and dynamically select adapters based on the task classifier.
- **Steps**:
  1. Install vLLM following the official documentation ([vLLM Documentation](https://docs.vllm.ai/en/latest/)).
  2. Load the Qwen-3 1.7B model into vLLM:
     - Use `vllm serve Qwen/Qwen3-1.7B` with appropriate parameters.
  3. Load the trained LoRA adapters into vLLM, assigning each a unique identifier (e.g., `code`, `legal`).
  4. Implement logic in `src/inference/engine.py` to:
     - Use the task classifier to predict the adapter for each input.
     - Pass the selected adapter to vLLM using `LoRARequest` (e.g., `llm.generate(prompt, lora_request=LoRARequest(adapter_id))`).
  5. Test inference with sample queries to verify dynamic adapter selection and response quality.
- **Expected Outcome**: Efficient inference with vLLM, dynamically selecting LoRA adapters based on the task classifier’s predictions.
- **Potential Issues**: Compatibility issues with Qwen-3, performance bottlenecks, or incorrect adapter selection.
- **Mitigation**: Verify Qwen-3 support in vLLM documentation, optimize batch sizes and memory usage, and debug classifier predictions.
- **Documentation**: Document vLLM setup and adapter integration in `README.md`.

### Phase 3: Basic RAG Integration

#### Task 3.1: Set Up FAISS

- **Objective**: Establish a vector store for RAG to enable document retrieval.
- **Steps**:
  1. Install `faiss-cpu` (or `faiss-gpu` for NVIDIA GPUs).
  2. Choose an embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
  3. Write `src/rag/vector_store.py` to initialize a FAISS index (e.g., `IndexFlatL2` for cosine similarity).
  4. Test index creation with sample embeddings.
- **Expected Outcome**: A functional FAISS index for storing document embeddings.
- **Potential Issues**: High memory usage or slow indexing for large datasets.
- **Mitigation**: Use smaller embedding models, optimize FAISS parameters (e.g., index type), and process documents in batches.
- **Documentation**: Document FAISS setup and configuration in `README.md`.
- **Reference**: [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)

#### Task 3.2: Implement Document Uploading and Indexing

- **Objective**: Enable users to upload and index documents for RAG.
- **Steps**:
  1. Write `src/rag/embedding.py` to:
     - Split documents into chunks (e.g., 512-token chunks).
     - Generate embeddings using the chosen embedding model.
  2. Add embeddings to the FAISS index in `src/rag/vector_store.py`.
  3. Implement a function to handle document uploads (e.g., text files, PDFs).
  4. Test with sample documents to verify indexing accuracy.
- **Expected Outcome**: Documents are successfully indexed and retrievable via FAISS.
- **Potential Issues**: Inefficient chunking, embedding generation bottlenecks, or file format issues.
- **Mitigation**: Optimize chunk size (e.g., 256-512 tokens), batch embedding generation, and support common file formats (e.g., use `PyPDF2` for PDFs).
- **Documentation**: Document document processing and indexing steps in `README.md`.

#### Task 3.3: Modify Inference Pipeline for RAG

- **Objective**: Integrate RAG into the inference pipeline to enhance responses with retrieved data.
- **Steps**:
  1. Write `src/rag/retrieval.py` to:
     - Encode the input query using the embedding model.
     - Retrieve top-k relevant document chunks from the FAISS index.
  2. Modify `src/inference/engine.py` to:
     - Construct a prompt combining the query and retrieved chunks.
     - Pass the prompt to vLLM with the selected LoRA adapter.
  3. Test end-to-end RAG functionality with sample queries.
- **Expected Outcome**: The model generates responses incorporating retrieved document data, using the dynamically selected adapter.
- **Potential Issues**: Poor retrieval accuracy, prompt formatting issues, or integration errors with vLLM.
- **Mitigation**: Fine-tune the embedding model, adjust prompt templates, and debug vLLM integration.
- **Documentation**: Document RAG pipeline and example queries in `README.md`.

### Phase 4: Inference Optimizations

#### Task 4.1: Apply Quantization

- **Objective**: Reduce model memory footprint and improve inference speed.
- **Steps**:
  1. Check vLLM’s support for quantization (e.g., FP8, AWQ for Qwen-3).
  2. Apply quantization if supported (e.g., `vllm serve Qwen/Qwen3-1.7B-AWQ`).
  3. Alternatively, use `bitsandbytes` for INT8 or INT4 quantization if vLLM lacks support.
  4. Test quantized model accuracy and speed on target hardware.
  5. Update `src/inference/optimizations.py` with quantization logic.
- **Expected Outcome**: A quantized model with minimal accuracy loss and improved performance.
- **Potential Issues**: Accuracy degradation or compatibility issues with vLLM.
- **Mitigation**: Test multiple quantization levels, validate outputs against the unquantized model, and consult vLLM documentation.
- **Documentation**: Document quantization setup and performance metrics in `README.md`.

#### Task 4.2: Implement Key-Value Caching

- **Objective**: Speed up sequential token generation using vLLM’s caching capabilities.
- **Steps**:
  1. Verify vLLM’s built-in Key-Value caching support.
  2. Configure caching parameters in `src/inference/engine.py` (e.g., cache size, eviction policy).
  3. Test caching effectiveness on long sequences (e.g., multi-turn dialogues).
- **Expected Outcome**: Faster inference for multi-token outputs with minimal memory overhead.
- **Potential Issues**: Increased memory usage or cache inefficiencies.
- **Mitigation**: Optimize cache size, monitor memory usage, and adjust vLLM parameters.
- **Documentation**: Document caching configuration in `README.md`.

#### Task 4.3: Explore Batching

- **Objective**: Improve throughput for multiple requests using vLLM’s batching capabilities.
- **Steps**:
  1. Configure dynamic batching in vLLM (e.g., set `max_batch_size`).
  2. Implement batch inference in `src/inference/engine.py`.
  3. Test batch processing with sample queries to measure throughput.
- **Expected Outcome**: Efficient handling of multiple queries with low latency.
- **Potential Issues**: Increased latency for small batches or memory constraints.
- **Mitigation**: Implement dynamic batching with a timeout, optimize batch sizes, and monitor performance.
- **Documentation**: Document batching setup and performance results in `README.md`.

### Phase 5: API Development

#### Task 5.1: Choose API Framework

- **Objective**: Select a framework for building a REST API.
- **Steps**:
  1. Choose FastAPI for its asynchronous capabilities and automatic OpenAPI documentation.
  2. Set up `src/api/main.py` with basic FastAPI configuration.
  3. Test the API server locally using `uvicorn`.
- **Expected Outcome**: A functional FastAPI server ready for endpoint development.
- **Potential Issues**: Configuration errors or dependency conflicts.
- **Mitigation**: Follow FastAPI tutorials, test locally, and resolve conflicts using `poetry`.
- **Documentation**: Document API setup in `README.md`.
- **Reference**: [FastAPI Documentation](https://fastapi.tiangolo.com/)

#### Task 5.2: Define API Endpoints

- **Objective**: Create endpoints for core functionality.
- **Steps**:
  1. Define endpoints in `src/api/routes.py`:
     - `GET /adapters`: List available LoRA adapters.
     - `POST /upload`: Upload documents for RAG indexing.
     - `POST /query`: Query the model with optional RAG and adapter selection.
  2. Create Pydantic models in `src/api/schemas.py` for request/response validation (e.g., `QueryRequest`, `QueryResponse`).
  3. Test endpoints using tools like Postman or curl.
- **Expected Outcome**: Well-defined API endpoints with validated inputs and outputs.
- **Potential Issues**: Incorrect endpoint design, validation errors, or security vulnerabilities.
- **Mitigation**: Use OpenAPI schema for validation, implement input sanitization, and test thoroughly.
- **Documentation**: Document endpoints and example requests in `README.md`.

#### Task 5.3: Implement Request Handling

- **Objective**: Process API requests and generate responses using the integrated pipeline.
- **Steps**:
  1. Parse incoming requests in `src/api/routes.py`.
  2. For `/query` endpoint:
     - Use the task classifier to select the appropriate LoRA adapter.
     - Retrieve relevant documents if RAG is enabled.
     - Construct the prompt and pass it to vLLM with the selected adapter.
  3. Return formatted responses with metadata (e.g., adapter used, latency).
  4. Implement error handling and logging in `src/utils/logging.py`.
  5. Test end-to-end request handling with sample queries.
- **Expected Outcome**: A functional API serving model responses with dynamic adapter selection and RAG support.
- **Potential Issues**: Slow response times, error handling issues, or integration errors.
- **Mitigation**: Optimize inference pipeline, implement robust logging, and test error scenarios.
- **Documentation**: Document request handling and error codes in `README.md`.

### Phase 6: Testing and Documentation

#### Task 6.1: Write Unit Tests

- **Objective**: Ensure individual components work correctly.
- **Steps**:
  1. Write tests in `tests/test_models.py` for model loading and inference.
  2. Write tests in `tests/test_moe.py` for task classifier accuracy and adapter selection.
  3. Write tests in `tests/test_rag.py` for document indexing and retrieval.
  4. Write tests in `tests/test_inference.py` for optimizations (e.g., quantization, caching).
  5. Write tests in `tests/test_api.py` for API endpoints.
  6. Use `pytest` to run tests and aim for >80% coverage.
- **Expected Outcome**: Comprehensive unit test coverage for all components.
- **Potential Issues**: Incomplete test coverage, flaky tests, or dependency issues.
- **Mitigation**: Use pytest fixtures, mock external dependencies, and regularly run tests in CI/CD.
- **Documentation**: Document test setup and coverage in `README.md`.

#### Task 6.2: Write Integration Tests

- **Objective**: Verify the end-to-end pipeline functionality.
- **Steps**:
  1. Write tests to simulate API requests and verify responses.
  2. Test combinations of adapters and RAG data (e.g., code query with RAG, legal query without RAG).
  3. Verify that the task classifier selects the correct adapter for various inputs.
- **Expected Outcome**: Reliable end-to-end functionality with dynamic adapter selection and RAG integration.
- **Potential Issues**: Complex test setup, integration errors, or performance issues.
- **Mitigation**: Mock external dependencies, use test data, and profile integration tests for performance.
- **Documentation**: Document integration test scenarios in `README.md`.

#### Task 6.3: Document the Project

- **Objective**: Provide clear, comprehensive documentation for developers and investors.
- **Steps**:
  1. Update `README.md` with:
     - Project overview and goals.
     - Setup instructions (environment,
