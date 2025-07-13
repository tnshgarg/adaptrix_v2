#!/usr/bin/env python3
"""
Test RAG System Integration for Adaptrix.

This script tests the complete RAG pipeline including document processing,
vector store, retrieval, and integration with the MoE engine.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.rag.vector_store import FAISSVectorStore
from src.rag.document_processor import DocumentProcessor
from src.rag.retriever import DocumentRetriever
from src.moe.moe_engine import MoEAdaptrixEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_documents():
    """Create sample documents for testing."""
    
    docs_dir = Path("test_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Sample documents
    documents = {
        "python_basics.txt": """
Python is a high-level programming language known for its simplicity and readability.
It supports multiple programming paradigms including procedural, object-oriented, and functional programming.

Key features of Python:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Large community support
- Interpreted language

Python is widely used in web development, data science, artificial intelligence, 
automation, and scientific computing.
        """,
        
        "machine_learning.txt": """
Machine Learning is a subset of artificial intelligence that enables computers to learn
and make decisions from data without being explicitly programmed.

Types of Machine Learning:
1. Supervised Learning - Learning with labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through interaction and feedback

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines
- Random Forest

Machine learning is used in recommendation systems, image recognition, 
natural language processing, and predictive analytics.
        """,
        
        "web_development.txt": """
Web development involves creating websites and web applications for the internet.
It consists of two main areas: frontend and backend development.

Frontend Development:
- HTML for structure
- CSS for styling
- JavaScript for interactivity
- Frameworks like React, Vue, Angular

Backend Development:
- Server-side languages (Python, Java, Node.js)
- Databases (MySQL, PostgreSQL, MongoDB)
- APIs and web services
- Cloud platforms (AWS, Azure, GCP)

Modern web development also includes:
- Responsive design
- Progressive Web Apps
- DevOps practices
- Security considerations
        """,
        
        "legal_contracts.txt": """
A legal contract is a binding agreement between two or more parties that is enforceable by law.
Essential elements of a valid contract include offer, acceptance, consideration, and legal capacity.

Types of contracts:
- Express contracts (explicitly stated terms)
- Implied contracts (inferred from conduct)
- Bilateral contracts (mutual promises)
- Unilateral contracts (one-sided promise)

Important contract clauses:
- Termination clauses
- Liability limitations
- Confidentiality provisions
- Dispute resolution mechanisms
- Force majeure clauses

Contract law varies by jurisdiction and requires careful consideration of
local regulations and precedents.
        """
    }
    
    # Write documents to files
    for filename, content in documents.items():
        file_path = docs_dir / filename
        with open(file_path, 'w') as f:
            f.write(content.strip())
    
    logger.info(f"Created {len(documents)} sample documents in {docs_dir}")
    return str(docs_dir)


def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    
    print("🔍" * 80)
    print("🔍 TESTING RAG SYSTEM PIPELINE")
    print("🔍" * 80)
    
    try:
        # Step 1: Create sample documents
        print("\n📄 Step 1: Creating Sample Documents")
        print("-" * 50)
        
        docs_dir = create_sample_documents()
        print(f"✅ Sample documents created in {docs_dir}")
        
        # Step 2: Initialize document processor
        print("\n⚙️ Step 2: Document Processing")
        print("-" * 50)
        
        processor = DocumentProcessor(
            chunk_size=300,
            chunk_overlap=50,
            min_chunk_size=100
        )
        
        # Process documents
        chunks = processor.process_directory(docs_dir)
        
        print(f"✅ Processed {len(chunks)} chunks from documents")
        
        # Convert to vector store format
        texts, metadata, doc_ids = processor.chunks_to_documents(chunks)
        
        print(f"   📊 Total texts: {len(texts)}")
        print(f"   📊 Sample chunk: {texts[0][:100]}...")
        
        # Step 3: Initialize and populate vector store
        print("\n🗄️ Step 3: Vector Store Creation")
        print("-" * 50)
        
        vector_store = FAISSVectorStore(
            embedding_model="all-MiniLM-L6-v2",
            index_type="flat"
        )
        
        if not vector_store.initialize():
            print("❌ Failed to initialize vector store")
            return False
        
        print("✅ Vector store initialized")
        
        # Add documents
        if not vector_store.add_documents(texts, metadata, doc_ids):
            print("❌ Failed to add documents to vector store")
            return False
        
        print(f"✅ Added {len(texts)} documents to vector store")
        
        # Step 4: Test retrieval
        print("\n🔍 Step 4: Document Retrieval")
        print("-" * 50)
        
        retriever = DocumentRetriever(
            vector_store=vector_store,
            top_k=3,
            rerank=True
        )
        
        test_queries = [
            "How to learn Python programming?",
            "What is machine learning?",
            "Frontend web development technologies",
            "Contract termination clauses"
        ]
        
        for query in test_queries:
            print(f"\n🔤 Query: {query}")
            
            results = retriever.retrieve(query, top_k=3)
            
            print(f"   📊 Retrieved {len(results)} documents")
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. Score: {result.score:.3f}")
                print(f"      Text: {result.document[:100]}...")
                print(f"      Source: {result.metadata.get('source_file', 'unknown')}")
        
        # Step 5: Save vector store
        print("\n💾 Step 5: Saving Vector Store")
        print("-" * 50)
        
        vector_store_path = "models/rag_vector_store"
        if vector_store.save(vector_store_path):
            print(f"✅ Vector store saved to {vector_store_path}")
        else:
            print("❌ Failed to save vector store")
            return False
        
        # Step 6: Test loading
        print("\n📂 Step 6: Loading Vector Store")
        print("-" * 50)
        
        new_vector_store = FAISSVectorStore()
        if new_vector_store.load(vector_store_path):
            print("✅ Vector store loaded successfully")
            
            # Quick test
            test_results = new_vector_store.search("Python programming", k=2)
            print(f"   🧪 Test search returned {len(test_results)} results")
        else:
            print("❌ Failed to load vector store")
            return False
        
        print("\n🎉" * 80)
        print("🎉 RAG PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("🎉" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moe_rag_integration():
    """Test MoE engine with RAG integration."""
    
    print("\n🤖" * 80)
    print("🤖 TESTING MOE ENGINE WITH RAG INTEGRATION")
    print("🤖" * 80)
    
    try:
        # Initialize MoE engine with RAG
        print("\n🚀 Initializing MoE Engine with RAG")
        print("-" * 50)
        
        engine = MoEAdaptrixEngine(
            model_id="Qwen/Qwen3-1.7B",
            device="cpu",
            adapters_dir="adapters",
            classifier_path="models/classifier",
            enable_auto_selection=True,
            rag_vector_store_path="models/rag_vector_store",
            enable_rag=True
        )
        
        start_time = time.time()
        success = engine.initialize()
        init_time = time.time() - start_time
        
        if not success:
            print("❌ MoE engine with RAG initialization failed!")
            return False
        
        print(f"✅ MoE engine with RAG initialized in {init_time:.2f}s")
        
        # Check status
        status = engine.get_moe_status()
        moe_info = status.get('moe', {})
        
        print(f"   🧠 Classifier: {moe_info.get('classifier_initialized', False)}")
        print(f"   🔍 RAG: {moe_info.get('rag_initialized', False)}")
        print(f"   📊 Vector Store Documents: {moe_info.get('vector_store_stats', {}).get('num_documents', 0)}")
        
        # Test document retrieval
        print("\n🔍 Testing Document Retrieval")
        print("-" * 50)
        
        test_query = "How to learn Python programming?"
        docs = engine.retrieve_documents(test_query, top_k=3)
        
        print(f"   📊 Retrieved {len(docs)} documents for: {test_query}")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. Score: {doc['score']:.3f} - {doc['document'][:80]}...")
        
        # Test RAG-enhanced generation
        print("\n💬 Testing RAG-Enhanced Generation")
        print("-" * 50)
        
        rag_queries = [
            "How do I start learning Python?",
            "What are the main types of machine learning?",
            "What technologies are used in frontend development?"
        ]
        
        for query in rag_queries:
            print(f"\n🔤 Query: {query}")
            
            # Generate with RAG
            start_time = time.time()
            response = engine.generate(
                query,
                max_length=150,
                task_type="auto",
                use_rag=True,
                rag_top_k=3
            )
            gen_time = time.time() - start_time
            
            print(f"⏱️ Generated in {gen_time:.2f}s")
            print(f"📝 Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Compare with and without RAG
        print("\n⚖️ Comparing With/Without RAG")
        print("-" * 50)
        
        test_query = "What is machine learning?"
        
        # Without RAG
        response_no_rag = engine.generate(
            test_query,
            max_length=100,
            use_rag=False
        )
        
        # With RAG
        response_with_rag = engine.generate(
            test_query,
            max_length=100,
            use_rag=True,
            rag_top_k=2
        )
        
        print(f"🔤 Query: {test_query}")
        print(f"📝 Without RAG: {response_no_rag[:150]}...")
        print(f"📝 With RAG: {response_with_rag[:150]}...")
        
        # Cleanup
        print("\n🧹 Cleanup")
        print("-" * 50)
        engine.cleanup()
        print("✅ Cleanup completed")
        
        print("\n🎉" * 80)
        print("🎉 MOE-RAG INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print("🎉" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ MoE-RAG integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    
    print("🚀" * 100)
    print("🚀 ADAPTRIX RAG SYSTEM TEST SUITE")
    print("🚀" * 100)
    
    tests = [
        ("RAG Pipeline", test_rag_pipeline),
        ("MoE-RAG Integration", test_moe_rag_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊" * 50)
    print("📊 TEST SUMMARY")
    print("📊" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! RAG SYSTEM IS READY!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED. PLEASE CHECK THE ERRORS ABOVE.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
