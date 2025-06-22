#!/usr/bin/env python3
"""
Test script for the advanced routing system.
Demonstrates semantic, keyword, and unified routing capabilities.
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.routing.semantic_router import SemanticRouter
from src.routing.keyword_router import KeywordRouter, KeywordRule
from src.routing.unified_router import UnifiedRouter, RoutingStrategy


def test_semantic_router():
    """Test the semantic router functionality."""
    print("🧠 TESTING SEMANTIC ROUTER")
    print("=" * 60)
    
    # Initialize semantic router
    router = SemanticRouter(confidence_threshold=0.5)
    
    # Register adapters with semantic profiles
    adapters = [
        {
            "name": "math_reasoning",
            "description": "Advanced mathematical problem solving and computation",
            "capabilities": ["arithmetic", "algebra", "calculus", "geometry", "statistics"],
            "example_queries": [
                "Solve the equation 2x + 5 = 13",
                "Calculate the derivative of x^2 + 3x",
                "Find the area of a circle with radius 5"
            ],
            "performance_metrics": {"accuracy": 0.92, "latency_ms": 150}
        },
        {
            "name": "code_generation",
            "description": "Programming and software development assistance",
            "capabilities": ["python", "javascript", "algorithms", "debugging", "api"],
            "example_queries": [
                "Write a Python function to sort a list",
                "Create a REST API endpoint",
                "Debug this JavaScript code"
            ],
            "performance_metrics": {"accuracy": 0.85, "latency_ms": 200}
        },
        {
            "name": "creative_writing",
            "description": "Creative content generation and storytelling",
            "capabilities": ["storytelling", "poetry", "dialogue", "narrative", "creative"],
            "example_queries": [
                "Write a short story about a robot",
                "Create a poem about nature",
                "Generate dialogue for a character"
            ],
            "performance_metrics": {"accuracy": 0.78, "latency_ms": 180}
        }
    ]
    
    # Register all adapters
    for adapter in adapters:
        success = router.register_adapter(**adapter)
        print(f"✅ Registered {adapter['name']}: {success}")
    
    # Test queries
    test_queries = [
        "What is the integral of sin(x) dx?",
        "Write a Python function to calculate fibonacci numbers",
        "Tell me a story about a magical forest",
        "How do I solve quadratic equations?",
        "Create a web scraper in Python",
        "Write a haiku about programming"
    ]
    
    print(f"\n📝 Testing {len(test_queries)} queries:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        result = router.route_query(query)
        print(f"\n{i}. Query: {query}")
        print(f"   → Adapter: {result.primary_adapter}")
        print(f"   → Confidence: {result.confidence:.3f}")
        print(f"   → Reasoning: {result.reasoning}")
        
        if result.secondary_adapters:
            print(f"   → Alternatives: {', '.join([f'{name}({score:.2f})' for name, score in result.secondary_adapters])}")
    
    # Show routing statistics
    stats = router.get_routing_stats()
    print(f"\n📊 Routing Statistics:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Success rate: {stats['successful_routes']}/{stats['total_queries']}")
    print(f"   Average confidence: {stats['avg_confidence']:.3f}")
    print(f"   Average processing time: {stats['avg_processing_time']*1000:.1f}ms")


def test_keyword_router():
    """Test the keyword router functionality."""
    print("\n\n🔤 TESTING KEYWORD ROUTER")
    print("=" * 60)
    
    # Initialize keyword router
    router = KeywordRouter(confidence_threshold=0.3)
    
    # Test with default rules (already initialized)
    test_queries = [
        "Calculate 15% of 240",
        "Write a Python script to read CSV files",
        "Analyze the logical structure of this argument",
        "Create a fantasy story with dragons",
        "Solve for x in the equation 3x - 7 = 14",
        "Debug my JavaScript function"
    ]
    
    print(f"📝 Testing {len(test_queries)} queries:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        result = router.route_query(query)
        print(f"\n{i}. Query: {query}")
        print(f"   → Adapter: {result.primary_adapter}")
        print(f"   → Confidence: {result.confidence:.3f}")
        print(f"   → Matched keywords: {', '.join(result.matched_keywords)}")
        print(f"   → Reasoning: {result.reasoning}")


def test_unified_router():
    """Test the unified router with multiple strategies."""
    print("\n\n🔀 TESTING UNIFIED ROUTER")
    print("=" * 60)
    
    # Initialize unified router
    router = UnifiedRouter(
        default_strategy=RoutingStrategy.HYBRID,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    
    # Register adapters
    adapters = [
        {
            "name": "math_reasoning",
            "description": "Mathematical problem solving and computation",
            "capabilities": ["math", "calculation", "algebra", "geometry"],
            "example_queries": [
                "Solve 2x + 5 = 13",
                "Calculate the area of a triangle"
            ],
            "keywords": ["calculate", "solve", "equation", "math", "number", "formula"],
            "performance_metrics": {"accuracy": 0.92, "latency_ms": 150}
        },
        {
            "name": "code_generation",
            "description": "Programming and software development",
            "capabilities": ["programming", "coding", "algorithms", "debugging"],
            "example_queries": [
                "Write a Python function",
                "Create an API endpoint"
            ],
            "keywords": ["python", "code", "function", "programming", "script", "debug"],
            "performance_metrics": {"accuracy": 0.85, "latency_ms": 200}
        }
    ]
    
    for adapter in adapters:
        success = router.register_adapter(**adapter)
        print(f"✅ Registered {adapter['name']}: {success}")
    
    # Test different strategies
    test_queries = [
        "What is 25% of 80?",
        "Write a Python function to sort numbers",
        "How do I calculate compound interest?"
    ]
    
    strategies = [
        RoutingStrategy.SEMANTIC_ONLY,
        RoutingStrategy.KEYWORD_ONLY,
        RoutingStrategy.HYBRID,
        RoutingStrategy.ENSEMBLE
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        for strategy in strategies:
            result = router.route_query(query, strategy=strategy)
            print(f"   {strategy.value:15} → {result.primary_adapter:15} (conf: {result.confidence:.3f})")
    
    # Test adaptive strategy
    print(f"\n🤖 Testing Adaptive Strategy:")
    print("-" * 40)
    
    adaptive_queries = [
        "Calculate",  # Short query
        "What is the derivative of x squared plus three x minus two?",  # Long query
        "Write a Python function to calculate fibonacci"  # Medium query
    ]
    
    for query in adaptive_queries:
        result = router.route_query(query, strategy=RoutingStrategy.ADAPTIVE)
        print(f"   '{query[:30]}...' → {result.strategy_used.value} → {result.primary_adapter}")
    
    # Show performance statistics
    stats = router.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    for strategy, perf in stats.items():
        if isinstance(perf, dict) and 'success_rate' in perf:
            print(f"   {strategy:15}: {perf['success_rate']:.1%} success, {perf['total_queries']} queries")


def benchmark_routing_performance():
    """Benchmark routing performance across different strategies."""
    print("\n\n⚡ ROUTING PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Initialize routers
    semantic_router = SemanticRouter()
    keyword_router = KeywordRouter()
    unified_router = UnifiedRouter()
    
    # Register a few adapters
    adapters = [
        {
            "name": "math",
            "description": "Math problem solving",
            "capabilities": ["math", "calculation"],
            "example_queries": ["Solve 2+2", "Calculate area"],
            "keywords": ["math", "calculate", "solve"]
        },
        {
            "name": "code",
            "description": "Code generation",
            "capabilities": ["programming", "coding"],
            "example_queries": ["Write function", "Debug code"],
            "keywords": ["code", "python", "function"]
        }
    ]
    
    # Register adapters
    for adapter in adapters:
        semantic_router.register_adapter(
            adapter["name"], adapter["description"], 
            adapter["capabilities"], adapter["example_queries"]
        )
        unified_router.register_adapter(**adapter)
    
    # Benchmark queries
    benchmark_queries = [
        "Calculate the square root of 144",
        "Write a Python function to reverse a string",
        "What is 15% of 200?",
        "Create a sorting algorithm",
        "Solve the equation x^2 - 4 = 0"
    ] * 10  # Repeat for better timing
    
    # Benchmark semantic routing
    start_time = time.time()
    for query in benchmark_queries:
        semantic_router.route_query(query)
    semantic_time = time.time() - start_time
    
    # Benchmark keyword routing
    start_time = time.time()
    for query in benchmark_queries:
        keyword_router.route_query(query)
    keyword_time = time.time() - start_time
    
    # Benchmark unified routing (hybrid)
    start_time = time.time()
    for query in benchmark_queries:
        unified_router.route_query(query, strategy=RoutingStrategy.HYBRID)
    unified_time = time.time() - start_time
    
    # Results
    num_queries = len(benchmark_queries)
    print(f"Processed {num_queries} queries:")
    print(f"   Semantic routing: {semantic_time:.3f}s ({semantic_time/num_queries*1000:.1f}ms per query)")
    print(f"   Keyword routing:  {keyword_time:.3f}s ({keyword_time/num_queries*1000:.1f}ms per query)")
    print(f"   Unified routing:  {unified_time:.3f}s ({unified_time/num_queries*1000:.1f}ms per query)")
    
    # Speed comparison
    fastest = min(semantic_time, keyword_time, unified_time)
    print(f"\n⚡ Speed comparison (vs fastest):")
    print(f"   Semantic: {semantic_time/fastest:.1f}x")
    print(f"   Keyword:  {keyword_time/fastest:.1f}x")
    print(f"   Unified:  {unified_time/fastest:.1f}x")


def main():
    """Run all routing tests."""
    print("🎯 ADVANCED ROUTING SYSTEM TEST SUITE")
    print("=" * 80)
    
    try:
        test_semantic_router()
        test_keyword_router()
        test_unified_router()
        benchmark_routing_performance()
        
        print("\n\n🎊 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ Semantic routing: Working")
        print("✅ Keyword routing: Working")
        print("✅ Unified routing: Working")
        print("✅ Multiple strategies: Working")
        print("✅ Performance benchmarks: Completed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
