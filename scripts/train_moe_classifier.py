#!/usr/bin/env python3
"""
Train MoE Task Classifier for Adaptrix.

This script trains the task classifier that automatically selects
the most appropriate LoRA adapter for each input query.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.moe.classifier import TaskClassifier
from src.moe.training_data import TrainingDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_classifier():
    """Train the MoE task classifier."""
    
    print("🧠" * 80)
    print("🧠 TRAINING MOE TASK CLASSIFIER FOR ADAPTRIX")
    print("🧠" * 80)
    
    try:
        # Step 1: Generate training data
        print("\n📊 Step 1: Generating Training Data")
        print("-" * 50)
        
        generator = TrainingDataGenerator()
        
        # Generate data for different domains
        domains = ["code", "legal", "general", "math"]
        samples_per_domain = 200  # Increase for better performance
        
        texts, labels = generator.generate_training_data(
            samples_per_domain=samples_per_domain,
            domains=domains
        )
        
        print(f"✅ Generated {len(texts)} training samples")
        print(f"   📋 Domains: {domains}")
        print(f"   📊 Samples per domain: {samples_per_domain}")
        
        # Save training data for future use
        data_dir = Path("models/classifier")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        generator.save_training_data(
            texts, labels, 
            str(data_dir / "training_data.json")
        )
        
        # Step 2: Initialize classifier
        print("\n🚀 Step 2: Initializing Task Classifier")
        print("-" * 50)
        
        classifier = TaskClassifier(
            embedding_model="all-MiniLM-L6-v2",
            classifier_type="logistic"
        )
        
        if not classifier.initialize():
            print("❌ Failed to initialize classifier")
            return False
        
        print("✅ Classifier initialized successfully")
        
        # Step 3: Add training data
        print("\n📝 Step 3: Adding Training Data")
        print("-" * 50)
        
        classifier.add_training_data(texts, labels)
        
        status = classifier.get_status()
        print(f"✅ Training data added")
        print(f"   📊 Total samples: {status['training_samples']}")
        print(f"   🏷️ Classes: {status['num_classes']}")
        print(f"   📋 Adapter mapping: {status['adapter_mapping']}")
        
        # Step 4: Train classifier
        print("\n🏋️ Step 4: Training Classifier")
        print("-" * 50)
        
        start_time = time.time()
        results = classifier.train(validation_split=0.2)
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f}s")
        print(f"   📊 Train Accuracy: {results['train_accuracy']:.3f}")
        print(f"   📊 Validation Accuracy: {results['validation_accuracy']:.3f}")
        print(f"   🏷️ Classes: {results['class_names']}")
        print(f"   📈 Training samples: {results['training_samples']}")
        print(f"   📈 Validation samples: {results['validation_samples']}")
        
        # Step 5: Test classifier
        print("\n🧪 Step 5: Testing Classifier")
        print("-" * 50)
        
        test_queries = [
            ("Write a Python function to sort a list", "code"),
            ("Analyze this contract for liability issues", "legal"),
            ("What is machine learning?", "general"),
            ("Solve: 2x + 5 = 15", "math"),
            ("Create a REST API endpoint", "code"),
            ("Draft a non-disclosure agreement", "legal"),
            ("Explain quantum computing", "general"),
            ("Calculate the area of a circle", "math")
        ]
        
        correct_predictions = 0
        
        for query, expected in test_queries:
            prediction = classifier.predict(query, return_probabilities=True)
            predicted_adapter = prediction["adapter_name"]
            confidence = prediction["confidence"]
            
            status = "✅" if predicted_adapter == expected else "❌"
            print(f"   {status} '{query[:50]}...' -> {predicted_adapter} ({confidence:.3f})")
            
            if predicted_adapter == expected:
                correct_predictions += 1
        
        test_accuracy = correct_predictions / len(test_queries)
        print(f"\n📊 Test Accuracy: {test_accuracy:.3f} ({correct_predictions}/{len(test_queries)})")
        
        # Step 6: Save classifier
        print("\n💾 Step 6: Saving Classifier")
        print("-" * 50)
        
        save_path = "models/classifier"
        classifier.save(save_path)
        
        print(f"✅ Classifier saved to {save_path}")
        
        # Step 7: Verify saved model
        print("\n🔍 Step 7: Verifying Saved Model")
        print("-" * 50)
        
        # Load and test saved model
        test_classifier = TaskClassifier()
        if test_classifier.load(save_path):
            print("✅ Model loaded successfully")
            
            # Quick test
            test_query = "Write a JavaScript function"
            prediction = test_classifier.predict(test_query)
            print(f"   🧪 Test prediction: '{test_query}' -> {prediction['adapter_name']} ({prediction['confidence']:.3f})")
        else:
            print("❌ Failed to load saved model")
            return False
        
        print("\n🎉" * 80)
        print("🎉 MOE TASK CLASSIFIER TRAINING COMPLETED SUCCESSFULLY!")
        print("🎉" * 80)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   🎯 Validation Accuracy: {results['validation_accuracy']:.3f}")
        print(f"   🎯 Test Accuracy: {test_accuracy:.3f}")
        print(f"   🏷️ Supported Adapters: {results['class_names']}")
        print(f"   💾 Model saved to: {save_path}")
        print(f"   ⏱️ Training time: {training_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_existing_classifier():
    """Test an existing trained classifier."""
    
    print("\n🔍" * 50)
    print("🔍 TESTING EXISTING CLASSIFIER")
    print("🔍" * 50)
    
    classifier_path = "models/classifier"
    
    if not Path(classifier_path).exists():
        print(f"❌ No trained classifier found at {classifier_path}")
        print("   Run training first!")
        return False
    
    try:
        # Load classifier
        classifier = TaskClassifier()
        if not classifier.load(classifier_path):
            print("❌ Failed to load classifier")
            return False
        
        print("✅ Classifier loaded successfully")
        
        # Interactive testing
        print("\n🎮 Interactive Testing (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            query = input("\n💬 Enter query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            try:
                prediction = classifier.predict(query, return_probabilities=True)
                
                print(f"🎯 Predicted Adapter: {prediction['adapter_name']}")
                print(f"🎯 Confidence: {prediction['confidence']:.3f}")
                
                if 'probabilities' in prediction:
                    print("📊 All Probabilities:")
                    for adapter, prob in prediction['probabilities'].items():
                        print(f"   {adapter}: {prob:.3f}")
                
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
        
        print("\n✅ Interactive testing completed")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False


def main():
    """Main function."""
    
    print("🚀" * 100)
    print("🚀 ADAPTRIX MOE TASK CLASSIFIER TRAINING")
    print("🚀" * 100)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test MoE task classifier")
    parser.add_argument("--test", action="store_true", help="Test existing classifier")
    parser.add_argument("--interactive", action="store_true", help="Interactive testing mode")
    
    args = parser.parse_args()
    
    if args.test or args.interactive:
        success = test_existing_classifier()
    else:
        success = train_classifier()
    
    if success:
        print("\n✅ Operation completed successfully!")
        return 0
    else:
        print("\n❌ Operation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
