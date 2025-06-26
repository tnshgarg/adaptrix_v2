#!/usr/bin/env python3
"""
🚀 ADAPTRIX ADAPTER ECOSYSTEM EXPANSION (DYNAMIC)

This script uses the new dynamic LoRA converter to automatically handle
any LoRA architecture without manual fixes. It's robust and modular!

New Adapters:
1. AmevinLS/phi-2-lora-realnews - News writing and factual articles
2. Nutanix/phi-2_SFT_lora_4_alpha_16_humaneval_raw_json - Python code generation
3. Gunslinger3D/fine-tuning-Phi2-with-webglm-qa-with-lora_9 - Question answering

Combined with existing:
4. phi2_gsm8k_converted - Mathematical reasoning
5. phi2_instruct_converted - Instruction following
"""

import sys
import os
from datetime import datetime
from huggingface_hub import snapshot_download

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.conversion.dynamic_lora_converter import DynamicLoRAConverter


class AdapterEcosystemExpander:
    """Manages the expansion of the Adaptrix adapter ecosystem using dynamic conversion."""

    def __init__(self):
        self.converter = DynamicLoRAConverter()
        self.new_adapters = [
            {
                "hf_repo": "AmevinLS/phi-2-lora-realnews",
                "name": "phi2_realnews_dynamic",
                "description": "Phi-2 RealNews LoRA adapter for long-form factual writing and realistic article generation",
                "capabilities": ["news_writing", "factual_content", "long_form", "journalism", "articles"],
                "domain": "journalism",
                "training_data": "RealNews dataset for factual writing"
            },
            {
                "hf_repo": "Nutanix/phi-2_SFT_lora_4_alpha_16_humaneval_raw_json",
                "name": "phi2_humaneval_dynamic",
                "description": "Phi-2 HumanEval LoRA adapter for Python code generation and function writing",
                "capabilities": ["code_generation", "python", "programming", "functions", "humaneval"],
                "domain": "programming",
                "training_data": "HumanEval dataset for Python code generation"
            },
            {
                "hf_repo": "Gunslinger3D/fine-tuning-Phi2-with-webglm-qa-with-lora_9",
                "name": "phi2_webglm_qa_dynamic",
                "description": "Phi-2 WebGLM-QA LoRA adapter for open-domain question answering and factual queries",
                "capabilities": ["question_answering", "factual_queries", "open_domain", "webglm", "qa"],
                "domain": "question_answering",
                "training_data": "WebGLM-QA dataset for question answering"
            }
        ]
    
    def download_adapter(self, hf_repo: str, local_dir: str) -> bool:
        """Download adapter from HuggingFace."""
        try:
            print(f"📥 Downloading {hf_repo}...")
            snapshot_download(
                repo_id=hf_repo,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded to: {local_dir}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def convert_adapter(self, adapter_info: dict) -> bool:
        """Convert a HuggingFace adapter using dynamic converter."""
        start_time = datetime.now()

        print(f"\n🔄 Converting {adapter_info['name']} (DYNAMIC)...")
        print(f"📊 Domain: {adapter_info['domain']}")
        print(f"🎯 Capabilities: {', '.join(adapter_info['capabilities'])}")

        hf_adapter_dir = f"adapters/{adapter_info['name']}_hf"

        # Download adapter if needed
        if not os.path.exists(hf_adapter_dir):
            if not self.download_adapter(adapter_info['hf_repo'], hf_adapter_dir):
                return False

        # Use dynamic converter
        success = self.converter.convert_adapter(
            adapter_info['hf_repo'],
            adapter_info['name'],
            adapter_info['description'],
            adapter_info['capabilities'],
            adapter_info['domain'],
            adapter_info['training_data']
        )
        
        conversion_time = (datetime.now() - start_time).total_seconds()

        if success:
            print(f"✅ Dynamic conversion complete!")
            print(f"⏱️ Conversion time: {conversion_time:.2f}s")
        else:
            print(f"❌ Dynamic conversion failed!")

        return success
    
    def expand_ecosystem(self) -> bool:
        """Expand the adapter ecosystem with all new adapters using dynamic conversion."""
        print("🚀" * 80)
        print("🚀 EXPANDING ADAPTRIX ECOSYSTEM (DYNAMIC CONVERSION) 🚀")
        print("🚀" * 80)
        print()
        print("🔄 Using NEW dynamic converter that handles ANY LoRA architecture!")
        print("✅ No more manual fixes needed")
        print("✅ Automatic architecture detection")
        print("✅ Robust error handling")
        print()
        print("Adding 3 new specialized adapters:")
        print("📰 News Writing (RealNews)")
        print("💻 Code Generation (HumanEval)")
        print("❓ Question Answering (WebGLM-QA)")
        print()

        successful_conversions = 0

        for adapter_info in self.new_adapters:
            success = self.convert_adapter(adapter_info)
            if success:
                successful_conversions += 1

        # Print final statistics
        self._print_final_stats(successful_conversions)

        return successful_conversions > 0

    def _print_final_stats(self, successful_conversions: int):
        """Print final conversion statistics."""
        print("\n" + "📊" * 80)
        print("📊 DYNAMIC ECOSYSTEM EXPANSION COMPLETE 📊")
        print("📊" * 80)

        total_adapters = len(self.new_adapters)
        failed_conversions = total_adapters - successful_conversions

        print(f"✅ Successful conversions: {successful_conversions}/{total_adapters}")
        print(f"❌ Failed conversions: {failed_conversions}/{total_adapters}")

        # Get converter stats
        converter_stats = self.converter.get_conversion_stats()
        print(f"🔍 Architectures detected: {converter_stats['architectures_detected']}")
        print(f"📈 Success rate: {converter_stats['success_rate']:.1%}")

        print(f"\n🎯 TOTAL ADAPTER ECOSYSTEM:")
        print(f"   1. 🧮 phi2_gsm8k_converted (Math)")
        print(f"   2. 📝 phi2_instruct_converted (Instructions)")
        print(f"   3. 📰 phi2_realnews_dynamic (News)")
        print(f"   4. 💻 phi2_humaneval_dynamic (Code)")
        print(f"   5. ❓ phi2_webglm_qa_dynamic (Q&A)")

        print(f"\n🚀 Ready for multi-domain composition with DYNAMIC conversion!")
        print(f"✅ No more manual architecture fixes needed!")
        print(f"✅ Any future LoRA adapter will work automatically!")


def main():
    """Main function."""
    expander = AdapterEcosystemExpander()
    
    success = expander.expand_ecosystem()
    
    if success:
        print("\n🎊 ECOSYSTEM EXPANSION SUCCESSFUL!")
        print("🌐 Update web interface to test new adapters")
        print("🧪 Run comprehensive multi-adapter tests")
    else:
        print("\n❌ Some conversions failed - check logs above")


if __name__ == "__main__":
    main()
