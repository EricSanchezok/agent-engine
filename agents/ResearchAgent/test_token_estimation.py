"""
Test script for token estimation using tiktoken

This script compares different token estimation methods.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file
while project_root.parent != project_root:
    if (project_root / "pyproject.toml").exists():
        break
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import tiktoken
from agent_engine.agent_logger.agent_logger import AgentLogger

logger = AgentLogger(__name__)


def simple_token_estimation(text: str) -> int:
    """Simple estimation: ~4 characters per token for English text"""
    return len(text) // 4


def tiktoken_estimation(text: str, model: str = "gpt-4.1") -> int:
    """Accurate token estimation using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Failed to estimate tokens with tiktoken: {e}")
        return simple_token_estimation(text)


def test_token_estimation():
    """Test different token estimation methods"""
    print("🧪 Testing Token Estimation Methods")
    print("=" * 50)
    
    # Test texts of different lengths and types
    test_texts = [
        "Hello, world!",
        "This is a simple English sentence.",
        "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "The quick brown fox jumps over the lazy dog. This is a longer sentence that contains more words and should give us a better comparison between different token estimation methods.",
        """Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.""",
        "def hello_world():\n    print('Hello, World!')\n    return 'success'",
        "SELECT * FROM users WHERE age > 18 AND status = 'active';",
        "🚀🎉💡🔥⭐️🌟✨💫🌈🎊🎈🎁🎂🎄🎃🎆🎇🎉🎊🎋🎌🎍🎎🎏🎐🎑🎒🎓🎖🎗🎙🎚🎛🎜🎝🎞🎟🎠🎡🎢🎣🎤🎥🎦🎧🎨🎩🎪🎫🎬🎭🎮🎯🎰🎱🎲🎳🎴🎵🎶🎷🎸🎹🎺🎻🎼🎽🎾🎿🏀🏁🏂🏃🏄🏅🏆🏇🏈🏉🏊🏋🏌🏍🏎🏏🏐🏑🏒🏓🏔🏕🏖🏗🏘🏙🏚🏛🏜🏝🏞🏟🏠🏡🏢🏣🏤🏥🏦🏧🏨🏩🏪🏫🏬🏭🏮🏯🏰🏱🏲🏳🏴🏵🏶🏷🏸🏹🏺🏻🏼🏽🏾🏿"
    ]
    
    print(f"{'Text Length':<12} {'Simple Est':<12} {'Tiktoken':<12} {'Difference':<12} {'Accuracy':<10}")
    print("-" * 70)
    
    total_simple = 0
    total_tiktoken = 0
    
    for i, text in enumerate(test_texts):
        simple_tokens = simple_token_estimation(text)
        tiktoken_tokens = tiktoken_estimation(text)
        
        difference = abs(simple_tokens - tiktoken_tokens)
        accuracy = (1 - difference / max(tiktoken_tokens, 1)) * 100
        
        print(f"{len(text):<12} {simple_tokens:<12} {tiktoken_tokens:<12} {difference:<12} {accuracy:<10.1f}%")
        
        total_simple += simple_tokens
        total_tiktoken += tiktoken_tokens
    
    print("-" * 70)
    print(f"{'TOTAL':<12} {total_simple:<12} {total_tiktoken:<12} {abs(total_simple - total_tiktoken):<12} {(1 - abs(total_simple - total_tiktoken) / max(total_tiktoken, 1)) * 100:<10.1f}%")
    
    print("\n📊 Analysis:")
    print(f"Simple estimation total: {total_simple}")
    print(f"Tiktoken estimation total: {total_tiktoken}")
    print(f"Average difference: {abs(total_simple - total_tiktoken) / len(test_texts):.1f} tokens")
    
    if total_tiktoken > 0:
        overall_accuracy = (1 - abs(total_simple - total_tiktoken) / total_tiktoken) * 100
        print(f"Overall accuracy: {overall_accuracy:.1f}%")
    
    print("\n💡 Recommendations:")
    if abs(total_simple - total_tiktoken) / max(total_tiktoken, 1) > 0.2:
        print("✅ Tiktoken provides significantly more accurate token estimation")
        print("✅ Recommended to use tiktoken for production systems")
    else:
        print("⚠️  Simple estimation is reasonably accurate for basic use cases")
        print("⚠️  Consider tiktoken for more precise token counting")


def test_model_compatibility():
    """Test tiktoken compatibility with different models"""
    print("\n🔍 Testing Model Compatibility")
    print("=" * 50)
    
    test_text = "This is a test sentence for model compatibility."
    
    models_to_test = [
        "gpt-4.1",
        "gpt-4",
        "gpt-3.5-turbo",
        "text-davinci-003",
        "text-davinci-002"
    ]
    
    print(f"{'Model':<20} {'Tokens':<10} {'Status':<10}")
    print("-" * 40)
    
    for model in models_to_test:
        try:
            tokens = tiktoken_estimation(test_text, model)
            print(f"{model:<20} {tokens:<10} {'✅ OK':<10}")
        except Exception as e:
            print(f"{model:<20} {'N/A':<10} {'❌ Error':<10}")


def main():
    """Run all tests"""
    print("🚀 Starting Token Estimation Tests")
    print("=" * 50)
    
    try:
        test_token_estimation()
        test_model_compatibility()
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        logger.error(f"Test error: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
