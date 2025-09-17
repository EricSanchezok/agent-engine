#!/usr/bin/env python3
"""
Test script to debug Azure client max_tokens issue
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from agent_engine.llm_client.azure_client import AzureClient
from agent_engine.agent_logger.agent_logger import AgentLogger

logger = AgentLogger('AzureMaxTokensTest')

async def test_max_tokens_values():
    """Test different max_tokens values to identify the issue"""
    
    # Get API key from environment
    api_key = os.getenv("AZURE_API_KEY")
    if not api_key:
        logger.error("❌ AZURE_API_KEY environment variable not set")
        return
    
    # Initialize Azure client
    client = AzureClient(api_key=api_key)
    
    # Test different max_tokens values
    test_values = [
        1000,      # Small value - should work
        4000,      # Medium value - should work  
        8000,      # Default value - should work
        16000,     # Large value - might work
        32000,     # Very large value - might fail
        64000,     # Extremely large value - likely to fail
        100000,    # Even larger - definitely should fail
    ]
    
    system_prompt = "You are a helpful assistant. Please provide a detailed response."
    user_prompt = "Please explain the concept of artificial intelligence in detail, including its history, current applications, and future prospects. Make your response comprehensive and informative."
    
    logger.info("🚀 Starting max_tokens testing...")
    
    for max_tokens in test_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 Testing max_tokens = {max_tokens}")
        logger.info(f"{'='*60}")
        
        try:
            # Test the chat method directly
            logger.info(f"📤 Sending request with max_tokens={max_tokens}")
            response = await client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name="gpt-4.1",
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if response is None:
                logger.error(f"❌ max_tokens={max_tokens}: Response is None")
            elif response == "":
                logger.error(f"❌ max_tokens={max_tokens}: Response is empty string")
            else:
                response_length = len(response)
                logger.info(f"✅ max_tokens={max_tokens}: Success! Response length: {response_length} characters")
                logger.info(f"📝 First 200 characters: {response[:200]}...")
                
                # Check if response seems truncated
                if response_length < max_tokens * 2:  # Rough estimate: 2 chars per token
                    logger.warning(f"⚠️ Response might be truncated (length: {response_length}, expected ~{max_tokens * 2})")
                
        except Exception as e:
            logger.error(f"❌ max_tokens={max_tokens}: Exception occurred: {type(e).__name__}: {e}")
            
        # Add a small delay between requests
        await asyncio.sleep(2)
    
    # Test the chat method as well
    logger.info(f"\n{'='*60}")
    logger.info("🧪 Testing chat method with large max_tokens")
    logger.info(f"{'='*60}")
    
    try:
        response = await client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name="gpt-4.1",
            max_tokens=64000,
            temperature=0.7
        )
        
        if response is None:
            logger.error("❌ chat method with max_tokens=64000: Response is None")
        else:
            logger.info(f"✅ chat method: Success! Response length: {len(response)} characters")
            
    except Exception as e:
        logger.error(f"❌ chat method: Exception occurred: {type(e).__name__}: {e}")
    
    # Close the client
    await client.close()
    logger.info("✅ Test completed")

async def test_parameter_preparation():
    """Test the parameter preparation logic"""
    logger.info(f"\n{'='*60}")
    logger.info("🧪 Testing parameter preparation")
    logger.info(f"{'='*60}")
    
    client = AzureClient(api_key="dummy_key")  # We won't make actual calls
    
    # Test different scenarios
    test_cases = [
        {"max_tokens": 8000, "temperature": 0.7, "expected_temp": 0.7},
        {"max_tokens": 64000, "temperature": 0.7, "expected_temp": 0.7},
        {"max_tokens": 8000, "temperature": None, "expected_temp": None},
        {"max_tokens": 64000, "temperature": None, "expected_temp": None},
    ]
    
    messages = [{"role": "user", "content": "test"}]
    
    for i, case in enumerate(test_cases):
        logger.info(f"\nTest case {i+1}: max_tokens={case['max_tokens']}, temperature={case['temperature']}")
        
        params = client._prepare_chat_params(
            model_name="gpt-4.1",
            messages=messages,
            max_tokens=case["max_tokens"],
            temperature=case["temperature"]
        )
        
        logger.info(f"Prepared params: {params}")
        
        # Check if temperature is correctly handled
        if case["expected_temp"] is None:
            if "temperature" in params:
                logger.warning(f"⚠️ Temperature should not be in params but found: {params['temperature']}")
            else:
                logger.info("✅ Temperature correctly omitted from params")
        else:
            if params.get("temperature") == case["expected_temp"]:
                logger.info("✅ Temperature correctly set in params")
            else:
                logger.warning(f"⚠️ Temperature mismatch: expected {case['expected_temp']}, got {params.get('temperature')}")

async def test_model_capabilities():
    """Test what the model actually supports"""
    logger.info(f"\n{'='*60}")
    logger.info("🧪 Testing model capabilities")
    logger.info(f"{'='*60}")
    
    api_key = os.getenv("AZURE_API_KEY")
    if not api_key:
        logger.error("❌ AZURE_API_KEY environment variable not set")
        return
    
    client = AzureClient(api_key=api_key)
    
    # Test with a simple request first
    simple_prompt = "Say hello in one sentence."
    
    try:
        response = await client.chat(
            system_prompt="You are a helpful assistant.",
            user_prompt=simple_prompt,
            model_name="gpt-4.1",
            max_tokens=1000,
            temperature=0.7
        )
        
        if response:
            logger.info("✅ Basic model test successful")
            logger.info(f"Response: {response}")
        else:
            logger.error("❌ Basic model test failed - no response")
            
    except Exception as e:
        logger.error(f"❌ Basic model test failed: {type(e).__name__}: {e}")
    
    await client.close()

async def main():
    """Main test function"""
    logger.info("🚀 Starting Azure max_tokens debugging tests")
    
    # Test 1: Parameter preparation
    await test_parameter_preparation()
    
    # Test 2: Model capabilities
    await test_model_capabilities()
    
    # Test 3: Different max_tokens values
    await test_max_tokens_values()
    
    logger.info("🏁 All tests completed")

if __name__ == "__main__":
    asyncio.run(main())
