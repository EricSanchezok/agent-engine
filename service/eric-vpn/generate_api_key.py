"""
API Key Generator for Eric VPN Proxy

This script generates secure API keys using Python's secrets module.
Generated keys are cryptographically secure and suitable for production use.
"""

import secrets
import string
from typing import List, Optional
import argparse


def generate_api_key(length: int = 32, prefix: str = "eric_vpn_") -> str:
    """
    Generate a cryptographically secure API key.
    
    Args:
        length: Length of the random part (default: 32)
        prefix: Optional prefix for the key (default: "eric_vpn_")
    
    Returns:
        Generated API key string
    """
    # Use URL-safe base64 characters for better compatibility
    alphabet = string.ascii_letters + string.digits + "-_"
    random_part = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}{random_part}"


def generate_multiple_keys(count: int = 1, length: int = 32, prefix: str = "eric_vpn_") -> List[str]:
    """
    Generate multiple API keys.
    
    Args:
        count: Number of keys to generate
        length: Length of each key's random part
        prefix: Optional prefix for all keys
    
    Returns:
        List of generated API keys
    """
    keys = []
    for _ in range(count):
        keys.append(generate_api_key(length, prefix))
    return keys


def update_config_file(api_keys: List[str], config_path: str = "config.py") -> None:
    """
    Update the config.py file with new API keys.
    
    Args:
        api_keys: List of API keys to set
        config_path: Path to config.py file
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find and replace the api_keys line
        import re
        pattern = r'"api_keys":\s*\[.*?\]'
        replacement = f'"api_keys": {api_keys}'
        
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        if new_content == content:
            print(f"Warning: Could not find api_keys configuration in {config_path}")
            print("Please manually update the config.py file with:")
            print(f'    "api_keys": {api_keys},')
            return
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Successfully updated {config_path} with new API keys")
        
    except FileNotFoundError:
        print(f"Error: {config_path} not found")
    except Exception as e:
        print(f"Error updating config file: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate secure API keys for Eric VPN Proxy")
    parser.add_argument("-c", "--count", type=int, default=1, help="Number of keys to generate (default: 1)")
    parser.add_argument("-l", "--length", type=int, default=32, help="Length of random part (default: 32)")
    parser.add_argument("-p", "--prefix", type=str, default="eric_vpn_", help="Key prefix (default: eric_vpn_)")
    parser.add_argument("-u", "--update-config", action="store_true", help="Update config.py with generated keys")
    parser.add_argument("--no-prefix", action="store_true", help="Generate keys without prefix")
    
    args = parser.parse_args()
    
    # Handle no-prefix option
    prefix = "" if args.no_prefix else args.prefix
    
    # Generate keys
    if args.count == 1:
        key = generate_api_key(args.length, prefix)
        print(f"Generated API Key: {key}")
        if args.update_config:
            update_config_file([key])
    else:
        keys = generate_multiple_keys(args.count, args.length, prefix)
        print(f"Generated {args.count} API Keys:")
        for i, key in enumerate(keys, 1):
            print(f"  {i}. {key}")
        
        if args.update_config:
            update_config_file(keys)
    
    print("\nUsage in requests:")
    print("  curl -H \"X-API-Key: <your-key>\" \"http://127.0.0.1:3000/r/{route}/{path}\"")


if __name__ == "__main__":
    main()
