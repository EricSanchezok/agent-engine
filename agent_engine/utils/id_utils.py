"""
ID generation utilities for agent engine.
"""
import time
import hashlib
import uuid
from typing import Optional


def generate_unique_id(input_str: str, prefix: Optional[str] = None) -> str:
    """
    Generate a unique ID based on current time and input string.
    
    Args:
        input_str: Input string to be included in the ID generation
        prefix: Optional prefix to add to the generated ID
        
    Returns:
        A unique string ID combining timestamp and input string hash
    """
    # Get current timestamp in microseconds for higher precision
    timestamp = int(time.time() * 1000000)
    
    # Create a hash of the input string
    input_hash = hashlib.md5(input_str.encode('utf-8')).hexdigest()[:8]
    
    # Generate a short UUID for additional uniqueness
    short_uuid = str(uuid.uuid4())[:8]
    
    # Combine all components
    unique_id = f"{timestamp}_{input_hash}_{short_uuid}"
    
    # Add prefix if provided
    if prefix:
        unique_id = f"{prefix}_{unique_id}"
    
    return unique_id


def generate_simple_unique_id(input_str: str, prefix: Optional[str] = None) -> str:
    """
    Generate a simpler unique ID based on current time and input string.
    
    Args:
        input_str: Input string to be included in the ID generation
        prefix: Optional prefix to add to the generated ID
        
    Returns:
        A unique string ID with timestamp and input string
    """
    # Get current timestamp
    timestamp = int(time.time())
    
    # Create a simple hash of the input string
    input_hash = hashlib.md5(input_str.encode('utf-8')).hexdigest()[:6]
    
    # Combine components
    unique_id = f"{timestamp}_{input_hash}"
    
    # Add prefix if provided
    if prefix:
        unique_id = f"{prefix}_{unique_id}"
    
    return unique_id


def generate_uuid_based_id(input_str: str, prefix: Optional[str] = None) -> str:
    """
    Generate a UUID-based unique ID with input string context.
    
    Args:
        input_str: Input string to be included in the ID generation
        prefix: Optional prefix to add to the generated ID
        
    Returns:
        A UUID-based unique string ID
    """
    # Create a namespace UUID from the input string
    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, input_str)
    
    # Generate a new UUID based on the namespace and current time
    current_time = str(int(time.time() * 1000000))
    unique_id = str(uuid.uuid5(namespace, current_time))
    
    # Add prefix if provided
    if prefix:
        unique_id = f"{prefix}_{unique_id}"
    
    return unique_id
