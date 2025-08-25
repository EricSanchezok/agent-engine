"""
Test file for network utility functions

This file tests the cross-platform IP address detection functionality.
"""

import sys
import os
import platform

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_engine.utils.network_utils import (
    get_local_ip, 
    get_all_local_ips, 
    get_public_ip,
    _is_valid_ip,
    _is_private_ip
)

def test_ip_validation():
    """Test IP address validation functions"""
    print("🧪 Testing IP validation functions...")
    
    # Test valid IPs
    valid_ips = [
        "192.168.1.1",
        "10.0.0.1",
        "172.16.0.1",
        "8.8.8.8",
        "127.0.0.1"
    ]
    
    for ip in valid_ips:
        assert _is_valid_ip(ip), f"IP {ip} should be valid"
        print(f"✅ {ip} is valid")
    
    # Test invalid IPs
    invalid_ips = [
        "256.1.2.3",
        "1.2.3.256",
        "192.168.1",
        "192.168.1.1.1",
        "abc.def.ghi.jkl",
        "192.168.1.abc"
    ]
    
    for ip in invalid_ips:
        assert not _is_valid_ip(ip), f"IP {ip} should be invalid"
        print(f"✅ {ip} is invalid")
    
    print("✅ IP validation tests passed\n")

def test_private_ip_detection():
    """Test private IP address detection"""
    print("🧪 Testing private IP detection...")
    
    # Test private IPs
    private_ips = [
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "127.0.0.1",
        "169.254.0.1"
    ]
    
    for ip in private_ips:
        assert _is_private_ip(ip), f"IP {ip} should be private"
        print(f"✅ {ip} is private")
    
    # Test public IPs
    public_ips = [
        "8.8.8.8",
        "1.1.1.1",
        "208.67.222.222"
    ]
    
    for ip in public_ips:
        assert not _is_private_ip(ip), f"IP {ip} should be public"
        print(f"✅ {ip} is public")
    
    print("✅ Private IP detection tests passed\n")

def test_local_ip_detection():
    """Test local IP address detection"""
    print("🧪 Testing local IP detection...")
    
    try:
        local_ip = get_local_ip()
        print(f"✅ Local IP detected: {local_ip}")
        
        # Verify it's a valid IP
        assert _is_valid_ip(local_ip), f"Detected IP {local_ip} should be valid"
        
        # Should not be empty
        assert local_ip, "Local IP should not be empty"
        
        # Should not be None
        assert local_ip is not None, "Local IP should not be None"
        
    except Exception as e:
        print(f"❌ Error getting local IP: {e}")
        raise
    
    print("✅ Local IP detection test passed\n")

def test_all_local_ips():
    """Test getting all local IP addresses"""
    print("🧪 Testing all local IPs detection...")
    
    try:
        all_ips = get_all_local_ips()
        print(f"✅ Found {len(all_ips)} local IP(s): {all_ips}")
        
        # Should return a list
        assert isinstance(all_ips, list), "Should return a list"
        
        # Should not be empty (at least localhost)
        assert len(all_ips) > 0, "Should find at least one IP"
        
        # All IPs should be valid
        for ip in all_ips:
            assert _is_valid_ip(ip), f"IP {ip} should be valid"
        
        # Should include the primary local IP
        primary_ip = get_local_ip()
        assert primary_ip in all_ips, f"Primary IP {primary_ip} should be in the list"
        
    except Exception as e:
        print(f"❌ Error getting all local IPs: {e}")
        raise
    
    print("✅ All local IPs detection test passed\n")

def test_public_ip_detection():
    """Test public IP address detection (requires internet)"""
    print("🧪 Testing public IP detection...")
    
    try:
        public_ip = get_public_ip()
        
        if public_ip:
            print(f"✅ Public IP detected: {public_ip}")
            assert _is_valid_ip(public_ip), f"Public IP {public_ip} should be valid"
            assert not _is_private_ip(public_ip), f"Public IP {public_ip} should not be private"
        else:
            print("⚠️ Public IP detection failed (may be offline or blocked)")
        
    except Exception as e:
        print(f"⚠️ Error getting public IP: {e}")
    
    print("✅ Public IP detection test completed\n")

def test_platform_info():
    """Display platform information for debugging"""
    print("🧪 Platform Information:")
    print(f"   OS: {platform.system()}")
    print(f"   Release: {platform.release()}")
    print(f"   Version: {platform.version()}")
    print(f"   Machine: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    print()

def main():
    """Run all tests"""
    print("🚀 Starting Network Utils Tests")
    print("=" * 50)
    
    try:
        test_platform_info()
        test_ip_validation()
        test_private_ip_detection()
        test_local_ip_detection()
        test_all_local_ips()
        test_public_ip_detection()
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
