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


Network utility functions for cross-platform network operations

This module provides utilities for network-related operations that work
across different operating systems (Windows, Linux, macOS).
"""

import subprocess
import re
import socket
import platform
from typing import Optional, List

from ..agent_logger.agent_logger import AgentLogger

logger = AgentLogger('NetworkUtils')

def get_local_ip() -> str:
    """
    Get the local IPv4 address of the machine
    
    This function tries multiple methods to find a valid local IP address:
    1. First tries to get the primary network interface IP
    2. Falls back to socket-based methods
    3. Returns '127.0.0.1' if all methods fail
    
    Returns:
        str: Local IPv4 address as string
    """
    # Try platform-specific methods first
    ip = _get_ip_by_platform()
    if ip and ip != '127.0.0.1':
        return ip
    
    # Fallback to socket-based methods
    ip = _get_ip_by_socket()
    if ip:
        return ip
    
    # Last resort
    logger.warning("Could not determine local IP address, using localhost")
    return '127.0.0.1'

def _get_ip_by_platform() -> Optional[str]:
    """
    Get IP address using platform-specific commands
    
    Returns:
        Optional[str]: IP address if found, None otherwise
    """
    system = platform.system().lower()
    
    if system == "windows":
        return _get_ip_windows()
    elif system in ["linux", "darwin"]:  # Linux and macOS
        return _get_ip_unix()
    else:
        logger.warning(f"Unsupported operating system: {system}")
        return None

def _get_ip_windows() -> Optional[str]:
    """
    Get IP address on Windows using 'ipconfig' command
    
    Returns:
        Optional[str]: IP address if found, None otherwise
    """
    try:
        result = subprocess.run(
            ['ipconfig'], 
            capture_output=True, 
            text=True, 
            check=True,
            encoding='gbk'  # Windows Chinese locale support
        )
        output = result.stdout
        
        # Look for active network adapters
        # Priority: Ethernet > Wi-Fi > others
        patterns = [
            r"以太网适配器.*?IPv4 地址.*?:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"Ethernet adapter.*?IPv4 Address.*?:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"无线局域网适配器.*?IPv4 地址.*?:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"Wireless LAN adapter.*?IPv4 Address.*?:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"本地连接.*?IPv4 地址.*?:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"Local Area Connection.*?IPv4 Address.*?:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if match:
                ip = match.group(1)
                if _is_valid_ip(ip) and not _is_private_ip(ip):
                    logger.info(f"Found Windows IP address: {ip}")
                    return ip
        
        # If no public IP found, look for any valid IP
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if match:
                ip = match.group(1)
                if _is_valid_ip(ip):
                    logger.info(f"Found Windows IP address: {ip}")
                    return ip
                    
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to execute 'ipconfig': {e}")
    except Exception as e:
        logger.warning(f"Error getting Windows IP: {e}")
    
    return None

def _get_ip_unix() -> Optional[str]:
    """
    Get IP address on Unix-like systems (Linux, macOS) using 'ip' or 'ifconfig'
    
    Returns:
        Optional[str]: IP address if found, None otherwise
    """
    # Try 'ip' command first (modern Linux systems)
    ip = _get_ip_using_ip_command()
    if ip:
        return ip
    
    # Fallback to 'ifconfig' (older systems, macOS)
    return _get_ip_using_ifconfig()

def _get_ip_using_ip_command() -> Optional[str]:
    """
    Get IP address using 'ip addr' command (modern Linux)
    
    Returns:
        Optional[str]: IP address if found, None otherwise
    """
    try:
        result = subprocess.run(
            ['ip', 'addr'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        output = result.stdout
        
        # Look for inet addresses (IPv4)
        # Priority: eth0 > en* > wlan* > others
        patterns = [
            r"eth0:.*?inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"en[spdxo]\S+:.*?inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"wlan\S+:.*?inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"(\w+):.*?inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    ip = match[1]  # For patterns with interface name
                else:
                    ip = match
                
                if _is_valid_ip(ip) and not _is_private_ip(ip):
                    logger.info(f"Found IP address using 'ip' command: {ip}")
                    return ip
        
        # If no public IP found, look for any valid IP
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    ip = match[1]
                else:
                    ip = match
                
                if _is_valid_ip(ip):
                    logger.info(f"Found IP address using 'ip' command: {ip}")
                    return ip
                    
    except subprocess.CalledProcessError as e:
        logger.debug(f"'ip addr' command failed: {e}")
    except Exception as e:
        logger.debug(f"Error using 'ip addr' command: {e}")
    
    return None

def _get_ip_using_ifconfig() -> Optional[str]:
    """
    Get IP address using 'ifconfig' command (legacy Unix systems, macOS)
    
    Returns:
        Optional[str]: IP address if found, None otherwise
    """
    try:
        result = subprocess.run(
            ['ifconfig'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        output = result.stdout
        
        # Look for inet addresses (IPv4)
        # Priority: eth0 > en* > wlan* > others
        patterns = [
            r"eth0:.*?inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"en[spdxo]\S+:.*?inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"wlan\S+:.*?inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"(\w+):.*?inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    ip = match[1]  # For patterns with interface name
                else:
                    ip = match
                
                if _is_valid_ip(ip) and not _is_private_ip(ip):
                    logger.info(f"Found IP address using 'ifconfig': {ip}")
                    return ip
        
        # If no public IP found, look for any valid IP
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    ip = match[1]
                else:
                    ip = match
                
                if _is_valid_ip(ip):
                    logger.info(f"Found IP address using 'ifconfig': {ip}")
                    return ip
                    
    except subprocess.CalledProcessError as e:
        logger.debug(f"'ifconfig' command failed: {e}")
    except FileNotFoundError:
        logger.debug("'ifconfig' command not found")
    except Exception as e:
        logger.debug(f"Error using 'ifconfig' command: {e}")
    
    return None

def _get_ip_by_socket() -> Optional[str]:
    """
    Get IP address using socket-based methods
    
    Returns:
        Optional[str]: IP address if found, None otherwise
    """
    try:
        # Method 1: Connect to external address to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if _is_valid_ip(ip) and ip != '127.0.0.1':
                logger.info(f"Found IP address using socket method: {ip}")
                return ip
    except Exception as e:
        logger.debug(f"Socket method failed: {e}")
    
    try:
        # Method 2: Get hostname and resolve
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if _is_valid_ip(ip) and ip != '127.0.0.1':
            logger.info(f"Found IP address using hostname method: {ip}")
            return ip
    except Exception as e:
        logger.debug(f"Hostname method failed: {e}")
    
    return None

def _is_valid_ip(ip: str) -> bool:
    """
    Check if the given string is a valid IPv4 address
    
    Args:
        ip (str): IP address string to validate
        
    Returns:
        bool: True if valid IPv4, False otherwise
    """
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        for part in parts:
            if not part.isdigit():
                return False
            num = int(part)
            if num < 0 or num > 255:
                return False
        
        return True
    except Exception:
        return False

def _is_private_ip(ip: str) -> bool:
    """
    Check if the given IP address is a private address
    
    Args:
        ip (str): IP address string to check
        
    Returns:
        bool: True if private IP, False otherwise
    """
    if not _is_valid_ip(ip):
        return False
    
    parts = [int(part) for part in ip.split('.')]
    
    # Private IP ranges
    private_ranges = [
        (10, 0, 0, 0, 10, 255, 255, 255),      # 10.0.0.0/8
        (172, 16, 0, 0, 172, 31, 255, 255),    # 172.16.0.0/12
        (192, 168, 0, 0, 192, 168, 255, 255),  # 192.168.0.0/16
        (127, 0, 0, 0, 127, 255, 255, 255),    # 127.0.0.0/8
        (169, 254, 0, 0, 169, 254, 255, 255),  # 169.254.0.0/16
    ]
    
    for start1, start2, start3, start4, end1, end2, end3, end4 in private_ranges:
        if (start1 <= parts[0] <= end1 and
            start2 <= parts[1] <= end2 and
            start3 <= parts[2] <= end3 and
            start4 <= parts[3] <= end4):
            return True
    
    return False

def get_all_local_ips() -> List[str]:
    """
    Get all local IPv4 addresses of the machine
    
    Returns:
        List[str]: List of all local IPv4 addresses
    """
    ips = []
    
    try:
        # Get all network interfaces
        for interface_name, interface_addresses in socket.getaddrinfo(socket.gethostname(), None):
            ip = interface_addresses[0]
            if _is_valid_ip(ip) and ip not in ips:
                ips.append(ip)
    except Exception as e:
        logger.debug(f"Error getting all IPs: {e}")
    
    # Add the primary IP if not already in the list
    primary_ip = get_local_ip()
    if primary_ip not in ips:
        ips.append(primary_ip)
    
    return ips

def get_public_ip() -> Optional[str]:
    """
    Get the public IP address of the machine (requires internet connection)
    
    Returns:
        Optional[str]: Public IP address if found, None otherwise
    """
    try:
        import urllib.request
        import urllib.error
        
        # Try multiple IP checking services
        services = [
            "https://api.ipify.org",
            "https://checkip.amazonaws.com",
            "https://icanhazip.com",
            "https://ident.me"
        ]
        
        for service in services:
            try:
                with urllib.request.urlopen(service, timeout=5) as response:
                    ip = response.read().decode('utf-8').strip()
                    if _is_valid_ip(ip):
                        logger.info(f"Found public IP: {ip}")
                        return ip
            except Exception as e:
                logger.debug(f"Service {service} failed: {e}")
                continue
                
    except ImportError:
        logger.debug("urllib not available for public IP check")
    except Exception as e:
        logger.debug(f"Error getting public IP: {e}")
    
    return None
