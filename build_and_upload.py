#!/usr/bin/env python3
"""
Build and upload agent-engine package to PyPI
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print("Return code:", e.returncode)
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def clean_build_dirs():
    """Clean build directories"""
    print("\n🧹 Cleaning build directories...")
    dirs_to_clean = ["build", "dist", "agent_engine.egg-info"]
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/")
        else:
            print(f"{dir_name}/ not found")

def check_environment():
    """Check if required tools are available"""
    print("\n🔍 Checking environment...")
    
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv is not available. Please install uv first.")
        return False
    
    # Check if twine is available
    try:
        subprocess.run(["twine", "--version"], check=True, capture_output=True)
        print("✅ twine is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ twine is not available. Installing...")
        if not run_command("uv add --dev twine", "Installing twine"):
            return False
    
    return True

def build_package():
    """Build the package"""
    print("\n📦 Building package...")
    
    # Sync dependencies first
    if not run_command("uv sync --extra opts", "Syncing dependencies"):
        return False
    
    # Build the package
    if not run_command("uv build", "Building package"):
        return False
    
    return True

def check_package():
    """Check the built package"""
    print("\n🔍 Checking package...")
    
    # Check if dist directory exists and has files
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ dist directory not found")
        return False
    
    files = list(dist_dir.glob("*"))
    if not files:
        print("❌ No files found in dist directory")
        return False
    
    print("✅ Package files found:")
    for file in files:
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    # Check package with twine
    if not run_command("twine check dist/*", "Checking package with twine"):
        return False
    
    return True

def upload_package(test=True):
    """Upload package to PyPI"""
    print(f"\n🚀 Uploading package to {'Test PyPI' if test else 'PyPI'}...")
    
    repository = "testpypi" if test else "pypi"
    cmd = f"twine upload --repository {repository} dist/*"
    
    if not run_command(cmd, f"Uploading to {repository}"):
        return False
    
    return True

def main():
    """Main function"""
    print("🎯 Agent Engine Package Builder")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Clean build directories
    clean_build_dirs()
    
    # Build package
    if not build_package():
        print("❌ Package build failed")
        sys.exit(1)
    
    # Check package
    if not check_package():
        print("❌ Package check failed")
        sys.exit(1)
    
    print("\n✅ Package built successfully!")
    print("\n📋 Next steps:")
    print("1. Test the package locally:")
    print("   pip install dist/agent_engine-*.whl")
    print("2. Upload to Test PyPI:")
    print("   python build_and_upload.py --test")
    print("3. Upload to PyPI:")
    print("   python build_and_upload.py --upload")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            if upload_package(test=True):
                print("\n✅ Package uploaded to Test PyPI successfully!")
            else:
                print("\n❌ Upload to Test PyPI failed")
                sys.exit(1)
        elif sys.argv[1] == "--upload":
            confirm = input("\n⚠️  Are you sure you want to upload to PyPI? (y/N): ")
            if confirm.lower() == 'y':
                if upload_package(test=False):
                    print("\n✅ Package uploaded to PyPI successfully!")
                else:
                    print("\n❌ Upload to PyPI failed")
                    sys.exit(1)
            else:
                print("Upload cancelled")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python build_and_upload.py [--test|--upload]")

if __name__ == "__main__":
    main()
