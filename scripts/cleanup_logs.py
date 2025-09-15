#!/usr/bin/env python3
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


Script to find and clean up all .log files in the project
"""

import os
import sys
from pathlib import Path
import shutil


def find_log_files(project_root: Path) -> list:
    """
    Find all .log files in the project
    
    Args:
        project_root: Project root directory
        
    Returns:
        List of Path objects for all .log files
    """
    log_files = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(project_root):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
        
        for file in files:
            if file.endswith('.log'):
                log_files.append(Path(root) / file)
    
    return log_files


def cleanup_logs(project_root: Path, dry_run: bool = True) -> None:
    """
    Clean up all .log files in the project
    
    Args:
        project_root: Project root directory
        dry_run: If True, only show what would be deleted without actually deleting
    """
    print("🔍 Searching for .log files...")
    
    # Find all log files
    log_files = find_log_files(project_root)
    
    if not log_files:
        print("✅ No .log files found in the project.")
        return
    
    print(f"📁 Found {len(log_files)} .log files:")
    print("-" * 60)
    
    # Group files by directory for better display
    files_by_dir = {}
    total_size = 0
    
    for log_file in log_files:
        dir_path = log_file.parent
        if dir_path not in files_by_dir:
            files_by_dir[dir_path] = []
        
        file_size = log_file.stat().st_size if log_file.exists() else 0
        total_size += file_size
        files_by_dir[dir_path].append((log_file.name, file_size))
    
    # Display files grouped by directory
    for dir_path, files in sorted(files_by_dir.items()):
        print(f"\n📂 {dir_path.relative_to(project_root)}/")
        for filename, file_size in sorted(files):
            size_str = f"{file_size / 1024:.1f} KB" if file_size > 0 else "0 KB"
            print(f"   └─ {filename} ({size_str})")
    
    print("-" * 60)
    total_size_mb = total_size / (1024 * 1024)
    print(f"📊 Total size: {total_size_mb:.2f} MB")
    
    if dry_run:
        print("\n🔍 This was a dry run. No files were deleted.")
        print("💡 To actually delete the files, run with --delete flag")
        return
    
    # Ask for confirmation
    print(f"\n⚠️  You are about to delete {len(log_files)} .log files ({total_size_mb:.2f} MB)")
    response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
    
    if response not in ['yes', 'y']:
        print("❌ Operation cancelled.")
        return
    
    # Delete the files
    deleted_count = 0
    deleted_size = 0
    errors = []
    
    print("\n🗑️  Deleting log files...")
    
    for log_file in log_files:
        try:
            if log_file.exists():
                file_size = log_file.stat().st_size
                log_file.unlink()
                deleted_count += 1
                deleted_size += file_size
                print(f"   ✅ Deleted: {log_file.relative_to(project_root)}")
            else:
                print(f"   ⚠️  File not found: {log_file.relative_to(project_root)}")
        except Exception as e:
            error_msg = f"Error deleting {log_file.relative_to(project_root)}: {str(e)}"
            errors.append(error_msg)
            print(f"   ❌ {error_msg}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Cleanup Summary:")
    print("=" * 60)
    print(f"✅ Successfully deleted: {deleted_count} files")
    print(f"📁 Total size freed: {deleted_size / (1024 * 1024):.2f} MB")
    
    if errors:
        print(f"❌ Errors encountered: {len(errors)}")
        print("\nError details:")
        for error in errors:
            print(f"   - {error}")
    
    print("\n🎉 Log cleanup completed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up all .log files in the project")
    parser.add_argument(
        "--delete", 
        action="store_true", 
        help="Actually delete the files (default is dry run)"
    )
    parser.add_argument(
        "--project-root", 
        type=str, 
        default=None,
        help="Project root directory (default: current working directory)"
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Try to find project root by looking for pyproject.toml
        current_dir = Path.cwd()
        project_root = current_dir
        
        # Look for pyproject.toml in current directory or parents
        for parent in current_dir.parents:
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
    
    if not project_root.exists():
        print(f"❌ Error: Project root directory does not exist: {project_root}")
        sys.exit(1)
    
    print(f"🏠 Project root: {project_root}")
    print(f"🔍 Mode: {'DELETE' if args.delete else 'DRY RUN'}")
    print("=" * 60)
    
    try:
        cleanup_logs(project_root, dry_run=not args.delete)
    except KeyboardInterrupt:
        print("\n\n❌ Operation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during cleanup: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
