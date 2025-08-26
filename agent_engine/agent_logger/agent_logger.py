import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import shutil


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset color
    }
    
    def format(self, record):
        # Get the original formatted message
        log_message = super().format(record)
        
        # Add color based on log level
        if record.levelname in self.COLORS:
            log_message = f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        
        return log_message


class AgentLogger:
    """Global logger class with colored output, rolling storage, and link file management"""
    
    def __init__(self, name: str = "AgentLogger", log_dir: Optional[str] = None, max_file_size: int = 100 * 1024 * 1024):
        """
        Initialize AgentLogger
        
        Args:
            name: Logger name
            log_dir: Directory to store log files (default: project_root/logs)
            max_file_size: Maximum log file size in bytes (default: 100MB)
        """
        self.name = name
        self.max_file_size = max_file_size
        
        # Get project root directory
        project_root = self._get_project_root()
        
        # Set log directory
        if log_dir is None:
            self.log_dir = project_root / "logs"
        else:
            self.log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear old link file content when reinitializing
        self._clear_old_link_file()
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"agent_logger_{timestamp}.log"
        self.log_filepath = self.log_dir / self.log_filename
        
        # Create and configure logger
        self.logger = self._setup_logger()
        
        # Create/update link file
        self._update_link_file()
        
        # Log initialization
        self.logger.info(f"AgentLogger initialized. Log file: {self.log_filepath}")
    
    def _get_project_root(self) -> Path:
        """Get the project root directory"""
        current_file = Path(__file__).resolve()
        # Navigate up to find the project root (where pyproject.toml is located)
        for parent in current_file.parents:
            if (parent / "pyproject.toml").exists():
                return parent
        # Fallback to current working directory
        return Path.cwd()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure the logger"""
        logger = logging.getLogger(self.name)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Set log level
        logger.setLevel(logging.DEBUG)
        
        # Create console handler with colored output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create colored formatter for console
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter for file (without colors)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
        
        return logger
    
    def _update_link_file(self):
        """Create/update the link file to point to the current log file"""
        # 使用 logger name 创建唯一的 link 文件名
        safe_name = self.name.replace('.', '_').replace('/', '_').replace('\\', '_')
        link_file = self.log_dir / f"{safe_name}_link.log"
        
        # Remove existing link if it exists
        if link_file.exists():
            link_file.unlink()
        
        # Create a text file with the current log filename and content
        try:
            with open(link_file, 'w', encoding='utf-8') as f:
                f.write(f"Logger: {self.name}\n")
                f.write(f"Current log file: {self.log_filename}\n")
                f.write(f"Created at: {datetime.now().isoformat()}\n")
                f.write(f"Full path: {self.log_filepath}\n")
                f.write("-" * 50 + "\n")
                # Copy the actual log content from the current log file
                if self.log_filepath.exists():
                    with open(self.log_filepath, 'r', encoding='utf-8') as log_f:
                        content = log_f.read()
                        f.write("Log content:\n")
                        f.write(content)
                else:
                    f.write("Log file not yet created.\n")
        except Exception as e:
            # Fallback: create a simple text file
            with open(link_file, 'w', encoding='utf-8') as f:
                f.write(f"Logger: {self.name}\n")
                f.write(f"Current log file: {self.log_filename}\n")
                f.write(f"Created at: {datetime.now().isoformat()}\n")
                f.write(f"Error: {str(e)}\n")
    
    def _clear_old_link_file(self):
        """Clear the old link file content when reinitializing"""
        # 使用 logger name 创建唯一的 link 文件名
        safe_name = self.name.replace('.', '_').replace('/', '_').replace('\\', '_')
        link_file = self.log_dir / f"{safe_name}_link.log"
        
        if link_file.exists():
            try:
                # Clear the content but keep the file
                with open(link_file, 'w', encoding='utf-8') as f:
                    f.write(f"Logger: {self.name}\n")
                    f.write("Log file cleared. Waiting for new content...\n")
            except Exception as e:
                # If we can't clear it, just remove it
                link_file.unlink()

    def _check_file_size(self):
        """Check if current log file exceeds size limit and rotate if necessary"""
        if self.log_filepath.exists() and self.log_filepath.stat().st_size > self.max_file_size:
            # Create new log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_log_filename = f"agent_logger_{timestamp}.log"
            new_log_filepath = self.log_dir / new_log_filename
            
            # Update logger with new file
            self.log_filename = new_log_filename
            self.log_filepath = new_log_filepath
            
            # Reconfigure logger with new file
            self.logger = self._setup_logger()
            
            # Update link file
            self._update_link_file()
            
            self.logger.info(f"Log file rotated. New log file: {self.log_filepath}")
    
    def get_latest_logs(self, n: int = 100) -> str:
        """
        Get the latest n lines from the current log file
        
        Args:
            n: Number of latest log lines to retrieve
            
        Returns:
            String containing the latest n log lines
        """
        try:
            if not self.log_filepath.exists():
                return "Log file not found."
            
            with open(self.log_filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                latest_lines = lines[-n:] if len(lines) > n else lines
                return ''.join(latest_lines)
        
        except Exception as e:
            return f"Error reading log file: {str(e)}"
    
    def cleanup_old_logs(self, keep_days: int = 30):
        """
        Clean up old log files
        
        Args:
            keep_days: Number of days to keep log files
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (keep_days * 24 * 60 * 60)
            
            for log_file in self.log_dir.glob("agent_logger_*.log"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.logger.info(f"Cleaned up old log file: {log_file.name}")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old logs: {str(e)}")
    
    # Proxy methods to the underlying logger
    def debug(self, msg, *args, **kwargs):
        self._check_file_size()
        self.logger.debug(msg, *args, **kwargs)
        self._update_link_file()
    
    def info(self, msg, *args, **kwargs):
        self._check_file_size()
        self.logger.info(msg, *args, **kwargs)
        self._update_link_file()
    
    def warning(self, msg, *args, **kwargs):
        self._check_file_size()
        self.logger.warning(msg, *args, **kwargs)
        self._update_link_file()
    
    def error(self, msg, *args, **kwargs):
        self._check_file_size()
        self.logger.error(msg, *args, **kwargs)
        self._update_link_file()
    
    def critical(self, msg, *args, **kwargs):
        self._check_file_size()
        self.logger.critical(msg, *args, **kwargs)
        self._update_link_file()
    
    def exception(self, msg, *args, **kwargs):
        self._check_file_size()
        self.logger.exception(msg, *args, **kwargs)
        self._update_link_file()
    
    def log(self, level, msg, *args, **kwargs):
        self._check_file_size()
        self.logger.log(level, msg, *args, **kwargs)
        self._update_link_file()
