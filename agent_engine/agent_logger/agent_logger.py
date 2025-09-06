import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
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
    """Global singleton logger class with colored output, rolling storage, and link file management"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, name: str = "AgentLogger", log_dir: Optional[str] = None, max_file_size: int = 100 * 1024 * 1024):
        """Singleton pattern - ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, name: str = "AgentLogger", log_dir: Optional[str] = None, max_file_size: int = 100 * 1024 * 1024):
        """
        Initialize AgentLogger (only once due to singleton pattern)
        
        Args:
            name: Logger name (used for filtering, not for creating new instances)
            log_dir: Directory to store log files (default: project_root/logs)
            max_file_size: Maximum log file size in bytes (default: 100MB)
        """
        # Only initialize once
        if self._initialized:
            return
            
        self.name = name
        self.max_file_size = max_file_size
        self.loggers: Dict[str, logging.Logger] = {}  # Store different named loggers
        
        # Get project root directory
        project_root = self._get_project_root()
        
        # Set log directory - priority: log_dir parameter > AGENT_LOG_DIR env var > default project_root/logs
        if log_dir is not None:
            self.log_dir = Path(log_dir)
        elif os.getenv('AGENT_LOG_DIR'):
            self.log_dir = Path(os.getenv('AGENT_LOG_DIR'))
        else:
            self.log_dir = project_root / "logs"
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear old link file content when reinitializing
        self._clear_old_link_file()
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"agent_logger_{timestamp}.log"
        self.log_filepath = self.log_dir / self.log_filename
        
        # Create/update link file
        self._update_link_file()
        
        # Mark as initialized
        self._initialized = True
        
        # Log initialization
        self._get_logger("AgentLogger").info(f"AgentLogger singleton initialized. Log file: {self.log_filepath}")
    
    def _get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name"""
        if name not in self.loggers:
            logger = self._setup_logger(name)
            self.loggers[name] = logger
        return self.loggers[name]
    
    def _get_project_root(self) -> Path:
        """Get the project root directory"""
        current_file = Path(__file__).resolve()
        # Navigate up to find the project root (where pyproject.toml is located)
        for parent in current_file.parents:
            if (parent / "pyproject.toml").exists():
                return parent
        # Fallback to current working directory
        return Path.cwd()
    
    def _setup_logger(self, name: str) -> logging.Logger:
        """Setup and configure a logger with the specified name"""
        logger = logging.getLogger(name)
        
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
        # Create a single link file for the main logger with "00_" prefix to ensure it appears first
        link_file = self.log_dir / "00_agent_logger_link.log"

        # Prepare content once
        content_lines = [
            "Main AgentLogger Link File\n",
            f"Current log file: {self.log_filename}\n",
            f"Created at: {datetime.now().isoformat()}\n",
            f"Full path: {self.log_filepath}\n",
            f"Active loggers: {', '.join(self.loggers.keys())}\n",
            "-" * 50 + "\n",
        ]
        try:
            if self.log_filepath.exists():
                with open(self.log_filepath, 'r', encoding='utf-8') as log_f:
                    content = log_f.read()
                content_lines.append("Log content:\n")
                content_lines.append(content)
            else:
                content_lines.append("Log file not yet created.\n")
        except Exception:
            # If reading current log fails, still proceed with basic info
            content_lines.append("Log file read failed.\n")

        data = ''.join(content_lines)

        # Best-effort atomic write with retries to avoid Windows sharing violations
        tmp_file = self.log_dir / f".{link_file.name}.tmp"
        pid_specific_file = self.log_dir / f"00_agent_logger_link_{os.getpid()}.log"
        for _ in range(3):
            try:
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    f.write(data)
                # Attempt atomic replace
                os.replace(tmp_file, link_file)
                return
            except Exception:
                # Small delay before retry
                time.sleep(0.05)
            finally:
                # Ensure tmp file does not linger in case of partial failures
                try:
                    if tmp_file.exists():
                        tmp_file.unlink()
                except Exception:
                    pass

        # Fallback: try direct write without replace
        try:
            with open(link_file, 'w', encoding='utf-8') as f:
                f.write(data)
            return
        except Exception:
            pass

        # Last resort: write to a process-specific link file to avoid contention
        try:
            with open(pid_specific_file, 'w', encoding='utf-8') as f:
                f.write(data)
        except Exception:
            # Give up quietly; link file is best-effort only
            pass
    
    def _clear_old_link_file(self):
        """Clear the old link file content when reinitializing"""
        link_file = self.log_dir / "00_agent_logger_link.log"
        
        if link_file.exists():
            try:
                # Best-effort truncate; avoid deleting to prevent Windows permission errors
                with open(link_file, 'w', encoding='utf-8') as f:
                    f.write("Main AgentLogger Link File\n")
                    f.write("Log file cleared. Waiting for new content...\n")
            except Exception:
                # Swallow errors; link file is optional convenience
                pass

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
            
            # Reconfigure all loggers with new file
            for name in self.loggers:
                self.loggers[name] = self._setup_logger(name)
            
            # Update link file
            self._update_link_file()
            
            self._get_logger("AgentLogger").info(f"Log file rotated. New log file: {self.log_filepath}")
    
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
                    self._get_logger("AgentLogger").info(f"Cleaned up old log file: {log_file.name}")
        
        except Exception as e:
            self._get_logger("AgentLogger").error(f"Error cleaning up old logs: {str(e)}")
    
    # Proxy methods to the underlying logger
    def debug(self, msg, *args, **kwargs):
        self._check_file_size()
        logger = self._get_logger(self.name)
        logger.debug(msg, *args, **kwargs)
        self._update_link_file()
    
    def info(self, msg, *args, **kwargs):
        self._check_file_size()
        logger = self._get_logger(self.name)
        logger.info(msg, *args, **kwargs)
        self._update_link_file()
    
    def warning(self, msg, *args, **kwargs):
        self._check_file_size()
        logger = self._get_logger(self.name)
        logger.warning(msg, *args, **kwargs)
        self._update_link_file()
    
    def error(self, msg, *args, **kwargs):
        self._check_file_size()
        logger = self._get_logger(self.name)
        logger.error(msg, *args, **kwargs)
        self._update_link_file()
    
    def critical(self, msg, *args, **kwargs):
        self._check_file_size()
        logger = self._get_logger(self.name)
        logger.critical(msg, *args, **kwargs)
        self._update_link_file()
    
    def exception(self, msg, *args, **kwargs):
        self._check_file_size()
        logger = self._get_logger(self.name)
        logger.exception(msg, *args, **kwargs)
        self._update_link_file()
    
    def log(self, level, msg, *args, **kwargs):
        self._check_file_size()
        logger = self._get_logger(self.name)
        logger.log(level, msg, *args, **kwargs)
        self._update_link_file()
