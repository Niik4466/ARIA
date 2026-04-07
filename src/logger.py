import os
import datetime
import builtins
from src.utils import Config

class MarkdownLogger:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MarkdownLogger, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        config = Config()
        self.verbose = config.get("verbose_mode", True)
        
        # Ensure path from root of ARIA
        _root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.logs_dir = os.path.join(_root_dir, "logs")
        
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        session_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(self.logs_dir, f"session_{session_time}.md")
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# ARIA Session Log - {session_time}\n\n")
            
        self.initialized = True

    def log(self, *args, **kwargs):
        if not self.verbose:
            return
            
        text = " ".join(str(a) for a in args)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            # We check if text is a header or just text, we format it as a markdown quote or list
            f.write(f"**[{timestamp}]** {text}  \n")

# Global logger instance
_logger = MarkdownLogger()

def mlog(*args, **kwargs):
    """
    Function to use instead of print() across the backend.
    """
    _logger.log(*args, **kwargs)

def redirect_print_to_logger():
    """
    Replaces builtins.print with our logger so third-party / legacy code uses it.
    """
    builtins.print = mlog
