"""Custom logging handlers for PostgreSQL and NFS."""

import logging
from datetime import datetime
from pathlib import Path


class PostgreSQLLogHandler(logging.Handler):
    """Log handler that writes to PostgreSQL database."""
    
    def __init__(self, study_id: int, trial_number: int, component: str, pg_url: str):
        super().__init__()
        self.study_id = study_id
        self.trial_number = trial_number
        self.component = component
        self.pg_url = pg_url
        self._connection = None
        
        # Setup formatter
        formatter = logging.Formatter('%(message)s')
        self.setFormatter(formatter)
    
    def emit(self, record: logging.LogRecord):
        """Write log record to PostgreSQL."""
        try:
            import psycopg2
            
            # Get worker node ID (Ray or local)
            worker_id = self._get_worker_id()
            
            # Insert log entry
            with psycopg2.connect(self.pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trial_logs 
                        (study_id, trial_number, component, timestamp, level, message, worker_node)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        self.study_id,
                        self.trial_number,
                        self.component,
                        datetime.fromtimestamp(record.created),
                        record.levelname,
                        record.getMessage(),
                        worker_id
                    ))
                    
        except Exception:
            # Don't fail trial execution due to logging errors
            # Could optionally write to stderr for debugging
            self.handleError(record)
    
    def _get_worker_id(self) -> str:
        """Get worker identifier (Ray node or local hostname)."""
        try:
            import ray
            if ray.is_initialized():
                return ray.get_runtime_context().get_node_id()
        except ImportError:
            pass
        
        # Fallback to hostname
        import socket
        return f"local_{socket.gethostname()}"


class LocalFileHandler(logging.Handler):
    """Log handler that writes to local files."""
    
    def __init__(self, study_id: int, trial_number: int, component: str, log_path: str):
        super().__init__()
        self.study_id = study_id
        self.trial_number = trial_number
        self.component = component
        
        # Create log file path
        self.log_file = Path(log_path) / f"study_{study_id}" / f"trial_{trial_number}" / f"{component}.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup formatter with timestamp and worker info
        formatter = logging.Formatter(
            '[%(asctime)s] [%(worker_id)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.setFormatter(formatter)
    
    def emit(self, record: logging.LogRecord):
        """Write log record to local file."""
        try:
            # Add worker ID to record
            record.worker_id = self._get_worker_id()
            
            # Format message
            message = self.format(record)
            
            # Append to log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
                f.flush()  # Ensure immediate write for real-time viewing
                
        except Exception:
            # Don't fail trial execution due to logging errors
            self.handleError(record)
    
    def _get_worker_id(self) -> str:
        """Get worker identifier (Ray node or local hostname)."""
        try:
            import ray
            if ray.is_initialized():
                node_id = ray.get_runtime_context().get_node_id()
                return node_id[:8]  # Shortened for readability
        except ImportError:
            pass
        
        # Fallback to hostname
        import socket
        return socket.gethostname()[:8]


# Backwards compatibility alias
NFSLogHandler = LocalFileHandler

class BufferedLogHandler(logging.Handler):
    """Buffered log handler for better performance with high-volume logging."""
    
    def __init__(self, target_handler: logging.Handler, buffer_size: int = 100):
        super().__init__()
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.buffer = []
        
    def emit(self, record: logging.LogRecord):
        """Buffer log records and flush when buffer is full."""
        self.buffer.append(record)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Flush buffered records to target handler."""
        if self.buffer:
            for record in self.buffer:
                self.target_handler.emit(record)
            self.buffer.clear()
    
    def close(self):
        """Close handler and flush remaining records."""
        self.flush()
        self.target_handler.close()
        super().close()