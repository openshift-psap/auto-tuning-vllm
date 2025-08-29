"""Centralized logging management."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .handlers import PostgreSQLLogHandler, LocalFileHandler

logger = logging.getLogger(__name__)


class CentralizedLogger:
    """Central logging manager with PostgreSQL and optional file support."""
    
    def __init__(
        self, 
        study_id: int, 
        pg_url: Optional[str] = None, 
        file_path: Optional[str] = None, 
        log_level: str = "INFO"
    ):
        self.study_id = study_id
        self.pg_url = pg_url
        self.file_path = file_path
        self.log_level = getattr(logging, log_level.upper())
        
        # Setup database schema if PostgreSQL is configured
        if self.pg_url:
            self._setup_log_tables()
    
    def _setup_log_tables(self):
        """Create log tables if they don't exist."""
        try:
            import psycopg2
            
            with psycopg2.connect(self.pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS trial_logs (
                            id SERIAL PRIMARY KEY,
                            study_id INTEGER NOT NULL,
                            trial_number INTEGER NOT NULL,
                            component VARCHAR(50) NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            level VARCHAR(10) NOT NULL,
                            message TEXT NOT NULL,
                            worker_node VARCHAR(100) NOT NULL
                        )
                    """)
                    
                    # Create indexes for efficient querying
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_trial_logs_study_trial 
                        ON trial_logs (study_id, trial_number, id)
                    """)
                    
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_trial_logs_study_trial_component
                        ON trial_logs (study_id, trial_number, component)
                    """)
            
            logger.info(f"Log tables ready for study {self.study_id}")
            
        except Exception as e:
            logger.warning(f"Failed to setup log tables: {e}")
    
    def get_trial_logger(self, trial_number: int, component: str) -> logging.Logger:
        """Get logger for specific trial and component."""
        logger_name = f"study_{self.study_id}.trial_{trial_number}.{component}"
        trial_logger = logging.getLogger(logger_name)
        
        # Only add handlers if not already configured
        if not trial_logger.handlers:
            # PostgreSQL handler (if configured)
            if self.pg_url:
                pg_handler = PostgreSQLLogHandler(
                    self.study_id, trial_number, component, self.pg_url
                )
                trial_logger.addHandler(pg_handler)
            
            # File handler (if configured)
            if self.file_path:
                file_handler = LocalFileHandler(
                    self.study_id, trial_number, component, self.file_path
                )
                trial_logger.addHandler(file_handler)
            
            # Fallback to console if no handlers configured
            if not self.pg_url and not self.file_path:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s'
                ))
                trial_logger.addHandler(console_handler)
            
            trial_logger.setLevel(self.log_level)
            trial_logger.propagate = False  # Don't propagate to root logger
        
        return trial_logger
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log entries (optional maintenance)."""
        try:
            import psycopg2
            
            with psycopg2.connect(self.pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM trial_logs 
                        WHERE study_id = %s 
                        AND timestamp < NOW() - INTERVAL '%s days'
                    """, (self.study_id, days_to_keep))
                    
                    deleted_count = cur.rowcount
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old log entries")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup old logs: {e}")


class LogStreamer:
    """Live log streaming from PostgreSQL."""
    
    def __init__(self, study_id: int, pg_url: str):
        self.study_id = study_id
        self.pg_url = pg_url
        self._ensure_log_tables_exist()
    
    def _ensure_log_tables_exist(self):
        """Ensure log tables exist, create them if they don't."""
        try:
            import psycopg2
            
            with psycopg2.connect(self.pg_url) as conn:
                with conn.cursor() as cur:
                    # Check if table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'trial_logs'
                        )
                    """)
                    
                    table_exists = cur.fetchone()[0]
                    
                    if not table_exists:
                        logger.info("Log tables don't exist, creating them...")
                        
                        # Create table using the same logic as CentralizedLogger
                        cur.execute("""
                            CREATE TABLE trial_logs (
                                id SERIAL PRIMARY KEY,
                                study_id INTEGER NOT NULL,
                                trial_number INTEGER NOT NULL,
                                component VARCHAR(50) NOT NULL,
                                timestamp TIMESTAMP NOT NULL,
                                level VARCHAR(10) NOT NULL,
                                message TEXT NOT NULL,
                                worker_node VARCHAR(100) NOT NULL
                            )
                        """)
                        
                        # Create indexes for efficient querying
                        cur.execute("""
                            CREATE INDEX idx_trial_logs_study_trial 
                            ON trial_logs (study_id, trial_number, id)
                        """)
                        
                        cur.execute("""
                            CREATE INDEX idx_trial_logs_study_trial_component
                            ON trial_logs (study_id, trial_number, component)
                        """)
                        
                        logger.info("Log tables created successfully")
                        
        except Exception as e:
            logger.warning(f"Failed to ensure log tables exist: {e}")
    
    async def stream_trial_logs(
        self, 
        trial_number: int, 
        component: Optional[str] = None,
        follow: bool = True,
        tail_lines: int = 100
    ):
        """Stream logs for a specific trial."""
        logger.info(f"Streaming logs for study {self.study_id}, trial {trial_number}" +
                   (f", component {component}" if component else ""))
        
        try:
            # First, fetch and display the last N lines (tail behavior)
            if follow:
                recent_logs = self._fetch_recent_logs(trial_number, component, tail_lines)
                for log_entry in recent_logs:
                    log_id, timestamp, level, comp, message, worker = log_entry
                    formatted_log = (
                        f"[{timestamp}] [{worker[:8]}] {comp}.{level}: {message}"
                    )
                    print(formatted_log)
                
                # Set last_seen_id to the highest ID from recent logs
                last_seen_id = max((log[0] for log in recent_logs), default=0)
            else:
                # For non-follow mode, start from beginning
                last_seen_id = 0
            
            # Now start the follow loop
            while True:
                new_logs = self._fetch_new_logs(trial_number, component, last_seen_id)
                
                for log_entry in new_logs:
                    log_id, timestamp, level, comp, message, worker = log_entry
                    
                    # Format log entry for display
                    formatted_log = (
                        f"[{timestamp}] [{worker[:8]}] {comp}.{level}: {message}"
                    )
                    print(formatted_log)
                    
                    last_seen_id = max(last_seen_id, log_id)
                
                if not follow and not new_logs:
                    break
                
                # Poll every 2 seconds
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("Log streaming stopped by user")
        except Exception as e:
            logger.error(f"Log streaming failed: {e}")
    
    def _fetch_recent_logs(
        self,
        trial_number: int,
        component: Optional[str],
        limit: int = 100
    ) -> list:
        """Fetch the most recent logs for a trial (tail behavior)."""
        try:
            import psycopg2
            
            with psycopg2.connect(self.pg_url) as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT id, timestamp, level, component, message, worker_node
                        FROM trial_logs 
                        WHERE study_id = %s AND trial_number = %s
                    """
                    params = [self.study_id, trial_number]
                    
                    if component:
                        query += " AND component = %s"
                        params.append(component)
                    
                    query += " ORDER BY id DESC LIMIT %s"
                    params.append(limit)
                    
                    cur.execute(query, params)
                    # Reverse to get chronological order (oldest to newest)
                    return list(reversed(cur.fetchall()))
                    
        except Exception as e:
            logger.error(f"Failed to fetch recent logs: {e}")
            return []

    def _fetch_new_logs(
        self, 
        trial_number: int, 
        component: Optional[str], 
        last_seen_id: int
    ) -> list:
        """Fetch new log entries from database."""
        try:
            import psycopg2
            
            with psycopg2.connect(self.pg_url) as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT id, timestamp, level, component, message, worker_node
                        FROM trial_logs 
                        WHERE study_id = %s AND trial_number = %s AND id > %s
                    """
                    params = [self.study_id, trial_number, last_seen_id]
                    
                    if component:
                        query += " AND component = %s"
                        params.append(component)
                    
                    query += " ORDER BY id ASC LIMIT 100"  # Batch fetch
                    
                    cur.execute(query, params)
                    return cur.fetchall()
                    
        except Exception as e:
            logger.error(f"Failed to fetch logs: {e}")
            return []
    
    async def stream_study_logs(self, follow: bool = True, tail_lines: int = 100):
        """Stream all logs for the entire study."""
        logger.info(f"Streaming all logs for study {self.study_id}")
        
        try:
            # First, fetch and display the last N lines (tail behavior)
            if follow:
                recent_logs = self._fetch_recent_study_logs(tail_lines)
                for log_entry in recent_logs:
                    log_id, timestamp, level, trial_num, comp, message, worker = log_entry
                    formatted_log = (
                        f"[{timestamp}] [T{trial_num}] [{worker[:8]}] {comp}.{level}: {message}"
                    )
                    print(formatted_log)
                
                # Set last_seen_id to the highest ID from recent logs
                last_seen_id = max((log[0] for log in recent_logs), default=0)
            else:
                # For non-follow mode, start from beginning
                last_seen_id = 0
            
            # Now start the follow loop
            while True:
                new_logs = self._fetch_study_logs(last_seen_id)
                
                for log_entry in new_logs:
                    log_id, timestamp, level, trial_num, comp, message, worker = log_entry
                    
                    formatted_log = (
                        f"[{timestamp}] [T{trial_num}] [{worker[:8]}] {comp}.{level}: {message}"
                    )
                    print(formatted_log)
                    
                    last_seen_id = max(last_seen_id, log_id)
                
                if not follow and not new_logs:
                    break
                
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("Study log streaming stopped by user")
        except Exception as e:
            logger.error(f"Study log streaming failed: {e}")
    
    def _fetch_recent_study_logs(self, limit: int = 100) -> list:
        """Fetch the most recent logs for the entire study (tail behavior)."""
        try:
            import psycopg2
            
            with psycopg2.connect(self.pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, timestamp, level, trial_number, component, message, worker_node
                        FROM trial_logs 
                        WHERE study_id = %s
                        ORDER BY id DESC LIMIT %s
                    """, (self.study_id, limit))
                    
                    # Reverse to get chronological order (oldest to newest)
                    return list(reversed(cur.fetchall()))
                    
        except Exception as e:
            logger.error(f"Failed to fetch recent study logs: {e}")
            return []

    def _fetch_study_logs(self, last_seen_id: int) -> list:
        """Fetch new log entries for entire study."""
        try:
            import psycopg2
            
            with psycopg2.connect(self.pg_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, timestamp, level, trial_number, component, message, worker_node
                        FROM trial_logs 
                        WHERE study_id = %s AND id > %s
                        ORDER BY id ASC LIMIT 100
                    """, (self.study_id, last_seen_id))
                    
                    return cur.fetchall()
                    
        except Exception as e:
            logger.error(f"Failed to fetch study logs: {e}")
            return []