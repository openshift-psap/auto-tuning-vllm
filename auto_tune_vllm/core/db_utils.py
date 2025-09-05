"""Database utilities for PostgreSQL database management."""

import logging
from typing import Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def parse_postgres_url(database_url: str) -> Tuple[str, str, str, int, str, str]:
    """
    Parse PostgreSQL URL and extract connection components.
    
    Args:
        database_url: PostgreSQL URL in format postgresql://user:password@host:port/database
        
    Returns:
        Tuple of (user, password, host, port, database_name, base_url_without_db)
        
    Raises:
        ValueError: If URL format is invalid
    """
    parsed = urlparse(database_url)
    
    if parsed.scheme not in ('postgresql', 'postgres'):
        raise ValueError(f"Invalid PostgreSQL URL scheme: {parsed.scheme}")
    
    user = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port or 5432
    database_name = parsed.path.lstrip('/')
    
    if not user or not host or not database_name:
        raise ValueError("PostgreSQL URL must include username, host, and database name")
    
    # Build base URL without database name for connecting to postgres system database
    password_part = f":{password}" if password else ""
    base_url = f"{parsed.scheme}://{user}{password_part}@{host}:{port}"
    
    return user, password, host, port, database_name, base_url


def create_database_if_not_exists(database_url: str) -> bool:
    """
    Create PostgreSQL database if it doesn't exist.
    
    Args:
        database_url: PostgreSQL URL for the target database
        
    Returns:
        True if database was created, False if it already existed
        
    Raises:
        Exception: If database creation fails
    """
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        raise ImportError("psycopg2 is required for PostgreSQL database operations")
    
    user, password, host, port, database_name, base_url = parse_postgres_url(database_url)
    
    # Connect to postgres system database to check if target database exists
    postgres_url = f"{base_url}/postgres"
    
    # Connect with autocommit mode to handle DDL operations
    conn = None
    try:
        conn = psycopg2.connect(postgres_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", 
                (database_name,)
            )
            
            if cur.fetchone():
                logger.info(f"Database '{database_name}' already exists")
                return False
            
            # Database doesn't exist, create it
            logger.info(f"Creating database '{database_name}'...")
            
            # Use identifier to safely quote database name
            cur.execute(
                f"CREATE DATABASE {psycopg2.extensions.quote_ident(database_name, cur)}"
            )
            
            logger.info(f"Successfully created database '{database_name}'")
            return True
            
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL error while creating database: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while creating database: {e}")
        raise
    finally:
        # Ensure connection is closed
        if conn:
            conn.close()


def verify_database_connection(database_url: str) -> bool:
    """
    Verify that we can connect to the database.
    
    Args:
        database_url: PostgreSQL URL for the target database
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        import psycopg2
    except ImportError:
        logger.debug("psycopg2 not installed; cannot verify database connection")
        return False

    try:
        with psycopg2.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except psycopg2.Error as e:
        logger.debug(f"Database connection failed: {e}")
        return False


def clear_study_data(study_name: str, database_url: str, clear_logs: bool = False, logs_database_url: str = None) -> dict:
    """
    Clear study-specific data from databases.
    
    Args:
        study_name: Name of the study to clear
        database_url: Database URL for Optuna study data
        clear_logs: Whether to also clear trial logs
        logs_database_url: Database URL for logs (if different from study database)
        
    Returns:
        Dict with success status and deletion counts
    """
    try:
        import psycopg2
    except ImportError:
        raise ImportError("psycopg2 is required for PostgreSQL database operations")
    
    result = {
        "success": False,
        "trials_deleted": 0,
        "logs_deleted": 0,
        "error": None
    }
    
    try:
        # Clear Optuna study data
        with psycopg2.connect(database_url) as conn:
            with conn.cursor() as cur:
                # Get study ID first
                cur.execute("SELECT study_id FROM studies WHERE study_name = %s", (study_name,))
                study_row = cur.fetchone()
                
                if not study_row:
                    result["error"] = f"Study '{study_name}' not found in database"
                    return result
                
                optuna_study_id = study_row[0]
                
                # Count trials before deletion
                cur.execute("SELECT COUNT(*) FROM trials WHERE study_id = %s", (optuna_study_id,))
                trials_count = cur.fetchone()[0]
                
                # Delete in order respecting foreign key constraints
                # Delete trial-related data first
                cur.execute("DELETE FROM trial_intermediate_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s)", (optuna_study_id,))
                cur.execute("DELETE FROM trial_system_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s)", (optuna_study_id,))
                cur.execute("DELETE FROM trial_user_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s)", (optuna_study_id,))
                cur.execute("DELETE FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s)", (optuna_study_id,))
                cur.execute("DELETE FROM trial_params WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = %s)", (optuna_study_id,))
                
                # Delete trials
                cur.execute("DELETE FROM trials WHERE study_id = %s", (optuna_study_id,))
                
                # Delete study
                cur.execute("DELETE FROM studies WHERE study_id = %s", (optuna_study_id,))
                
                result["trials_deleted"] = trials_count
                
                logger.info(f"Deleted {trials_count} trials and study '{study_name}' from Optuna database")
        
        # Clear trial logs if requested
        if clear_logs:
            logs_db_url = logs_database_url or database_url
            
            with psycopg2.connect(logs_db_url) as conn:
                with conn.cursor() as cur:
                    # Count logs before deletion
                    cur.execute("SELECT COUNT(*) FROM trial_logs WHERE study_name = %s", (study_name,))
                    logs_count = cur.fetchone()[0]
                    
                    # Delete trial logs
                    cur.execute("DELETE FROM trial_logs WHERE study_name = %s", (study_name,))
                    
                    result["logs_deleted"] = logs_count
                    
                    logger.info(f"Deleted {logs_count} log entries for study '{study_name}'")
        
        result["success"] = True
        return result
        
    except psycopg2.Error as e:
        error_msg = f"PostgreSQL error while clearing study data: {e}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result
    except Exception as e:
        error_msg = f"Unexpected error while clearing study data: {e}"
        logger.error(error_msg)
        result["error"] = error_msg
        return result
