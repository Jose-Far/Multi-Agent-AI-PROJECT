"""
database — PhishDec SQLite Persistence Layer
============================================
Provides relational persistence and historical investigation tracking
for the Multi-Agent Cybersecurity Platform.
"""

from .database import get_connection, initialize_database, get_db_path
from .repository import AnalysisRepository

__all__ = [
    "get_connection",
    "initialize_database",
    "get_db_path",
    "AnalysisRepository",
]
