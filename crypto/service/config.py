"""
Configuration management using Pydantic Settings

Environment variables:
- NOVA_DB_PATH: Path to DuckDB database (required)
- NOVA_CACHE_TTL: Cache TTL in seconds (default: 60)
- NOVA_CACHE_CAPACITY: Cache capacity (default: 2048)
- NOVA_LOG_LEVEL: Logging level (default: INFO)
"""

from pydantic import BaseSettings, Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    db_path: str = Field(
        ...,
        env="NOVA_DB_PATH",
        description="Path to DuckDB database file"
    )
    
    # Cache
    cache_ttl: int = Field(
        60,
        env="NOVA_CACHE_TTL",
        ge=1,
        le=3600,
        description="Cache TTL in seconds"
    )
    
    cache_capacity: int = Field(
        2048,
        env="NOVA_CACHE_CAPACITY",
        ge=64,
        le=10000,
        description="Maximum cache entries"
    )
    
    # Logging
    log_level: str = Field(
        "INFO",
        env="NOVA_LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    log_jsonl_path: Optional[str] = Field(
        None,
        env="NOVA_LOG_JSONL",
        description="Path to JSONL metrics log (optional)"
    )
    
    # API
    chain_id: int = Field(
        11155111,
        env="NOVA_CHAIN_ID",
        description="EVM chain ID (default: Sepolia)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

