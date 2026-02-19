from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    ch_host: str
    ch_port: int
    ch_protocol: str
    ch_database: str
    ch_user: str
    ch_password: str
    ch_verify_certificate: bool = False
    ch_ca_cert: str | None = None
    poll_interval_sec: float = 5.0

    # Pydantic settings config
    model_config = {
        "env_file": Path("/home/ubuntu/monitoring-observability/.env"),  # loads .env from current working directory
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_prefix": "",  # no prefix; use exact variable names
    }


settings = Settings()
