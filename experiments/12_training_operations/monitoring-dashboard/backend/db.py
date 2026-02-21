import urllib3
import clickhouse_connect
from clickhouse_connect.driver.client import Client

from config import settings

# Suppress SSL verification warnings for self-signed certificates
if not settings.ch_verify_certificate:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        client_kwargs = {
            "host": settings.ch_host,
            "port": settings.ch_port,
            "interface": settings.ch_protocol,
            "database": settings.ch_database,
            "username": settings.ch_user,
            "password": settings.ch_password,
        }
        # Pass SSL verification setting
        if settings.ch_ca_cert:
            client_kwargs["ca_cert"] = settings.ch_ca_cert
        else:
            # Disable SSL verification for self-signed certificates
            client_kwargs["verify"] = settings.ch_verify_certificate
        _client = clickhouse_connect.get_client(**client_kwargs)
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
