from typing import Dict, Any
import logging
from logging.config import dictConfig
from typing import Optional
from pathlib import Path
import json
from config import settings

def configure_logging(log_config_path: Optional[str] = None) -> None:
    """Configure application logging"""
    default_config : Dict[str, Any]  = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': True
            },
            'app': {
                'handlers': ['console'],
                'level': 'DEBUG' if settings.DEBUG else 'INFO',
                'propagate': False
            },
        }
    }

    try:
        if log_config_path and Path(log_config_path).exists():
            with open(log_config_path) as f:
                config = json.load(f)
            dictConfig(config)
        else:
            dictConfig(default_config)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error(
            f"Failed to configure logging: {str(e)}", 
            exc_info=True
        )