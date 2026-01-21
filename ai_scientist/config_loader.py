import os
import sys
from omegaconf import OmegaConf
from dotenv import load_dotenv

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache for configuration
_CONFIG = None

def setup_environment():
    """
    Load environment variables from .env file automatically.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".env")
    
    if not os.path.exists(env_path):
        logger.warning(f"Environment file not found at {env_path}. System environment variables will be used.")
        return

    try:
        # load_dotenv loads variables from .env file into os.environ
        # override=True ensures .env values overwrite existing system variables if conflict
        # verbose=True allows python-dotenv to output warnings about parsing errors
        load_dotenv(dotenv_path=env_path, override=True, verbose=True)
        logger.info(f"Successfully loaded environment variables from {env_path}")
    except Exception as e:
        logger.error(f"Failed to load environment variables from {env_path}: {e}")

# Automatically load environment variables when module is imported
setup_environment()

def load_config(config_path=None):
    """
    Load the project configuration from a YAML file.
    
    Args:
        config_path (str, optional): Path to the configuration file. 
                                     If None, defaults to 'bfts_config.yaml' in the project root.
    
    Returns:
        OmegaConf: The loaded configuration object.
    """
    global _CONFIG
    
    # Environment variables are already loaded by setup_environment() on import
    # But we can re-ensure it if needed, though module level execution is standard.
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if config_path is None:
        # Default to bfts_config.yaml in project root
        config_path = os.path.join(project_root, "bfts_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    try:
        conf = OmegaConf.load(config_path)
        # Resolve interpolations immediately to catch errors early
        OmegaConf.resolve(conf)
        _CONFIG = conf
        return conf
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

def get_config():
    """
    Get the cached configuration object. Loads it if not already loaded.
    """
    global _CONFIG
    if _CONFIG is None:
        load_config()
    return _CONFIG

def get_llm_config(model_name=None):
    """
    Helper to get LLM configuration for a specific model.
    
    Args:
        model_name (str, optional): The model name. If None, uses the default model from config.
        
    Returns:
        dict: The configuration dictionary for the model.
    """
    config = get_config()
    llm_config = config.get("llm_config", {})
    
    if model_name is None:
        model_name = llm_config.get("default_model")
        
    if not model_name:
        raise ValueError("No model name provided and no default model configured.")
        
    models = llm_config.get("models", {})
    model_conf = models.get(model_name)
    
    if not model_conf:
        # Check for provider defaults to handle dynamic model names
        defaults = llm_config.get("defaults", {})
        
        # Heuristic for Ollama models
        if model_name.startswith("ollama/") and "ollama" in defaults:
             # Create a config based on the default template
             conf = defaults["ollama"].copy()
             # We can't easily deep copy OmegaConf node with .copy(), so we use container
             conf = OmegaConf.to_container(conf, resolve=True)
             conf["model"] = model_name
             return conf
             
        raise ValueError(f"Model '{model_name}' is not defined in bfts_config.yaml")
        
    # Return a copy to avoid mutation issues
    # Add the model name itself to the config
    conf_dict = OmegaConf.to_container(model_conf, resolve=True)
    conf_dict["model"] = model_name
    return conf_dict
