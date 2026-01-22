"""
配置加载模块
============

本模块负责加载项目配置和设置环境变量。
使用 OmegaConf 处理 YAML 配置文件，并支持 .env 文件加载。

主要功能：
1. setup_environment: 自动从 .env 文件加载环境变量。
2. load_config: 从 YAML 文件加载配置。
3. get_config: 获取全局配置对象（单例模式）。
4. get_llm_config: 获取特定 LLM 模型的配置。

作者: AI Scientist Team
日期: 2025-01-22
"""

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
    设置环境变量。

    自动加载 .env 文件中的环境变量。
    如果文件不存在，将记录警告并使用系统环境变量。
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
    加载配置文件。

    Args:
        config_path (str, optional): 配置文件路径。
                                     如果为 None，默认为项目根目录下的 'bfts_config.yaml'。

    Returns:
        OmegaConf: 加载的配置对象。

    Raises:
        FileNotFoundError: 如果配置文件不存在。
        RuntimeError: 如果加载配置文件失败。
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
    获取全局配置对象。

    如果尚未加载配置，则尝试加载默认配置。

    Returns:
        OmegaConf: 配置对象。
    """
    global _CONFIG
    if _CONFIG is None:
        load_config()
    return _CONFIG

def get_llm_config(model_name=None):
    """
    获取特定 LLM 模型的配置。

    Args:
        model_name (str, optional): 模型名称。如果为 None，则使用配置中的默认模型。

    Returns:
        dict: 模型配置字典。

    Raises:
        ValueError: 如果未提供模型名称且未配置默认模型，或者模型在配置中未定义。
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
