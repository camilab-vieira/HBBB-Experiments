import os
import ast
import configparser

def load_config(file_path: str, section: str) -> dict:
    """Load configuration parameters with support for complex structures (e.g., lists, dicts)."""
    
    # Check if the config file exists
    if os.path.exists(file_path):
        config = configparser.ConfigParser()
        config.read(file_path)

        # Check if the specified section exists
        if config.has_section(section):
            config_dict = {key: value for key, value in config.items(section)}
            
            # Parse complex structures like lists or dicts
            for key, value in config_dict.items():
                try:
                    # Attempt to evaluate the value as a literal (e.g., list, dict)
                    config_dict[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # If evaluation fails, leave the value as a string
                    continue
            
            return config_dict
        else:
            raise ValueError(f"Section '{section}' not found in config file.")
    else:
        raise ValueError("Config file not found.")