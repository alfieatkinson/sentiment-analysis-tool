import json

import toolkit

def get_config(key: str) -> any:
    with open(toolkit.get_dir() + '/src/config.json', 'r') as f:
        return json.load(f)[key]
    
def set_config(key: str, value: any) -> None:
    with open(toolkit.get_dir() + '/src/config.json', 'r') as f:
        config = json.load(f)

    config[key] = value

    with open(toolkit.get_dir() + '/src/config.json', 'w') as f:
        json.dump(config, f)

def get_profiles() -> dict[str, dict[str, str]]:
    with open(toolkit.get_dir() + '/src/profiles.json', 'r') as f:
        return json.load(f)
    
def get_profile(key: str) -> dict[str, str]:
    with open(toolkit.get_dir() + '/src/profiles.json', 'r') as f:
        return json.load(f)[key]
