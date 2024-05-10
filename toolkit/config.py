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

def get_profiles() -> dict[str, any]:
    with open(toolkit.get_dir() + '/src/profiles/profiles.json', 'r') as f:
        return json.load(f)
    
def set_profiles(profiles: dict[str, any]) -> None:
    with open(toolkit.get_dir() + '/src/profiles/profiles.json', 'w') as f:
        json.dump(profiles, f)

def increment_profiles_next_index() -> None:
    profiles = get_profiles()
    profiles['next_index'] += 1
    set_profiles(profiles)

def get_profile(ID: int) -> dict[str, any]:
    profiles = get_profiles()
    return profiles[str(ID)]

def set_profile(ID: int, profile: dict[str, any]) -> None:
    profiles = get_profiles()
    profiles[str(ID)] = profile
    set_profiles(profiles)

def del_profile(ID: int) -> None:
    profiles = get_profiles()
    del profiles[str(ID)]
    set_profiles(profiles)