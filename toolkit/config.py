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

def get_subreddits() -> list[str]:
    return get_config('subreddits')

def set_subreddits(subreddits: list[str]) -> None:
    set_config('subreddits', subreddits)

def get_score_weighting() -> bool:
    return get_config('score_weighting')

def set_score_weighting(score_weighting: bool) -> None:
    set_config('score_weighting', score_weighting)