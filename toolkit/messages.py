import toolkit

def alert(message):
    print(toolkit.time(), message)

def error(message):
    print(toolkit.time(), message)

def console(message: str) -> None:
    print(f"{toolkit.time()}: {message}")