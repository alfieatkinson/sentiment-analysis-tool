import PySimpleGUI as ps

#import toolkit

def settings() -> None:
    pass

def main() -> None:
    layout = [[ps.Text("Hello World!")], [ps.Button("OK")]]
    window = ps.Window("Demo", layout)

    while True:
        event, values = window.read()
        if event == "OK" or event == ps.WIN_CLOSED:
            break

#main()