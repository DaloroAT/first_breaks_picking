import click

from first_breaks.desktop.main_gui import run_app

cli_commands = click.Group()

help_desktop_app = f"Launch desktop application for picking. 'app' and 'desktop' are aliases"


@cli_commands.command(help=help_desktop_app)
def app():
    run_app()


@cli_commands.command(help=help_desktop_app)
def desktop():
    run_app()



