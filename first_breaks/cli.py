import click

from first_breaks.desktop.main_gui import run_app

cli_commands = click.Group()

help_desktop_app = "Launch desktop application for picking. 'app' and 'desktop' are aliases"


@cli_commands.command(help=help_desktop_app)
def app() -> None:
    run_app()


@cli_commands.command(help=help_desktop_app)
def desktop() -> None:
    run_app()
