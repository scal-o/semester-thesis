import click
from static_assignment import conv_to_aequilibrae, sta_run, sta_test


@click.group("sta")
def cli():
    pass


cli.add_command(conv_to_aequilibrae.convert_to_omx)
cli.add_command(sta_run.run_sta)
cli.add_command(sta_test.test_sta)
