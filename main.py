#!/usr/bin/env python3
"""
Command-line interface for B-SOiD

TODO: Commands to implement:
    - clear logs
    - clear output folder (+ prompt for confirm)
"""

import argparse

import bsoid
from bsoid.pipeline import *  # This line is required for Streamlit to load Pipeline objects. Do not delete. For a more robust solution, consider: https://rebeccabilbro.github.io/module-main-has-no-attribute/

logger = bsoid.config.initialize_logger(__file__)


########################################################################################################################

bsoid_runtime_description = 'BSOID command line utility. Do BSOID stuff. Expand on this later.'

map_command_to_func = {

    # 'clean': bsoid.app.clear_output_folders,  # TODO: review clear output folders function for
    # 'cleanoutput': bsoid.app.clear_output_folders,
    'buildandrunlegacy': bsoid.main_LEGACY.test_function_to_build_then_run_py,
    'newbuild': bsoid.app.build_classifier_new_pipeline,
    'streamlit': bsoid.streamlit_bsoid.home,
    'test': lambda *args, **kwargs: print(args, kwargs),
}


########################################################################################################################

def parse_args() -> argparse.Namespace:
    """
    Instantiate arguments that will be parsed from the command-line here.
    Regarding HOW these commands will be carried out, implement that elsewhere.
    """
    # Instantiate parser, add arguments as expected on command-line
    parser = argparse.ArgumentParser(description=bsoid_runtime_description)
    parser.add_argument(f'command', help=f'HELP: TODO: command. Valid commands: '
                                         f'{[""+x for x in list(map_command_to_func.keys())]}')
    parser.add_argument('-p', help=f'HELP: TODO: PIPELINE LOC')
    # TODO: add commands, sub-commands as necessary

    # Parse args, return
    args: argparse.Namespace = parser.parse_args()

    # TODO: uncomment below later
    logger.debug(f'ARGS: {args}')
    logger.debug(f'args.command = {args.command}')
    logger.debug(f'args.p = {args.p}')

    return args


def execute_command(args: argparse.Namespace) -> None:
    kwargs = {}

    if args.p:
        kwargs['pipeline_path'] = args.p

        # TODO: uncoment below later any time
        logger.debug(f'main.py: arg.p parsed as: {args.p}')

    return map_command_to_func[args.command](**kwargs)


#### Stand-in functions ################################################################################################

def parse_args_using_sysargv() -> List[str]:
    """
    Stand-in function for parse_args().
    """

    # args = sys.argv
    # return args
    raise NotImplementedError(f'Deprecated')


def do_command_from_sysargv_parse(args: List[str]) -> None:
    """
    Stand-in function for do_command(). Because parsing functions using argparse is not complete/ready,
    we use this function for now to execute command-line commands
    """
    # kwargs = {}
    # if len(args) < 2:
    #     err = f'No command detected. Command-line args = {args}.'
    #     logger.error(err)
    #     raise NotImplementedError(err)
    #
    # cmd = args[1]
    # if cmd in map_command_to_func:
    #     if len(args) >= 3:
    #         kwargs['subcommand'] = args[2]
    #
    #     map_command_to_func[cmd](**kwargs)
    # else:
    #     err = f'Command was not found: `{cmd}` (args: {args}). ' \
    #           f'Check {os.path.abspath(__file__)} to see if application implemented.'
    #     logger.error(err)
    #     raise ValueError(err)
    raise NotImplementedError(f'Deprecated')


### Main execution #####################################################################################################

def main():
    ### Parse args
    args = parse_args()

    print(f'args: {args}')
    print(f'args.command: {args.command}')

    ### Do command
    execute_command(args)

    ### End
    pass


if __name__ == '__main__':
    main()
