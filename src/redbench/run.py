import argparse
import os

from utils.load_and_preprocess_redset import NoDataInClusterError
from utils.log import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build workload with optional config path."
    )
    parser.add_argument(
        "--redset_path", type=str, required=True, help="Path to the redset file."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory."
    )

    # by default, we execute both strategies
    parser.add_argument(
        "--generation_strategy",
        nargs="+",
        choices=["matching", "generation"],
        default=["matching", "generation"],
        help="List of generation strategies to use: matching, generation. Default is both.",
    )

    # optional argument to specify the database or instance IDs
    parser.add_argument(
        "--instance_id",
        type=int,
        default=None,
        help="Instance ID to use. If not provided, will auto-determine a suitable and diverse instance ID.",
    )
    parser.add_argument(
        "--database_id",
        type=int,
        default=None,
        help="Database ID to use. If not provided, will auto-determine a suitable and diverse database ID.",
    )

    # overwrite existing workloads (with same config hash)
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Overwrite existing workloads with the same config hash.",
    )

    ###
    # Generator Specific Arguments
    ###
    parser.add_argument(
        "--config_path_generation",
        type=str,
        default=None,
        help="Path to the configuration file for the generator method. If not provided, defaults will be used.",
    )
    parser.add_argument(
        "--config_path_matching",
        type=str,
        default=None,
        help="Path to the configuration file for the wmatching method. If not provided, defaults will be used.",
    )

    ###
    # Print ASCII Art
    ###

    args = parser.parse_args()

    ascii_art = r"""
██████╗ ███████╗██████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██████╔╝█████╗  ██║  ██║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██╔══██╗██╔══╝  ██║  ██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██║  ██║███████╗██████╔╝██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═╝  ╚═╝╚══════╝╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
    """
    log(ascii_art)

    ###
    # Auto determine suitable and diverse cluster/database ID if not provided by user
    ###
    if args.instance_id is None or args.database_id is None:
        

        ###
        # Set a default cluster ID and database ID
        ###
        if 'provisioned' in args.redset_path.lower():
            # instance / database used in the paper experiments
            instance_id = 186
            database_id = 8
        else:
            # random database in serverless dataset
            instance_id = 0
            database_id = 0
        ####

        log(f"Selected cluster ID: {instance_id}")
        log(f"Selected database ID: {database_id}")
    else:
        instance_id = args.instance_id
        database_id = args.database_id

        log(f"Using provided cluster ID: {instance_id}")
        log(f"Using provided database ID: {database_id}")

    # make sure the redset path exists
    if not os.path.exists(args.redset_path):
        raise FileNotFoundError(
            f"The specified redset path does not exist: {args.redset_path}"
        )

    ###
    # Generate Workload
    ###
    for strategy in args.generation_strategy:
        log(f"Generating workload with strategy: >>{strategy}<<")
        try:
            if strategy == "matching":
                from redbench.matching.run import generate_workload

                generate_workload(
                    args.output_dir,
                    args.redset_path,
                    instance_id,
                    database_id,
                    args.config_path_matching,
                    args.overwrite_existing,
                )
            elif strategy == "generation":
                from redbench.generation.run import generate_workload

                generate_workload(
                    args.output_dir,
                    args.redset_path,
                    instance_id,
                    database_id,
                    args.config_path_generation,
                    args.overwrite_existing,
                )
            else:
                raise ValueError(
                    f"Unknown generation strategy: {strategy}. Choose from 'matching' or 'generation'."
                )
        except NoDataInClusterError as e:
            log(e, log_mode="error")
            import sys

            sys.exit(1)

    log("Workload generation completed successfully.")
