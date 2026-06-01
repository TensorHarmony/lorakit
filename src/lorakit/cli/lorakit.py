import argparse

from lorakit.utils import get_job


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "configs", nargs="+", type=str, help="Name of or or more config files to run sequentially"
    )

    args = parser.parse_args()

    configs = args.configs
    if len(configs) == 0:
        raise ValueError(
            "No configuration files provided. Please specify at least one config file."
        )

    print(f"Starting {len(configs)} job(s)")

    for config in configs:
        print(f"Starting job {config}")
        try:
            # Run the job
            job = get_job(config)
            job.run()
            print(f"Job {config} completed successfully")
        except Exception as e:
            import traceback

            print(f"Job {config} failed with error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
