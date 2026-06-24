from src.experiments.runner import ExperimentRunner


ANCHOR_COUNT = 6
TARGET_COUNT = 3
RUN_COUNT = 500
NOISE_STD = 2.0
PATH_LOSS_EXPONENT = 2.2


def main():
    runner = ExperimentRunner(
        anchor_count=ANCHOR_COUNT,
        target_count=TARGET_COUNT,
        noise_std=NOISE_STD,
        path_loss_exponent=PATH_LOSS_EXPONENT,
    )

    print("Running batch experiments...")

    results = runner.run_batch(run_count=RUN_COUNT)

    runner.save(results)

    print("\nDONE")
    print(results.head())

    print("\nSTATS:")
    print(results["error"].describe())


if __name__ == "__main__":
    main()
