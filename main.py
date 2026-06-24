from src.experiments.runner import ExperimentRunner


def main():

    runner = ExperimentRunner(
        N=6,
        T=3,
        sigma=2.0,
        n=2.2
    )

    print("Running batch experiments...")

    df = runner.run_batch(L=50)  # 50 Monte Carlo runs

    runner.save(df)

    print("\nDONE")
    print(df.head())

    print("\nSTATS:")
    print(df["error"].describe())


if __name__ == "__main__":
    main()