import argparse
from src.experiment_manager import ExperimentManager
from src.utils.config_loader import ConfigLoader


def main():
    parser = argparse.ArgumentParser(description='Run RL Experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    config = ConfigLoader.load(args.config)
    experiment = ExperimentManager(config)
    experiment.run()


if __name__ == '__main__':
    main()
