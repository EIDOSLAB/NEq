import argparse

import experiments


def run_experiment(experiment, opts):
    experiment = experiment(opts)
    experiment.initialize()
    experiment.run()
    
    print(f"{experiment} over")


def main(opts):
    # Get the experiment using the name given as positional argument
    Experiment = experiments.get_experiment_by_name(opts.experiment)
    print(f'Got experiment: {Experiment}')
    
    # Initialize and run the experiment
    run_experiment(Experiment, opts)


if __name__ == '__main__':
    # Initialize parent parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add experiment-specific subparser used to get the experiment's name
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    
    for experiment in experiments.__all__:
        # For each available experiment add the experiment's parser
        subparser = subparsers.add_parser(experiment.__name__, help=experiment.__help__,
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        experiment.load_config(subparser)
    
    # Parse the arguments
    opts = parser.parse_args()
    
    # Launch the experiment
    main(opts)
