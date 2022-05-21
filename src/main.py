import argparse

import experiments


def main(opts):
    Experiment = experiments.get_experiment_by_name(opts.experiment)
    print('Got experiment:', Experiment)
    
    experiment = Experiment(opts)
    experiment.initialize()
    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    for experiment in experiments.__all__:
        subparser = subparsers.add_parser(experiment.__name__, help=experiment.__help__,
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        experiment.load_config(subparser)
    
    opts = parser.parse_args()
    main(opts)
