from experiments.templates.learning import LearningExperiment


class ClassificationLearningExperiment(LearningExperiment):
    def __init__(self, opts):
        super().__init__(opts)
        self.project_name = 'classification-learning'
