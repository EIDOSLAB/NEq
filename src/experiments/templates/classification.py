from experiments.templates.learning import LearningExperiment


# TODO is this needed?

class ClassificationLearningExperiment(LearningExperiment):
    
    def __init__(self, opts):
        super().__init__(opts)
        self.project_name = 'NEq'
        self.tag = "classification-base"
    
    def get_experiment_key(self):
        raise NotImplementedError
    
    def load_model(self):
        raise NotImplementedError
    
    def load_optimizer(self):
        raise NotImplementedError
    
    def load_scheduler(self):
        raise NotImplementedError
    
    def compute_loss(self, outputs, minibatch):
        raise NotImplementedError
    
    def init_meters(self):
        raise NotImplementedError
    
    def update_meters(self, meters, outputs, minibatch):
        raise NotImplementedError
    
    def optimize_metric(self):
        raise NotImplementedError
    
    def load_data(self):
        raise NotImplementedError
