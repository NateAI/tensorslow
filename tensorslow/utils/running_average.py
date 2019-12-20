class RunningAverage:
    """Utility class to keep running average of performance accross epoch"""
    def __init__(self):

        self._performance_dict_average = {}
        self._num_examples_so_far = 0

    @property
    def average(self):
        return self._performance_dict_average

    @average.setter
    def average(self, value):
        raise Exception('Cannot edit the value of average directly - use the update method')

    def update(self, performance_dict, num_examples):
        """
        Update running average
        Parameters
        ----------
        performance_dict: dict
            dict mapping metric names to score (output of model.evaluate()) e.g. {'loss': 1.23, ..}
        num_examples: int
            number of examples in batch that was evaluated on to give this performance_dict

        """

        if not self._performance_dict_average:
            self._performance_dict_average = performance_dict
        else:
            for metric, current_avg in self._performance_dict_average.items():
                new_avg = ((current_avg * self._num_examples_so_far) + (performance_dict[metric] * num_examples)) / \
                          (self._num_examples_so_far + num_examples)

                self._performance_dict_average[metric] = new_avg

        self._num_examples_so_far += num_examples