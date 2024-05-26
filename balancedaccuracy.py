import torch

# NOTE: This will be the calculation of balanced accuracy for your classification task
# The balanced accuracy is defined as the average accuracy for each class.
# The accuracy for an indiviual class is the ratio between correctly classified example to all examples of that class.
# The code in train.py will instantiate one instance of this class.
# It will call the reset method at the beginning of each epoch. Use this to reset your
# internal states. The update method will be called multiple times during an epoch, once for each batch of the training.
# You will receive the network predictions, a Tensor of Size (BATCHSIZExCLASSES) containing the logits (output without Softmax).
# You will also receive the groundtruth, an integer (long) Tensor with the respective class index per example.
# For each class, count how many examples were correctly classified and how many total examples exist.
# Then, in the getBACC method, calculate the balanced accuracy by first calculating each individual accuracy
# and then taking the average.

# Balanced Accuracy
"""
class BalancedAccuracy:
    def __init__(self, nClasses):
        # TODO: Setup internal variables
        # NOTE: It is good practive to all reset() from here to make sure everything is properly initialized
        pass
    def reset(self):
        # TODO: Reset internal states.
        # Called at the beginning of each epoch
        pass
    def update(self, predictions, groundtruth):
        # TODO: Implement the update of internal states
        # based on current network predictios and the groundtruth value.
        #
        # Predictions is a Tensor with logits (non-normalized activations)
        # It is a BATCH_SIZE x N_CLASSES float Tensor. The argmax for each samples
        # indicated the predicted class.
        #
        # Groundtruth is a BATCH_SIZE x 1 long Tensor. It contains the index of the
        # ground truth class.
        pass
    def getBACC(self):
        # TODO: Calculcate and return balanced accuracy 
        # based on current internal state
        pass
    """


class BalancedAccuracy:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.correct_predictions = [0] * nClasses  # list for the correct predictions
        self.total_examples = [0] * nClasses  # total number of examples

    def reset(self):  # resets the internal state of the class
        self.correct_predictions = [0] * self.nClasses
        self.total_examples = [0] * self.nClasses

    def update(self, predictions, groundtruth):
        # calculates the predicted class for each example
        predicted_classes = torch.argmax(predictions, dim=1)
        for pred, gt in zip(predicted_classes, groundtruth):
            if (
                pred == gt
            ):  # updates the correct_predictions if the prediction matches the ground truth
                self.correct_predictions[gt] += 1
            self.total_examples[gt] += 1  # updates total_examples

    def getBACC(self):
        # calculates indovidual accuracies for each class (correct predictions / total number of examples)
        individual_accuracies = []
        for i in range(self.nClasses):
            if self.total_examples[i] == 0:
                individual_accuracies.append(0)
            else:
                individual_accuracies.append(
                    self.correct_predictions[i] / self.total_examples[i]
                )
        balanced_accuracy = sum(individual_accuracies) / self.nClasses
        return balanced_accuracy
