import torch

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class LogisticRegression:
    def __init__(self, random_state: int):
        self._weights: torch.Tensor = None
        self.random_state: int = random_state

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float,
        epochs: int,
    ):
        """
        Train the logistic regression model using pre-processed features and labels.

        Args:
            features (torch.Tensor): The bag of words representations of the training examples.
            labels (torch.Tensor): The target labels.
            learning_rate (float): The learning rate for gradient descent.
            epochs (int): The number of iterations over the training dataset.

        Returns:
            None: The function updates the model weights in place.
        """
        # TODO: Implement gradient-descent algorithm to optimize logistic regression weights
        self.weights = self.initialize_parameters(features.shape[1], self.random_state)
        for e in range(epochs):
            predictions = self.predict_proba(features)
            ce_loss = self.binary_cross_entropy_loss(predictions, labels)
            grad_weights = torch.zeros(features.shape[1],)
            grad_bias = torch.zeros(1,)

            for i in range(len(features)):
                grad_weights += ((predictions[i]-labels[i])*features[i])/features.shape[0]
                grad_bias += (predictions[i]-labels[i])/features.shape[0]
            
            gradient = torch.cat((grad_weights,grad_bias))
            self.weights -= learning_rate*gradient

            print(ce_loss)
            
        return

    def predict(self, features: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
        """
        Predict class labels for given examples based on a cutoff threshold.

        Args:
            features (torch.Tensor): The bag of words representations of the input examples.
            cutoff (float): The threshold for classifying a sample as positive. Defaults to 0.5.

        Returns:
            torch.Tensor: Predicted class labels (0 or 1).
        """
        decisions: torch.Tensor = []
        probabilities = self.predict_proba(features)

        for p in list(probabilities):
            if  p >= 0.5:
                decisions.append(1)
            else:
                decisions.append(0)
        
        decisions = torch.tensor(decisions)

        return decisions

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predicts the probability of each sample belonging to the positive class using pre-processed features.

        Args:
            features (torch.Tensor): The bag of words representations of the input examples.

        Returns:
            torch.Tensor: A tensor of probabilities for each input sample being in the positive class.

        Raises:
            ValueError: If the model weights are not initialized (model not trained).
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call the 'train' method first.")
        
        probabilities: torch.Tensor = []

        for f in list(features):
            b = self.weights[-1]
            z = self.weights[:-1].dot(f.float())+b
            result = self.sigmoid(z)
            probabilities.append(result)

        probabilities = torch.tensor(probabilities)
        
        return probabilities

    def initialize_parameters(self, dim: int, random_state: int) -> torch.Tensor:
        """
        Initialize the weights for logistic regression using a normal distribution.

        This function initializes the weights (and bias as the last element) with values drawn from a normal distribution.
        The use of random weights can help in breaking the symmetry and improve the convergence during training.

        Args:
            dim (int): The number of features (dimension) in the input data.
            random_state (int): A seed value for reproducibility of results.

        Returns:
            torch.Tensor: Initialized weights as a tensor with size (dim + 1,).
        """
        torch.manual_seed(random_state)
        
        params: torch.Tensor = None
        params = torch.randn((dim + 1,))*0.01
        return params

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        """
        Compute the sigmoid of z.

        This function applies the sigmoid function, which is defined as 1 / (1 + exp(-z)).
        It is used to map predictions to probabilities in logistic regression.

        Args:
            z (torch.Tensor): A tensor containing the linear combination of weights and features.

        Returns:
            torch.Tensor: The sigmoid of z.
        """
        result: torch.Tensor = torch.tensor(1/(1+torch.exp(-z)))
        return result

    @staticmethod
    def binary_cross_entropy_loss(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss.

        The binary cross-entropy loss is a common loss function for binary classification. It calculates the difference
        between the predicted probabilities and the actual labels.

        Args:
            predictions (torch.Tensor): Predicted probabilities from the logistic regression model.
            targets (torch.Tensor): Actual labels (0 or 1).

        Returns:
            torch.Tensor: The computed binary cross-entropy loss.
        """
        ce_loss: torch.Tensor = None
        N = len(predictions)
        tensor_suma = targets*torch.log(predictions)+(1-targets)*torch.log(1-predictions)
    
        ce_loss = torch.tensor(-(1/N)*torch.sum(tensor_suma))
        return ce_loss
    

    @property
    def weights(self):
        """Get the weights of the logistic regression model."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Set the weights of the logistic regression model."""
        self._weights: torch.Tensor = value

