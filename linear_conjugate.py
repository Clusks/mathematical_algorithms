import numpy as np

class linearConjugate(object):
    """
    Goal is to build linear conjugate gradient based on the description here:
    https://en.wikipedia.org/wiki/Conjugate_gradient_method

    Values are stored in lists for a better understanding of what the process is doing.

    Args:
        A: numpy.array
            Squqare numpy matrix. Should be n x n size.
        b: numpy.array
            Output vector. Should be m x n size.
        init_x: None or numpy.array
            The initial guess for the x vector. If left blank then a random set is 
    """
    def __init__(self, A, b, init_x=None):
        if init_x is None:
            self.x = np.random.rand(A.shape[1], 1)
        else:
            self.x = init_x
        self.x_store = [self.x]
        self.a_store = []
        self.p_store = []
        self.r_store= []
        self.beta_store = []
        self.A = A
        self.b = b
    
    def get_alpha(self, epoch):
        """
        Get the alpha values for the current cycle.

        Args:
            epoch: int
                The current cycle.

        Returns: numpy.array
            Array/matrix of float numbers.
        """
        return (self.r_store[epoch].T @ self.r_store[epoch]) / (self.r_store[epoch].T@A@self.p_store[epoch])
    
    def get_beta(self, epoch):
        """
        Get the beta values for the current cycle. Please note, this is not run on the first
        epoch due to the fact that there is not enough residual values.

        Args:
            epoch: int
                The current cycle.

        Returns: numpy.array
            Array/matrix of float numbers.
        """
        return (self.r_store[epoch].T @ self.r_store[epoch]) / (self.r_store[epoch-1].T@self.r_store[epoch-1])
    
    def get_p_val(self, epoch):
        """
        Get the beta values for the current cycle. Please note, this is not run on the first epoch
        due to the fact that there is not enough residual values.

        Args:
            epoch: int
                The current cycle.

        Returns: numpy.array
            Array/matrix of float numbers.
        """
        return self.r_store[epoch] + self.beta_store[epoch-1]*self.p_store[epoch-1]
    
    def get_residual(self, epoch):
        """
        Get the residual value for the current epoch run.

        Args:
            epoch: int
                The current cycle.

        Returns: numpy.array
            Array/matrix of float numbers.
        """
        if epoch == 0:
            return self.b - self.A@self.x
        else:
            return self.r_store[epoch-1] - self.a_store[epoch-1]*self.A@self.p_store[epoch-1]
    
    def compute_x(self, epoch):
        """
        Recalcuate the x matrix based on the updated values in alpha and p.

        Args:
            epoch: int
                The current cycle.

        Returns: numpy.array
            Array/matrix of float numbers.
        """
        return self.x + self.a_store[epoch]*self.p_store[epoch]
    
    def conjugate(self, epochs):
        """
        Run the conjugate gradient values for chosen number of epochs. Please note, that there is
        currently no function to stop once the function converges.

        Args:
            epochs: int
                Number of cycles to run the function.

        Returns: numpy.array
            Calculated x matrix/vector. Please note, all values are stored within the class
            initialisation, so a full history can be seen.
        """
        for i in range(epochs):
            print(i)
            if i == 0:
                initial_resid = self.get_residual(i)
                
                self.r_store.append(initial_resid)
                self.p_store.append(initial_resid)
                
                self.a_store.append(self.get_alpha(i))
                
                self.x = self.compute_x(i)
                self.x_store.append(self.x)
            else:
                self.r_store.append(self.get_residual(i))
                self.beta_store.append(self.get_beta(i))
                self.p_store.append(self.get_p_val(i))
                self.a_store.append(self.get_alpha(i))
                self.x = self.compute_x(i)
                self.x_store.append(self.x)

        return self.x

if __name__ == "__main__":

    # This is based on example in https://en.wikipedia.org/wiki/Conjugate_gradient_method
    A = np.array([[4, 1], [1, 3]])
    x = np.array([[2],[1]]) #initial guess
    b = np.array([[1],[2]])

    lc = linearConjugate(A, b, x)    
    lc.conjugate(2)