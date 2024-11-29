import numpy as np

class ICUZen():
    """Alarm prediction based on a simple model with learned parameters.

       The alarm system recommends an alarm if the vitals signs deviate critically from their expected values.

        Args:
            threshold (float): threshold for throwing and alarm, used by the predict method.
            sensitivity (float): defines how sensitive the system is to the input signal.

    """

    def __init__(self, threshold=0.76, sensitivity=30):
        self.params = np.array([0.027, 0.00793, 0.0123, 0.012, 0.066]).reshape(-1,1)
        self.sensitivity = sensitivity
        self.threshold = threshold

    def predict_proba(self, X):
        """Returns the score for the alarm system based on the vital signs input.

           A higher scores means, that it is more likely that a critical situation occurred.

        Args:
            X (np.array):  Vitals as a MxN matrix, where M in the number of observations
                           and N number of vital signs. The vital signs are expected in the following
                           order:
                           - body_temperature [in Degree Celsius]
                           - blood_pressure_systolic [in mmHg]
                           - blood_pressure_diastolic [in mmHg]
                           - heart_rate [in bpm]
                           - respiratory_rate [in breaths per minute]
        Returns:
            (np.array): The scores of the model
        """

        X = (X.T*self.params).T
        X = np.exp(self.sensitivity*np.abs(1-X))
        Y = 1-(np.mean(X, axis=1)/np.max(X, axis=1))

        return Y


    def predict(self, X):
        """Returns if an alarm is predicted based on the vital signs input.

            True means, that the systems recommends to throw an alarm
            False means, that the systems recommends not to throw an alarm.

        Args:
            X (np.array): Vitals as a MxN matrix, where M in the number of observations
                           and N number of vital signs. The vital signs are expected in the following
                           order:
                           - body_temperature [in Degree Celsius]
                           - blood_pressure_systolic [in mmHg]
                           - blood_pressure_diastolic [in mmHg]
                           - heart_rate [in bpm]
                           - respiratory_rate [in breaths per minute]

        Returns:
            (np.array): A boolean array stating if an alarm should be thrown
        """

        Y = self.predict_proba(X)
        Y = Y > self.threshold

        return Y