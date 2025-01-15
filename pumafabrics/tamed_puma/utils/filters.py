import numpy as np

class PDController:
    def __init__(self, Kp, Kd, dt=0.01):
        self.Kp = Kp
        self.Kd = Kd
        self.previous_error = 0
        self.dt=dt

    def control(self, desired_velocity, current_velocity):
        error = (desired_velocity - current_velocity)/self.dt
        control_value = self.Kp * error + self.Kd * (error - self.previous_error)
        self.previous_error = error
        return control_value

    def control_pos_vel(self, x, xdot, x_d, xdot_d=np.zeros((7,)), Kp=1, Kd=0.1):
        error_pos = x_d - x
        error_vel = xdot_d - xdot
        control_value = Kp * error_pos + Kd * (error_vel)
        return control_value

def ema_filter(signal, alpha=0.5):
        """
        Apply an Exponential Moving Average (EMA) filter to a signal.

        Parameters:
        signal : list or array-like
            The signal to filter.
        alpha : float
            The decay factor. This should be between 0 and 1, where higher values give more importance to recent values.

        Returns:
        filtered_signal : list
            The filtered signal.
        """
        filtered_signal = []
        for value in signal: #.transpose():
            if filtered_signal:
                previous_ema = filtered_signal[-1]
                filtered_signal.append(alpha * value + (1 - alpha) * previous_ema)
            else:
                filtered_signal.append(value)

        return filtered_signal[-1]

def ema_filter_deriv(signal:np.ndarray, alpha=0.01, dt=0.01):
    dim_array = signal.shape
    if dim_array[1]>1:
        signal_deriv = np.gradient(signal, axis=0) / dt
    else:
        signal_deriv = signal / dt
    filtered_signal = ema_filter(signal_deriv, alpha=alpha)
    filtered_deriv = np.array(filtered_signal)
    return filtered_deriv

