from dataclasses import dataclass

@dataclass
class StftArgument:
    sample_rate: int = 44100  # sample rate of the data
    t_int: float = 0.05       # sec, the half-width of the window function
    divide_step: int = 1      # window step = T/divide_step

    @staticmethod
    def wavelet_connector_factory(f = lambda z:1/(1+(2*z)**6)):
        if f(0) != 1 or f(0.5) != 0.5:
            raise ValueError('f(0) must be 1, f(0.5) must be 0.5')
        def window(x):
            if x<-1:
                return 0
            elif x<-0.5:
                return 1-f(x+1)
            elif x<0.5:
                return f(x)
            elif x<1:
                return 1-f(x-1)
            else:
                return 0
        return window

    def stft_default_window(self, x, power=6):
        f = lambda z:1/(1+(2*z)**power)
        if x<-1:
            return 0
        elif x<-0.5:
            return 1-f(x+1)
        elif x<0.5:
            return f(x)
        elif x<1:
            return 1-f(x-1)
        else:
            return 0