class SteadyStateDetector:

    def __init__(self, threshold=1e-10, duration=10.0):

        self.threshold = threshold

        self.duration = duration



        self.prev_E = None

        self.last_time = None



        self.steady_start_time = None



    def update(self, E, t):


        if self.prev_E is None:

            self.prev_E = E

            self.last_time = t

            return False



        dt = t - self.last_time

        dE_dt = abs((E - self.prev_E) / dt)



        self.prev_E = E

        self.last_time = t



        if dE_dt < self.threshold:

            if self.steady_start_time is None:

                self.steady_start_time = t

            elif (t - self.steady_start_time) >= self.duration:

                return True

        else:

            # 一旦不满足，重置

            self.steady_start_time = None



        return False
