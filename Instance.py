import sys

from Scenario import *
from Agent import *
from Sensor import *
from Hardware import *

class Instance:
    def __init__(self, size, N, S, L, seedId = 0, isCNAD = False):
        self.name = f"({size},{N},{S},{L})"
        self.size = size
        self.N = N
        self.S = S
        self.isCNAD = isCNAD
        if L == 'short':
            self.L = 17280 * 1 # periods in 1 year
        if L == 'long':
            self.L = 17280 * 2 # 1.5 # periods in 1.5 years

        self.scenario = Scenario(self.size, self.N, self.S, self.L, seedId)
        self.scenario.display_poses()
        # self.fig, self.ax = self.scenario.plot_poses()

        self.sensor = Sensor(self.S)
        self.sensor.display_models()
        self.scenario.onePeriod = self.sensor.get_period_length()
        print(f"In scenario one period is {self.scenario.onePeriod} secs")

        pose1 = self.scenario.poses[1]
        pose2 = self.scenario.poses[2]
        isSense = self.scenario.isSense("S1", pose1, pose2, self.sensor)
        print(f"The coverage status of sensor S1 is {isSense}")

        self.box = Box()
        self.box.display_info()

        self.battery = Battery()
        self.battery.display_models()

        self.memory = Memory()
        self.memory.display_models()

        self.transceiver = Transceiver(self.scenario.onePeriod, self.isCNAD)
        self.transceiver.display_models()

        pose1 = {'x': 10, 'y': 20}
        pose2 = {'x': 10, 'y': 19}
        result = self.scenario.isCommunicate('T1', pose1, pose2, self.transceiver)
        print(f"Can T1 communicate between pose1 and pose2? {'Yes' if result else 'No'}")

        self.agent = Agent()
        self.agent.display_types()

        result = self.scenario.isCommunicate('Router', pose1, pose2, self.agent)
        print(f"Can Router communicate between pose1 and pose2? {'Yes' if result else 'No'}")
