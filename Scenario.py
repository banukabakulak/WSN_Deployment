import numpy as np
import matplotlib.pyplot as plt
import random

class Scenario:
    def __init__(self, size, N, S, L, seedId = 0): # seed=1982
        self.seeds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        np.random.seed(self.seeds[seedId])  # Set seed for reproducibility
        random.seed(self.seeds[seedId]/2)
        self.regionMap ={'demo': {'W': 5, 'H': 4},
                         'small' : {'W' : 10, 'H' : 8},
                         'medium' : {'W' : 15, 'H' : 14},
                         'large' : {'W' : 20, 'H' : 18},
                         'custom' : {'W' : 20, 'H' : 18}}
        self.size = size
        self.L = L  # Planning horizon, #periods
        self.poses = []

        for _ in range(N):
            pose = {
                'x': np.random.uniform(0, self.regionMap[self.size]['W']),
                'y': np.random.uniform(0, self.regionMap[self.size]['H']),
                'coverReq': [0] * S # np.random.randint(0, 2, size=S).tolist()  # random ints 0,1,2 for S sensors
            }
            self.poses.append(pose)

        for pose in self.poses:
            while sum(pose['coverReq']) == 0:
                pose['coverReq'] = np.random.randint(0, 2, size=S).tolist()  # random ints 0,1,2 for S sensors

        # set actuator positions
        # self.xa = np.random.randint(0, 2, size=N).tolist()
        num_actuators = N // 4
        self.xa = [0] * N
        actuator_indices = random.sample(range(N), num_actuators)
        for idx in actuator_indices:
            self.xa[idx] = 1
        print(self.xa)

    def display_poses(self):
        print("List of (x, y) poses with cover requirements:")
        for i, pose in enumerate(self.poses):
            print(f"Pose {i+1}: x = {pose['x']:.2f}, y = {pose['y']:.2f}, coverReq = {pose['coverReq']}")

    def plot_poses(self):
        xs = [pose['x'] for pose in self.poses]
        ys = [pose['y'] for pose in self.poses]

        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot base target poses (in blue)
        ax.scatter(xs, ys, c='blue', edgecolors='black', s=80, label='Target Poses')
        # If actuator selection list is provided, plot actuators
        if self.xa is not None:
            for j, is_actuator in enumerate(self.xa):
                if is_actuator:
                    x = self.poses[j]['x']
                    y = self.poses[j]['y']
                    ax.scatter(x, y, c='orange', edgecolors='black', s=100, marker='^',
                               label='Actuator' if 'Actuator' not in ax.get_legend_handles_labels()[1] else "")
        ax.set_title(f"Scenario Poses with Actuators in {self.regionMap[self.size]['W']}×{self.regionMap[self.size]['H']} Area")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_xlim(0, self.regionMap[self.size]['W'])
        ax.set_ylim(0, self.regionMap[self.size]['H'])
        ax.grid(True)
        # ax.legend()
        # Clean legend (avoid duplicates)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper right')
        fig.tight_layout()
        # plt.show()
        return fig, ax

    def isSense(self, name, pose1, pose2, sensor):
        """
        Returns 1 if the sensor with the given name can sense pose2 from pose1, else 0.
        pose1, pose2: tuples of (x, y)
        sensor: Sensor object with attribute .models (dict of dicts)
        """
        s = sensor.models.get(name)
        if s:
            distance = np.linalg.norm(np.array([pose1['x'], pose1['y']]) - np.array([pose2['x'], pose2['y']]))
            return int(distance <= 5 * s["senseRange"])
        return 0

    def isCommunicate(self, name, pose1, pose2, node):
        """
        Returns 1 if the transceiver with the given name can communicate between pose1 and pose2, else 0.

        Parameters:
        - name: str, name of the transceiver model (e.g., 'T1')
        - pose1, pose2: tuple of floats, (x, y) coordinates
        - transceiver: Transceiver object containing .availableModels dictionary

        Returns:
        - int: 1 if communication is possible, 0 otherwise
        """
        node = node.models.get(name)
        if node:
            distance = np.linalg.norm(np.array([pose1['x'], pose1['y']]) - np.array([pose2['x'], pose2['y']]))
            return int(distance <= node["commRange"])
        return 0



