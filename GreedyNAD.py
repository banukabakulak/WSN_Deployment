from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import pandas as pd
import time
import math
import random
import os

class GreedyNAD:
    def __init__(self, epsilon, delta, isRelease = False):
        self.epsilon = epsilon
        self.delta = delta
        self.isRelease = isRelease
        self.isVisualize = {'SensorNode': False,
                            'Actuator': True,
                            'Router': False,
                            'Gateway': False
                            }

        self.isFeasible = {'GD': False,
                           'SD': False,
                           'NTD': False,
                           'NMD': False,
                           'RD': False,
                           'NBD': False,
                           'CF': False}
        self.point_gate_cover = 1

    def initializeModels(self, instance):
        # Index sets
        self.N = range(instance.N)
        self.K = range(4)
        self.S = range(instance.S)
        self.B = range(len(instance.battery.models))
        self.M = range(len(instance.memory.models))
        self.T = range(len(instance.transceiver.models))

        self.nodeMap = {
            0: 'SensorNode',
            1: 'Actuator',
            2: 'Router',
            3: 'Gateway'
        }

        # Big number H
        maxH = 0
        for s in self.S:
            if instance.sensor.models[f"S{s + 1}"]['dataSize'] > maxH:
                maxH = instance.sensor.models[f"S{s + 1}"]['dataSize']
        self.H = instance.box.attributes['P'] * maxH

    def solveGatewayDeployment(self, instance):
        start_time = time.time()
        # Initialize gateway deployment and undercover status
        self.xg = {i: 0 for i in self.N}
        self.setUndercover = {i: 1 for i in self.N}

        # Loop until all nodes are covered by at least one gateway
        while any(self.setUndercover.values()):
            best_pose = -1
            best_gain = 0

            # Evaluate all unused candidate positions
            for j in self.N:
                if self.xg[j] == 0:
                    gain = sum(
                        self.setUndercover[i] * instance.scenario.isCommunicate('Gateway', instance.scenario.poses[j],
                                                                                instance.scenario.poses[i],
                                                                                instance.agent)
                        for i in self.N
                    )

                    if gain > best_gain:
                        best_pose = j
                        best_gain = gain

            # Place a gateway at the best position found
            if best_gain > 0:
                self.xg[best_pose] = 1

                # Update undercover status for all nodes
                for i in self.N:
                    if instance.scenario.isCommunicate('Gateway', instance.scenario.poses[best_pose],
                                                       instance.scenario.poses[i], instance.agent):
                        self.setUndercover[i] = 0
            else:
                print(f"Greedy-GD is infeasible...")
                break
        end_time = time.time()
        self.GD_runtime = end_time - start_time

        self.GD_ObjValue = sum([instance.agent.models['Gateway']['cost'] * self.xg[j] for j in self.N])

        print("Solution found. Gateway placements:")
        print(f"Greedy-GD Objective Value: {self.GD_ObjValue}")
        print(f"Greedy-GD model execution time is {self.GD_runtime :.4f} seconds.")

        for j in self.N:
            if self.xg[j] > 0.5 and not self.isRelease:  # print only non‑zeros
                print(f"xg[{j + 1}] = {self.xg[j]}")

        for j in self.N:
            if self.setUndercover[j] > 0.5 and not self.isRelease:  # print only non‑zeros
                print(f"setU[{j + 1}] = {self.setUndercover[j]}")

        self.gateway_indices = [j for j in self.N if self.xg[j] > 0.5]
        self.isVisualize['Gateway'] = True
        self.isFeasible['GD'] = True

    def solveSensorDeployment(self, instance):
        start_time = time.time()
        # Initialize gateway deployment and undercover status
        self.xn = {j: 0 for j in self.N}
        self.sn = {(j, s): 0 for j in self.N for s in self.S}

        self.underSensed = {(i, s): instance.scenario.poses[i]['coverReq'][s]
                            for i in self.N for s in self.S}
        self.setUndersensed = {(i, s): 1 for i in self.N for s in self.S}

        self.nodeVolume = {j: instance.box.attributes['volume'] for j in self.N}
        self.nodePin = {j: instance.box.attributes['P'] for j in self.N}

        # Loop until all nodes are covered by at least one gateway
        for s in self.S:
            while any(self.setUndersensed[i, s] > 0 for i in self.N):
                best_pose_with_node = -1
                best_pose_no_node = -1
                best_gain_with_node = 0
                best_gain_no_node = 0

                for j in self.N:
                    if (self.sn[j, s] == 0 and instance.sensor.models[f"S{s + 1}"]["volume"] <= self.nodeVolume[j]
                            and self.nodePin[j] > 0):
                        gain = sum(
                            self.setUndersensed[i, s] * instance.scenario.isSense(f"S{s + 1}", instance.scenario.poses[j],
                                                                             instance.scenario.poses[i], instance.sensor)

                            for i in self.N
                        )
                        if self.xn[j] == 1:
                            if gain > best_gain_with_node:
                                best_pose_with_node = j
                                best_gain_with_node = gain
                        elif self.xn[j] == 0:
                            if gain > best_gain_no_node:
                                best_pose_no_node = j
                                best_gain_no_node = gain

                if best_gain_with_node > 0:
                    self.sn[best_pose_with_node, s] = 1
                    for i in self.N:
                        self.underSensed[i, s] -= instance.scenario.isSense(f"S{s + 1}", instance.scenario.poses[best_pose_with_node],
                                                                             instance.scenario.poses[i], instance.sensor)
                        if self.underSensed[i, s] <= 0:
                            self.underSensed[i, s] = 0
                            self.setUndersensed[i, s] = 0

                    self.nodeVolume[best_pose_with_node] -= instance.sensor.models[f"S{s + 1}"]["volume"]
                    self.nodePin[best_pose_with_node] -= 1

                elif best_gain_no_node > 0:
                    self.xn[best_pose_no_node] = 1
                    self.sn[best_pose_no_node, s] = 1
                    for i in self.N:
                        self.underSensed[i, s] -= instance.scenario.isSense(f"S{s + 1}", instance.scenario.poses[best_pose_no_node],
                                                                             instance.scenario.poses[i], instance.sensor)
                        if self.underSensed[i, s] <= 0:
                            self.underSensed[i, s] = 0
                            self.setUndersensed[i, s] = 0

                    self.nodeVolume[best_pose_no_node] -= instance.sensor.models[f"S{s + 1}"]["volume"]
                    self.nodePin[best_pose_no_node] -= 1
                else:
                    print(f"Greedy-SD is infeasible for sensor type S{s + 1}")
                    break
        end_time = time.time()
        self.SD_runtime = end_time - start_time

        self.SD_ObjValue = (sum([instance.box.attributes['cost'] * self.xn[j] for j in self.N]) +
                            sum([instance.sensor.models[f"S{s + 1}"]['cost'] * self.sn[j, s] for j in self.N for s in self.S]))

        print("Solution found. Sensor Node and Sensor placements:")
        print(f"Greedy-SD Objective Value: {self.SD_ObjValue}")
        print(f"Greedy-SD model execution time is {self.SD_runtime :.4f} seconds.")

        for j in self.N:
            if self.xn[j] > 0.5 and not self.isRelease:  # print only non‑zeros
                print(f"xn[{j + 1}] = {self.xn[j]}")

        for s in self.S:
            for j in self.N:
                if self.sn[j, s] > 0.5 and not self.isRelease:  # print only non‑zeros
                    print(f"s[{j + 1, s + 1}] = {self.sn[j, s]}")

        for s in self.S:
            for j in self.N:
                if self.setUndersensed[j, s] > 0.5 and not self.isRelease:  # print only non‑zeros
                    print(f"setU[{j + 1, s + 1}] = {self.setUndersensed[j, s]}")

        self.hn = {}  # Dictionary to store h values
        self.wn = {}  # Dictionary to store w values

        # Calculate Sensor Node Data Size
        for j in self.N:
            dataSize = 0
            for s in self.S:
                dataSize += instance.sensor.models[f"S{s + 1}"]['dataSize'] * self.sn[j, s]
            self.hn[j] = dataSize
            if self.hn[j] > 0.5 and not self.isRelease:
                print(f"h[{j + 1}] = {self.hn[j]}")

        # Calculate Sensor Node Protocol Overhead
        for j in self.N:
            self.wn[j] = instance.agent.models['SensorNode']['gateCommand'] * self.xn[j] + self.hn[j]
            for s in self.S:
                self.wn[j] += instance.sensor.models[f"S{s + 1}"]['gateCommand'] * self.sn[j, s]
            if self.wn[j] > 0.5 and not self.isRelease:
                print(f"w[{j + 1}] = {self.wn[j]}")

        self.sensor_node_indices = [j for j in self.N if self.xn[j] > 0.5]
        self.sensor_assignments = {(j, s): val for (j, s), val in self.sn.items() if val > 0.5}

        self.isVisualize['SensorNode'] = True
        self.isFeasible['SD'] = True

    def solveNodeTransceiverDeployment(self, instance):
        start_time = time.time()
        self.tn = {(j, t): 0 for j in self.N for t in self.T}

        self.underComm = {(j, t): -instance.agent.models["SensorNode"]["lambda"] * self.xn[j]
                          for j in self.N for t in self.T} # >= 0 pass

        self.underBandwidth = {(j, t): instance.transceiver.models[f"T{t + 1}"]["bandwidth"]
                               for j in self.N for t in self.T} # >= 0 pass

        for j in self.N:
            if self.xn[j] > 0.5:
                # Check comm feasibility for transceivers at a node j
                isCommFeas = False
                for t in self.T:
                    for i in self.N:
                        commStatus = instance.scenario.isCommunicate(f"T{t + 1}", instance.scenario.poses[j],
                                                                     instance.scenario.poses[i], instance.transceiver)
                        self.underComm[j, t] += commStatus * self.xg[i]
                        if not i == j:
                            self.underComm[j, t] += commStatus * self.xn[i]

                    if self.underComm[j, t] >= 0:
                        isCommFeas = True

                if not isCommFeas:
                    print(f"There is no comm feasible transceiver for sensor node {j + 1}...")
                    break

        loop = True
        while loop:
            for j in self.N:
                if self.xn[j] > 0.5:
                    # Assign transceiver to node j
                    isBandFeas = False
                    for t in self.T:
                        if self.underComm[j, t] >= 0:
                            # Check bandwidth feasibility with the current tx assignments
                            self.underBandwidth[j, t] -= (self.hn[j] + instance.agent.models['Router']['lambda'] * instance.agent.models['Router']['dataSize'])
                            for i in self.N:
                                commStatus = instance.scenario.isCommunicate('Actuator', instance.scenario.poses[i],
                                                                             instance.scenario.poses[j], instance.agent)
                                self.underBandwidth[j, t] -= commStatus * instance.agent.models['Actuator']['dataSize'] * instance.scenario.xa[i]

                                for tt in self.T:
                                    self.underBandwidth[j, t] -= (instance.scenario.isCommunicate(f"T{tt + 1}", instance.scenario.poses[i],
                                                                             instance.scenario.poses[j], instance.transceiver)
                                                          * self.tn[i, tt] * self.hn[i])

                        if self.underBandwidth[j, t] >= 0:
                            isBandFeas = True

                    if not isBandFeas:
                        print(f"There is no band feasible transceiver for sensor node {j + 1}...")
                        loop = False
                        break

            if not loop: break

            # self.tn = {(j, t): 0 for j in self.N for t in self.T}
            loop = False

            for j in self.N:
                if self.xn[j] > 0.5:
                    best_tx = -1
                    best_gain = -1
                    for t in self.T:
                        if self.underComm[j, t] >= 0 and self.underBandwidth[j, t] >= 0:
                            # gain = (self.underComm[j, t] * self.underBandwidth[j, t]
                            #         / instance.transceiver.models[f"T{t + 1}"]['cost'])
                            gain = (1 / instance.transceiver.models[f"T{t + 1}"]['cost'])
                            if gain > best_gain:
                                best_tx = t
                                best_gain = gain
                    if best_gain >= 0:
                        if self.tn[j, best_tx] == 0:
                            for t in self.T:
                                self.tn[j, t] = 0
                            self.tn[j, best_tx] = 1
                            loop = True

        end_time = time.time()
        self.NTD_runtime = end_time - start_time

        self.NTD_ObjValue = sum([instance.transceiver.models[f"T{t + 1}"]['cost'] * self.tn[j, t]
                                 for j in self.N for t in self.T])

        print("Solution found. Transceiver placements:")
        print(f"Greedy-NTD Objective Value: {self.NTD_ObjValue}")
        print(f"Greedy-NTD model execution time is {self.NTD_runtime :.4f} seconds.")

        for j in self.N:
            for t in self.T:
                if self.tn[j, t] > 0.5 and not self.isRelease:  # print only non‑zeros
                    print(f"t[{j + 1, t + 1}] = {self.tn[j, t]}")

        self.qn = {}  # Dictionary to store q values
        self.dn = {}  # Dictionary to store d values
        self.ea = {}  # Dictionary to store e_a values
        self.erx = {}  # Dictionary to store e_rx values
        self.etx = {}  # Dictionary to store e_tx values
        self.eHat = {}  # Dictionary to store e_Hat values

        # Calculate Sensor Node Comm Range Specifications
        for j in self.N:
            for i in self.N:
                self.qn[j, i] = sum(instance.scenario.isCommunicate(f"T{t + 1}", instance.scenario.poses[j],
                                    instance.scenario.poses[i], instance.transceiver) * self.tn[j, t] for t in self.T)
                if self.qn[j, i] > 0.5 and not self.isRelease:
                    print(f"q[{j + 1}, {i + 1}] = {self.qn[j, i]}")

        # Calculate Sensor Node Bandwidth Specifications
        for j in self.N:
            self.dn[j] = sum(instance.transceiver.models[f"T{t + 1}"]["bandwidth"] * self.tn[j, t] for t in self.T)
            if self.dn[j] > 1e-6 and not self.isRelease:
                print(f"d[{j + 1}] = {self.dn[j]}")

        # Calculate Sensor Node Energy Specifications
        for j in self.N:
            activeEnergy = 0
            for t in self.T:
                activeEnergy += instance.transceiver.models[f"T{t + 1}"]['activeEnergy'] * self.tn[j, t]
            self.ea[j] = activeEnergy

            rxEnergy = 0
            for t in self.T:
                rxEnergy += instance.transceiver.models[f"T{t + 1}"]['rxEnergy'] * self.tn[j, t]
            self.erx[j] = rxEnergy

            txEnergy = 0
            for t in self.T:
                txEnergy += instance.transceiver.models[f"T{t + 1}"]['txEnergy'] * self.tn[j, t]
            self.etx[j] = txEnergy

            self.node_lambda = instance.agent.models['SensorNode']['lambda']
            estEnergy = (self.ea[j] + self.node_lambda * self.H * self.erx[j]
                         + (self.node_lambda + 1) * self.H * self.etx[j])
            for s in self.S:
                estEnergy += instance.sensor.models[f"S{s + 1}"]['senseEnergy'] * self.sn[j, s]
            self.eHat[j] = estEnergy

        self.isFeasible['NTD'] = True
      
    def solveNodeMemoryDeployment(self, instance):
        start_time = time.time()
        self.mn = {(j, m): 0 for j in self.N for m in self.M}
        self.nodeMemory = {j: 0 for j in self.N}

        for j in self.N:
            if self.xn[j] > 0.5:
                best_memory = -1
                best_gain = 100000
                self.nodeMemory[j] = (self.hn[j] + instance.agent.models['Router']['lambda']
                                      * instance.agent.models['Router']['dataSize'])
                for i in self.N:
                    commStatus = instance.scenario.isCommunicate('Actuator', instance.scenario.poses[i],
                                                                 instance.scenario.poses[j], instance.agent)
                    self.nodeMemory[j] += (commStatus * instance.agent.models['Actuator']['dataSize']
                                           * instance.scenario.xa[i])

                    self.nodeMemory[j] += self.qn[i, j] * self.hn[i]

                for m in self.M:
                    memorySlot =  math.floor(self.nodeMemory[j] / instance.memory.models[f"M{m + 1}"]["capacity"])
                    gain = memorySlot * instance.memory.models[f"M{m + 1}"]["cost"]
                    if gain < best_gain:
                        best_memory = m
                        best_gain = gain

                if best_memory >= 0:
                    self.mn[j, best_memory] = min(instance.box.attributes["C"] - 1, math.floor(self.nodeMemory[j]
                                                  / instance.memory.models[f"M{best_memory + 1}"]["capacity"]))
                    self.nodeMemory[j] -= self.mn[j, best_memory] * instance.memory.models[f"M{best_memory + 1}"]["capacity"]

                if self.nodeMemory[j] > 0:
                    best_memory = -1
                    best_gain = 100000
                    for m in self.M:
                        if instance.memory.models[f"M{m + 1}"]["capacity"] >= self.nodeMemory[j]:
                            gain = instance.memory.models[f"M{m + 1}"]["cost"]
                            if gain < best_gain:
                                best_memory = m
                                best_gain = gain

                    if best_memory >= 0:
                        self.mn[j, best_memory] += 1

        end_time = time.time()
        self.NMD_runtime = end_time - start_time

        self.NMD_ObjValue = sum([instance.memory.models[f"M{m + 1}"]['cost'] * self.mn[j, m]
                                 for j in self.N for m in self.M])

        print("Solution found. Memory placements:")
        print(f"Greedy-NMD Objective Value: {self.NMD_ObjValue}")
        print(f"Greedy-NMD model execution time is {self.NMD_runtime :.4f} seconds.")

        for j in self.N:
            for m in self.M:
                if self.mn[j, m] > 0.5 and not self.isRelease:  # print only non‑zeros
                    print(f"m[{j + 1, m + 1}] = {self.mn[j, m]}")

        self.Mn = {}  # Dictionary to store None Memory specification values

        # Calculate Sensor Node Memory Specifications
        for j in self.N:
            self.Mn[j] = sum(instance.memory.models[f"M{m + 1}"]["capacity"] * self.mn[j, m] for m in self.M)
            if self.Mn[j] > 0 and not self.isRelease:
                print(f"M[{j + 1}] = {self.Mn[j]}")

        self.isFeasible['NMD'] = True

    def compute_disconnected_nodes(self, G):
        """
        Updates self.setDiscomm by marking nodes as disconnected (1)
        if they cannot reach any gateway node (type 3) within self.delta hops.

        Parameters:
            G (nx.Graph): Graph with nodes labeled as (j, k) tuples.
        """

        # Identify gateway nodes (type 3)
        gateway_nodes = [n for n in G.nodes() if n[1] == 3]

        # For each node, check connectivity to a gateway within delta hops
        for node in G.nodes():
            reachable = False
            lengths = nx.single_source_shortest_path_length(G, node, cutoff=self.delta)

            for reachable_node in lengths:
                if reachable_node in gateway_nodes:
                    reachable = True
                    break

            if reachable:
                self.setDiscomm[node] = 0  # Mark as connected

        # Print only disconnected nodes (for debugging or logging)
        print({n: v for n, v in self.setDiscomm.items() if v > 0.5})

    def solveRouterDeployment(self, instance):
        start_time = time.time()
        self.xr = {j: 0 for j in self.N}
        self.outReachable = {j: 0 for j in self.N}
        self.inReachable = {j: 0 for j in self.N}
        self.setDiscomm = {}

        # Construct current graph G
        G = nx.Graph()  # Use nx.DiGraph() for directed graphs

        # Add nodes
        for j in self.N:
            if self.xn[j] > 0.5:
                G.add_node((j, 0))
            if instance.scenario.xa[j] > 0.5:
                G.add_node((j, 1))
            if self.xg[j] > 0.5:
                G.add_node((j, 3))

        # Add edges
        for (j, k) in G.nodes():
            for (i, l) in G.nodes():
                if not (i == j and k == l):
                    commStatus = instance.scenario.isCommunicate(self.nodeMap[k], instance.scenario.poses[j],
                                                                 instance.scenario.poses[i], instance.agent)
                    if commStatus:
                        G.add_edge((j, k), (i, l))

        # print(list(G.nodes()))
        # print(list(G.edges()))
        print([(j + 1, k) for (j, k) in G.nodes()])
        print([((j + 1, k), (i + 1, l)) for (j, k), (i, l) in G.edges()])

        # Set all nodes as disconnected initially
        for node in G.nodes():
            self.setDiscomm[node] = 1

        self.compute_disconnected_nodes(G)

        while any(v > 0 for v in self.setDiscomm.values()):
            print("There is a disconnection.")
            best_pose = -1
            best_gain = 0
            for j in self.N:
                self.outReachable[j] = sum(instance.scenario.isCommunicate('Router', instance.scenario.poses[j],
                                                                            instance.scenario.poses[i], instance.agent)
                                            * (self.xn[i] + self.xr[i] + self.xg[i]) for i in self.N)

                if self.outReachable[j] >= instance.agent.models['Router']['lambda']:

                    self.inReachable[j] = (sum(self.setDiscomm[(i, 0)] * self.qn[i, j]
                                              for i in self.N if (i, 0) in self.setDiscomm)
                                           + sum(self.setDiscomm[(i, l)] * instance.scenario.isCommunicate(self.nodeMap[l], instance.scenario.poses[i],
                                                                                                           instance.scenario.poses[j], instance.agent)
                                              for i in self.N for l in (1, 2) if (i, l) in self.setDiscomm))

                    gain = self.outReachable[j] * self.inReachable[j]
                    if gain > best_gain:
                        best_pose = j
                        best_gain = gain

            if best_gain > 0:
                self.xr[best_pose] = 1
                G.add_node((best_pose, 2))
                self.compute_disconnected_nodes(G)
            else:
                print("No feasible router deployment found.")
                break

        end_time = time.time()
        self.RD_runtime = end_time - start_time

        self.RD_ObjValue = sum(instance.agent.models["Router"]['cost'] * self.xr[j] for j in self.N)

        print("Solution found. Router placements:")
        print(f"Greedy-RD Objective Value: {self.RD_ObjValue}")
        print(f"Greedy-RD model execution time is {self.RD_runtime :.4f} seconds.")

        for j in self.N:
                if self.xr[j] > 0.5 and not self.isRelease:  # print only non‑zeros
                    print(f"xr[{j + 1}] = {self.xr[j]}")

        self.router_indices = [j for j, val in self.xr.items() if val > 0.5]
        self.isVisualize['Router'] = True
        self.isFeasible['RD'] = True

    def solveNodeBatteryDeployment(self, instance):
        start_time = time.time()
        self.bn = {(j, b): 0 for j in self.N for b in self.B}
        self.nodeEnergy = {j: 0 for j in self.N}

        for j in self.N:
            if self.xn[j] > 0.5:
                best_battery = -1
                best_gain = 100000
                self.nodeEnergy[j] = self.eHat[j] * instance.L

                for b in self.B:
                    batterySlot =  math.floor(self.nodeEnergy[j] / instance.battery.models[f"B{b + 1}"]["capacity"])
                    gain = batterySlot * instance.battery.models[f"B{b + 1}"]["cost"]
                    if gain < best_gain:
                        best_battery = b
                        best_gain = gain

                if best_battery >= 0:
                    self.bn[j, best_battery] = min(instance.box.attributes["D"] - 1, math.floor(self.nodeEnergy[j]
                                                  / instance.battery.models[f"B{best_battery + 1}"]["capacity"]))

                    self.nodeEnergy[j] -= (self.bn[j, best_battery] *
                                           instance.battery.models[f"B{best_battery + 1}"]["capacity"])

                if self.nodeEnergy[j] > 0:
                    best_battery = -1
                    best_gain = 100000
                    for b in self.B:
                        if instance.battery.models[f"B{b + 1}"]["capacity"] >= self.nodeEnergy[j]:
                            gain = instance.battery.models[f"B{b + 1}"]["cost"]
                            if gain < best_gain:
                                best_battery = b
                                best_gain = gain

                    if best_battery >= 0:
                        self.bn[j, best_battery] += 1

        end_time = time.time()
        self.NBD_runtime = end_time - start_time

        self.NBD_ObjValue = sum([instance.battery.models[f"B{b + 1}"]['cost'] * self.bn[j, b]
                                 for j in self.N for b in self.B])

        print("Solution found. Battery placements:")
        print(f"Greedy-NBD Objective Value: {self.NBD_ObjValue}")
        print(f"Greedy-NBD model execution time is {self.NBD_runtime :.4f} seconds.")

        for j in self.N:
            for b in self.B:
                if self.bn[j, b] > 0.5 and not self.isRelease:  # print only non‑zeros
                    print(f"b[{j + 1, b + 1}] = {self.bn[j, b]}")

        self.En = {}  # Dictionary to store None Energy specification values

        # Calculate Sensor Node Energy Specifications
        for j in self.N:
            self.En[j] = sum(instance.battery.models[f"B{b + 1}"]["capacity"] * self.bn[j, b] for b in self.B)
            if self.En[j] > 0 and not self.isRelease:
                print(f"E[{j + 1}] = {self.En[j]}")

        self.isFeasible['NBD'] = True

    def solveClusterFormation(self, instance):
        start_time = time.time()
        self.uu = {(j, k, i): 0 for j in self.N for k in (0, 1, 2) for i in self.N}
        self.OO = {i: 0 for i in self.N}

        # Initialize unclustered set for the deployed agents
        self.setUnclustered = {}
        for j in self.N:
            self.setUnclustered[j, 0] = self.xn[j]
            self.setUnclustered[j, 1] = instance.scenario.xa[j]
            self.setUnclustered[j, 2] = self.xr[j]

        # Calculate average cluster overhead
        self.avgOverhead = 0
        self.actuatorOverhead = (instance.agent.models["Actuator"]["gateCommand"]
                                 + instance.agent.models["Actuator"]["dataSize"])
        self.routerOverhead = (instance.agent.models["Router"]["gateCommand"]
                                 + instance.agent.models["Router"]["dataSize"])
        for j in self.N:
            self.avgOverhead += self.wn[j] * self.xn[j]
            self.avgOverhead += (self.actuatorOverhead * instance.scenario.xa[j]
                                 + self.routerOverhead * self.xr[j])
        self.avgOverhead /= sum(self.xg[j] for j in self.N)

        # while any(self.setUnclustered[j, k] > 0 for j in self.N for k in (0, 1, 2)):
        while True:
            # Get all (j, k) where setUnclustered[(j, k)] > 0
            candidates = [key for key, val in self.setUnclustered.items() if val > 0]

            # Pick a random one if the list is not empty
            if candidates:
                curr_pose, curr_agent = random.choice(candidates)
                print(f"Selected: (j, k) = {curr_pose, curr_agent}")
            else:
                print("Clustering completed.")
                break

            if curr_agent == 0:
                agentOverhead = self.wn[curr_pose]
            elif curr_agent == 1:
                agentOverhead = self.actuatorOverhead
            elif curr_agent == 2:
                agentOverhead = self.routerOverhead

            for i in self.N:
                if self.xg[i] and instance.scenario.isCommunicate("Gateway", instance.scenario.poses[i],
                                                                  instance.scenario.poses[curr_pose], instance.agent):

                    if (self.OO[i] < (1 - self.epsilon) * self.avgOverhead):
                        self.OO[i] += agentOverhead
                        self.uu[curr_pose, curr_agent, i] = 1
                        self.setUnclustered[curr_pose, curr_agent] = 0
                        break

            if self.setUnclustered[curr_pose, curr_agent] == 1:
                for i in self.N:
                    if self.xg[i] and instance.scenario.isCommunicate("Gateway", instance.scenario.poses[i],
                                                                      instance.scenario.poses[curr_pose],
                                                                      instance.agent):
                        if ((self.OO[i] + agentOverhead) <= (1 + self.epsilon) * self.avgOverhead):
                            self.OO[i] += agentOverhead
                            self.uu[curr_pose, curr_agent, i] = 1
                            self.setUnclustered[curr_pose, curr_agent] = 0
                            break

            if self.setUnclustered[curr_pose, curr_agent] == 1:
                print(f"No available cluster for agent {curr_pose, curr_agent}")
                break

        end_time = time.time()
        self.CF_runtime = end_time - start_time

        self.CF_ObjValue = 0

        print("Solution found. Cluster assignments:")
        print(f"Greedy-CF Objective Value: {self.CF_ObjValue}")
        print(f"Greedy-CF model execution time is {self.CF_runtime :.4f} seconds.")

        for j in self.N:
            if self.OO[j] > 1e-6 and not self.isRelease:
                print(f"O[{j + 1}] = {self.OO[j]}")
        for j in self.N:
            for k in 0, 1, 2:
                for i in self.N:
                    if self.uu[j, k, i] > 0.5 and not self.isRelease:
                        print(f"u[{j + 1}, {k}, {i + 1}] = {self.uu[j, k, i]}")

        self.isFeasible['CF'] = True

    def solve(self, instance):
        self.initializeModels(instance)
        self.solveGatewayDeployment(instance)
        if self.isFeasible['GD']:
            self.solveSensorDeployment(instance)
        if self.isFeasible['SD']:
            self.solveNodeTransceiverDeployment(instance)
        if self.isFeasible['NTD']:
            self.solveNodeMemoryDeployment(instance)
        if self.isFeasible['NMD']:
            self.solveRouterDeployment(instance)
        if self.isFeasible['RD']:
            self.solveNodeBatteryDeployment(instance)
        if self.isFeasible['NBD']:
            self.solveClusterFormation(instance)
        if self.isFeasible['CF']:
            return True
        else:
            return False

    def reportOutput(self, filePath):
        """Append one block of model results to <filePath>."""
        if not self.isFeasible['CF']:  # nothing to report
            return

        # ------------------------------------------------------------------
        # column layout (fixed‑width strings for nice plain‑text view)
        widths = (6, 12, 15)  # Model, ObjVal, CPU
        cols = [
            f"{'Model':<{widths[0]}}",
            f"{'ObjVal':>{widths[1]}}",
            f"{'CPU (secs)':>{widths[2]}}"
        ]

        # helper that formats one model row
        def make_row(tag, obj, cpu):
            return [
                f"{tag:<{widths[0]}}",
                f"{obj:>{widths[1]}.3f}",
                f"{cpu:>{widths[2]}.5f}"
            ]

        self.sumObj = self.GD_ObjValue + self.SD_ObjValue + self.NTD_ObjValue + self.NMD_ObjValue \
                      + self.RD_ObjValue + self.NBD_ObjValue + self.CF_ObjValue

        self.sumRuntime = self.GD_runtime + self.SD_runtime + self.NTD_runtime + self.NMD_runtime \
                          + self.RD_runtime + self.NBD_runtime + self.CF_runtime

        # ------------------------------------------------------------------
        rows = [
            make_row('GD', self.GD_ObjValue, self.GD_runtime),
            make_row('SD', self.SD_ObjValue, self.SD_runtime),
            make_row('NTD', self.NTD_ObjValue, self.NTD_runtime),
            make_row('NMD', self.NMD_ObjValue, self.NMD_runtime),
            make_row('RD', self.RD_ObjValue, self.RD_runtime),
            make_row('NBD', self.NBD_ObjValue, self.NBD_runtime),
            make_row('CF', self.CF_ObjValue, self.CF_runtime),
            make_row('Total', self.sumObj, self.sumRuntime),
        ]

        df = pd.DataFrame(rows, columns=cols)

        # ------------------------------------------------------------------
        # append to CSV (write header only if file does not yet exist)
        file_exists = os.path.isfile(filePath)
        df.to_csv(filePath,
                  mode='a',  # append mode
                  header=not file_exists,  # write header once
                  index=False)

        # ── add a blank line separator at the end ──────────────────────
        with open(filePath, 'a', newline='') as f:
            f.write('\n')

        print(f"Results appended to {filePath}")

    def visualizeClusterSolution(self, instance):

        # self.setVisualization(True, True, True, True)

        def polar_offset(x0, y0, r, theta_deg):
            theta = np.deg2rad(theta_deg)
            return x0 + r * np.cos(theta), y0 + r * np.sin(theta)

        xs = [p['x'] for p in instance.scenario.poses]
        ys = [p['y'] for p in instance.scenario.poses]

        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(16, 6))
        fig.canvas.manager.set_window_title("Greedy Solution")

        # -------------------- Full Deployment Plot (left: ax2) --------------------
        ax2.set_title("Agent Deployment")
        W = instance.scenario.regionMap[instance.scenario.size]['W']
        H = instance.scenario.regionMap[instance.scenario.size]['H']
        ax2.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor='black', linestyle='-', linewidth=2.5))

        # 1) candidate poses + ID
        ax2.scatter(xs, ys, c='blue', s=80, edgecolors='black', label='Pose')
        for pid, (x, y) in enumerate(zip(xs, ys)):
            ax2.text(x - 0.04, y, f"P{pid + 1}", ha='right', va='center', fontsize=8, color='blue')

        # 2) Actuators
        if self.isVisualize.get('Actuator', False) and instance.scenario.xa is not None:
            for j, is_act in enumerate(instance.scenario.xa):
                if is_act:
                    x, y = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
                    ax2.scatter(*polar_offset(x, y, 0.08, 90), c='orange', marker='^', s=80, edgecolors='black',
                                label='Actuator' if 'Actuator' not in ax2.get_legend_handles_labels()[1] else "")

        # 3) Gateways
        if self.isVisualize.get('Gateway', False):
            gate_r = instance.agent.models['Gateway']['commRange']
            for k, j in enumerate(self.gateway_indices):
                gx0, gy0 = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
                x, y = polar_offset(gx0, gy0, 0.09, 0)
                ax2.scatter(x, y, c='red', marker='s', s=80, edgecolors='black', label='Gateway' if k == 0 else "")
                ax2.add_patch(Circle((gx0, gy0), gate_r, alpha=.12, color='red', linewidth=0))

        # 3B) Routers
        if hasattr(self, 'router_indices') and self.router_indices and self.isVisualize.get('Router', True):
            for k, j in enumerate(self.router_indices):
                xr, yr = polar_offset(instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y'], 0.1, 135)
                ax2.scatter(xr, yr, c='magenta', marker='D', s=70, edgecolors='black', label='Router' if k == 0 else "")

        # 4) Sensor nodes
        if self.isVisualize.get('SensorNode', False):
            sensor_colors = list(mcolors.TABLEAU_COLORS.values())
            max_per_row, cell, dot_r = 2, 0.06, 0.025
            sensor_types_used = set()

            for node_id in self.sensor_node_indices:
                x0, y0 = polar_offset(instance.scenario.poses[node_id]['x'], instance.scenario.poses[node_id]['y'],
                                      0.08, 270)
                sens_here = [s for (pid, s), v in self.sensor_assignments.items() if pid == node_id and v > 0.5]
                if not sens_here:
                    continue
                rows = (len(sens_here) + max_per_row - 1) // max_per_row
                cols = min(max_per_row, len(sens_here))
                w, h = cols * cell, rows * cell
                ax2.add_patch(Rectangle((x0 - w / 2, y0 - h / 2), w, h, fc='lightgreen', ec='green', lw=1.2))
                ax2.text(x0, y0 - h / 2 - 0.02, f"Node {node_id + 1}", ha='center', va='top', fontsize=8, color='green')
                for idx, s in enumerate(sens_here):
                    gx = (idx % max_per_row + 0.5) * cell - w / 2
                    gy = h / 2 - (idx // max_per_row + 0.5) * cell
                    ax2.add_patch(Circle((x0 + gx, y0 + gy), dot_r, color=sensor_colors[s % len(sensor_colors)]))
                    sensor_types_used.add(s)

        # Legend
        handles = []
        if self.isVisualize.get('SensorNode', False):
            sensor_patches = [mpatches.Patch(color=sensor_colors[s % len(sensor_colors)], label=f"Sensor {s + 1}") for s
                              in sorted(sensor_types_used)]
            node_patch = mpatches.Patch(facecolor='lightgreen', edgecolor='green', label='Sensor Node')
            handles.extend(sensor_patches)
            handles.append(node_patch)

        if self.isVisualize.get('Gateway', False):
            handles.append(
                mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=9, markeredgecolor='black',
                              label='Gateway'))

        if self.isVisualize.get('Router', True) and self.router_indices:
            handles.append(mlines.Line2D([], [], color='magenta', marker='D', linestyle='None', markersize=9,
                                         markeredgecolor='black', label='Router'))

        if self.isVisualize.get('Actuator', False):
            handles.append(mlines.Line2D([], [], color='orange', marker='^', linestyle='None', markersize=9,
                                         markeredgecolor='black', label='Actuator'))

        handles.append(
            mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, markeredgecolor='black',
                          label='Pose'))
        ax2.legend(handles=handles, loc='lower left')

        ax2.set_xlabel("X (km)")
        ax2.set_ylabel("Y (km)")
        ax2.set_xlim(-0.5, W + 0.5)
        ax2.set_ylim(-0.5, H + 0.5)
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True)

        # -------------------- Candidate Poses Only Plot (right: ax1) --------------------
        ax1.set_title("Clusters and Connectivity")
        ax1.scatter(xs, ys, c='blue', s=60, edgecolors='black', label='Pose')
        for pid, (x, y) in enumerate(zip(xs, ys)):
            ax1.text(x - 0.03, y + 0.03, f"P{pid + 1}", ha='right', va='bottom', fontsize=8, color='blue')

        ax1.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor='black', linestyle='-', linewidth=2.5))
        ax1.set_xlabel("X (km)")
        ax1.set_ylabel("Y (km)")
        ax1.set_xlim(-0.5, W + 0.5)
        ax1.set_ylim(-0.5, H + 0.5)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True)
        # ax1.legend(loc='upper right')

        # --------- Cluster Visualization Based on Gateway Activation (xg) ---------
        cluster_colors = ['yellow', 'olive', 'orange', 'brown', 'red']
        cluster_legend = []
        cluster_index = 1  # for legend labels

        for idx, I in enumerate(self.N):
            if self.xg.get(I, 0) > 0.5:
                color = cluster_colors[(cluster_index - 1) % len(cluster_colors)]
                label = f"cluster {cluster_index}"
                cluster_index += 1

                # Plot gateway I as rectangle in cluster color
                gx0, gy0 = instance.scenario.poses[I]['x'], instance.scenario.poses[I]['y']
                gx, gy = polar_offset(gx0, gy0, 0.09, 0)
                ax1.scatter(gx, gy, c=color, marker='s', s=90, edgecolors='black')

                # Plot agents (j, k) in cluster I
                for j in self.N:
                    for k in (0, 1, 2):
                        if self.uu.get((j, k, I), 0) > 0.5:
                            px, py = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
                            if k == 0:
                                # Rectangle for k=0 (below)
                                x0, y0 = polar_offset(px, py, 0.08, 270)
                                width, height = 0.15, 0.08
                                rect = Rectangle((x0 - width / 2, y0 - height / 2), width, height,
                                                 facecolor=color, edgecolor='black')
                                ax1.add_patch(rect)
                            elif k == 1:
                                # Triangle pointing up for k=1
                                angle = 90
                                xk, yk = polar_offset(px, py, 0.08, angle)
                                ax1.scatter(xk, yk, c=color, marker='^', s=80, edgecolors='black')
                            elif k == 2:
                                # Diamond for k=2
                                angle = 135
                                xk, yk = polar_offset(px, py, 0.08, angle)
                                ax1.scatter(xk, yk, c=color, marker='D', s=80, edgecolors='black')

                # Add to legend (only once per cluster)
                cluster_legend.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                                    markersize=9, markeredgecolor='black', label=label))

        # Add cluster legend to ax2
        ax1.legend(handles=cluster_legend, loc='upper right')

        fig.tight_layout()
        plt.show()

    def visualizeDeploymentSolution(self, instance):
        # --------------------------------------------------------------- helpers
        #self.setVisualization(True, True, True, True)  # ensure flags exist

        def polar_offset(x0, y0, r, theta_deg):
            theta = np.deg2rad(theta_deg)
            return x0 + r * np.cos(theta), y0 + r * np.sin(theta)

        # ---------------------------------------------------------------- poses
        xs = [p['x'] for p in instance.scenario.poses]
        ys = [p['y'] for p in instance.scenario.poses]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title("Greedy Solution")
        ax.set_title("Greedy Agent Deployment")

        # ───────────────────── scenario boundary ──────────────────────────
        W = instance.scenario.regionMap[instance.scenario.size]['W']
        H = instance.scenario.regionMap[instance.scenario.size]['H']
        ax.add_patch(Rectangle(
            (0, 0), W, H,
            fill=False,
            edgecolor='black',  # darker color for visibility
            linestyle='-',  # solid line for stronger visibility
            linewidth=2.5,  # thicker border
            label='Scenario Boundary'
        ))
        # ──────────────────────────────────────────────────────────────────

        # 1) candidate poses + ID
        ax.scatter(xs, ys, c='blue', s=80, edgecolors='black', label='Pose')
        for pid, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x - 0.04, y, f"P{pid + 1}",
                    ha='right', va='center', fontsize=8, color='blue')

        # 2) actuators (above)
        if self.isVisualize.get('Actuator', False) and instance.scenario.xa is not None:
            for j, is_act in enumerate(instance.scenario.xa):
                if is_act:
                    x, y = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
                    ax.scatter(*polar_offset(x, y, 0.08, 90),
                               c='orange', marker='^', s=80, edgecolors='black',
                               label='Actuator' if 'Actuator' not in ax.get_legend_handles_labels()[1] else "")

        # 3) gateways (right)
        if self.isVisualize.get('Gateway', False):
            gate_r = instance.agent.models['Gateway']['commRange']
            for k, j in enumerate(self.gateway_indices):
                gx0, gy0 = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
                x, y = polar_offset(gx0, gy0, 0.09, 0)
                ax.scatter(x, y, c='red', marker='s', s=80, edgecolors='black',
                           label='Gateway' if k == 0 else "")
                ax.add_patch(Circle((gx0, gy0), gate_r, alpha=.12, color='red', linewidth=0))

        # 3‑B) routers (above–left at 135°)
        if hasattr(self, 'router_indices') and self.router_indices \
                and self.isVisualize.get('Router', True):

            for k, j in enumerate(self.router_indices):
                xr, yr = polar_offset(
                    instance.scenario.poses[j]['x'],
                    instance.scenario.poses[j]['y'],
                    0.1,  # radius
                    135  # 135°  = up‑left
                )
                ax.scatter(xr, yr, c='magenta', marker='D', s=70, edgecolors='black',
                           label='Router' if k == 0 else "")

                # pose‑ID label next to the router icon (little offset further up‑left)
                # ax.text(xr - 0.03, yr + 0.03, f"P{j + 1}", ha='right', va='bottom', fontsize=7, color='magenta')

        # 4) sensor‑node rectangles (below) + dots
        if self.isVisualize.get('SensorNode', False):
            sensor_colors = list(mcolors.TABLEAU_COLORS.values())
            max_per_row, cell, dot_r = 2, 0.06, 0.025
            sensor_types_used = set()

            for node_id in self.sensor_node_indices:
                x0, y0 = polar_offset(instance.scenario.poses[node_id]['x'],
                                      instance.scenario.poses[node_id]['y'], 0.08, 270)
                sens_here = [s for (pid, s), v in self.sensor_assignments.items()
                             if pid == node_id and v > 0.5]
                if not sens_here:
                    continue
                rows = (len(sens_here) + max_per_row - 1) // max_per_row
                cols = min(max_per_row, len(sens_here))
                w, h = cols * cell, rows * cell
                ax.add_patch(Rectangle((x0 - w / 2, y0 - h / 2), w, h,
                                       fc='lightgreen', ec='green', lw=1.2))
                ax.text(x0, y0 - h / 2 - 0.02, f"Node {node_id + 1}",
                        ha='center', va='top', fontsize=8, color='green')
                for idx, s in enumerate(sens_here):
                    gx = (idx % max_per_row + 0.5) * cell - w / 2
                    gy = h / 2 - (idx // max_per_row + 0.5) * cell
                    ax.add_patch(Circle((x0 + gx, y0 + gy), dot_r,
                                        color=sensor_colors[s % len(sensor_colors)]))
                    sensor_types_used.add(s)

        # ---------------------------------------------------------------- legend
        handles = []

        if self.isVisualize.get('SensorNode', False):
            sensor_patches = [mpatches.Patch(color=sensor_colors[s % len(sensor_colors)],
                                             label=f"Sensor {s + 1}")
                              for s in sorted(sensor_types_used)]
            node_patch = mpatches.Patch(facecolor='lightgreen', edgecolor='green',
                                        label='Sensor Node')
            handles.extend(sensor_patches)
            handles.append(node_patch)

        if self.isVisualize.get('Gateway', False):
            handles.append(mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                                         markersize=9, markeredgecolor='black', label='Gateway'))

        if self.isVisualize.get('Router', True) and self.router_indices:
            handles.append(mlines.Line2D([], [], color='magenta', marker='D', linestyle='None',
                                         markersize=9, markeredgecolor='black', label='Router'))

        if self.isVisualize.get('Actuator', False):
            handles.append(mlines.Line2D([], [], color='orange', marker='^', linestyle='None',
                                         markersize=9, markeredgecolor='black', label='Actuator'))

        handles.append(mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                     markersize=8, markeredgecolor='black', label='Pose'))

        ax.legend(handles=handles, loc='lower left')

        # ---------------------------------------------------------------- axes cosmetics
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_xlim(-0.5, instance.scenario.regionMap[instance.scenario.size]['W'] + 0.5)
        ax.set_ylim(-0.5, instance.scenario.regionMap[instance.scenario.size]['H'] + 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    def visualizeSolution(self, instance, isCluster):
        if isCluster:
            self.visualizeClusterSolution(instance)
        else:
            self.visualizeDeploymentSolution(instance)

