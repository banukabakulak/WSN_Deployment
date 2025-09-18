# import cplex
from docplex.mp.model import Model
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import time
import os

class CentralizedNAD:
    def __init__(self, epsilon, delta, isRelease = False):
        self.epsilon = epsilon
        self.delta = delta
        self.isRelease = isRelease

    def initializeModel(self, instance):
        # Create CPLEX model
        self.CNAD_mdl = Model(name="CNAD model")
        self.isFeasible = False

        # Index sets
        self.N = range(instance.N)
        self.K = range(4)
        self.S = range(instance.S)
        self.B = range(len(instance.battery.models))
        self.M = range(len(instance.memory.models))
        self.T = range(len(instance.transceiver.models))
        # self.L = range(instance.L)

        self.nodeMap = {
            0: 'SensorNode',
            1: 'Actuator',
            2: 'Router',
            3: 'Gateway'
        }

        # Create decision variables
        # x[j, k] - binary
        self.x = self.CNAD_mdl.binary_var_dict(
            ((j, k) for j in self.N for k in self.K),
            name="x"
        )

        # ss[j, s] - binary
        self.ss = self.CNAD_mdl.binary_var_dict(
            ((j, s) for j in self.N for s in self.S),
            name="s"
        )

        # bb[j, b] - integer
        self.bb = self.CNAD_mdl.integer_var_dict(
            ((j, b) for j in self.N for b in self.B),
            name="b"
        )

        # mm[j, m] - integer
        self.mm = self.CNAD_mdl.integer_var_dict(
            ((j, m) for j in self.N for m in self.M),
            name="m"
        )

        # tt[j, t] - binary
        self.tt = self.CNAD_mdl.binary_var_dict(
            ((j, t) for j in self.N for t in self.T),
            name="t"
        )

        self.y = self.CNAD_mdl.continuous_var_dict(
            ((j, k, i, l) for j in self.N for k in self.K for i in self.N for l in self.K),
            name="y"
        )

        self.u = self.CNAD_mdl.binary_var_dict(
            ((j, k, i) for j in self.N for k in self.K for i in self.N),
            name="u"
        )

        # Create auxiliary decision variables
        # chi[i, l, j] - binary
        self.chi = self.CNAD_mdl.binary_var_dict(
            ((i, l, j) for i in self.N for l in self.K for j in self.N),
            name="chi"
        )

        # phi[i, j] - continuous
        self.phi = self.CNAD_mdl.continuous_var_dict(
            ((i, j) for i in self.N for j in self.N),
            name="phi"
        )

        # psi[i, j] - continuous
        self.psi = self.CNAD_mdl.continuous_var_dict(
            ((i, j) for i in self.N for j in self.N),
            name="psi"
        )

        # theta[i, j] - continuous
        self.theta = self.CNAD_mdl.continuous_var_dict(
            ((i, j) for i in self.N for j in self.N),
            name="theta"
        )

        # Create sensor node specification variables
        # 1D continuous variables: c[j], h[j], w[j], d[j], e_a[j], e_tx[j], e_rx[j], E[j], M[j]
        self.c = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="c")
        self.h = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="h")
        self.w = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="w")
        self.d = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="d")
        self.e_a = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="e_a")
        self.e_tx = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="e_tx")
        self.e_rx = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="e_rx")
        self.Energy = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="E")
        self.Memory = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="M")

        self.E_hat = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="E_hat")
        self.O = self.CNAD_mdl.continuous_var_dict((j for j in self.N), name="O")

        # 2D binary variable: q[j, i]
        self.q = self.CNAD_mdl.binary_var_dict(
            ((j, i) for j in self.N for i in self.N),
            name="q"
        )

        # Big number H
        maxH = 0
        for s in self.S:
            if instance.sensor.models[f"S{s+1}"]['dataSize'] > maxH:
                maxH = instance.sensor.models[f"S{s+1}"]['dataSize']
        self.H = instance.box.attributes['P'] * maxH

    def defineObjective(self, instance):
        obj_expr = self.CNAD_mdl.linear_expr()

        # First term: c[j]
        for j in self.N:
            obj_expr += self.c[j]

        # Second term: x[j, r] + x[j, g]
        for j in self.N:
            obj_expr += (instance.agent.models['Router']['cost'] * self.x[j, 2] +
                         instance.agent.models['Gateway']['cost'] * self.x[j, 3])
            # for i in self.N:
            #     obj_expr += (self.psi[j, i] + self.theta[j, i])

        # Set the objective
        self.CNAD_mdl.minimize(obj_expr)

    def actuatorAssignmentConstraint(self, instance):
        # Constraints : Set the positions of actuators
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.x[j, 1]
            self.CNAD_mdl.add_constraint(lin_expr == instance.scenario.xa[j], "actuator_poses")

    def sensorNodeCostConstraint(self, instance):
        # Constraints: Sensor Node Cost
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.c[j] - instance.box.attributes['cost'] * self.x[j, 0]
            for s in self.S:
                lin_expr -= instance.sensor.models[f"S{s + 1}"]['cost'] * self.ss[j, s]
            for t in self.T:
                lin_expr -= instance.transceiver.models[f"T{t + 1}"]['cost'] * self.tt[j, t]
            for m in self.M:
                lin_expr -= instance.memory.models[f"M{m + 1}"]['cost'] * self.mm[j, m]
            for b in self.B:
                lin_expr -= instance.battery.models[f"B{b + 1}"]['cost'] * self.bb[j, b]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "node_cost")

    def sensorAssignmentConstraint(self, instance):
        # Constraints: Gateway Coverage
        for j in self.N:
            for k in (0, 1, 2):
                lin_expr = self.CNAD_mdl.linear_expr()
                for i in self.N:
                    commStatus = instance.scenario.isCommunicate('Gateway', instance.scenario.poses[i],
                                                                instance.scenario.poses[j], instance.agent)
                    lin_expr += commStatus * self.x[i, 3]
                lin_expr -= instance.agent.models[self.nodeMap[k]]['gateCover']* self.x[j, k]
                self.CNAD_mdl.add_constraint(lin_expr >= 0, "gate_cover")

        # Constraints: Feasible Sensor to Node
        for j in self.N:
            for s in self.S:
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += (self.ss[j, s] - self.x[j, 0])
                self.CNAD_mdl.add_constraint(lin_expr <= 0, "feas_sensor1")

        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.x[j, 0]
            for s in self.S:
                lin_expr -= self.ss[j, s]
            self.CNAD_mdl.add_constraint(lin_expr <= 0, "feas_sensor2")

        # Constraints: Feasible Sensor in Box
        for j in self.N:
                lin_expr = self.CNAD_mdl.linear_expr()
                for s in self.S:
                    lin_expr += self.ss[j, s]
                #self.CNAD_mdl.add_constraint(lin_expr <= instance.box.attributes['P'], "feas_sensor_box")
                self.CNAD_mdl.add_constraint(lin_expr <= instance.S,
                                             "feas_sensor_box")
        # Constraints: Feasible Sensor in Volume
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            for s in self.S:
                lin_expr += instance.sensor.models[f"S{s+1}"]['volume'] * self.ss[j, s]
            lin_expr -= instance.box.attributes['volume'] * self.x[j, 0]
            self.CNAD_mdl.add_constraint(lin_expr <= 0,"feas_sensor_vol")

        # Constraints: Sensor Coverage
        for i in self.N:
            for s in self.S:
                lin_expr = self.CNAD_mdl.linear_expr()
                for j in self.N:
                    coverStatus = instance.scenario.isSense(f"S{s+1}", instance.scenario.poses[j],
                                                            instance.scenario.poses[i], instance.sensor)
                    lin_expr += coverStatus * self.ss[j, s]
                self.CNAD_mdl.add_constraint(lin_expr >= instance.scenario.poses[i]['coverReq'][s],
                                             "sensor_cover")

        # Constraints: Sensor Node Data Size
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.h[j]
            for s in self.S:
                lin_expr -= instance.sensor.models[f"S{s+1}"]['dataSize'] * self.ss[j, s]
            self.CNAD_mdl.add_constraint(lin_expr == 0,"sensor_node_data")

    def transceiverAssignmentConstraint(self, instance):
        # Constraints: Feasible Transceiver to Node
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            for t in self.T:
                lin_expr += self.tt[j, t]
            lin_expr -= self.x[j, 0]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "feas_transceiver_node")

        # Constraints: Node CommRange
        for i in self.N:
            for j in self.N:
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.q[j, i]
                for t in self.T:
                    commStatus = instance.scenario.isCommunicate(f"T{t+1}", instance.scenario.poses[j],
                                                        instance.scenario.poses[i], instance.transceiver)
                    lin_expr -= commStatus * self.tt[j, t]
                self.CNAD_mdl.add_constraint(lin_expr == 0, "node_commRange")

        '''
        # Constraints: Node Connectivity -- NONLINEAR
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr -= instance.agent.models['SensorNode']['lambda'] * self.x[j, 0]
            for i in self.N:
                lin_expr += self.q[j, i] * (self.x[i, 2] + self.x[i, 3])
                if i != j:
                    lin_expr += self.q[j, i] * self.x[i, 0]
            self.CNAD_mdl.add_constraint(lin_expr >= 0, "node_connectivity")
        '''


        # Constraints: Node Connectivity -- Linearized
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr -= instance.agent.models['SensorNode']['lambda'] * self.x[j, 0]
            for i in self.N:
                lin_expr += (self.chi[i, 2, j] + self.chi[i, 3, j])
                if not (i == j):
                    lin_expr += self.chi[i, 0, j]
            self.CNAD_mdl.add_constraint(lin_expr >= 0, "node_comm_lin1")

        for i in self.N:
            for l in (0, 2, 3):
                for j in self.N:
                    lin_expr = self.CNAD_mdl.linear_expr()
                    lin_expr += (self.chi[i, l, j] - self.x[i, l])
                    self.CNAD_mdl.add_constraint(lin_expr <= 0, "node_comm_lin2")

                    lin_expr = self.CNAD_mdl.linear_expr()
                    lin_expr += (self.chi[i, l, j] - self.q[j, i])
                    self.CNAD_mdl.add_constraint(lin_expr <= 0, "node_comm_lin3")

                    lin_expr = self.CNAD_mdl.linear_expr()
                    lin_expr += (self.chi[i, l, j] - self.x[i, l] - self.q[j, i] + 1)
                    self.CNAD_mdl.add_constraint(lin_expr >= 0, "node_comm_lin4")

        # Constraints: Actuator, Router Connectivity
        for k in (1, 2):
            for j in self.N:
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr -= instance.agent.models[self.nodeMap[k]]['lambda'] * self.x[j, k]
                for i in self.N:
                    for l in 0, 2, 3:
                        if not (i == j and k == l):
                            commStatus = instance.scenario.isCommunicate(self.nodeMap[k], instance.scenario.poses[j],
                                                                         instance.scenario.poses[i], instance.agent)
                            lin_expr += commStatus * self.x[i, l]
                self.CNAD_mdl.add_constraint(lin_expr >= 0, "a_r_connectivity")

        # Constraints: Node Bandwidth
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.d[j]
            for t in self.T:
                lin_expr -= instance.transceiver.models[f"T{t+1}"]["bandwidth"] * self.tt[j, t]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "node_bandwidth")

        '''
        # Constraints: Feasible Node Bandwidth -- NONLINEAR
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += (self.d[j] - self.h[j])
            for i in self.N:
                lin_expr -= (self.q[i, j] * self.h[i])
                for l in (1, 2):
                    commStatus = instance.scenario.isCommunicate(self.nodeMap[l], instance.scenario.poses[i],
                                                                 instance.scenario.poses[j], instance.agent)
                    lin_expr -= commStatus * instance.agent.models[self.nodeMap[l]]['dataSize'] * self.x[i, l]
            self.CNAD_mdl.add_constraint(lin_expr >= 0, "feas_node_bandwidth")
        '''

        # Constraints: Feasible Node Bandwidth -- Linearized
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += (self.d[j] - self.h[j])
            for i in self.N:
                lin_expr -= self.phi[i, j]
                for l in (1, 2):
                    commStatus = instance.scenario.isCommunicate(self.nodeMap[l], instance.scenario.poses[i],
                                                                 instance.scenario.poses[j], instance.agent)
                    lin_expr -= commStatus * instance.agent.models[self.nodeMap[l]]['dataSize'] * self.x[i, l]
            lin_expr += 3 * self.H * instance.N * (1 - self.x[j, 0])
            self.CNAD_mdl.add_constraint(lin_expr >= 0, "feas_node_bandwidth_lin1")

        for i in self.N:
            for j in self.N:

                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.phi[i, j]
                lin_expr -= 3 * self.H * instance.N * self.q[i, j]
                self.CNAD_mdl.add_constraint(lin_expr <= 0, "feas_node_bandwidth_lin2")

                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.phi[i, j]
                lin_expr -= (self.h[i] - 3 * self.H * instance.N * (1 - self.q[i, j]))
                self.CNAD_mdl.add_constraint(lin_expr >= 0, "feas_node_bandwidth_lin3")

    def memoryAssignmentConstraint(self, instance):
        # Constraints: Feasible Memory to Node
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            for m in self.M:
                lin_expr += self.mm[j, m]
            lin_expr -= self.x[j, 0]
            self.CNAD_mdl.add_constraint(lin_expr >= 0, "feas1_memory_node")

            lin_expr = self.CNAD_mdl.linear_expr()
            for m in self.M:
                lin_expr += self.mm[j, m]
            self.CNAD_mdl.add_constraint(lin_expr <= instance.box.attributes['C'], "feas2_memory_node")

            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.Memory[j]
            for m in self.M:
                lin_expr -= instance.memory.models[f"M{m+1}"]['capacity'] * self.mm[j, m]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "feas3_memory_node")

        # Constraints: Feasible Node Memory -- Linearized
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += (self.Memory[j] - self.h[j])
            for i in self.N:
                lin_expr -= self.phi[i, j]
                for l in (1, 2):
                    commStatus = instance.scenario.isCommunicate(self.nodeMap[l], instance.scenario.poses[i],
                                                                 instance.scenario.poses[j], instance.agent)
                    lin_expr -= commStatus * instance.agent.models[self.nodeMap[l]]['dataSize'] * self.x[i, l]
                    lin_expr += self.H * instance.N * (1 - self.x[j, 0])
            self.CNAD_mdl.add_constraint(lin_expr >= 0, "feas_node_memory_lin")

    def flowConstraints(self, instance):
        # Constraints: delta-hop flow
        for j in self.N:
            for k in (0, 2):
                lin_expr = self.CNAD_mdl.linear_expr()
                for i in self.N:
                    for l in (0, 1 ,2):
                        lin_expr += self.y[i, l, j, k]
                lin_expr -= self.delta * self.x[j, k]
                self.CNAD_mdl.add_constraint(lin_expr <= 0, "n_r_delta_hop_flow")

        for j in self.N:
            for i in self.N:
                for l in (0, 1, 2):
                    lin_expr = self.CNAD_mdl.linear_expr()
                    lin_expr += self.y[i, l, j, 3]
                    lin_expr -= self.delta * self.x[j, 3]
                    self.CNAD_mdl.add_constraint(lin_expr <= 0, "g_delta_hop_flow")

        # Constraints: node outflow
        for j in self.N:
            for k in (0, 1, 2):
                lin_expr = self.CNAD_mdl.linear_expr()
                for i in self.N:
                    for l in (0, 2, 3):
                        lin_expr += self.y[j, k, i, l]
                lin_expr -= self.x[j, k]
                self.CNAD_mdl.add_constraint(lin_expr >= 0, "node_outflow")

        # Constraints: node flow balance
        for j in self.N:
            for k in (0, 2):
                lin_expr = self.CNAD_mdl.linear_expr()
                for i in self.N:
                    for l in (0, 2, 3):
                        lin_expr += self.y[j, k, i, l]
                    for l in (0, 1, 2):
                        lin_expr -= self.y[i, l, j, k]
                lin_expr -= self.x[j, k]
                self.CNAD_mdl.add_constraint(lin_expr == 0, "n_r_node_flow_balance")

        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr += self.y[j, 1, i, l]
            lin_expr -= self.x[j, 1]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "a_node_flow_balance")

        # Constraints: gateway inflow
        lin_expr = self.CNAD_mdl.linear_expr()
        for j in self.N:
            for k in (0, 1, 2):
                for i in self.N:
                    lin_expr += self.y[j, k, i, 3]
                lin_expr -= self.x[j, k]
        self.CNAD_mdl.add_constraint(lin_expr == 0, "gate_inflow")

        # Constraints: Feasible Flow
        for j in self.N:
            for k in (1, 2):
                for i in self.N:
                    for l in (0, 2, 3):
                        lin_expr = self.CNAD_mdl.linear_expr()
                        lin_expr += self.y[j, k, i, l]
                        commStatus = instance.scenario.isCommunicate(self.nodeMap[k], instance.scenario.poses[j],
                                                                     instance.scenario.poses[i], instance.agent)
                        self.CNAD_mdl.add_constraint(lin_expr <= self.delta * commStatus, "feas1_flow")

        for j in self.N:
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr = self.CNAD_mdl.linear_expr()
                    lin_expr += (self.y[j, 0, i, l] - self.delta * self.q[j, i])
                    self.CNAD_mdl.add_constraint(lin_expr <= 0, "feas2_flow")

        for j in self.N:
            for i in self.N:
                for l in (0, 1, 2):
                    lin_expr = self.CNAD_mdl.linear_expr()
                    lin_expr += self.y[i, l, j, 1]
                    self.CNAD_mdl.add_constraint(lin_expr == 0, "feas3_flow")

        for j in self.N:
            for i in self.N:
                for l in (0, 1, 2):
                    lin_expr = self.CNAD_mdl.linear_expr()
                    lin_expr += self.y[j, 3, i, l]
                    self.CNAD_mdl.add_constraint(lin_expr == 0, "feas4_flow")

        for i in self.N:
            for l in self.K:
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.y[i, l, i, l]
                self.CNAD_mdl.add_constraint(lin_expr == 0, "feas5_flow")

    def energyConstraints(self, instance):
        # Constraints: node energy specifications
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.e_a[j]
            for t in self.T:
                lin_expr -= instance.transceiver.models[f"T{t+1}"]['activeEnergy'] * self.tt[j, t]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "node_active_energy")

            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.e_rx[j]
            for t in self.T:
                lin_expr -= instance.transceiver.models[f"T{t + 1}"]['rxEnergy'] * self.tt[j, t]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "node_rx_energy")

            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.e_tx[j]
            for t in self.T:
                lin_expr -= instance.transceiver.models[f"T{t + 1}"]['txEnergy'] * self.tt[j, t]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "node_tx_energy")

        # Constraints: node estimated period energy
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += (self.E_hat[j] - self.e_a[j])
            for s in self.S:
                lin_expr -= instance.sensor.models[f"S{s+1}"]['senseEnergy'] * self.ss[j, s]
            # node lamda connectivity
            self.node_lambda = instance.agent.models['SensorNode']['lambda']
            lin_expr -= self.node_lambda * self.H * self.e_rx[j]
            lin_expr -= (self.node_lambda + 1) * self.H * self.e_tx[j]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "node_estimated_energy")

    def batteryAssignmentConstraint(self, instance):
        # Constraints : feasible Battery to Node
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr -= self.x[j, 0]
            for b in self.B:
                lin_expr += self.bb[j, b]
            self.CNAD_mdl.add_constraint(lin_expr >= 0, "feas1_node_battery")

            lin_expr = self.CNAD_mdl.linear_expr()
            for b in self.B:
                lin_expr += self.bb[j, b]
            self.CNAD_mdl.add_constraint(lin_expr <= instance.box.attributes['D'], "feas2_node_battery")

            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.Energy[j]
            for b in self.B:
                lin_expr -= instance.battery.models[f"B{b+1}"]['capacity'] * self.bb[j, b]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "node_battery_energy")

            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.Energy[j]
            lin_expr -= instance.L * self.E_hat[j]
            self.CNAD_mdl.add_constraint(lin_expr >= 0, "node_battery_lifetime")

    def clusterConstraints(self, instance):
        # Constraints: Feasible cluster
        for j in self.N:
            for k in (0, 1, 2):
                lin_expr = self.CNAD_mdl.linear_expr()
                for i in self.N:
                    lin_expr += self.u[j, k, i]
                lin_expr -= self.x[j, k]
                self.CNAD_mdl.add_constraint(lin_expr == 0, "feas1_cluster")

        # Constraints: Feasible cluster
        for i in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            for j in self.N:
                for k in (0, 1, 2):
                    lin_expr += self.u[j, k, i]
            lin_expr -= 3 * instance.N * self.x[i, 3]
            self.CNAD_mdl.add_constraint(lin_expr <= 0, "feas2_cluster")

        # Constraints: Feasible cluster
        for i in self.N:
            for j in self.N:
                for k in (0, 1, 2):
                    lin_expr = self.CNAD_mdl.linear_expr()
                    lin_expr += self.u[j, k, i]
                    commStatus = instance.scenario.isCommunicate('Gateway', instance.scenario.poses[i],
                                                                 instance.scenario.poses[j], instance.agent)
                    lin_expr -= commStatus * self.x[j, k]
                    self.CNAD_mdl.add_constraint(lin_expr <= 0, "feas3_cluster")

        # Constraints: Node overhead
        for j in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += (self.w[j] - instance.agent.models['SensorNode']['gateCommand'] * self.x[j, 0] - self.h[j])
            for s in self.S:
                lin_expr -= instance.sensor.models[f"S{s+1}"]['gateCommand'] * self.ss[j, s]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "node_overhead")

        '''
        # Constraints: Cluster overhead -- NONLINEAR
        for i in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.O[i]
            for j in self.N:
                lin_expr -= self.w[j] * self.u[j, 0, i]
                for k in (1, 2):
                    lin_expr -= (instance.agent.models[self.nodeMap[k]]['gateCommand'] +
                                 instance.agent.models[self.nodeMap[k]]['dataSize']) * self.u[j, k, i]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "cluster_overhead")
        '''

        # Constraints: Cluster overhead -- Linearized
        for i in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            lin_expr += self.O[i]
            for j in self.N:
                lin_expr -= self.psi[j, i]
                for k in (1, 2):
                    lin_expr -= (instance.agent.models[self.nodeMap[k]]['gateCommand'] +
                                 instance.agent.models[self.nodeMap[k]]['dataSize']) * self.u[j, k, i]
            self.CNAD_mdl.add_constraint(lin_expr == 0, "cluster_overhead_lin1")

            for j in self.N:
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.psi[j, i]
                lin_expr -= self.H * instance.N * self.u[j, 0, i]
                self.CNAD_mdl.add_constraint(lin_expr <= 0, "cluster_overhead_lin2")

                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.psi[j, i]
                lin_expr -= (self.w[j] - self.H  * instance.N * (1 - self.u[j, 0, i]))
                self.CNAD_mdl.add_constraint(lin_expr >= 0, "cluster_overhead_lin3")
                '''
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.psi[j, i]
                lin_expr -= (self.w[j] + self.H * instance.N * (1 - self.u[j, 0, i]))
                self.CNAD_mdl.add_constraint(lin_expr <= 0, "cluster_overhead_lin4")
                '''
        '''        
        # Constraints : Cluster deviation -- NONLINEAR 
        for i in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            for j in self.N:
                lin_expr += (1 - self.epsilon) * self.O[j]
                lin_expr -= self.x[j, 3] * self.O[i]
            self.CNAD_mdl.add_constraint(lin_expr <= 0, "cluster_deviation1")

            lin_expr = self.CNAD_mdl.linear_expr()
            for j in self.N:
                lin_expr -= (1 + self.epsilon) * self.O[j]
                lin_expr += self.x[j, 3] * self.O[i]
            self.CNAD_mdl.add_constraint(lin_expr <= 0, "cluster_deviation2")
        '''

        # Constraints : Cluster deviation -- Linearized
        for i in self.N:
            lin_expr = self.CNAD_mdl.linear_expr()
            for j in self.N:
                lin_expr += (1 - self.epsilon) * self.O[j]
                lin_expr -= self.theta[i, j]
            lin_expr -= 3 * self.H * instance.N * (1 - self.x[i, 3])
            self.CNAD_mdl.add_constraint(lin_expr <= 0, "cluster_deviation1_lin")

            lin_expr = self.CNAD_mdl.linear_expr()
            for j in self.N:
                lin_expr -= (1 + self.epsilon) * self.O[j]
                lin_expr += self.theta[i, j]
            lin_expr -= 3 * self.H * instance.N * (1 - self.x[i, 3])
            self.CNAD_mdl.add_constraint(lin_expr <= 0, "cluster_deviation2_lin")

        for i in self.N:
            for j in self.N:
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.theta[i, j]
                lin_expr -= 3 * self.H * instance.N * self.x[j, 3]
                self.CNAD_mdl.add_constraint(lin_expr <= 0, "cluster_dev_lin1")
                '''
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.theta[i, j]
                lin_expr -= (self.O[i] + 3 * self.H * instance.N * (1 - self.x[j, 3]))
                self.CNAD_mdl.add_constraint(lin_expr <= 0, "cluster_dev_lin1.2")
                '''
                lin_expr = self.CNAD_mdl.linear_expr()
                lin_expr += self.theta[i, j]
                lin_expr -= (self.O[i] - 3 * self.H * instance.N * (1 - self.x[j, 3]))
                self.CNAD_mdl.add_constraint(lin_expr >= 0, "cluster_dev_lin2")

    def defineConstraints(self, instance):
        self.actuatorAssignmentConstraint(instance)
        self.sensorNodeCostConstraint(instance)
        self.sensorAssignmentConstraint(instance)
        self.transceiverAssignmentConstraint(instance)
        self.memoryAssignmentConstraint(instance)
        self.flowConstraints(instance)
        self.energyConstraints(instance)
        self.batteryAssignmentConstraint(instance)
        self.clusterConstraints(instance)

    def solveModel(self, instance):
        # self.CNAD_mdl.export_as_lp("CNAD_Model.lp")
        start_time = time.time()
        # Solve the model
        self.CNAD_mdl.context.cplex_parameters.timelimit = 300  # 5 minutes

        # (Optional) Limit effort at root node
        # self.CNAD_mdl.context.cplex_parameters.mip.limits.cutpasses = 1
        # self.CNAD_mdl.context.cplex_parameters.mip.strategy.probe = -1
        # self.CNAD_mdl.context.cplex_parameters.preprocessing.presolve = 0
        # self.CNAD_mdl.context.cplex_parameters.mip.limits.strengthen = 0
        # self.CNAD_mdl.context.cplex_parameters.mip.strategy.variableselect = 3  # Fast var select

        self.solution = self.CNAD_mdl.solve(log_output = True)
        end_time = time.time()
        self.CNAD_runtime = end_time - start_time
        print(f"CNAD model execution time is {self.CNAD_runtime:.4f} seconds.")

        # Print solution if available
        if self.solution:
            print("Solution found. Network agent placements:")
            self.isFeasible = True
            self.CNAD_ObjValue = self.solution.objective_value  # or solution.get_objective_value()
            self.CNAD_RelGap = self.solution.solve_details.mip_relative_gap  # 0 if proven optimal
            self.CNAD_LowerBound = self.solution.solve_details.best_bound
            self.CNAD_ProcessedNodes = self.solution.solve_details.nb_nodes_processed

            self.getSolution(instance)

    def reportOutput(self, filePath):
        if not self.isFeasible: return

        # ------------------------------------------------------------------
        # column layout (fixed‑width strings for nice plain‑text view)
        widths = (6, 15, 12, 15, 10, 10)  # Model, LowerBound, ObjVal, CPU, Gap %, #Nodes
        cols = [
            f"{'Model':<{widths[0]}}",
            f"{'LowerBound':>{widths[1]}}",
            f"{'ObjVal':>{widths[2]}}",
            f"{'CPU (secs)':>{widths[3]}}",
            f"{'Gap %':>{widths[4]}}",
            f"{'#Nodes':>{widths[5]}}"
        ]

        # helper that formats one model row
        def make_row(tag, lb, obj, cpu, gap, nodes):
            return [
                f"{tag:<{widths[0]}}",
                f"{lb:>{widths[1]}.5f}",
                f"{obj:>{widths[2]}.3f}",
                f"{cpu:>{widths[3]}.5f}",
                f"{gap * 100:>{widths[4]}.2f} %",
                f"{nodes:>{widths[5]}}"
            ]

        # ------------------------------------------------------------------
        rows = [
            make_row('CNAD', self.CNAD_LowerBound, self.CNAD_ObjValue, self.CNAD_runtime, self.CNAD_RelGap, self.CNAD_ProcessedNodes),
            make_row('GD', 0, self.GD_ObjValue, 0, 0, 0),
            make_row('SD', 0, self.SD_ObjValue, 0, 0, 0),
            make_row('NTD', 0, self.NTD_ObjValue, 0, 0, 0),
            make_row('NMD',0,  self.NMD_ObjValue, 0, 0, 0),
            make_row('RD', 0, self.RD_ObjValue, 0, 0, 0),
            make_row('NBD', 0, self.NBD_ObjValue, 0, 0, 0),
            make_row('CF', 0, self.CF_ObjValue, 0, 0, 0),
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

    def getSolution(self, instance):
        if not self.isFeasible: return

        self.xg = {}
        self.xn = {}
        self.sn = {}
        self.dn = {}
        self.qn = {}
        self.tn = {}
        self.Mn = {}
        self.mn = {}
        self.xr = {}
        self.En = {}
        self.bn = {}
        self.OO = {}
        self.uu = {}

        # Print solution if available
        print("Solution found. Network agent placements:")
        for j in self.N:
            val = self.x[j, 3].solution_value
            self.xg[j] = val  # keep full dictionary
            if val > 1e-6 and not self.isRelease:  # print only non‑zeros
                print(f"x_g[{j + 1}] = {val}")

        for j in self.N:
            val = self.x[j, 0].solution_value
            self.xn[j] = val  # Store all values
            if val > 1e-6 and not self.isRelease:
                print(f"x_n[{j + 1}] = {val}")

            for s in self.S:
                val = self.ss[j, s].solution_value
                self.sn[j, s] = val  # Store all values
                if self.xn[j] > 1e-6 and not self.isRelease:
                    print(f"s[{j + 1}, {s + 1}] = {val}")

        for j in self.N:
            val = self.d[j].solution_value
            self.dn[j] = val  # Store in dictionary
            if val > 1e-6 and not self.isRelease:
                print(f"d[{j + 1}] = {val}")
            for i in self.N:
                val = self.q[j, i].solution_value
                self.qn[j, i] = val  # Store in dictionary
                if val > 1e-6 and not self.isRelease:
                    print(f"q[{j + 1}, {i + 1}] = {val}")
            for t in self.T:
                val = self.tt[j, t].solution_value
                self.tn[j, t] = val  # Store in dictionary
                if val > 1e-6 and not self.isRelease:
                    print(f"t[{j + 1}, {t + 1}] = {val}")

        for j in self.N:
            val = self.Memory[j].solution_value
            self.Mn[j] = val  # Store in dictionary
            if val > 1e-6 and not self.isRelease:
                print(f"M[{j + 1}] = {val}")
            for m in self.M:
                val = self.mm[j, m].solution_value
                self.mn[j, m] = val  # Store in dictionary
                if val > 1e-6 and not self.isRelease:
                    print(f"m[{j + 1}, {m + 1}] = {val}")

        for j in self.N:
            val = self.x[j, 2].solution_value
            self.xr[j] = val  # Store in dictionary
            if val > 1e-6 and not self.isRelease:
                print(f"xr[{j + 1}] = {val}")

        for j in self.N:
            for k in (0, 1, 2):
                for i in self.N:
                    for l in self.K:
                        val = self.y[j, k, i, l].solution_value
                        if val > 0 and not self.isRelease:
                            print(f"y[{j + 1}, {k}, {i + 1}, {l}] = {val}")

        for j in self.N:
            val = self.Energy[j].solution_value
            self.En[j] = val  # Store in dictionary
            if val > 1e-6 and not self.isRelease:
                print(f"E[{j + 1}] = {val}")
            for b in self.B:
                val = self.bb[j, b].solution_value
                self.bn[j, b] = val  # Store in dictionary
                if val > 1e-6 and not self.isRelease:
                    print(f"b[{j + 1}, {b + 1}] = {val}")

        for j in self.N:
            val = self.O[j].solution_value
            self.OO[j] = val  # Store in dictionary
            if val > 1e-6 and not self.isRelease:
                print(f"O[{j + 1}] = {val}")
            for k in 0, 1, 2:
                for i in self.N:
                    val = self.u[j, k, i].solution_value
                    self.uu[j, k, i] = val  # Store in dictionary
                    if val > 1e-6 and not self.isRelease:
                        print(f"u[{j + 1}, {k}, {i + 1}] = {val}")

        for j in self.N:
            val = self.w[j].solution_value
            if val > 1e-6 and not self.isRelease:
                print(f"w[{j + 1}] = {val}")

        for i in self.N:
            for j in self.N:
                val = self.psi[i, j].solution_value
                if val > 1e-6 and not self.isRelease:
                    print(f"psi[{i + 1}, {j + 1}] = {val}")
        for i in self.N:
            for j in self.N:
                val = self.theta[i, j].solution_value
                if val > 1e-6 and not self.isRelease:
                    print(f"theta[{i + 1}, {j + 1}] = {val}")

        # Calculate subproblem objectives for comparison
        self.GD_ObjValue = sum([instance.agent.models['Gateway']['cost'] * self.xg[j] for j in self.N])
        self.SD_ObjValue = (sum([instance.box.attributes['cost'] * self.xn[j] for j in self.N]) +
                            sum([instance.sensor.models[f"S{s + 1}"]['cost'] * self.sn[j, s] for j in self.N for s in
                                 self.S]))
        self.NTD_ObjValue = sum([instance.transceiver.models[f"T{t + 1}"]['cost'] * self.tn[j, t]
                                 for j in self.N for t in self.T])
        self.NMD_ObjValue = sum([instance.memory.models[f"M{m + 1}"]['cost'] * self.mn[j, m]
                                 for j in self.N for m in self.M])
        self.RD_ObjValue = sum(instance.agent.models["Router"]['cost'] * self.xr[j] for j in self.N)
        self.NBD_ObjValue = sum([instance.battery.models[f"B{b + 1}"]['cost'] * self.bn[j, b]
                                 for j in self.N for b in self.B])
        self.CF_ObjValue = 0

        # Plot the gateway
        self.gateway_indices = [j for j in self.N if self.xg[j] > 0.5]
        self.sensor_node_indices = [j for j, val in self.xn.items() if val > 0.5]
        self.sensor_assignments = {(j, s): val for (j, s), val in self.sn.items() if val > 0.5}
        self.router_indices = [j for j, val in self.xr.items() if val > 0.5]

    def visualizeClusterSolution(self, instance):
        # --------------------------------------------------------------- helpers
        #self.getSolution(instance)

        def polar_offset(x0, y0, r, theta_deg):
            theta = np.deg2rad(theta_deg)
            return x0 + r * np.cos(theta), y0 + r * np.sin(theta)

        # ---------------------------------------------------------------- poses
        xs = [p['x'] for p in instance.scenario.poses]
        ys = [p['y'] for p in instance.scenario.poses]

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.canvas.manager.set_window_title("CNAD Solution")

        # ───────────────────── scenario boundary ──────────────────────────
        W = instance.scenario.regionMap[instance.scenario.size]['W']
        H = instance.scenario.regionMap[instance.scenario.size]['H']
        for axis in [ax, ax2]:
            axis.add_patch(Rectangle(
                (0, 0), W, H,
                fill=False,
                edgecolor='black',
                linestyle='-',
                linewidth=2.5
            ))

        # ───────────────────── FULL DEPLOYMENT VIEW ──────────────────────────
        ax.set_title("Agent Deployment")

        # 1) candidate poses + ID
        ax.scatter(xs, ys, c='blue', s=80, edgecolors='black', label='Pose')
        for pid, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x - 0.04, y, f"P{pid + 1}",
                    ha='right', va='center', fontsize=8, color='blue')

        # 2) actuators (above)
        if instance.scenario.xa is not None:
            for j, is_act in enumerate(instance.scenario.xa):
                if is_act:
                    x, y = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
                    ax.scatter(*polar_offset(x, y, 0.08, 90),
                               c='orange', marker='^', s=80, edgecolors='black',
                               label='Actuator' if 'Actuator' not in ax.get_legend_handles_labels()[1] else "")

        # 3) gateways (right)
        gate_r = instance.agent.models['Gateway']['commRange']
        for k, j in enumerate(self.gateway_indices):
            gx0, gy0 = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
            x, y = polar_offset(gx0, gy0, 0.09, 0)
            ax.scatter(x, y, c='red', marker='s', s=80, edgecolors='black',
                       label='Gateway' if k == 0 else "")
            ax.add_patch(Circle((gx0, gy0), gate_r, alpha=.12, color='red', linewidth=0))

        # 3-B) routers (above–left)
        if hasattr(self, 'router_indices') and self.router_indices:
            for k, j in enumerate(self.router_indices):
                xr, yr = polar_offset(
                    instance.scenario.poses[j]['x'],
                    instance.scenario.poses[j]['y'],
                    0.1, 135
                )
                ax.scatter(xr, yr, c='magenta', marker='D', s=70, edgecolors='black',
                           label='Router' if k == 0 else "")

        # 4) sensor nodes
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

        # Legend

        handles = []

        sensor_patches = [mpatches.Patch(color=sensor_colors[s % len(sensor_colors)],
                                         label=f"Sensor {s + 1}")
                          for s in sorted(sensor_types_used)]
        node_patch = mpatches.Patch(facecolor='lightgreen', edgecolor='green',
                                    label='Sensor Node')
        handles.extend(sensor_patches)
        handles.append(node_patch)

        handles.append(mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                                     markersize=9, markeredgecolor='black', label='Gateway'))

        if self.router_indices:
            handles.append(mlines.Line2D([], [], color='magenta', marker='D', linestyle='None',
                                         markersize=9, markeredgecolor='black', label='Router'))

        handles.append(mlines.Line2D([], [], color='orange', marker='^', linestyle='None',
                                     markersize=9, markeredgecolor='black', label='Actuator'))

        handles.append(mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                     markersize=8, markeredgecolor='black', label='Pose'))

        ax.legend(handles=handles, loc='lower left')

        # Cosmetics
        for axis in [ax, ax2]:
            axis.set_xlabel("X (km)")
            axis.set_ylabel("Y (km)")
            axis.set_xlim(-0.5, W + 0.5)
            axis.set_ylim(-0.5, H + 0.5)
            axis.set_aspect('equal', adjustable='box')
            axis.grid(True)

        # ───────────────────── SIMPLIFIED VIEW: Poses + Actuators Only ─────────────────────
        ax2.set_title("Clusters and Connectivity")
        ax2.scatter(xs, ys, c='blue', s=80, edgecolors='black', label='Pose')
        for pid, (x, y) in enumerate(zip(xs, ys)):
            ax2.text(x - 0.04, y, f"P{pid + 1}",
                     ha='right', va='center', fontsize=8, color='blue')

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
                ax2.scatter(gx, gy, c=color, marker='s', s=90, edgecolors='black')

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
                                ax2.add_patch(rect)
                            elif k == 1:
                                # Triangle pointing up for k=1
                                angle = 90
                                xk, yk = polar_offset(px, py, 0.08, angle)
                                ax2.scatter(xk, yk, c=color, marker='^', s=80, edgecolors='black')
                            elif k == 2:
                                # Diamond for k=2
                                angle = 135
                                xk, yk = polar_offset(px, py, 0.08, angle)
                                ax2.scatter(xk, yk, c=color, marker='D', s=80, edgecolors='black')

                # Add to legend (only once per cluster)
                cluster_legend.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                                    markersize=9, markeredgecolor='black', label=label))

        # Add cluster legend to ax2
        ax2.legend(handles=cluster_legend, loc='upper right')

        fig.tight_layout()
        plt.show()

    def visualizeDeploymentSolution(self, instance):
        # --------------------------------------------------------------- helpers
        # self.setVisualization(True, True, True, True)  # ensure flags exist

        # self.getSolution(instance)

        def polar_offset(x0, y0, r, theta_deg):
            theta = np.deg2rad(theta_deg)
            return x0 + r * np.cos(theta), y0 + r * np.sin(theta)

        # ---------------------------------------------------------------- poses
        xs = [p['x'] for p in instance.scenario.poses]
        ys = [p['y'] for p in instance.scenario.poses]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title("CNAD Solution")
        ax.set_title("CNAD Agent Deployment")

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
        if instance.scenario.xa is not None:
            for j, is_act in enumerate(instance.scenario.xa):
                if is_act:
                    x, y = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
                    ax.scatter(*polar_offset(x, y, 0.08, 90),
                               c='orange', marker='^', s=80, edgecolors='black',
                               label='Actuator' if 'Actuator' not in ax.get_legend_handles_labels()[1] else "")

        # 3) gateways (right)
        gate_r = instance.agent.models['Gateway']['commRange']
        for k, j in enumerate(self.gateway_indices):
            gx0, gy0 = instance.scenario.poses[j]['x'], instance.scenario.poses[j]['y']
            x, y = polar_offset(gx0, gy0, 0.09, 0)
            ax.scatter(x, y, c='red', marker='s', s=80, edgecolors='black',
                       label='Gateway' if k == 0 else "")
            ax.add_patch(Circle((gx0, gy0), gate_r, alpha=.12, color='red', linewidth=0))

        # 3‑B) routers (above–left at 135°)
        if hasattr(self, 'router_indices') and self.router_indices:

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

        sensor_patches = [mpatches.Patch(color=sensor_colors[s % len(sensor_colors)],
                                         label=f"Sensor {s + 1}")
                          for s in sorted(sensor_types_used)]
        node_patch = mpatches.Patch(facecolor='lightgreen', edgecolor='green',
                                    label='Sensor Node')
        handles.extend(sensor_patches)
        handles.append(node_patch)

        handles.append(mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                                         markersize=9, markeredgecolor='black', label='Gateway'))

        if self.router_indices:
            handles.append(mlines.Line2D([], [], color='magenta', marker='D', linestyle='None',
                                         markersize=9, markeredgecolor='black', label='Router'))


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
        ax.grid(True);
        fig.tight_layout();
        plt.show()

    def visualizeSolution(self, instance, isCluster):
        if isCluster:
            self.visualizeClusterSolution(instance)
        else:
            self.visualizeDeploymentSolution(instance)

    def solve(self, instance):
        self.initializeModel(instance)
        self.defineObjective(instance)
        self.defineConstraints(instance)
        self.solveModel(instance)
        if self.isFeasible:
            return True
        else: return False




