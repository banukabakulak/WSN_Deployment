from docplex.mp.model import Model
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import time
import numpy as np
import os
import pandas as pd

class DecomposedNAD:
    def __init__(self, epsilon, delta, isRelease = False):
        self.epsilon = epsilon
        self.delta = delta
        self.isRelease = isRelease
        self.isVisualize = {'SensorNode' : False,
                            'Actuator': True,
                            'Router': False,
                            'Gateway' : False
                            }
        self.isFeasible = {'GD' : False,
                           'SD' : False,
                           'NTD' : False,
                           'NMD' : False,
                           'RD' : False,
                           'NBD': False,
                           'CF' : False}

    def setVisualization(self, sensorNode, actuator, router, gateway):
        self.isVisualize['SensorNode'] = sensorNode
        self.isVisualize['Actuator'] = actuator
        self.isVisualize['Router'] = router
        self.isVisualize['Gateway'] = gateway

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
        print("Solving Gateway Deployment...")
        # Create CPLEX model
        self.GD_mdl = Model(name="DeNAD GD model")

        self.x_g = self.GD_mdl.binary_var_dict((j for j in self.N), name="xg")

        # Set the objective
        obj_expr = self.GD_mdl.linear_expr()
        for j in self.N:
            obj_expr += instance.agent.models['Gateway']['cost'] * self.x_g[j]
        self.GD_mdl.minimize(obj_expr)

        # Constraints: Gateway Coverage
        for i in self.N:
            lin_expr = self.GD_mdl.linear_expr()
            for j in self.N:
                commStatus = instance.scenario.isCommunicate('Gateway', instance.scenario.poses[j],
                                                             instance.scenario.poses[i], instance.agent)
                lin_expr += commStatus * self.x_g[j]
            self.point_gate_cover = 1
            self.GD_mdl.add_constraint(lin_expr >= self.point_gate_cover, "gate_cover")

        self.GD_mdl.export_as_lp("GD_Model.lp")
        start_time = time.time()
        # Solve the model
        self.GD_mdl.context.cplex_parameters.timelimit = 300  # 5 minutes
        solution = self.GD_mdl.solve(log_output=True)
        end_time = time.time()
        self.GD_runtime = end_time - start_time
        print(f"GD model execution time is {self.GD_runtime :.4f} seconds.")

        self.xg = {}  # Dictionary to store x_g values

        # Print solution if available
        if solution:
            print("Solution found. Gateway placements:")
            self.GD_ObjValue = solution.objective_value  # or solution.get_objective_value()
            self.GD_RelGap = solution.solve_details.mip_relative_gap  # 0 if proven optimal
            for j in self.N:
                val = self.x_g[j].solution_value
                self.xg[j] = val  # keep full dictionary
                if val > 1e-6 and not self.isRelease:  # print only non‑zeros
                    print(f"x_g[{j + 1}] = {val}")
        else:
            print("No feasible solution found.")
            self.xg = None

        # Plot the gateway
        if solution:
            self.gateway_indices = [j for j in self.N if self.x_g[j].solution_value > 0.5]
            self.isVisualize['Gateway'] = True
            self.isFeasible['GD'] = True

    def solveSensorDeployment(self, instance):
        print("Solving Sensor Deployment...")
        # Create CPLEX model
        self.SD_mdl = Model(name="DeNAD SD model")

        self.x_n = self.SD_mdl.binary_var_dict((j for j in self.N), name="xn")
        self.ss = self.SD_mdl.binary_var_dict( ((j, s) for j in self.N for s in self.S), name="s")

        # Set the objective
        obj_expr = self.SD_mdl.linear_expr()
        for j in self.N:
            obj_expr += instance.box.attributes['cost'] * self.x_n[j]
            for s in self.S:
                obj_expr += instance.sensor.models[f"S{s+1}"]['cost'] * self.ss[j, s]
        self.SD_mdl.minimize(obj_expr)

        # Constraints: Feasible Sensor to Node
        for j in self.N:
            for s in self.S:
                lin_expr = self.SD_mdl.linear_expr()
                lin_expr += (self.ss[j, s] - self.x_n[j])
                self.SD_mdl.add_constraint(lin_expr <= 0, "feas_sensor1")

        for j in self.N:
            lin_expr = self.SD_mdl.linear_expr()
            lin_expr += self.x_n[j]
            for s in self.S:
                lin_expr -= self.ss[j, s]
            self.SD_mdl.add_constraint(lin_expr <= 0, "feas_sensor2")

        # Constraints: Feasible Sensor in Box
        for j in self.N:
            lin_expr = self.SD_mdl.linear_expr()
            for s in self.S:
                lin_expr += self.ss[j, s]
            # self.SD_mdl.add_constraint(lin_expr <= instance.box.attributes['P'], "feas_sensor_box")
            self.SD_mdl.add_constraint(lin_expr <= instance.S,
                                       "feas_sensor_box")
        # Constraints: Feasible Sensor in Volume
        for j in self.N:
            lin_expr = self.SD_mdl.linear_expr()
            for s in self.S:
                lin_expr += instance.sensor.models[f"S{s + 1}"]['volume'] * self.ss[j, s]
            lin_expr -= instance.box.attributes['volume'] * self.x_n[j]
            self.SD_mdl.add_constraint(lin_expr <= 0, "feas_sensor_vol")

        # Constraints: Sensor Coverage
        for i in self.N:
            for s in self.S:
                lin_expr = self.SD_mdl.linear_expr()
                for j in self.N:
                    coverStatus = instance.scenario.isSense(f"S{s + 1}", instance.scenario.poses[j],
                                                            instance.scenario.poses[i], instance.sensor)
                    lin_expr += coverStatus * self.ss[j, s]
                self.SD_mdl.add_constraint(lin_expr >= instance.scenario.poses[i]['coverReq'][s],
                                             "sensor_cover")

        # Constraints: Actuator Connectivity
        for j in self.N:
            lin_expr = self.SD_mdl.linear_expr()
            lin_expr -= instance.agent.models['Actuator']['lambda'] * instance.scenario.xa[j]
            for i in self.N:
                commStatus = instance.scenario.isCommunicate('Actuator', instance.scenario.poses[j],
                                                             instance.scenario.poses[i], instance.agent)
                lin_expr += commStatus * (self.x_n[i] + self.xg[i])
            self.SD_mdl.add_constraint(lin_expr >= 0, "a_connectivity")

        self.SD_mdl.export_as_lp("SD_Model.lp")
        start_time = time.time()
        # Solve the model
        self.SD_mdl.context.cplex_parameters.timelimit = 300  # 5 minutes
        solution = self.SD_mdl.solve(log_output=True)
        end_time = time.time()
        self.SD_runtime = end_time - start_time
        print(f"SD model execution time is {self.SD_runtime:.4f} seconds.")

        self.xn = {}  # Dictionary to store x_n values
        self.sn = {}  # Dictionary to store s values
        self.hn = {}  # Dictionary to store h values
        self.wn = {}  # Dictionary to store w values

        # Print solution if available
        if solution:
            print("Solution found. Sensor Nodes with Sensor placements:")
            self.SD_ObjValue = solution.objective_value  # or solution.get_objective_value()
            self.SD_RelGap = solution.solve_details.mip_relative_gap  # 0 if proven optimal
            for j in self.N:
                val = self.x_n[j].solution_value
                self.xn[j] = val  # Store all values
                if val > 1e-6 and not self.isRelease:
                    print(f"x_n[{j + 1}] = {val}")

                for s in self.S:
                    val = self.ss[j, s].solution_value
                    self.sn[j, s] = val  # Store all values
                    if self.xn[j]  > 1e-6 and not self.isRelease:
                        print(f"s[{j + 1}, {s + 1}] = {val}")
        else:
            print("No feasible solution found.")
            self.xn = None
            self.sn = None

        if solution:
            # Calculate Sensor Node Data Size
            for j in self.N:
                dataSize = 0
                for s in self.S:
                    dataSize += instance.sensor.models[f"S{s + 1}"]['dataSize'] * self.sn[j, s]
                self.hn[j] = dataSize
                if self.hn[j] > 1e-6 and not self.isRelease:
                    print(f"h[{j + 1}] = {self.hn[j]}")

            # Calculate Sensor Node Protocol Overhead
            for j in self.N:
                self.wn[j] = instance.agent.models['SensorNode']['gateCommand'] * self.xn[j] + self.hn[j]
                for s in self.S:
                   self.wn[j] += instance.sensor.models[f"S{s + 1}"]['gateCommand'] * self.sn[j, s]
                if self.wn[j] > 1e-6 and not self.isRelease:
                    print(f"w[{j + 1}] = {self.wn[j]}")

            # Plot the sensor nodes with sensors
            self.sensor_node_indices = [j for j, val in self.xn.items() if val > 0.5]
            self.sensor_assignments = {(j, s): val for (j, s), val in self.sn.items() if val > 0.5}

            self.isVisualize['SensorNode'] = True
            self.isFeasible['SD'] = True

    def solveNodeTransceiverDeployment(self, instance):
        print("Solving Node Transceiver Deployment...")
        # Create CPLEX model
        self.NTD_mdl = Model(name="DeNAD NTD model")

        self.q = self.NTD_mdl.binary_var_dict(((j, i) for j in self.N for i in self.N),name="q")
        self.d = self.NTD_mdl.continuous_var_dict((j for j in self.N), name="d")
        self.tt = self.NTD_mdl.binary_var_dict(((j, t) for j in self.N for t in self.T),name="t")

        # Set the objective
        obj_expr = self.NTD_mdl.linear_expr()
        for j in self.N:
            for t in self.T:
                obj_expr += instance.transceiver.models[f"T{t + 1}"]['cost'] * self.tt[j, t]
        self.NTD_mdl.minimize(obj_expr)

        # Constraints: Feasible Transceiver to Node
        for j in self.N:
            lin_expr = self.NTD_mdl.linear_expr()
            for t in self.T:
                lin_expr += self.tt[j, t]
            lin_expr -= self.xn[j]
            self.NTD_mdl.add_constraint(lin_expr == 0, "feas_transceiver_node")

        # Constraints: Node CommRange
        for i in self.N:
            for j in self.N:
                lin_expr = self.NTD_mdl.linear_expr()
                lin_expr += self.q[j, i]
                for t in self.T:
                    commStatus = instance.scenario.isCommunicate(f"T{t + 1}", instance.scenario.poses[j],
                                                                 instance.scenario.poses[i], instance.transceiver)
                    lin_expr -= commStatus * self.tt[j, t]
                self.NTD_mdl.add_constraint(lin_expr == 0, "node_commRange")

        # Constraints: Node Bandwidth
        for j in self.N:
            lin_expr = self.NTD_mdl.linear_expr()
            lin_expr += self.d[j]
            for t in self.T:
                lin_expr -= instance.transceiver.models[f"T{t + 1}"]["bandwidth"] * self.tt[j, t]
            self.NTD_mdl.add_constraint(lin_expr == 0, "node_bandwidth")

        # Constraints: Node Connectivity -- NONLINEAR
        for j in self.N:
            lin_expr = self.NTD_mdl.linear_expr()
            lin_expr -= instance.agent.models['SensorNode']['lambda'] * self.xn[j]
            for i in self.N:
                lin_expr += self.q[j, i] * self.xg[i]
                if i != j:
                    lin_expr += self.q[j, i] * self.xn[i]
            self.NTD_mdl.add_constraint(lin_expr >= 0, "node_connectivity")

        # Constraints: Feasible Node Bandwidth -- NONLINEAR
        for j in self.N:
            lin_expr = self.NTD_mdl.linear_expr()
            lin_expr += (self.d[j] - self.hn[j])
            for i in self.N:
                lin_expr -= (self.q[i, j] * self.hn[i])
                commStatus = instance.scenario.isCommunicate('Actuator', instance.scenario.poses[i],
                                                             instance.scenario.poses[j], instance.agent)
                lin_expr -= commStatus * instance.agent.models['Actuator']['dataSize'] * instance.scenario.xa[i]
                lin_expr -= instance.agent.models['Router']['lambda'] * instance.agent.models['Router']['dataSize']
                lin_expr += self.H * instance.N * (1 - self.xn[j])
            self.NTD_mdl.add_constraint(lin_expr >= 0, "feas_node_bandwidth")

        self.NTD_mdl.export_as_lp("NTD_mdl.lp")
        start_time = time.time()
        # Solve the model
        self.NTD_mdl.context.cplex_parameters.timelimit = 300  # 5 minutes
        solution = self.NTD_mdl.solve(log_output=True)
        end_time = time.time()
        self.NTD_runtime = end_time - start_time
        print(f"NTD model execution time is {self.NTD_runtime:.4f} seconds.")

        self.tn = {}  # Dictionary to store t values
        self.qn = {}  # Dictionary to store q values
        self.dn = {}  # Dictionary to store d values
        self.ea = {}  # Dictionary to store e_a values
        self.erx = {}  # Dictionary to store e_rx values
        self.etx = {}  # Dictionary to store e_tx values
        self.eHat = {}  # Dictionary to store e_Hat values

        # Print solution if available
        if solution:
            print("Solution found. Sensor Nodes with Sensor placements:")
            self.NTD_ObjValue = solution.objective_value  # or solution.get_objective_value()
            self.NTD_RelGap = solution.solve_details.mip_relative_gap  # 0 if proven optimal
            for j in self.N:
                val = self.d[j].solution_value
                self.dn[j] = val  # Store in dictionary
                if val > 1e-6 and not self.isRelease:
                    print(f"d[{j + 1}] = {val}")
                for i in self.N:
                    val = self.q[j, i].solution_value
                    self.qn[j, i] = val  # Store in dictionary
                    if val > 1e-6 and not self.isRelease:
                        print(f"q[{j+ 1}, {i + 1}] = {val}")
                for t in self.T:
                    val = self.tt[j, t].solution_value
                    self.tn[j, t] = val  # Store in dictionary
                    if val > 1e-6 and not self.isRelease:
                        print(f"t[{j+ 1}, {t + 1}] = {val}")
        else:
            print("No feasible solution found.")
            self.tn = None
            self.qn = None
            self.dn = None

        if solution:
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
        print("Solving Node Memory Deployment...")
        # Create CPLEX model
        self.NMD_mdl = Model(name="DeNAD NMD model")

        self.Memory = self.NMD_mdl.continuous_var_dict((j for j in self.N), name="M")
        self.mm = self.NMD_mdl.binary_var_dict(((j, m) for j in self.N for m in self.M), name="m")

        # Set the objective
        obj_expr = self.NMD_mdl.linear_expr()
        for j in self.N:
            for m in self.M:
                obj_expr += instance.memory.models[f"M{m + 1}"]['cost'] * self.mm[j, m]
        self.NMD_mdl.minimize(obj_expr)

        # Constraints: Feasible Memory to Node
        for j in self.N:
            lin_expr = self.NMD_mdl.linear_expr()
            for m in self.M:
                lin_expr += self.mm[j, m]
            lin_expr -= self.xn[j]
            self.NMD_mdl.add_constraint(lin_expr >= 0, "feas1_memory_node")

            lin_expr = self.NMD_mdl.linear_expr()
            for m in self.M:
                lin_expr += self.mm[j, m]
            self.NMD_mdl.add_constraint(lin_expr <= instance.box.attributes['C'], "feas2_memory_node")

            lin_expr = self.NMD_mdl.linear_expr()
            lin_expr += self.Memory[j]
            for m in self.M:
                lin_expr -= instance.memory.models[f"M{m + 1}"]['capacity'] * self.mm[j, m]
            self.NMD_mdl.add_constraint(lin_expr == 0, "feas3_memory_node")

        # Constraints: Feasible Node Memory -- Linearized
        for j in self.N:
            lin_expr = self.NMD_mdl.linear_expr()
            lin_expr += (self.Memory[j] - self.hn[j])
            for i in self.N:
                lin_expr -= (self.qn[i, j] * self.hn[i])
                commStatus = instance.scenario.isCommunicate('Actuator', instance.scenario.poses[i],
                                                             instance.scenario.poses[j], instance.agent)
                lin_expr -= commStatus * instance.agent.models['Actuator']['dataSize'] * instance.scenario.xa[i]
                lin_expr -= instance.agent.models['Router']['lambda'] * instance.agent.models['Router']['dataSize']
                lin_expr += self.H * instance.N * (1 - self.xn[j])
            self.NMD_mdl.add_constraint(lin_expr >= 0, "feas_node_memory_lin")

        self.NMD_mdl.export_as_lp("NMD_mdl.lp")
        start_time = time.time()
        # Solve the model
        self.NMD_mdl.context.cplex_parameters.timelimit = 300  # 5 minutes
        solution = self.NMD_mdl.solve(log_output=True)
        end_time = time.time()
        self.NMD_runtime = end_time - start_time
        print(f"NMD model execution time is {self.NMD_runtime:.4f} seconds.")

        self.Mn = {}  # Dictionary to store None Memory specification values
        self.mn = {}  # Dictionary to store m values

        # Print solution if available
        if solution:
            print("Solution found. Sensor Nodes with Sensor placements:")
            self.NMD_ObjValue = solution.objective_value  # or solution.get_objective_value()
            self.NMD_RelGap = solution.solve_details.mip_relative_gap  # 0 if proven optimal
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

            self.isFeasible['NMD'] = True
        else:
            print("No feasible solution found.")
            self.Mn = None
            self.mn = None

    def solveRouterDeployment(self, instance):
        print("Solving Router Deployment...")
        # Create CPLEX model
        self.RD_mdl = Model(name="DeNAD RD model")

        self.x_r = self.RD_mdl.binary_var_dict((j for j in self.N), name="xr")
        self.y = self.RD_mdl.continuous_var_dict(
            ((j, k, i, l) for j in self.N for k in self.K for i in self.N for l in self.K),
            name="y")

        # Set the objective
        obj_expr = self.RD_mdl.linear_expr()
        for j in self.N:
            obj_expr += instance.agent.models['Router']['cost'] * self.x_r[j]
        self.RD_mdl.minimize(obj_expr)

        # Constraints: delta-hop flow
        for j in self.N:
            lin_expr = self.RD_mdl.linear_expr()
            lin_expr -= self.delta * self.xn[j]
            for i in self.N:
                for l in (0, 1, 2):
                    lin_expr += self.y[i, l, j, 0]
            self.RD_mdl.add_constraint(lin_expr <= 0, "delta_hop_n_flow")

        #for j in self.N:
        #    lin_expr = self.RD_mdl.linear_expr()
        #    lin_expr -= self.delta * instance.scenario.xa[j]
        #     for i in self.N:
        #        for l in self.K:
        #            lin_expr += self.y[i, l, j, 1]
        #    self.RD_mdl.add_constraint(lin_expr <= 0, "delta_hop_a_flow")

        for j in self.N:
            lin_expr = self.RD_mdl.linear_expr()
            lin_expr -= self.delta * self.x_r[j]
            for i in self.N:
                for l in (0, 1, 2):
                    lin_expr += self.y[i, l, j, 2]
            self.RD_mdl.add_constraint(lin_expr <= 0, "delta_hop_r_flow")

        for j in self.N:
            for i in self.N:
                for l in (0, 1, 2):
                    lin_expr = self.RD_mdl.linear_expr()
                    lin_expr -= self.delta * self.xg[j]
                    lin_expr += self.y[i, l, j, 3]
                    self.RD_mdl.add_constraint(lin_expr <= 0, "delta_hop_g_flow")

        # Constraints: node outflow
        for j in self.N:
            lin_expr = self.RD_mdl.linear_expr()
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr += self.y[j, 0, i, l]
            lin_expr -= self.xn[j]
            self.RD_mdl.add_constraint(lin_expr >= 0, "n_node_outflow")

        for j in self.N:
            lin_expr = self.RD_mdl.linear_expr()
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr += self.y[j, 1, i, l]
            lin_expr -= instance.scenario.xa[j]
            self.RD_mdl.add_constraint(lin_expr >= 0, "a_node_outflow")

        for j in self.N:
            lin_expr = self.RD_mdl.linear_expr()
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr += self.y[j, 2, i, l]
            lin_expr -= self.x_r[j]
            self.RD_mdl.add_constraint(lin_expr >= 0, "r_node_outflow")

        # Constraints: node flow balance
        for j in self.N:
            lin_expr = self.RD_mdl.linear_expr()
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr += self.y[j, 0, i, l]
                for l in (0, 1, 2):
                    lin_expr -= self.y[i, l, j, 0]
            lin_expr -= self.xn[j]
            self.RD_mdl.add_constraint(lin_expr == 0, "n_node_flow_balance")

        for j in self.N:  
            lin_expr = self.RD_mdl.linear_expr()
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr += self.y[j, 1, i, l]
            lin_expr -= instance.scenario.xa[j]
            self.RD_mdl.add_constraint(lin_expr == 0, "a_node_flow_balance")

        for j in self.N:
            lin_expr = self.RD_mdl.linear_expr()
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr += self.y[j, 2, i, l]
                for l in (0, 1, 2):
                    lin_expr -= self.y[i, l, j, 2]
            lin_expr -= self.x_r[j]
            self.RD_mdl.add_constraint(lin_expr == 0, "r_node_flow_balance")

        # Constraints: gateway inflow
        lin_expr = self.RD_mdl.linear_expr()
        for j in self.N:
            for k in (0, 1, 2):
                for i in self.N:
                    lin_expr += self.y[j, k, i, 3]
            lin_expr -= (self.xn[j] + instance.scenario.xa[j] + self.x_r[j])
        self.RD_mdl.add_constraint(lin_expr == 0, "gate_inflow")

        # Constraints: Feasible Flow
        for j in self.N:
            for k in (1, 2):
                for i in self.N:
                    for l in (0, 2, 3):
                        lin_expr = self.RD_mdl.linear_expr()
                        lin_expr += self.y[j, k, i, l]
                        commStatus = instance.scenario.isCommunicate(self.nodeMap[k], instance.scenario.poses[j],
                                                                     instance.scenario.poses[i], instance.agent)
                        self.RD_mdl.add_constraint(lin_expr <= self.delta * commStatus, "feas1_flow")

        for j in self.N:
            for i in self.N:
                for l in (0, 2, 3):
                    lin_expr = self.RD_mdl.linear_expr()
                    lin_expr -= self.delta * self.qn[j, i]
                    lin_expr += self.y[j, 0, i, l]
                    self.RD_mdl.add_constraint(lin_expr <= 0, "feas2_flow")

        for j in self.N:
            for i in self.N:
                for l in (0, 1, 2):
                    lin_expr = self.RD_mdl.linear_expr()
                    lin_expr += self.y[i, l, j, 1]
                    self.RD_mdl.add_constraint(lin_expr == 0, "feas3_flow")

        for j in self.N:
            for i in self.N:
                for l in (0, 1, 2):
                    lin_expr = self.RD_mdl.linear_expr()
                    lin_expr += self.y[j, 3, i, l]
                    self.RD_mdl.add_constraint(lin_expr == 0, "feas4_flow")

        for i in self.N:
            for l in self.K:
                lin_expr = self.RD_mdl.linear_expr()
                lin_expr += self.y[i, l, i, l]
                self.RD_mdl.add_constraint(lin_expr == 0, "feas5_flow")

        # Constraints: Router Connectivity
        for j in self.N:
            lin_expr = self.RD_mdl.linear_expr()
            lin_expr -= instance.agent.models['Router']['lambda'] * self.x_r[j]
            for i in self.N:
                commStatus = instance.scenario.isCommunicate('Router', instance.scenario.poses[j],
                                                             instance.scenario.poses[i], instance.agent)
                lin_expr += commStatus * (self.xn[i] + self.xg[i])
                if not i == j:
                    lin_expr += commStatus * self.x_r[i]
            self.RD_mdl.add_constraint(lin_expr >= 0, "r_connectivity")

        self.RD_mdl.export_as_lp("RD_mdl.lp")
        start_time = time.time()
        # Solve the model
        self.RD_mdl.context.cplex_parameters.timelimit = 300  # 5 minutes
        solution = self.RD_mdl.solve(log_output=True)
        end_time = time.time()
        self.RD_runtime = end_time - start_time
        print(f"RD model execution time is {self.RD_runtime :.4f} seconds.")

        self.xr = {}  # Dictionary to store xr values

        # Print solution if available
        if solution:
            print("Solution found. Sensor Nodes with Sensor placements:")
            self.RD_ObjValue = solution.objective_value  # or solution.get_objective_value()
            self.RD_RelGap = solution.solve_details.mip_relative_gap  # 0 if proven optimal
            for j in self.N:
                val = self.x_r[j].solution_value
                self.xr[j] = val  # Store in dictionary
                if val > 0.5 and not self.isRelease:
                    print(f"xr[{j + 1}] = {val}")

            for j in self.N:
                for k in (0, 1, 2):
                    for i in self.N:
                        for l in self.K:
                            val = self.y[j, k, i, l].solution_value
                            if val > 0.5 and not self.isRelease:
                                print(f"y[{j + 1}, {k}, {i + 1}, {l}] = {val}")

        else:
            print("No feasible solution found.")
            self.xr = None
        if solution:
            # Plot the routers
            # self.xr[5] = 1
            self.router_indices = [j for j, val in self.xr.items() if val > 0.5]
            self.isVisualize['Router'] = True
            self.isFeasible['RD'] = True

    def solveNodeBatteryDeployment(self, instance):
        print("Solving Node Battery Deployment...")
        # Create CPLEX model
        self.NBD_mdl = Model(name="DeNAD NBD model")

        self.Energy = self.NBD_mdl.continuous_var_dict((j for j in self.N), name="E")
        self.bb = self.NBD_mdl.integer_var_dict(((j, b) for j in self.N for b in self.B), name="b")

        # Set the objective
        obj_expr = self.NBD_mdl.linear_expr()
        for j in self.N:
            for b in self.B:
                obj_expr += instance.battery.models[f"B{b + 1}"]['cost'] * self.bb[j, b]
        self.NBD_mdl.minimize(obj_expr)

        # Constraints : feasible Battery to Node
        for j in self.N:
            lin_expr = self.NBD_mdl.linear_expr()
            lin_expr -= self.xn[j]
            for b in self.B:
                lin_expr += self.bb[j, b]
            self.NBD_mdl.add_constraint(lin_expr >= 0, "feas1_node_battery")

            lin_expr = self.NBD_mdl.linear_expr()
            for b in self.B:
                lin_expr += self.bb[j, b]
            self.NBD_mdl.add_constraint(lin_expr <= instance.box.attributes['D'], "feas2_node_battery")

            lin_expr = self.NBD_mdl.linear_expr()
            lin_expr += self.Energy[j]
            for b in self.B:
                lin_expr -= instance.battery.models[f"B{b + 1}"]['capacity'] * self.bb[j, b]
            self.NBD_mdl.add_constraint(lin_expr == 0, "node_battery_energy")

            lin_expr = self.NBD_mdl.linear_expr()
            lin_expr += self.Energy[j]
            lin_expr -= instance.L * self.eHat[j]
            self.NBD_mdl.add_constraint(lin_expr >= 0, "node_battery_lifetime")

        self.NBD_mdl.export_as_lp("NBD_mdl.lp")
        start_time = time.time()
        # Solve the model
        self.NBD_mdl.context.cplex_parameters.timelimit = 300  # 5 minutes
        solution = self.NBD_mdl.solve(log_output=True)
        end_time = time.time()
        self.NBD_runtime = end_time - start_time
        print(f"NBD model execution time is {self.NBD_runtime:.4f} seconds.")

        self.En = {}  # Dictionary to store Node Energy Capacity specification values
        self.bn = {}  # Dictionary to store b values

        # Print solution if available
        if solution:
            print("Solution found. Sensor Nodes with Sensor placements:")
            self.NBD_ObjValue = solution.objective_value  # or solution.get_objective_value()
            self.NBD_RelGap = solution.solve_details.mip_relative_gap  # 0 if proven optimal
            for j in self.N:
                val = self.Energy[j].solution_value
                self.En[j] = val  # Store in dictionary
                if val > 1e-6 and not self.isRelease:
                    print(f"E[{j + 1}] = {val}")
                for b in self.B:
                    val = self.bb[j, b].solution_value
                    self.bn[j, b] = val  # Store in dictionary
                    if val > 1e-6 and not self.isRelease:
                        print(f"b[{j+ 1}, {b + 1}] = {val}")

            self.isFeasible['NBD'] = True
        else:
            print("No feasible solution found.")
            self.En = None
            self.bn = None

    def solveClusterFormation(self, instance):
        print("Solving Cluster Formation...")
        # Create CPLEX model
        self.CF_mdl = Model(name="DeNAD CF model")

        self.u = self.CF_mdl.binary_var_dict(((j, k, i) for j in self.N for k in self.K for i in self.N),name="u")
        self.O = self.CF_mdl.continuous_var_dict((j for j in self.N), name="O")

        # Set the objective
        obj_expr = self.CF_mdl.linear_expr()
        obj_expr += 0
        self.CF_mdl.minimize(obj_expr)

        # Constraints: Feasible cluster
        for j in self.N:
            lin_expr = self.CF_mdl.linear_expr()
            for i in self.N:
                lin_expr += self.u[j, 0, i]
            self.CF_mdl.add_constraint(lin_expr == self.xn[j], "n_feas1_cluster")

            lin_expr = self.CF_mdl.linear_expr()
            for i in self.N:
                lin_expr += self.u[j, 1, i]
            self.CF_mdl.add_constraint(lin_expr == instance.scenario.xa[j], "a_feas1_cluster")

            lin_expr = self.CF_mdl.linear_expr()
            for i in self.N:
                lin_expr += self.u[j, 2, i]
            self.CF_mdl.add_constraint(lin_expr == self.xr[j], "r_feas1_cluster")

        # Constraints: Feasible cluster
        for i in self.N:
            lin_expr = self.CF_mdl.linear_expr()
            for j in self.N:
                for k in (0, 1, 2):
                    lin_expr += self.u[j, k, i]
            lin_expr -= 3 * instance.N * self.xg[i]
            self.CF_mdl.add_constraint(lin_expr <= 0, "feas2_cluster")
        '''
        # Constraints: Feasible cluster
        for i in self.N:
            for j in self.N:
                commStatus = instance.scenario.isCommunicate('Gateway', instance.scenario.poses[i],
                                                             instance.scenario.poses[j], instance.agent)
                lin_expr = self.CF_mdl.linear_expr()
                lin_expr += self.u[j, 0, i]
                lin_expr -= commStatus * self.xn[j]
                self.CF_mdl.add_constraint(lin_expr <= 0, "n_feas3_cluster")

                lin_expr = self.CF_mdl.linear_expr()
                lin_expr += self.u[j, 1, i]
                lin_expr -= commStatus * instance.scenario.xa[j]
                self.CF_mdl.add_constraint(lin_expr <= 0, "a_feas3_cluster")

                lin_expr = self.CF_mdl.linear_expr()
                lin_expr += self.u[j, 2, i]
                lin_expr -= commStatus * self.xr[j]
                self.CF_mdl.add_constraint(lin_expr <= 0, "r_feas3_cluster")
        '''
        # Constraints: Cluster overhead -- NONLINEAR
        for i in self.N:
            lin_expr = self.CF_mdl.linear_expr()
            lin_expr += self.O[i]
            for j in self.N:
                lin_expr -= self.wn[j] * self.u[j, 0, i]
                for k in (1, 2):
                    lin_expr -= (instance.agent.models[self.nodeMap[k]]['gateCommand'] +
                                 instance.agent.models[self.nodeMap[k]]['dataSize']) * self.u[j, k, i]
            self.CF_mdl.add_constraint(lin_expr == 0, "cluster_overhead")

        # Constraints : Cluster deviation -- NONLINEAR
        for i in self.N:
            lin_expr = self.CF_mdl.linear_expr()
            for j in self.N:
                lin_expr += (1 - self.epsilon) * self.O[j]
                lin_expr -= 3 * self.H * instance.N * (1 - self.xg[i])
                lin_expr -= self.xg[j] * self.O[i]
            self.CF_mdl.add_constraint(lin_expr <= 0, "cluster_deviation1")

            lin_expr = self.CF_mdl.linear_expr()
            for j in self.N:
                lin_expr -= (1 + self.epsilon) * self.O[j]
                lin_expr -= 3 * self.H * instance.N * (1 - self.xg[i])
                lin_expr += self.xg[j] * self.O[i]
            self.CF_mdl.add_constraint(lin_expr <= 0, "cluster_deviation2")

        self.CF_mdl.export_as_lp("CF_mdl.lp")
        start_time = time.time()
        # Solve the model
        self.CF_mdl.context.cplex_parameters.timelimit = 300  # 5 minutes
        solution = self.CF_mdl.solve(log_output=True)
        end_time = time.time()
        self.CF_runtime = end_time - start_time
        print(f"CF model execution time is {self.CF_runtime:.4f} seconds.")

        self.OO = {}  # Dictionary to store O values
        self.uu = {}  # Dictionary to store u values

        # Print solution if available
        if solution:
            print("Solution found. Sensor Nodes with Sensor placements:")
            self.CF_ObjValue = solution.objective_value  # or solution.get_objective_value()
            self.CF_RelGap = solution.solve_details.mip_relative_gap  # 0 if proven optimal
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

            self.isFeasible['CF'] = True
        else:
            print("No feasible solution found.")
            self.OO = None
            self.uu = None

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
        widths = (6, 12, 15, 10)  # Model, ObjVal, CPU, Gap %
        cols = [
            f"{'Model':<{widths[0]}}",
            f"{'ObjVal':>{widths[1]}}",
            f"{'CPU (secs)':>{widths[2]}}",
            f"{'Gap %':>{widths[3]}}"
        ]

        # helper that formats one model row
        def make_row(tag, obj, cpu, gap):
            return [
                f"{tag:<{widths[0]}}",
                f"{obj:>{widths[1]}.3f}",
                f"{cpu:>{widths[2]}.5f}",
                f"{gap * 100:>{widths[3]}.2f} %"
            ]

        self.sumObj = self.GD_ObjValue + self.SD_ObjValue + self.NTD_ObjValue + self.NMD_ObjValue \
                    + self.RD_ObjValue + self.NBD_ObjValue + self.CF_ObjValue

        self.sumRuntime = self.GD_runtime + self.SD_runtime + self.NTD_runtime + self.NMD_runtime \
                          + self.RD_runtime + self.NBD_runtime + self.CF_runtime

        # ------------------------------------------------------------------
        rows = [
            make_row('GD', self.GD_ObjValue, self.GD_runtime, self.GD_RelGap),
            make_row('SD', self.SD_ObjValue, self.SD_runtime, self.SD_RelGap),
            make_row('NTD', self.NTD_ObjValue, self.NTD_runtime, self.NTD_RelGap),
            make_row('NMD', self.NMD_ObjValue, self.NMD_runtime, self.NMD_RelGap),
            make_row('RD', self.RD_ObjValue, self.RD_runtime, self.RD_RelGap),
            make_row('NBD', self.NBD_ObjValue, self.NBD_runtime, self.NBD_RelGap),
            make_row('CF', self.CF_ObjValue, self.CF_runtime, self.CF_RelGap),
            make_row('Total', self.sumObj, self.sumRuntime, 0  ),
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

        self.setVisualization(True, True, True, True)

        def polar_offset(x0, y0, r, theta_deg):
            theta = np.deg2rad(theta_deg)
            return x0 + r * np.cos(theta), y0 + r * np.sin(theta)

        xs = [p['x'] for p in instance.scenario.poses]
        ys = [p['y'] for p in instance.scenario.poses]

        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(16, 6))
        fig.canvas.manager.set_window_title("DeNAD Solution")

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
        self.setVisualization(True, True, True, True)  # ensure flags exist

        def polar_offset(x0, y0, r, theta_deg):
            theta = np.deg2rad(theta_deg)
            return x0 + r * np.cos(theta), y0 + r * np.sin(theta)

        # ---------------------------------------------------------------- poses
        xs = [p['x'] for p in instance.scenario.poses]
        ys = [p['y'] for p in instance.scenario.poses]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title("DeNAD Solution")
        ax.set_title("DeNAD Agent Deployment")

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
        ax.grid(True);
        fig.tight_layout();
        plt.show()

    def visualizeSolution(self, instance, isCluster):
        if isCluster:
            self.visualizeClusterSolution(instance)
        else:
            self.visualizeDeploymentSolution(instance)
