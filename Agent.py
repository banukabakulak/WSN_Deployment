class Agent:
    def __init__(self):
        self.models = {
            'SensorNode': {'cost': 0.0, 'dataSize': 0.24, 'gateCommand': 0.28, 'commRange': 0, 'lambda': 1, 'gateCover': 1},
            'Actuator': {'cost': 45.0, 'dataSize': 0.24, 'gateCommand': 0.28, 'commRange': 0.8, 'lambda': 1, 'gateCover': 1}, # Rain Bird CP075 irrigation valve
            'Router': {'cost': 30.0, 'dataSize': 0.24, 'gateCommand': 0.28, 'commRange': 1.5, 'lambda': 1, 'gateCover': 1}, # XBee 3 Zigbee Module
            'Gateway': {'cost': 150.0, 'dataSize': 0, 'gateCommand': 0.28, 'commRange': 2, 'lambda': 0, 'gateCover': 1} # Digi XBee Gateway 
        }

    def display_types(self):
        print("Agent Types:")
        for name, specs in self.models.items():
            print(f"{name}: Cost = ${specs['cost']:.2f}, "
                  f"dataSize = {specs['dataSize']} Kbits/period, "
                  f"gateCommand = {specs['gateCommand']} Kbits, "
                  f"commRange = {specs['commRange']} km, "
                  f"lambda = {specs['lambda']}, "
                  f"gateCover = {specs['gateCover']}")

class Box:
    def __init__(self):
        # Store properties in a dictionary
        self.attributes = { # Hammond 1554F2GYCL, IP67, clear lid, size: 160×90×60 mm
            'cost': 25,  # USD
            'volume': 5,  # 864 cm³
            'P': 3,  # number of sensor pins, 5-8 sensors can be connected to the SoC transceivers
            'D': 3, # number of battery docks, 2-3 parallel connected batteries to increase mAh capacity, voltage is the same.
            'C': 2 # number of memory compartments, 4-6 SPI connections to the SoC transceivers
        }

    def display_info(self):
        print("Protective Box Parameters:") # (cost [$], volume [cm³], pin[#])
        print(f"Cost: ${self.attributes['cost']}, "
              f"Volume: {self.attributes['volume']} cm³, "
              f"S_Pin: {self.attributes['P']}, "
              f"B_Dock: {self.attributes['D']}, "
              f"M_Comp: {self.attributes['C']}")
