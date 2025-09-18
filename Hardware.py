class Battery:
    def __init__(self):
        # Dictionary of predefined battery models
        self.models = {  # capacity in mAh
            'B1': {'cost': 6, 'capacity': 2400}, # Tadiran TL-5903 3.6V primary battery
            'B2': {'cost': 8, 'capacity': 3400}  # Panasonic NCR18650B 3.6V rechargable battery
        }

    def display_models(self):
        print("Battery Models:")
        for name, spec in self.models.items():
            print(f"{name}: Cost = ${spec['cost']}, Capacity = {spec['capacity']} mAh")

class Memory:
    def __init__(self):
        # Memory model: cost in USD, capacity in Kbits
        self.models = {
            'M1': {'cost': 2.5, 'capacity': 8 * 8 * 1024}, # Adesto AT45DB641E Flash, 8 MB = 65,536 Kbits
            'M2': {'cost': 1.5, 'capacity': 1 * 1024} # Microchip 25AA1024 EEPROM, 128 KB = 1,024 Kbits
        }

    def display_models(self):
        print("Memory Models (Cost in $, Capacity in Kbits):")
        for name, specs in self.models.items():
            print(f"{name}: Cost = ${specs['cost']:.2f}, Capacity = {specs['capacity']} Kbits")

class Transceiver:
    def __init__(self, onePeriod, isCNAD):
        # Dictionary of available transceiver models
        self.dutyCycle = 6 # seconds
        self.txTime = 1 # second
        self.models = {
            'T1': {'cost': 4.5, 'commRange': 1,  'bandwidth': 250 * self.txTime, 'activeEnergy': 0.00264 * self.dutyCycle, 'txEnergy': 0.1, 'rxEnergy': 0.011}, # EFR32MG21 SoC transceiver
            'T2': {'cost': 2.5, 'commRange': 0.8,  'bandwidth': 150 * self.txTime, 'activeEnergy': 0.00147 * self.dutyCycle, 'txEnergy': 0.009, 'rxEnergy': 0.005}, # nRF52840 SoC transceiver  
        }

        # Conditionally add Silicon Labs MGM210PA model, cost 12, commRange 1.5 - 2 km
        if not isCNAD:
            self.models['T3'] = {
                'cost': 40.0,  # USD
                'commRange': 2,  # km
                'bandwidth': 250 * self.txTime,  # kilobits
                'activeEnergy': 0.001 * self.dutyCycle,  # mAh
                'txEnergy': 0.000111,  # mAh/kbit
                'rxEnergy': 0.0000106  # mAh/kbit
            }

    def display_models(self):
        print("Transceiver Models:")
        for name, t in self.models.items():
            print(f"{name}: Cost = ${t['cost']:.2f}, commRange = {t['commRange']} km, "
                  f"Bandwidth = {t['bandwidth']} Kbits/period, ActiveEnergy = {t['activeEnergy']} mAh, "
                  f"TxEnergy = {t['txEnergy']} mAh/Kbit, RxEnergy = {t['rxEnergy']} mAh/Kbit")


