class Sensor:
    def __init__(self, S):
        # Full dictionary of available sensor models
        self.availableModels = {
            'S1':  {'senseRange': 0.1, 'volume': 0.1, 'cost': 20.0, 'sensePeriod': 30 * 60, 'dataSize': 0.36, 'gateCommand': 0.3, 'senseEnergy': 0.00028}, # DFRobot Gravity Analog Waterproof Capacitive Soil Moisture Sensor V2
            'S2':  {'senseRange': 0.1, 'volume': 0.1, 'cost': 12.0, 'sensePeriod': 30 * 60, 'dataSize': 0.32, 'gateCommand': 0.28, 'senseEnergy': 0.000078}, # SHT31‑D temperature & humidity sensor
            'S3':  {'senseRange': 0.1,  'volume': 0.92, 'cost': 6.0,  'sensePeriod': 30 * 60, 'dataSize': 0.32, 'gateCommand': 0.28, 'senseEnergy': 0.00006}, # Zio Qwiic TSL2561 Light intensity sensor
            'S4':  {'senseRange': 0.1, 'volume': 5.6, 'cost': 33.0, 'sensePeriod': 30 * 60, 'dataSize': 0.32, 'gateCommand': 0.28, 'senseEnergy': 0.00066}, # SCD30 (Sensirion CO₂ + Temp + RH) Air quality sensor
            'S5':  {'senseRange': 0.1,  'volume': 0, 'cost': 2.5,  'sensePeriod': 30 * 60, 'dataSize': 0.24, 'gateCommand': 0.28, 'senseEnergy': 0.000028}, # YL‑83 (FC‑37) Rain Sensor Module
        }

        # Select the first S sensor models
        if S > len(self.availableModels):
            raise ValueError(f"S = {S} exceeds the number of available models ({len(self.availableModels)}).")

        self.models = dict(list(self.availableModels.items())[:S])

    def get_period_length(self):
        onePeriod = max(sensor['sensePeriod'] for sensor in self.models.values())
        print(f"Length of a period is {onePeriod} seconds")
        return onePeriod

    def display_models(self):
        print("Sensor Models:")
        for name, sensor in self.models.items():
            print(f"{name}: "
                  f"senseRange = {sensor['senseRange']} km, Volume = {sensor['volume']} cm³, "
                  f"Cost = ${sensor['cost']}, sensePeriod = {sensor['sensePeriod']} s, "
                  f"dataSize = {sensor['dataSize']} Kbits/period, "
                  f"gateCommand = {sensor['gateCommand']} Kbits, "
                  f"senseEnergy = {sensor['senseEnergy']} mAh/period")
