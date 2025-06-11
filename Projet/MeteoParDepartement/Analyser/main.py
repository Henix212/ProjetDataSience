from tempAltitude.temperatureAltitude import TemperatureAltitudeAnalyzer
from tempAltitude.temperatureAltitudeCorr import TemperatureAltitudeCorrelation

def main():
    TempAltAnalyzer = TemperatureAltitudeAnalyzer()
    TempAltCoor = TemperatureAltitudeCorrelation()

    TempAltAnalyzer.analyze_all_departments()
    TempAltCoor.analyze_all_departments()

    print("\nAnalyse terminée ! Les graphiques ont été sauvegardés dans le dossier 'images'.")

if __name__ == "__main__":
    main()
