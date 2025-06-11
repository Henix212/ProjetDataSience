import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class TemperatureAltitudeAnalyzer:
    def __init__(self):
        # Obtenir le chemin absolu du répertoire de travail
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.project_dir = self.script_dir.parent.parent.parent
        self.csv_trier_path = self.project_dir / 'CSVTrier'
        self.images_path = self.project_dir / 'Images/ImagesTempAlt/NuagesDePoints'
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.departments = ['13', '64', '67', '72']
        
    def load_department_data(self, dept):
        """Charge toutes les données d'un département"""
        all_data = []
        
        # Parcourir toutes les années
        for year_dir in self.csv_trier_path.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                dept_path = year_dir / dept
                if dept_path.exists():
                    # Parcourir tous les fichiers CSV du département
                    for csv_file in dept_path.glob('*.csv'):
                        try:
                            df = pd.read_csv(csv_file, sep=';')
                            # Vérifier que les colonnes nécessaires existent
                            if 'T' in df.columns and 'ALTI' in df.columns:
                                df['city'] = csv_file.stem  # Nom de la ville (sans extension)
                                all_data.append(df)
                        except Exception as e:
                            print(f"Erreur lors de la lecture de {csv_file}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def analyze_department(self, dept):
        """Analyse les données d'un département et crée le graphique"""
        df = self.load_department_data(dept)
        if df is not None:
            # Calculer la moyenne de température par ville
            city_stats = df.groupby('city').agg({
                'T': 'mean',
                'ALTI': 'first'
            }).reset_index()
            
            # Filtrer les données invalides
            city_stats = city_stats[
                (city_stats['T'].notna()) & 
                (city_stats['ALTI'].notna()) & 
                (city_stats['T'] != 0) & 
                (city_stats['ALTI'] != 0)
            ]
            
            if len(city_stats) < 2:
                print(f"Pas assez de données valides pour le département {dept}")
                return
            
            # Créer le graphique
            plt.figure(figsize=(12, 8))
            
            # Tracer les points
            plt.scatter(city_stats['T'], city_stats['ALTI'], alpha=0.6, label='Villes')
            
            # Ajouter les noms des villes
            for idx, row in city_stats.iterrows():
                plt.annotate(row['city'], 
                           (row['T'], row['ALTI']),
                           xytext=(5, 5), textcoords='offset points')
            
            try:
                # Ajouter une ligne de régression
                z = np.polyfit(city_stats['T'], city_stats['ALTI'], 1)
                p = np.poly1d(z)
                plt.plot(city_stats['T'], p(city_stats['T']), "r--", alpha=0.8, 
                        label=f'Régression (y = {z[0]:.2f}x + {z[1]:.2f})')
            except Exception as e:
                print(f"Erreur lors du calcul de la régression pour le département {dept}: {e}")
            
            plt.title(f'Relation entre Température et Altitude - Département {dept}')
            plt.xlabel('Température moyenne (°C)')
            plt.ylabel('Altitude (m)')
            plt.grid(True)
            plt.legend()
            
            # Sauvegarder le graphique
            plt.savefig(self.images_path / f'temperature_altitude_dept_{dept}.png')
            plt.close()
            
            # Afficher les statistiques
            print(f"\nStatistiques pour le département {dept}:")
            print(f"Nombre de villes: {len(city_stats)}")
            print(f"Altitude moyenne: {city_stats['ALTI'].mean():.2f}m")
            print(f"Température moyenne: {city_stats['T'].mean():.2f}°C")
            print(f"Corrélation altitude-température: {city_stats['ALTI'].corr(city_stats['T']):.3f}")
            
            # Afficher les villes avec leurs températures et altitudes
            print("\nDétail par ville:")
            for _, row in city_stats.sort_values('ALTI').iterrows():
                print(f"{row['city']}: {row['T']:.2f}°C à {row['ALTI']}m")

    def analyze_all_departments(self):
        """Analyse tous les départements"""
        for dept in self.departments:
            print(f"\nAnalyse du département {dept}...")
            self.analyze_department(dept)

def main():
    analyzer = TemperatureAltitudeAnalyzer()
    analyzer.analyze_all_departments()
    print("\nAnalyse terminée ! Les graphiques ont été sauvegardés dans le dossier 'Images'.")

if __name__ == "__main__":
    main()


