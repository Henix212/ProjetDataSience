import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class TemperatureAltitudeCorrelation:
    def __init__(self):
        # Obtenir le chemin absolu du répertoire de travail
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.project_dir = self.script_dir.parent.parent.parent
        self.csv_trier_path = self.project_dir / 'CSVTrier'
        self.images_path = self.project_dir / 'Images/ImagesTempAlt/MatricesCorrelation'
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
        """Analyse les données d'un département et crée la matrice de corrélation"""
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
            
            # Créer la matrice de corrélation
            plt.figure(figsize=(10, 8))
            
            # Calculer la matrice de corrélation
            corr_matrix = city_stats[['T', 'ALTI']].corr()
            
            # Créer la heatmap
            sns.heatmap(corr_matrix, 
                       annot=True,  # Afficher les valeurs
                       cmap='coolwarm',  # Utiliser une palette de couleurs
                       vmin=-1, vmax=1,  # Échelle de corrélation
                       center=0,  # Centre de l'échelle
                       square=True,  # Forme carrée
                       fmt='.3f')  # Format des nombres
            
            plt.title(f'Matrice de Corrélation - Département {dept}\nTempérature vs Altitude')
            
            # Sauvegarder le graphique
            plt.savefig(self.images_path / f'correlation_matrix_dept_{dept}.png')
            plt.close()
            
            # Afficher les statistiques
            print(f"\nStatistiques de corrélation pour le département {dept}:")
            print(f"Nombre de villes: {len(city_stats)}")
            print(f"Corrélation température-altitude: {corr_matrix.loc['T', 'ALTI']:.3f}")
            
            # Afficher les villes avec leurs températures et altitudes
            print("\nDétail par ville:")
            for _, row in city_stats.sort_values('ALTI').iterrows():
                print(f"{row['city']}: {row['T']:.2f}°C à {row['ALTI']}m")

    def analyze_all_departments(self):
        """Analyse tous les départements"""
        for dept in self.departments:
            print(f"\nAnalyse du département {dept}...")
            self.analyze_department(dept)
