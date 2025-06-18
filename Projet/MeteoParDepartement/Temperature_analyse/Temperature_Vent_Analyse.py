import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import seaborn as sns

class AnalyseDirectionVent:
    def __init__(self):
        self.result_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self.result_path.parent.parent / "CSVTrier"
        self.images_path = self.result_path / "Images" / "DirectionVent"
        os.makedirs(self.images_path, exist_ok=True)
        
        self.departements = ['13', '67', '72', '64']
        
        # Définition des directions cardinales avec leurs plages de degrés
        self.directions = {
            'N': (348.75, 11.25),    # 0° ± 11.25°
            'NE': (33.75, 56.25),    # 45° ± 11.25°
            'E': (78.75, 101.25),    # 90° ± 11.25°
            'SE': (123.75, 146.25),  # 135° ± 11.25°
            'S': (168.75, 191.25),   # 180° ± 11.25°
            'SO': (213.75, 236.25),  # 225° ± 11.25°
            'O': (258.75, 281.25),   # 270° ± 11.25°
            'NO': (303.75, 326.25)   # 315° ± 11.25°
        }
        
        # Define the order of directions for consistent plotting
        self.directions_ordre = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']

    def verifier_donnees_brutes(self, departement):
        """Vérifie les données brutes de direction avant conversion"""
        print(f"\nVérification des données brutes pour le département {departement}:")
        
        for year in sorted(self.data_dir.iterdir()):
            if year.is_dir() and year.name.isdigit():
                path_departement = year / departement
                if path_departement.exists():
                    for csv_file in path_departement.glob('*.csv'):
                        try:
                            df = pd.read_csv(csv_file, sep=';')
                            # Vérifier quelle colonne de direction est disponible
                            colonne_direction = None
                            if 'DD' in df.columns:
                                colonne_direction = 'DD'
                            elif 'DG' in df.columns:
                                colonne_direction = 'DG'
                            
                            if 'T' in df.columns and colonne_direction is not None:
                                print(f"\nFichier: {csv_file.name}")
                                print(f"Utilisation de la colonne: {colonne_direction}")
                                
                                # Nettoyer les données
                                df = df[df[colonne_direction].notna()]
                                df = df[df[colonne_direction] != -9999]
                                df = df[df[colonne_direction].between(0, 360)]
                                
                                # Créer des intervalles de 45 degrés
                                bins = np.arange(0, 405, 45)
                                labels = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
                                df['DIRECTION_TEST'] = pd.cut(df[colonne_direction], bins=bins, labels=labels, include_lowest=True)
                                
                                print("\nDistribution des directions (par intervalles de 45°):")
                                print(df['DIRECTION_TEST'].value_counts())
                                
                                print("\nDistribution des degrés:")
                                print(df[colonne_direction].value_counts().sort_index())
                                
                                # Tester la conversion sur quelques valeurs
                                print("\nTest de conversion sur quelques valeurs:")
                                test_values = df[colonne_direction].dropna().unique()[:10]
                                for val in test_values:
                                    print(f"{val}° -> {self.get_direction_vent(val)}")
                                
                                return
                        except Exception as e:
                            print(f"Erreur lors de la lecture de {csv_file}: {str(e)}")

    def get_direction_vent(self, degres):
        """Convertit les degrés en direction cardinale"""
        if pd.isna(degres) or degres == -9999:
            return None
            
        # Normaliser les degrés entre 0 et 360
        degres = degres % 360
        
        # Définir les intervalles pour chaque direction
        if degres >= 337.5 or degres < 22.5:
            return 'N'
        elif 22.5 <= degres < 67.5:
            return 'NE'
        elif 67.5 <= degres < 112.5:
            return 'E'
        elif 112.5 <= degres < 157.5:
            return 'SE'
        elif 157.5 <= degres < 202.5:
            return 'S'
        elif 202.5 <= degres < 247.5:
            return 'SO'
        elif 247.5 <= degres < 292.5:
            return 'O'
        else:  # 292.5 <= degres < 337.5
            return 'NO'

    def charger_donnees(self, departement):
        """Charge et nettoie les données pour un département"""
        all_data = []
        
        for year in sorted(self.data_dir.iterdir()):
            if year.is_dir() and year.name.isdigit():
                path_departement = year / departement
                if path_departement.exists():
                    for csv_file in path_departement.glob('*.csv'):
                        try:
                            df = pd.read_csv(csv_file, sep=';')
                            # Vérifier quelle colonne de direction est disponible
                            colonne_direction = None
                            if 'DD' in df.columns:
                                colonne_direction = 'DD'
                            elif 'DG' in df.columns:
                                colonne_direction = 'DG'
                            
                            if 'T' in df.columns and colonne_direction is not None:
                                # Nettoyer les données
                                df = df[df['T'].notna() & df[colonne_direction].notna()]
                                df = df[(df['T'] != -9999) & (df[colonne_direction] != -9999)]
                                df = df[df['T'].between(-50, 50)]
                                df = df[df[colonne_direction].between(0, 360)]
                                
                                df['NOM_USUEL'] = csv_file.stem
                                df['DIRECTION'] = df[colonne_direction].apply(self.get_direction_vent)
                                
                                all_data.append(df)
                        except Exception as e:
                            print(f"Erreur lors de la lecture de {csv_file}: {str(e)}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def analyser_correlations(self, departement):
        """Analyse les corrélations entre direction du vent et température"""
        print(f"\nAnalyse des corrélations pour le département {departement}:")
        
        df = self.charger_donnees(departement)
        if df is None or df.empty:
            print(f"Pas de données disponibles pour le département {departement}")
            return
        
        # Déterminer quelle colonne de direction est utilisée
        colonne_direction = 'DD' if 'DD' in df.columns else 'DG'
        
        # Créer une figure avec 4 sous-graphiques
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Distribution des directions
        ax1 = fig.add_subplot(gs[0, 0])
        direction_counts = df['DIRECTION'].value_counts()
        # Réorganiser selon l'ordre défini
        direction_counts = direction_counts.reindex(self.directions_ordre)
        direction_counts.plot(kind='bar', ax=ax1)
        ax1.set_title(f'Distribution des Directions de Vent\nDépartement {departement}')
        ax1.set_xlabel('Direction')
        ax1.set_ylabel('Nombre d\'observations')
        
        # Ajouter les pourcentages
        total = direction_counts.sum()
        for i, v in enumerate(direction_counts):
            ax1.text(i, v, f'{v/total*100:.1f}%', ha='center', va='bottom')
        
        # 2. Température moyenne par direction
        ax2 = fig.add_subplot(gs[0, 1])
        temp_by_dir = df.groupby('DIRECTION')['T'].agg(['mean', 'std', 'count'])
        # Réorganiser selon l'ordre défini
        temp_by_dir = temp_by_dir.reindex(self.directions_ordre)
        temp_by_dir['mean'].plot(kind='bar', yerr=temp_by_dir['std'], ax=ax2, capsize=5)
        ax2.set_title(f'Température Moyenne par Direction\nDépartement {departement}')
        ax2.set_xlabel('Direction')
        ax2.set_ylabel('Température (°C)')
        
        # Ajouter les valeurs sur les barres
        for i, (mean, std, count) in enumerate(zip(temp_by_dir['mean'], temp_by_dir['std'], temp_by_dir['count'])):
            ax2.text(i, mean, f'{mean:.1f}°C\nn={count:,}', ha='center', va='bottom')
        
        # 3. Rose des vents (température moyenne)
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        # Définir les angles pour chaque direction (en radians)
        angles = {
            'N': 0,      # 0°
            'NE': 45,    # 45°
            'E': 90,     # 90°
            'SE': 135,   # 135°
            'S': 180,    # 180°
            'SO': 225,   # 225°
            'O': 270,    # 270°
            'NO': 315    # 315°
        }
        
        # Convertir les angles en radians
        angles_rad = np.array([np.radians(angles[d]) for d in self.directions_ordre])
        angles_rad = np.concatenate((angles_rad, [angles_rad[0]]))  # Fermer le cercle
        
        # Obtenir les températures dans le bon ordre
        temps = [temp_by_dir.loc[d, 'mean'] if d in temp_by_dir.index else 0 for d in self.directions_ordre]
        temps = np.concatenate((temps, [temps[0]]))  # Fermer le cercle
        
        # Tracer la rose des vents
        ax3.plot(angles_rad, temps)
        ax3.fill(angles_rad, temps, alpha=0.25)
        
        # Configurer l'axe polaire
        ax3.set_theta_zero_location('N')  # Placer le Nord en haut
        ax3.set_theta_direction(-1)  # Sens horaire
        ax3.set_xticks(angles_rad[:-1])
        ax3.set_xticklabels(self.directions_ordre)
        ax3.set_title(f'Rose des Vents (Température Moyenne)\nDépartement {departement}')
        
        # 4. Box plot des températures par direction
        ax4 = fig.add_subplot(gs[1, 1])
        # Réorganiser les données pour le box plot
        df['DIRECTION'] = pd.Categorical(df['DIRECTION'], categories=self.directions_ordre, ordered=True)
        sns.boxplot(x='DIRECTION', y='T', data=df, ax=ax4, order=self.directions_ordre)
        ax4.set_title(f'Distribution des Températures par Direction\nDépartement {departement}')
        ax4.set_xlabel('Direction')
        ax4.set_ylabel('Température (°C)')
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = self.images_path / f'analyse_direction_vent_{departement}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nAnalyse détaillée pour le département {departement}:")
        print("\nStatistiques par direction:")
        print(temp_by_dir.round(2))
        
        # Trouver la direction dominante
        direction_dominante = direction_counts.index[0]
        print(f"\nDirection dominante: {direction_dominante}")
        print(f"Nombre d'observations: {direction_counts[direction_dominante]:,}")
        print(f"Pourcentage: {direction_counts[direction_dominante]/total*100:.1f}%")
        print(f"Température moyenne: {temp_by_dir.loc[direction_dominante, 'mean']:.1f}°C")
        
        # Afficher les statistiques par ville
        print("\nStatistiques par ville:")
        stats_par_ville = df.groupby('NOM_USUEL').agg({
            'DIRECTION': ['count', 'nunique'],
            'T': ['count', 'mean', 'std', 'min', 'max'],
            colonne_direction: ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        print(stats_par_ville)
        
        print(f"\nGraphique sauvegardé dans {output_path}")

    def analyser_donnees_brutes(self, departement):
        """Analyse en détail les données brutes de direction"""
        print(f"\nAnalyse détaillée des données pour le département {departement}:")
        
        all_data = []
        for year in sorted(self.data_dir.iterdir()):
            if year.is_dir() and year.name.isdigit():
                path_departement = year / departement
                if path_departement.exists():
                    for csv_file in path_departement.glob('*.csv'):
                        try:
                            df = pd.read_csv(csv_file, sep=';')
                            # Vérifier quelle colonne de direction est disponible
                            colonne_direction = None
                            if 'DD' in df.columns:
                                colonne_direction = 'DD'
                            elif 'DG' in df.columns:
                                colonne_direction = 'DG'
                            
                            if 'T' in df.columns and colonne_direction is not None:
                                # Nettoyer les données
                                df = df[df[colonne_direction].notna()]
                                df = df[df[colonne_direction] != -9999]
                                df = df[df[colonne_direction].between(0, 360)]
                                all_data.append(df)
                        except Exception as e:
                            print(f"Erreur lors de la lecture de {csv_file}: {str(e)}")
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            # Déterminer quelle colonne a été utilisée
            colonne_direction = 'DD' if 'DD' in final_df.columns else 'DG'
            
            print("\nStatistiques des directions en degrés:")
            print("Min:", final_df[colonne_direction].min())
            print("Max:", final_df[colonne_direction].max())
            print("Moyenne:", final_df[colonne_direction].mean())
            print("Médiane:", final_df[colonne_direction].median())
            
            print("\nDistribution des degrés:")
            print(final_df[colonne_direction].value_counts().sort_index())
            
            # Créer un histogramme des directions
            plt.figure(figsize=(15, 6))
            plt.hist(final_df[colonne_direction], bins=36, range=(0, 360))
            plt.title(f'Distribution des Directions de Vent (en degrés)\nDépartement {departement}')
            plt.xlabel('Direction (degrés)')
            plt.ylabel('Nombre d\'observations')
            plt.grid(True, alpha=0.3)
            
            # Ajouter les limites des directions cardinales
            directions = {
                'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
                'S': 180, 'SO': 225, 'O': 270, 'NO': 315
            }
            for direction, angle in directions.items():
                plt.axvline(angle, color='r', linestyle='--', alpha=0.3)
                plt.text(angle, plt.ylim()[1]*0.9, direction, rotation=90)
            
            output_path = self.images_path / f'distribution_degres_{departement}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nGraphique de distribution sauvegardé dans {output_path}")

    def main(self):
        """Exécute l'analyse pour tous les départements"""
        # D'abord analyser les données brutes
        for departement in self.departements:
            self.analyser_donnees_brutes(departement)
        
        # Puis faire l'analyse des corrélations
        for departement in self.departements:
            self.analyser_correlations(departement)

if __name__ == "__main__":
    analyse = AnalyseDirectionVent()
    analyse.main()