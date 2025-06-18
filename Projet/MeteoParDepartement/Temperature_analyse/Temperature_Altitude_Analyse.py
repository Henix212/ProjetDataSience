"""
Analyse de la relation entre l'altitude et la température pour 4 départements français.

Ce script effectue une analyse statistique approfondie de la relation entre l'altitude et la température
moyenne dans différents départements. Il utilise des techniques de régression polynomiale pour modéliser
cette relation et fournit des visualisations détaillées des résultats.

Aspects statistiques principaux :
1. Régression polynomiale :
   - Utilise des polynômes de degré 1 à 5 pour modéliser la relation
   - Sélectionne le meilleur degré basé sur le R² et la stabilité du modèle ( La stabilité est importante pour la metéo )
   - Normalise les données pour améliorer la stabilité numérique

2. Métriques d'évaluation :
   - R² (coefficient de détermination) : mesure la qualité d'ajustement du modèle
   - RMSE (Root Mean Square Error) : erreur quadratique moyenne, en °C
   - MAE (Mean Absolute Error) : erreur absolue moyenne, en °C
   - Validation croisée : évalue la stabilité du modèle

3. Analyse de corrélation :
   - Calcule la corrélation entre altitude et température
   - Visualise les matrices de corrélation par département

4. Filtrage des données :
   - Seuils de données minimum par ville pour assurer la fiabilité
   - Bornes d'altitude spécifiques par département ( Altitude maximal connu dans le département )
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import seaborn as sns

class TemperatureAltitude:
    def __init__(self) -> None:
        """
        Initialise l'analyse de température par altitude.
        
        Définit les chemins de données et les seuils statistiques :
        - Seuils de données minimum par ville pour assurer la fiabilité statistique
        - Bornes d'altitude par département pour limiter l'extrapolation
        """
        self.result_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self.result_path.parent.parent / "CSVTrier"
        self.images_path = self.result_path / "Images" / "TempAltitude"
        os.makedirs(self.images_path, exist_ok=True)
        
        self.departements = ['13', '67', '72', '64']
        self.seuils = {
            '13': 140000,
            '67': 132500,
            '72': 140000,
            '64': 100000
        }
        
        self.bornes_altitude = {
            '13': {'min': -12, 'max': 1658},
            '64': {'min': 0, 'max': 3247},
            '67': {'min': 104, 'max': 1274},
            '72': {'min': 20, 'max': 341}
        }

    def load_departements_data(self, departement):
        all_data = []

        for year in self.data_dir.iterdir():
            if year.is_dir() and year.name.isdigit():
                path_departement = year / departement
                if path_departement.exists():
                    for csv_file in path_departement.glob('*.csv'):
                        try: 
                            df = pd.read_csv(csv_file, sep=';')
                            if 'T' in df.columns and 'ALTI' in df.columns:
                                df = df[df['T'].notna() & df['ALTI'].notna()]
                                df = df[df['T'] != -9999] 
                                df['NOM_USUEL'] = csv_file.stem
                                all_data.append(df)
                        except Exception as e:
                            print(f"  Erreur lors de la lecture de {csv_file}: {str(e)}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def filtre_ville_seuille_nb_donnees(self, df, departement):
        if df is None or df.empty:
            return None
        
        ville_counts = df['NOM_USUEL'].value_counts()
        villes_selectionnees = ville_counts[ville_counts >= self.seuils[departement]].index
        df_filtre = df[df['NOM_USUEL'].isin(villes_selectionnees)]
        
        print(f"\nDépartement {departement}:")
        print(f"Seuil: {self.seuils[departement]:,} données")
        print("Villes sélectionnées:")
        for ville in villes_selectionnees:
            count = ville_counts[ville]
            print(f"- {ville}: {count:,} données")
        
        return df_filtre

    def evaluate_model_stability(self, model, X, y, cv=5):
        """
        Évalue la stabilité du modèle par validation croisée.
        
        Args:
            model: Le modèle de régression à évaluer
            X: Variables explicatives (altitude)
            y: Variable à prédire (température)
            cv: Nombre de folds pour la validation croisée
            
        Returns:
            tuple: (MSE moyen, écart-type du MSE)
            
        Note statistique:
            La validation croisée permet d'estimer la performance réelle du modèle
            en évitant le surapprentissage. Un écart-type faible indique un modèle stable.
        """
        # Évaluer la stabilité du modèle avec validation croisée
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        mse_cv = -cv_scores.mean()  # Convertir en MSE positif
        mse_std = cv_scores.std()   # Écart-type des scores
        return mse_cv, mse_std

    def find_best_polynomial(self, X, y, max_degree=5):
        """
        Trouve le meilleur degré polynomial pour modéliser la relation altitude-température.
        
        Args:
            X: Altitudes
            y: Températures
            max_degree: Degré maximum du polynôme à tester
            
        Returns:
            tuple: (meilleur degré, coefficients, intercept, R², scaler_X, scaler_y, modèle, features polynomiales, MSE, écart-type MSE)
            
        Note statistique:
            - Teste les degrés 1 à max_degree
            - Sélectionne le modèle avec le meilleur R² tout en évitant le surapprentissage
            - Utilise la validation croisée pour évaluer la stabilité
            - Normalise X et y pour améliorer la stabilité numérique
        """
        best_r2 = -np.inf
        best_degree = 1
        best_model = None
        best_coef = None
        best_intercept = None
        best_scaler_X = None
        best_scaler_y = None
        best_poly = None
        best_mse = float('inf')
        best_mse_std = float('inf')
        
        # Normaliser X et y
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, 1))
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        for degree in range(1, max_degree + 1):
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X_scaled)
            model = LinearRegression()
            model.fit(X_poly, y_scaled)
            r2 = model.score(X_poly, y_scaled)
            
            mse_cv, mse_std = self.evaluate_model_stability(model, X_poly, y_scaled)
            
            if r2 > best_r2 and mse_cv < best_mse * 1.5: 
                best_r2 = r2
                best_degree = degree
                best_model = model
                best_coef = model.coef_
                best_intercept = model.intercept_
                best_scaler_X = scaler_X
                best_scaler_y = scaler_y
                best_poly = poly
                best_mse = mse_cv
                best_mse_std = mse_std
        
        return best_degree, best_coef, best_intercept, best_r2, best_scaler_X, best_scaler_y, best_model, best_poly, best_mse, best_mse_std

    def get_polynomial_equation(self, coef, intercept, degree, scaler_X, scaler_y):
        """
        Génère l'équation polynomiale avec les coefficients dénormalisés.
        
        Args:
            coef: Coefficients du modèle
            intercept: Intercept du modèle
            degree: Degré du polynôme
            scaler_X: StandardScaler pour X
            scaler_y: StandardScaler pour y
            
        Returns:
            str: Équation polynomiale avec coefficients dénormalisés
        """
        # Dénormaliser les coefficients en tenant compte du centrage
        coef_denorm = coef.copy()
        
        # Ajuster les coefficients pour la dénormalisation
        for i in range(degree + 1):
            if i == 0:
                # Pour le terme constant, on doit tenir compte du centrage de X
                coef_denorm[i] = intercept
                for j in range(1, degree + 1):
                    coef_denorm[i] += coef[j] * (-scaler_X.mean_ / scaler_X.scale_) ** j
                coef_denorm[i] = scaler_y.inverse_transform([[coef_denorm[i]]])[0][0]
            else:
                # Pour les autres termes, on ajuste en tenant compte du centrage
                coef_denorm[i] = coef[i] * (scaler_y.scale_ / (scaler_X.scale_ ** i))
        
        terms = []
        for i in range(degree + 1):
            if i == 0:
                terms.append(f"{coef_denorm[i]:.2f}")
            else:
                terms.append(f"{coef_denorm[i]:.2f}x^{i}")
        return " + ".join(terms)

    def calculate_model_metrics(self, model, X, y, X_poly, scaler_y):
        """
        Calcule les métriques d'évaluation du modèle.
        
        Args:
            model: Modèle de régression
            X: Données d'origine
            y: Valeurs réelles
            X_poly: Features polynomiales
            scaler_y: StandardScaler pour y
            
        Returns:
            tuple: (MAE, RMSE, R²)
            
        Note statistique:
            - MAE: Erreur absolue moyenne, plus intuitive que le RMSE
            - RMSE: Erreur quadratique moyenne, pénalise plus les grandes erreurs
            - R²: Proportion de variance expliquée par le modèle
        """
        y_pred_scaled = model.predict(X_poly)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        return mae, rmse, r2

    def nuage_de_point_TempAlt(self):
        """
        Crée un graphique de dispersion altitude-température avec régression polynomiale.
        
        Pour chaque département:
        1. Filtre les données selon les seuils
        2. Calcule la température moyenne par station
        3. Ajuste un modèle polynomial
        4. Visualise les données et la courbe de régression
        
        Note statistique:
            - Les points représentent la température moyenne par station
            - La courbe représente le modèle polynomial optimal
            - Les métriques (R², RMSE, MAE) indiquent la qualité du modèle
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Relation Altitude-Température par Département\n(Villes avec données > seuil)', fontsize=16)
        
        colors = ['red', 'blue', 'green', 'purple']
        
        for idx, (departement, ax) in enumerate(zip(self.departements, axes.flat)):
            df = self.load_departements_data(departement)
            df = self.filtre_ville_seuille_nb_donnees(df, departement)
            
            if df is not None and not df.empty:
                station_temp_avg = df.groupby(['NOM_USUEL', 'ALTI'])['T'].mean().reset_index()
                
                if len(station_temp_avg) > 1: 
                    scatter = ax.scatter(station_temp_avg['ALTI'], station_temp_avg['T'],
                                       label=f'Stations', alpha=0.6, color=colors[idx])
                    
                    try:
                        X = station_temp_avg['ALTI'].values
                        y = station_temp_avg['T'].values
                        
                        degree, coef, intercept, r2, scaler_X, scaler_y, best_model, best_poly, mse, mse_std = self.find_best_polynomial(X, y)
                        
                        # Calculer les métriques supplémentaires
                        X_scaled = scaler_X.transform(X.reshape(-1, 1))
                        X_poly = best_poly.transform(X_scaled)
                        mae, rmse, r2_final = self.calculate_model_metrics(best_model, X, y, X_poly, scaler_y)
                        
                        # Générer plus de points pour une courbe plus lisse
                        x_range = np.linspace(self.bornes_altitude[departement]['min'],
                                            self.bornes_altitude[departement]['max'], 200)
                        
                        # Normaliser les points de prédiction
                        x_range_scaled = scaler_X.transform(x_range.reshape(-1, 1))
                        
                        # Calculer les prédictions
                        X_poly_range = best_poly.transform(x_range_scaled)
                        y_pred_scaled = best_model.predict(X_poly_range)
                        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                        
                        # Créer le texte de la légende avec toutes les informations
                        equation = self.get_polynomial_equation(coef, intercept, degree, scaler_X, scaler_y)
                        legend_text = (
                            f'Modèle polynomial (degré {degree})\n'
                            f'Équation: T = {equation}\n'
                            f'R² = {r2_final:.3f}\n'
                            f'MSE = {mse:.2f} ± {mse_std:.2f}\n'
                            f'RMSE = {rmse:.2f}°C\n'
                            f'MAE = {mae:.2f}°C\n'
                            f'Nombre de points: {len(X)}'
                        )
                        
                        ax.plot(x_range, y_pred, 
                               color=colors[idx], linestyle='--', 
                               label=legend_text)
                        
                        ax.axvline(x=self.bornes_altitude[departement]['min'], 
                                 color='gray', linestyle=':', alpha=0.5)
                        ax.axvline(x=self.bornes_altitude[departement]['max'], 
                                 color='gray', linestyle=':', alpha=0.5)
                        
                    except Exception as e:
                        print(f"Erreur lors de la régression pour le département {departement}: {str(e)}")
                    
                    ax.set_title(f'Département {departement}\nSeuil: {self.seuils[departement]:,} données')
                    ax.set_xlabel('Altitude (m)')
                    ax.set_ylabel('Température (°C)')
                    ax.grid(True, alpha=0.3)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    for i, row in station_temp_avg.iterrows():
                        ax.annotate(row['NOM_USUEL'], 
                                  (row['ALTI'], row['T']),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=5)
                else:
                    ax.text(0.5, 0.5, f'Pas assez de données\npour le département {departement}',
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'Pas de données\npour le département {departement}',
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        output_path = self.images_path / 'temperature_altitude_par_departement_filtre.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nGraphique sauvegardé dans {output_path}")

    def nb_donnee_par_villes(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Nombre de Données de Température par Ville', fontsize=16)
        
        for idx, (departement, ax) in enumerate(zip(self.departements, axes.flat)):
            df = self.load_departements_data(departement)
            if df is not None and not df.empty:
                ville_counts = df['NOM_USUEL'].value_counts()
                
                bars = ax.bar(ville_counts.index, ville_counts.values)
                
                ax.axhline(y=self.seuils[departement], color='r', linestyle='--', 
                          label=f'Seuil ({self.seuils[departement]:,} données)')
                
                ax.set_title(f'Département {departement}')
                ax.set_xlabel('Villes')
                ax.set_ylabel('Nombre de données')
                ax.grid(True, alpha=0.3)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}',
                           ha='center', va='bottom')
                
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'Pas de données\npour le département {departement}',
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        output_path = self.images_path / 'nombre_donnees_par_ville.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nGraphique sauvegardé dans {output_path}")

    def afficher_tous_polynomes(self):
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'purple']
        legend_entries = []
        
        for idx, departement in enumerate(self.departements):
            df = self.load_departements_data(departement)
            df = self.filtre_ville_seuille_nb_donnees(df, departement)
            
            if df is not None and not df.empty:
                station_temp_avg = df.groupby(['NOM_USUEL', 'ALTI'])['T'].mean().reset_index()
                
                if len(station_temp_avg) > 1:
                    try:
                        X = station_temp_avg['ALTI'].values
                        y = station_temp_avg['T'].values
                        
                        degree, coef, intercept, r2, scaler_X, scaler_y, best_model, best_poly, mse, mse_std = self.find_best_polynomial(X, y)
                        
                        # Calculer les métriques supplémentaires
                        X_scaled = scaler_X.transform(X.reshape(-1, 1))
                        X_poly = best_poly.transform(X_scaled)
                        mae, rmse, r2_final = self.calculate_model_metrics(best_model, X, y, X_poly, scaler_y)
                        
                        # Générer les points pour la courbe sur l'intervalle [0, altitude_max_departement]
                        altitude_max = self.bornes_altitude[departement]['max']
                        x_range = np.linspace(0, altitude_max, 200)
                        x_range_scaled = scaler_X.transform(x_range.reshape(-1, 1))
                        X_poly_range = best_poly.transform(x_range_scaled)
                        y_pred_scaled = best_model.predict(X_poly_range)
                        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                        
                        # Créer le texte de la légende
                        equation = self.get_polynomial_equation(coef, intercept, degree, scaler_X, scaler_y)
                        legend_text = (
                            f'Département {departement}\n'
                            f'Équation: T = {equation}\n'
                            f'R² = {r2_final:.3f}\n'
                            f'RMSE = {rmse:.2f}°C\n'
                            f'Altitude max: {altitude_max}m'
                        )
                        
                        # Tracer la courbe
                        plt.plot(x_range, y_pred, 
                                color=colors[idx], 
                                linestyle='-', 
                                linewidth=2,
                                label=legend_text)
                        
                        # Ajouter les points de données
                        plt.scatter(X, y, 
                                  color=colors[idx], 
                                  alpha=0.3, 
                                  s=20)
                        
                        # Ajouter une ligne verticale à l'altitude maximale
                        plt.axvline(x=altitude_max, 
                                  color=colors[idx], 
                                  linestyle=':', 
                                  alpha=0.5)
                        
                    except Exception as e:
                        print(f"Erreur lors de la régression pour le département {departement}: {str(e)}")
        
        plt.title('Comparaison des Relations Altitude-Température par Département\nChaque courbe s\'arrête à l\'altitude maximale du département', fontsize=14)
        plt.xlabel('Altitude (m)', fontsize=12)
        plt.ylabel('Température (°C)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Ajuster la légende
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Ajuster les marges pour la légende
        plt.tight_layout()
        
        output_path = self.images_path / 'comparaison_polynomes_departements.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nGraphique de comparaison sauvegardé dans {output_path}")

    def matrice_correlation(self):
        """
        Crée des matrices de corrélation pour chaque département.
        
        Note statistique:
            - La corrélation varie de -1 à 1
            - Ici une corrélation négative indique que la température diminue avec l'altitude
            - La valeur absolue indique la force de la relation
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Matrices de Corrélation Altitude-Température par Département', fontsize=16)
        
        for idx, (departement, ax) in enumerate(zip(self.departements, axes.flat)):
            df = self.load_departements_data(departement)
            df = self.filtre_ville_seuille_nb_donnees(df, departement)
            
            if df is not None and not df.empty:
                # Calculer la température moyenne par station
                station_temp_avg = df.groupby(['NOM_USUEL', 'ALTI'])['T'].mean().reset_index()
                
                if len(station_temp_avg) > 1:
                    # Créer la matrice de corrélation
                    corr_matrix = station_temp_avg[['ALTI', 'T']].corr()
                    
                    # Créer le heatmap
                    sns.heatmap(corr_matrix, 
                              annot=True, 
                              cmap='coolwarm', 
                              vmin=-1, 
                              vmax=1,
                              ax=ax,
                              fmt='.3f',
                              square=True)
                    
                    # Ajouter le coefficient de corrélation dans le titre
                    correlation = corr_matrix.loc['ALTI', 'T']
                    ax.set_title(f'Département {departement}\nCorrélation: {correlation:.3f}')
                    
                    # Ajuster les labels
                    ax.set_xticklabels(['Altitude', 'Température'])
                    ax.set_yticklabels(['Altitude', 'Température'])
                else:
                    ax.text(0.5, 0.5, f'Pas assez de données\npour le département {departement}',
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'Pas de données\npour le département {departement}',
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        output_path = self.images_path / 'matrices_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nMatrices de corrélation sauvegardées dans {output_path}")

    def main(self):
        self.nb_donnee_par_villes()
        self.nuage_de_point_TempAlt()
        self.afficher_tous_polynomes()
        self.matrice_correlation()

if __name__ == "__main__":
    Temp = TemperatureAltitude()
    Temp.main()