import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA

class DepartmentClassification:
    def __init__(self):
        self.result_path = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self.result_path.parent.parent / "CSVTrier"
        self.images_path = self.result_path / "Images" / "Classification"
        
        # Créer explicitement le dossier
        try:
            self.images_path.mkdir(parents=True, exist_ok=True)
            print(f"Dossier créé : {self.images_path}")
        except Exception as e:
            print(f"Erreur lors de la création du dossier : {str(e)}")
        
        self.departements = ['13', '67', '72', '64']
        self.n_folds = 5  # Nombre de folds pour la validation croisée

    def charger_donnees(self):
        """Charge et prépare les données pour tous les départements"""
        all_data = []
        
        for departement in self.departements:
            for year in sorted(self.data_dir.iterdir()):
                if year.is_dir() and year.name.isdigit():
                    path_departement = year / departement
                    if path_departement.exists():
                        for csv_file in path_departement.glob('*.csv'):
                            try:
                                df = pd.read_csv(csv_file, sep=';')
                                if 'T' in df.columns:
                                    # Nettoyer les données
                                    df = df[df['T'].notna()]
                                    df = df[df['T'] != -9999]
                                    df = df[df['T'].between(-50, 50)]
                                    df['NOM_USUEL'] = csv_file.stem
                                    df['DEPARTEMENT'] = departement
                                    all_data.append(df)
                            except Exception as e:
                                print(f"Erreur lors de la lecture de {csv_file}: {str(e)}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def preparer_features(self, df):
        """Prépare les features pour la classification en utilisant toutes les températures brutes"""
        # Convertir la date en datetime
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        
        # Créer une feature pour chaque jour de l'année (1-365)
        df['jour_annee'] = df['date'].dt.dayofyear
        
        # Pivoter les données pour avoir une colonne par jour de l'année
        features = df.pivot_table(
            index=['NOM_USUEL', 'DEPARTEMENT'],
            columns='jour_annee',
            values='T'
        ).reset_index()
        
        # Séparer les features et la cible
        X = features.drop(['NOM_USUEL', 'DEPARTEMENT'], axis=1)
        y = features['DEPARTEMENT']
        
        # Remplacer les valeurs NaN par la moyenne des températures pour chaque jour
        X = X.fillna(X.mean())
        
        return X, y, features['NOM_USUEL']

    def preparer_train_test(self, X, y, villes):
        """Prépare les ensembles d'entraînement et de test en utilisant les villes comme unité"""
        # Créer un DataFrame avec les features et la cible
        df = pd.DataFrame(X)
        df['DEPARTEMENT'] = y
        df['NOM_USUEL'] = villes
        
        # Sélectionner aléatoirement 80% des villes de chaque département
        villes_train = []
        for dept in self.departements:
            villes_dept = df[df['DEPARTEMENT'] == dept]['NOM_USUEL'].unique()
            n_villes = len(villes_dept)
            n_train = int(0.8 * n_villes)
            villes_train.extend(np.random.choice(villes_dept, n_train, replace=False))
        
        # Séparer les données
        mask_train = df['NOM_USUEL'].isin(villes_train)
        X_train = df[mask_train].drop(['DEPARTEMENT', 'NOM_USUEL'], axis=1)
        y_train = df[mask_train]['DEPARTEMENT']
        X_test = df[~mask_train].drop(['DEPARTEMENT', 'NOM_USUEL'], axis=1)
        y_test = df[~mask_train]['DEPARTEMENT']
        
        return X_train, X_test, y_train, y_test

    def entrainer_modeles(self, X, y):
        """Entraîne et compare KNN et LDA avec validation croisée"""
        # Normaliser les features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialiser les modèles
        knn = KNeighborsClassifier(n_neighbors=5)
        lda = LinearDiscriminantAnalysis()
        
        # Validation croisée pour KNN
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        knn_scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
        
        # Validation croisée pour LDA
        lda_scores = cross_val_score(lda, X_scaled, y, cv=kf, scoring='accuracy')
        
        print("\nScores de validation croisée (accuracy):")
        print(f"KNN - Moyenne: {knn_scores.mean():.3f} (+/- {knn_scores.std() * 2:.3f})")
        print(f"LDA - Moyenne: {lda_scores.mean():.3f} (+/- {lda_scores.std() * 2:.3f})")
        
        # Entraîner les modèles finaux sur toutes les données
        knn.fit(X_scaled, y)
        lda.fit(X_scaled, y)
        
        return knn, lda, scaler, X_scaled, y

    def visualiser_projection_et_frontieres(self, knn, lda, X, y):
        """
        Visualise la projection LDA, l'importance des features, et les frontières de décision KNN.
        Génère des graphiques pédagogiques et explicites.
        """
        # --- PROJECTION LDA ---
        X_lda = lda.transform(X)
        plt.figure(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, dept in enumerate(self.departements):
            mask = y == dept
            plt.scatter(X_lda[mask, 0], X_lda[mask, 1], label=f'Département {dept}', alpha=0.7, color=colors[i])
        plt.title('Projection LDA (2 premières composantes)')
        plt.xlabel('Composante 1')
        plt.ylabel('Composante 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.images_path / 'projection_lda.png', dpi=300, bbox_inches='tight')
        plt.close()

        # --- ANALYSE DÉTAILLÉE DES COEFFICIENTS LDA ---
        # Créer un DataFrame avec les coefficients
        feature_names = [f'Jour {i+1}' for i in range(X.shape[1])]
        coef_df = pd.DataFrame(lda.coef_, 
                             columns=feature_names,
                             index=[f'Département {d}' for d in self.departements])
        
        # 1. Heatmap des coefficients
        plt.figure(figsize=(20, 6))
        sns.heatmap(coef_df, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Poids discriminant'})
        plt.title('Importance des températures quotidiennes pour chaque département (LDA)')
        plt.xlabel('Jour de l\'année')
        plt.ylabel('Département')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.images_path / 'lda_coefficients_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Graphique des coefficients moyens par mois
        mean_coef = coef_df.abs().mean()
        # Convertir les jours en mois
        jours_par_mois = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        mois_coef = []
        jour_actuel = 0
        for jours in jours_par_mois:
            mois_coef.append(mean_coef[jour_actuel:jour_actuel+jours].mean())
            jour_actuel += jours
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, 13), mois_coef)
        plt.title('Importance moyenne des températures par mois')
        plt.xlabel('Mois')
        plt.ylabel('Importance absolue moyenne')
        plt.xticks(range(1, 13), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'])
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.images_path / 'lda_mois_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Top 10 des jours les plus importants
        top_features = mean_coef.nlargest(10)
        
        plt.figure(figsize=(12, 6))
        top_features.plot(kind='bar')
        plt.title('Top 10 des jours les plus discriminants')
        plt.xlabel('Jour de l\'année')
        plt.ylabel('Importance absolue moyenne')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.images_path / 'lda_top_jours.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Afficher les informations détaillées dans la console
        print("\n=== Analyse détaillée des coefficients LDA ===")
        print("\nTop 5 des jours les plus importants :")
        for jour, importance in top_features.head().items():
            print(f"- Jour {jour}: {importance:.3f}")
        
        print("\nImportance moyenne par mois :")
        mois_noms = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                    'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
        for mois, importance in zip(mois_noms, mois_coef):
            print(f"- {mois}: {importance:.3f}")
        
        print("\nInterprétation :")
        print("1. Les coefficients positifs indiquent une contribution positive à la séparation")
        print("2. Les coefficients négatifs indiquent une contribution négative")
        print("3. Plus la valeur absolue est grande, plus la température de ce jour est importante")
        print("4. Les features sont organisées par jour de l'année (1-365)")

        # --- FRONTIERES KNN (PCA 2D) ---
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        knn_2d = KNeighborsClassifier(n_neighbors=5)
        knn_2d.fit(X_pca, y)
        Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array([self.departements.index(z) for z in Z]).reshape(xx.shape)
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, levels=len(self.departements), cmap='Pastel1')
        for i, dept in enumerate(self.departements):
            mask = y == dept
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Département {dept}', color=colors[i], edgecolor='k', alpha=0.7)
        plt.title('Frontières de décision KNN (projection PCA)')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.images_path / 'frontieres_knn.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualiser_performances(self, knn, lda, X, y, villes):
        """
        Visualise les performances des modèles avec des matrices de confusion normalisées et affiche un résumé utile.
        """
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        # Séparation train/test basée sur les villes
        X_train, X_test, y_train, y_test = self.preparer_train_test(X, y, villes)
        # KNN
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        cm_knn = confusion_matrix(y_test, y_pred_knn, normalize='true')
        disp_knn = ConfusionMatrixDisplay(cm_knn, display_labels=self.departements)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp_knn.plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title('Matrice de confusion normalisée - KNN')
        plt.tight_layout()
        plt.savefig(self.images_path / 'matrice_confusion_knn_norm.png', dpi=300, bbox_inches='tight')
        plt.close()
        # LDA
        lda.fit(X_train, y_train)
        y_pred_lda = lda.predict(X_test)
        cm_lda = confusion_matrix(y_test, y_pred_lda, normalize='true')
        disp_lda = ConfusionMatrixDisplay(cm_lda, display_labels=self.departements)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp_lda.plot(ax=ax, cmap='Oranges', colorbar=False)
        plt.title('Matrice de confusion normalisée - LDA')
        plt.tight_layout()
        plt.savefig(self.images_path / 'matrice_confusion_lda_norm.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Résumé console
        print("\n--- Résumé des performances ---")
        print(f"Score test KNN (80% villes train, 20% test): {knn.score(X_test, y_test):.3f}")
        print(f"Score test LDA (80% villes train, 20% test): {lda.score(X_test, y_test):.3f}")
        print("\nImages générées :")
        print(f"- Projection LDA : {self.images_path / 'projection_lda.png'}")
        print(f"- Heatmap coefficients LDA : {self.images_path / 'lda_coefficients_heatmap.png'}")
        print(f"- Frontières KNN : {self.images_path / 'frontieres_knn.png'}")
        print(f"- Matrice confusion KNN : {self.images_path / 'matrice_confusion_knn_norm.png'}")
        print(f"- Matrice confusion LDA : {self.images_path / 'matrice_confusion_lda_norm.png'}")
        print("\nPour interpréter LDA : les features avec les plus grands coefficients (en valeur absolue) sont les plus discriminants pour séparer les départements.")
        print("Pour interpréter KNN : la frontière sépare les groupes dans l'espace réduit (PCA).")
        # Appel à la visualisation avancée
        self.visualiser_projection_et_frontieres(knn, lda, X, y)

    def analyser_resultats(self):
        """Analyse complète des données"""
        print("\nChargement et préparation des données...")
        df = self.charger_donnees()
        if df is None or df.empty:
            print("Pas de données disponibles")
            return
        
        print("\nPréparation des features...")
        X, y, villes = self.preparer_features(df)
        
        print("\nEntraînement des modèles...")
        knn, lda, scaler, X_scaled, y = self.entrainer_modeles(X, y)
        
        print("\nVisualisation des performances...")
        self.visualiser_performances(knn, lda, X_scaled, y, villes)
        
        # Afficher les statistiques par département
        print("\nStatistiques par département:")
        stats = df.groupby('DEPARTEMENT')['T'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        print(stats)

    def main(self):
        """Exécute l'analyse"""
        self.analyser_resultats()

if __name__ == "__main__":
    classification = DepartmentClassification()
    classification.main()
