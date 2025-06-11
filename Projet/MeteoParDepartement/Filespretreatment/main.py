import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# Utiliser un import relatif pour le module csvCleaner
from csvCleaner import treatment as depTreatment

def create_directory_structure():
    # Obtenir le chemin absolu du répertoire de travail
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_dir = script_dir.parent.parent
    
    # Créer les dossiers principaux
    base_path = project_dir / 'CSVTrier'
    
    # Créer les dossiers pour chaque année
    for year in range(2000, 2026):
        year_path = base_path / str(year)
        year_path.mkdir(parents=True, exist_ok=True)
        
        # Créer les sous-dossiers pour chaque département
        for dept in ['13', '64', '67', '72']:
            dept_path = year_path / dept
            dept_path.mkdir(exist_ok=True)
            print(f"Created directory: {dept_path}")

def process_department_files():
    # Obtenir le chemin absolu du répertoire de travail
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_dir = script_dir.parent.parent
    base_path = project_dir / 'CSVTrier'
    
    periods = {
        '2000-2009': ['H_13_2000-2009.csv.gz', 'H_64_2000-2009.csv.gz', 
                      'H_67_2000-2009.csv.gz', 'H_72_2000-2009.csv.gz'],
        '2010-2019': ['H_13_2010-2019.csv.gz', 'H_64_2010-2019.csv.gz', 
                      'H_67_2010-2019.csv.gz', 'H_72_2010-2019.csv.gz'],
        '2020-2025': ['H_13_previous-2020-2023.csv.gz', 'H_64_previous-2020-2023.csv.gz',
                      'H_67_previous-2020-2023.csv.gz', 'H_72_previous-2020-2023.csv.gz',
                      'H_13_latest-2024-2025.csv.gz', 'H_64_latest-2024-2025.csv.gz',
                      'H_67_latest-2024-2025.csv.gz', 'H_72_latest-2024-2025.csv.gz']
    }
    
    # Traiter chaque période
    for period, files in periods.items():
        for file in files:
            # Extraire le numéro de département du nom du fichier
            dept = file.split('_')[1]
            
            # Chemin du fichier source
            source_path = script_dir.parent / dept / file
            
            if source_path.exists():
                # Lire et traiter le fichier
                df = pd.read_csv(source_path, sep=';')
                print(f"\nProcessing {file}...")
                
                cleaned_df = depTreatment.deleteEmptyColumns(df)
                processed_df = depTreatment.dateReformatage(cleaned_df)
                
                # Convertir la colonne AAAAMMJJHH en datetime
                processed_df['date'] = pd.to_datetime(processed_df['AAAAMMJJHH'], format='%Y%m%d%H')
                
                # Grouper par année et ville
                for year, year_df in processed_df.groupby(processed_df['date'].dt.year):
                    # Pour chaque ville dans cette année
                    for (station_id, station_name), city_df in year_df.groupby(['NUM_POSTE', 'NOM_USUEL']):
                        # Créer un nom de dossier sécurisé pour la ville
                        safe_city_name = "".join(c for c in station_name if c.isalnum() or c in (' ', '-', '_')).strip()
                        
                        # Créer le chemin de sortie pour cette année et département
                        output_path = base_path / str(year) / dept / f'{safe_city_name}.csv'
                        
                        # Sauvegarder le fichier
                        city_df.to_csv(output_path, sep=';', index=False)
                        print(f'Processed {file} for year {year}, city {station_name} -> {output_path}')
            else:
                print(f'File not found: {source_path}')

def main():
    print("Creating directory structure...")
    create_directory_structure()
    
    print("\nProcessing files...")
    process_department_files()
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()