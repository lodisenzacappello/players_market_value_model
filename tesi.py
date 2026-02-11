#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#FASE 0- IMPORTAZIONE DELLE LIBRERIE
#Importo le librerie necessarie per la corretta esecuzione dello script. Le metto tutte all'inizio 
#per comodità e per avere accesso a questa parte dello script ogni volta che mi serve senza dover
#andare a cercare nelle righe di codice.
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#SEZIONE WEB-DRIVER
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options

#SEZIONE GRAFICI
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
import time

#SEZIONE DATA SCRAPING 2
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import unicodedata

#SEZIONE LASSO
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

#SEZIONE SVR
from sklearn.svm import SVR 
from sklearn.inspection import permutation_importance

#SEZIONE XGboost
#conda install -c conda-forge xgboost da installare su Anaconda prompt
from xgboost import XGBRegressor
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#FASE 1- CREAZIONE DELLE LISTE PER IL CORRETTO DOWNLOAD DEI DATI DA FBREF
#Delineo le liste con all'interno i nomi delle colonne che la funzione andrà a scaricare dal sito
#e gli urls delle pagine web da cui dovranno essere scaricati i dati.
#In questo modo il programma andrà automaticamente a eseguire il download dei dati.
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
col_std = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born',
           'MP', 'Starts', 'Min', '90s', 'Goals', 'Ast', 'G+A', 'G-PK',
           'PK', 'PKatt', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG',
           'PrgC', 'PrgP', 'PrgR', 'Gls_per90', 'Ast_per90', 'G+A_per90',
           'G-PK_per90', 'G+A-PK_per90', 'xG_per90', 'xAG_per90', 'xG+xAG_per90',
           'npxG_per90', 'npxG+xAG_per90', 'Matches']

col_gk= ["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "MP", "Starts",
             "Min","90s", "Gag", "Gag_per90", "SoTA", "Saves", "Save%", "W" ,"D","L","CS",
             "CS%","PK_att_ag", "PKall", "PKsaved", "PKmiss","PK_Save%","Matches"]

col_adv_gk=["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "90s",	
            "Gag", "PKall", "FK","CK", "OG",
            "PSxG", "PSxG/SoTA", "PSxG-GA", "PSxG-GA_per90",
            "pass_40yds_Cmp", "pass_40yds_Att",	"pass_40yds_Cmp%",	
            "pass_att(no GK)", "thr_att", "pass_launched_%", "avg_len_pass",	
            "GK_Att", "%GK_launched_40yds", "GK_AvgLen",
            "cross_faced", "cross_stp", "cross_stp%", 
            "OPA", "OPA_per90", "OPA_avg_dist", "Matches"]

col_shoot=["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "90s",	
           "Goals", "Sh", "SoT", "SoT%", "Sh_per90", "SoT_per90", "G/Sh", "G/SoT", "avg_shot_dist",	
           "shot_FK", "PK_made", "PKatt",	
           "xG", "npxG", "npxG/Sh", "G-xG", "npG-npxG", "Matches"]

col_pass=["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "90s",	
          "pass_cmp", "pass_att", "pass_cmp%", "pass_tot_dist", "pass_prg_dist",
          "short_pass_cmp", "short_pass_att", "short_pass_cmp%",
          "med_pass_cmp", "med_pass_att", "med_pass_cmp%",
          "long_pass_cmp", "long_pass_att", "long_pass_cmp%",
          "ast", "xAG", "xA", "A-xAG",
          "key_pass", "pass_into_att_3rd", "PPA", "cross_into_PA",
          "prog_pass", "Matches"]

col_pass_types=["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "90s",	
                "pass_att", "live_passes", "dead_passes",
                "fk", "through_balls", "pass_switches", "cross", "throw_in", "ck",
                "corner_inwards", "corner_outwards", "corner_straight",
                "passes_cmp", "passes_offside", "passes_blocked_by_opp",
                "Matches"]

col_GCA_SCA=["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "90s",	
             "SCA", "SCA_per90", "SCA_live_pass", "SCA_dead_pass", "SCA_take_on", 
             "SCA_shot", "SCA_fouls_drawn", "SCA_def_action",	
             "GCA", "GCA_per90", "GCA_live_pass", "GCA_dead_pass", "GCA_take_on", 
             "GCA_shot", "GCA_fouls_drawn", "GCA_def_action",	
             "Matches"]

col_def=["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "90s",	
         "tackles", "tackles_won", "tackles_def_3rd", "tackle_mid_3rd", "tackle_att_3rd",	
         "dribblers_tackled", "dribblers_challenged", "%dribblers_tackled", "challenges_lost",
         "blocks", "shot_blocked", "passes_blocked",	
         "interceptions", "tackles + int", "clearance", "errors_leading_shot",	
         "Matches"]

col_poss=["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "90s",	
          "touches", "touches_def_PA", "touches_def_3rd", "touches_mid_3rd", "touches_att_3rd", "touches_att_PA", "touches_live_ball",	
          "dribbling_att", "dribbling_succ", "dribbling_succ%", "dribbling_tkld", "dribbling_Tkld%",	
          "carries", "carries_tot_dist", "prg_carries_tot_dist", "prg_carries", "carries_into_att_3rd", "carries_into_PA", 
          "miscontrols", "dispossessed",
          "passes_rec",	"prg_passes_rec",	
          "Matches"]

col_misc=["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "90s",	
          "CrdY", "CrdR", "2CrdY",	
          "fouls_com","fouls_drawn",
          "offsides", "crosses", "interceptions",
          "tackles_won", "PK_won", "PK_con", "own_goals", "ball_recoveries",
          "aerial_duels_won", "aerial_duels_lost", "aerial_duels%",	
          "Matches"]

id_tables=["stats_standard","stats_keeper","stats_keeper_adv","stats_shooting","stats_passing",
           "stats_passing_types","stats_gca","stats_defense","stats_possession","stats_misc"]

urls_2425=["https://fbref.com/en/comps/11/2024-2025/stats/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/keepers/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/keepersadv/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/shooting/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/passing/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/passing_types/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/gca/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/defense/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/possession/2024-2025-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2024-2025/misc/2024-2025-Serie-A-Stats",]

urls_2324=["https://fbref.com/en/comps/11/2023-2024/stats/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/keepers/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/keepersadv/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/shooting/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/passing/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/passing_types/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/gca/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/defense/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/possession/2023-2024-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2023-2024/misc/2023-2024-Serie-A-Stats"]

urls_2223=["https://fbref.com/en/comps/11/2022-2023/stats/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/keepers/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/keepersadv/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/shooting/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/passing/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/passing_types/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/gca/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/defense/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/possession/2022-2023-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2022-2023/misc/2022-2023-Serie-A-Stats"]

urls_2122=["https://fbref.com/en/comps/11/2021-2022/stats/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/keepers/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/keepersadv/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/shooting/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/passing/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/passing_types/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/gca/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/defense/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/possession/2021-2022-Serie-A-Stats",
          "https://fbref.com/en/comps/11/2021-2022/misc/2021-2022-Serie-A-Stats"]

urls_2021=["https://fbref.com/en/comps/11/2020-2021/stats/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/keepers/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/keepersadv/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/shooting/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/passing/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/passing_types/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/gca/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/defense/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/possession/2020-2021-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2020-2021/misc/2020-2021-Serie-A-Stats"]

urls_1920=["https://fbref.com/en/comps/11/2019-2020/stats/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/keepers/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/keepersadv/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/shooting/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/passing/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/passing_types/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/gca/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/defense/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/possession/2019-2020-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2019-2020/misc/2019-2020-Serie-A-Stats"]

urls_1819=["https://fbref.com/en/comps/11/2018-2019/stats/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/keepers/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/keepersadv/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/shooting/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/passing/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/passing_types/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/gca/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/defense/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/possession/2018-2019-Serie-A-Stats",
           "https://fbref.com/en/comps/11/2018-2019/misc/2018-2019-Serie-A-Stats"]
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#FASE 2- PARSING E DOWNLOAD DELLE TABELLE DA FBREF
#Creo il programma per scaricare autonomamente tutte le 70 tabelle dal sito FBref. 
#Le tabelle sono 10 per ognuna delle 7 stagioni (dal 2018-2019 al 2024-2025) prese in considerazione 
#per la creazione del dataset finale.
#In pratica, sto costruendo un web scraper automatizzato che va su FBref, simula un utente reale, 
#e trasforma le tabelle HTML in dati pronti per essere analizzati.

#Nel contesto del web scraping, il parsing è l'operazione di analisi di un ammasso di testo 
#grezzo (il codice HTML della pagina web) per trasformarlo in una struttura organizzata 
#su cui lavorare.

"""Selenium: libreria specializzata in browser automation
Selenium serve a controllare il browser.
Apre il browser e cerca l'urls richiesto. Lo uso per automatizzare il processo di download dei dati 
(uso sleep per aspettare il caricamento dei dati). """

"""BeautifulSoup: libreria specializzata nel parsing
Beautiful soup è il parser. Una volta che Selenium ha ottenuto il codice della pagina, scorre l'html del sito e 
cerca la tabella con table_id richiesto."""

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#dizionario che associa ogni tabella FBref alla lista di colonne
col_dict = {
    "stats_standard": col_std,
    "stats_keeper": col_gk,
    "stats_keeper_adv": col_adv_gk,
    "stats_shooting": col_shoot,
    "stats_passing": col_pass,
    "stats_passing_types": col_pass_types,
    "stats_gca": col_GCA_SCA,
    "stats_defense": col_def,
    "stats_possession": col_poss,
    "stats_misc": col_misc,
}

#Funzione per scaricare la singola tabella
def get_fbref_table(url, table_id, season, colnames):
    # Driver headless (posso cambiare path se necessario in base al browser usato)
    options = Options()
    options.headless = True
    service = Service(r"C:/Users/ludov/OneDrive/Desktop/msedgedriver.exe")   
    driver = webdriver.Edge(service=service, options=options)
    # Apri URL
    driver.get(url)
    time.sleep(5)  # tempo per il caricamento della pagina
    # Parsing HTML
    soup = BeautifulSoup(driver.page_source, "lxml")
    driver.quit()
    # Trovo la tabella
    table = soup.find("table", {"id": table_id})
    if table is None:
        print(f" Tabella {table_id} non trovata in {url}")
        return None
    
    df = pd.read_html(str(table))[0]

    # Rinomino le colonne se il numero corrisponde
    if len(df.columns) == len(colnames):
        df.columns = colnames
    else:
        print(f" Numero colonne diverso per {table_id}: trovato {len(df.columns)}, atteso {len(colnames)}")
    # Aggiungo stagione
    df["Season"] = season
    return df

#Funzione per scaricare tutte le tabelle  di una stagione (10 dataframe per stagione)
#Parametri: lista urls, stagione in versione stringa (es:"2024-2025"), lista con id tabelle, 
#dizionario con id tabella associata a lista nomi colonne
#Output: dizionario con key="id tabella" e dataframe con dati della tabella 

def get_all_tables_for_season(url_list, season, id_tables=id_tables, col_dict=col_dict):
    all_dfs = {}
    for i, url in enumerate(url_list):
        table_id = id_tables[i]
        colnames = col_dict[table_id]

        print(f"Scarico {table_id} ({season})...")
        df = get_fbref_table(url, table_id, season, colnames)
        if df is not None:
            all_dfs[table_id] = df
    
    return all_dfs

df_2425=get_all_tables_for_season(urls_2425, "2024-2025", id_tables=id_tables, col_dict=col_dict)   #dizionari con all'interno 10 df con tabelle per ogni anno
df_2324=get_all_tables_for_season(urls_2324, "2023-2024", id_tables=id_tables, col_dict=col_dict)
df_2223=get_all_tables_for_season(urls_2223, "2022-2023", id_tables=id_tables, col_dict=col_dict)
df_2122=get_all_tables_for_season(urls_2122, "2021-2022", id_tables=id_tables, col_dict=col_dict)
df_2021=get_all_tables_for_season(urls_2021, "2020-2021", id_tables=id_tables, col_dict=col_dict)
df_1920=get_all_tables_for_season(urls_1920, "2019-2020", id_tables=id_tables, col_dict=col_dict)
df_1819=get_all_tables_for_season(urls_1819, "2018-2019", id_tables=id_tables, col_dict=col_dict)
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#FASE 3 - DATA CLEANING DEI DATASET DI FBREF
#Elimino le variabili ripetute tra le diverse tabelle allo scopo di poter
#eseguire la concatenazione per creare i 7 dataframe completi per le diverse stagioni.
#Infatti in tabelle diverse può capitare di trovare le stesse variabili ripetute (ES: Gol in stats_standard e stats_GCA).
#Se ho due dataset con 2 colonne uguali non riesco a unire orizzontalmente i datset
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# Dizionario con le liste (già definito in precedenza)
col_dict = {
    "stats_standard": col_std,
    "stats_keeper": col_gk,
    "stats_keeper_adv": col_adv_gk,
    "stats_shooting": col_shoot,
    "stats_passing": col_pass,
    "stats_passing_types": col_pass_types,
    "stats_gca": col_GCA_SCA,
    "stats_defense": col_def,
    "stats_possession": col_poss,
    "stats_misc": col_misc,
}

# Creo un dizionario che mi dice in quali tabelle compare ogni variabile 
col_to_tables = {}
for table_name, cols in col_dict.items():
    for col in cols:
        if col not in col_to_tables:
            col_to_tables[col] = []
        col_to_tables[col].append(table_name)

# Ora creo una lista con solo le variabili che compaiono in più di una tabella
duplicates = {col: tables for col, tables in col_to_tables.items() if len(tables) > 1}

# Converto in DataFrame per visualizzazione più comoda
df_duplicates = pd.DataFrame([
    {"Column": col, "Tables": ", ".join(tables)} 
    for col, tables in duplicates.items()
])

# Mostro il risultato ordinato in ordine alfabetico
df_duplicates = df_duplicates.sort_values("Column").reset_index(drop=True)

print(df_duplicates)

# Colonne da eliminare
cols_to_drop = ["Pos", "Squad", "Age", "90s"]                           #elimino da tutti i df tranne stats_standard
cols_to_drop_all = ["Rk","Starts", "MP", "Matches"]                     #elimino da tutti i df
cols_to_drop_misc = ["CrdR", "CrdY","interceptions","tackles_won"]      #elimino solo da stats_misc
cols_to_drop_keeper_adv = ["PKall", "Gag"]                              #elimino solo da stats_keeper_adv
cols_to_drop_keeper = ["Min"]                                           #elimino solo da stats_keeper
cols_to_drop_standard = ["Goals", "PKatt", "npxG", "xG", "xAG"]         #elimino solo da stats_standard
cols_to_drop_passing_types=["pass_att"]                                 #elimino solo da stats_passing_types

def clean_all(d):
    new_d = {}
    for key, df in d.items():
        # Drop generale
        df = df.drop(columns=[c for c in cols_to_drop_all if c in df.columns], errors="ignore")
        
        # Drop per tutte le tabelle tranne stats_standard
        if key != "stats_standard":
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        
        # Drop extra per stats_misc
        if key == "stats_misc":
            df = df.drop(columns=[c for c in cols_to_drop_misc if c in df.columns], errors="ignore")
        
        # Drop extra per stats_keeper_adv
        if key == "stats_keeper_adv":
            df = df.drop(columns=[c for c in cols_to_drop_keeper_adv if c in df.columns], errors="ignore")
        
        #Drop extra per stas_keeper
        if key == "stats_keeper":
            df = df.drop(columns=[c for c in cols_to_drop_keeper if c in df.columns], errors="ignore")
        
        #Drop extra per stats_standard
        if key == "stats_standard":
            df = df.drop(columns=[c for c in cols_to_drop_standard if c in df.columns], errors="ignore")
        
        #Drop extra per stats_passing_types
        if key == "stats_passing_types":
            df = df.drop(columns=[c for c in cols_to_drop_passing_types if c in df.columns], errors="ignore")
            
        new_d[key] = df
    return new_d

# Applicazione alle stagioni
df_cleaned_1819 = clean_all(df_1819)
df_cleaned_1920 = clean_all(df_1920)
df_cleaned_2021 = clean_all(df_2021)
df_cleaned_2122 = clean_all(df_2122)
df_cleaned_2223 = clean_all(df_2223)
df_cleaned_2324 = clean_all(df_2324)
df_cleaned_2425 = clean_all(df_2425)

#Elimino gli header ripetuti necessari per la fruizione sul sito di FBRef
#Su FBRef ogni 70-80 entries vengono ripetuti i nomi delle colonne.
#Per eliminare queste righe elimino le linee in cui nella variabile "Player" trovo la 
#scritta "Player" e non il nome del calciatore.
def clean_headers(df, key_col="Player"):
    return df[df[key_col] != key_col]

# Lista dei dizionari stagionali
season_dicts = [
    df_cleaned_1819, df_cleaned_1920, df_cleaned_2021,
    df_cleaned_2122, df_cleaned_2223, df_cleaned_2324, df_cleaned_2425]

# Applico clean_headers a tutti i dataframe interni
for season_dict in season_dicts:
    for key, df in season_dict.items():
        season_dict[key] = clean_headers(df, key_col="Player")

#Aggrego le entries di uno stesso giocatore nel caso in cui in una stessa 
#stagione giochi con due squadre diverse. Per farlo mi baso sull'uguaglianza nelle due entries 
#del multi-indice quindi stesso nome, stessa nazionalità, stesso anno di nascita, stessa stagione 
#implica che si tratta dello stesso giocatore.

def aggregate_duplicates(df, keys=["Player", "Nation", "Born", "Season"]):
    import numpy as np
    import pandas as pd

    # 🔹 1. Definiamo le colonne da escludere
    exclude_cols = ["Player", "Nation", "Age", "Pos", "Squad", "Matches", "Season"] + keys

    # 🔹 2. Convertiamo in numeriche solo le altre
    cols_to_convert = [c for c in df.columns if c not in exclude_cols]
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")

    # 🔹 3. Controlliamo che tutte le chiavi esistano nel DataFrame
    missing_keys = [k for k in keys if k not in df.columns]
    if missing_keys:
        print(f" Attenzione: le seguenti chiavi mancano nel DataFrame: {missing_keys}")
        return df

    # 🔹 4. Identifichiamo colonne numeriche e non numeriche
    num_cols = df.select_dtypes(include=[np.number]).columns
    non_num_cols = [c for c in df.columns if c not in num_cols and c not in keys]

    # 🔹 5. Costruiamo il dizionario di aggregazione
    agg_dict = {}
    for col in num_cols:
        if "%" in col or "per90" in col:
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "sum"

    for col in non_num_cols:
        agg_dict[col] = "first"

    # 🔹 6. Gestione speciale per la colonna "Squad"
    if "Squad" in df.columns:
        def merge_squads(x):
            squads = x.dropna().unique()
            if len(squads) == 1:
                return squads[0]
            else:
                return f"{len(squads)}squads({'+'.join(squads)})"

        agg_dict["Squad"] = merge_squads

    # 🔹 7. Aggregazione effettiva
    df_agg = df.groupby(keys, as_index=False).agg(agg_dict)

    # 🔹 8. Log informativo
    if len(df_agg) < len(df):
        print(f" Aggregazione effettuata: {len(df) - len(df_agg)} righe aggregate su {len(df)} totali.")
    else:
        print(" Nessuna aggregazione necessaria (nessun duplicato trovato).")

    return df_agg

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#FASE 4 - CREAZIONE DI UN UNICO DATASET CON LE STATISTICHE DI CAMPO PER OGNI STAGIONE (DF_FINAL)
#Unisce i 10 dataframe di una stagione in un unico dataframe finale, utilizzando 
#un multi-indice ["Player", "Nation", "Born", "Season"].
#Alla fine ottengo 10 dataframe (1 per stagione) chiamati "df_final_XXXX" con 187 colonne di variabili di interesse.

#df_final1819 num= 413 giocatori
#df_final1920 num= 438 giocatori
#df_final2021 num= 436 giocatori
#df_final2122 num= 445 giocatori
#df_final2223 num= 441 giocatori
#df_final2324 num= 446 giocatori
#df_final2425 num= 451 giocatori
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# Stagione 2018-2019
# -----------------------------------------------------------------------------------------------------
#Applico l'aggregazione dei duplicati a tutti i dataframe della stagione
for key in df_cleaned_1819:
    df_cleaned_1819[key] = aggregate_duplicates(df_cleaned_1819[key])

#Imposto l'indice multi-colonna per tutti i dataframe
for key in df_cleaned_1819:
    df_cleaned_1819[key] = df_cleaned_1819[key].set_index(["Player", "Nation", "Born", "Season"])

#Concateno i 10 dataframe in uno solo
df_final_1819 = pd.concat(df_cleaned_1819.values(), axis=1).reset_index()
# -----------------------------------------------------------------------------------------------------
# Stagione 2019-2020
# -----------------------------------------------------------------------------------------------------
for key in df_cleaned_1920:
    df_cleaned_1920[key] = aggregate_duplicates(df_cleaned_1920[key])

for key in df_cleaned_1920:
    df_cleaned_1920[key] = df_cleaned_1920[key].set_index(["Player", "Nation", "Born", "Season"])

df_final_1920 = pd.concat(df_cleaned_1920.values(), axis=1).reset_index()
# -----------------------------------------------------------------------------------------------------
# Stagione 2020-2021
# -----------------------------------------------------------------------------------------------------
for key in df_cleaned_2021:
    df_cleaned_2021[key] = aggregate_duplicates(df_cleaned_2021[key])

for key in df_cleaned_2021:
    df_cleaned_2021[key] = df_cleaned_2021[key].set_index(["Player", "Nation", "Born", "Season"])

df_final_2021 = pd.concat(df_cleaned_2021.values(), axis=1).reset_index()
# -----------------------------------------------------------------------------------------------------
# Stagione 2021-2022
# -----------------------------------------------------------------------------------------------------
for key in df_cleaned_2122:
    df_cleaned_2122[key] = aggregate_duplicates(df_cleaned_2122[key])

for key in df_cleaned_2122:
    df_cleaned_2122[key] = df_cleaned_2122[key].set_index(["Player", "Nation", "Born", "Season"])

df_final_2122 = pd.concat(df_cleaned_2122.values(), axis=1).reset_index()
# ------------------------------------------------------------------------------------------------------
# Stagione 2022-2023
# ------------------------------------------------------------------------------------------------------
for key in df_cleaned_2223:
    df_cleaned_2223[key] = aggregate_duplicates(df_cleaned_2223[key])

for key in df_cleaned_2223:
    df_cleaned_2223[key] = df_cleaned_2223[key].set_index(["Player", "Nation", "Born", "Season"])

df_final_2223 = pd.concat(df_cleaned_2223.values(), axis=1).reset_index()
# ------------------------------------------------------------------------------------------------------
# Stagione 2023-2024
# ------------------------------------------------------------------------------------------------------
for key in df_cleaned_2324:
    df_cleaned_2324[key] = aggregate_duplicates(df_cleaned_2324[key])

for key in df_cleaned_2324:
    df_cleaned_2324[key] = df_cleaned_2324[key].set_index(["Player", "Nation", "Born", "Season"])

df_final_2324 = pd.concat(df_cleaned_2324.values(), axis=1).reset_index()
# ------------------------------------------------------------------------------------------------------
# Stagione 2024-2025
# ------------------------------------------------------------------------------------------------------
for key in df_cleaned_2425:
    df_cleaned_2425[key] = aggregate_duplicates(df_cleaned_2425[key])

for key in df_cleaned_2425:
    df_cleaned_2425[key] = df_cleaned_2425[key].set_index(["Player", "Nation", "Born", "Season"])

df_final_2425 = pd.concat(df_cleaned_2425.values(), axis=1).reset_index()

#Cambio del format delle nazioni in modo da farle combaciare con il format derivante dalle tabelle 
#di TransferMarkt.
#Converto i codici delle nazionali di df_final nel formato esteso usato da Transfermarkt.
#Creo una nuova colonna 'Nation2' con la versione estesa.
#Mostro un log con il numero di conversioni riuscite e non riuscite.

nation_map = {
    "al ALB": "Albania",
    "am ARM": "Armenia",
    "ao ANG": "Angola",
    "ar ARG": "Argentina",
    "at AUT": "Austria",
    "au AUS": "Australia",
    "ba BIH": "Bosnia-Erzegovina",
    "be BEL": "Belgio",
    "bf BFA": "Burkina Faso",
    "bg BUL": "Bulgaria",
    "bo BOL": "Bolivia",
    "br BRA": "Brasile",
    "by BLR": "Bielorussia",
    "ca CAN": "Canada",
    "cd COD": "RD del Congo",
    "cg CGO": "Congo",
    "ch SUI": "Svizzera",
    "ci CIV": "Costa d'Avorio",
    "cl CHI": "Cile",
    "cm CMR": "Camerun",
    "co COL": "Colombia",
    "cr CRC": "Costarica",
    "cv CPV": "Capo Verde",
    "cy CYP": "Cipro",
    "cz CZE": "Repubblica Ceca",
    "de GER": "Germania",
    "dk DEN": "Danimarca",
    "dz ALG": "Algeria",
    "ec ECU": "Ecuador",
    "ee EST": "Estonia",
    "eg EGY": "Egitto",
    "eng ENG": "Inghilterra",
    "es ESP": "Spagna",
    "fi FIN": "Finlandia",
    "fr FRA": "Francia",
    "ga GAB": "Gabon",
    "ge GEO": "Georgia",
    "gh GHA": "Ghana",
    "gp GLP": "Guadalupa",
    "wls WAL": "Galles",
    "gm GAM": "Gambia",
    "gn GUI": "Guinea",
    "gq EQG": "Guinea Equatoriale",
    "gr GRE": "Grecia",
    "gt GUA": "Guatemala",
    "gw GNB": "Guinea-Bissau",
    "hr CRO": "Croazia",
    "hu HUN": "Ungheria",
    "id IDN": "Indonesia",
    "ie IRL": "Irlanda",
    "il ISR": "Israele",
    "in IND": "India",
    "iq IRQ": "Iraq",
    "ir IRN": "Iran",
    "is ISL": "Islanda",
    "it ITA": "Italia",
    "jm JAM": "Giamaica",
    "jp JPN": "Giappone",
    "kr KOR": "Corea del Sud",
    "ly LBY": "Libia",
    "lt LTU": "Lituania",
    "lu LUX": "Lussemburgo",
    "lv LVA": "Lettonia",
    "ma MAR": "Marocco",
    "md MDA": "Moldavia",
    "me MNE": "Montenegro",
    "mg MAD": "Madagascar",
    "mk MKD": "Macedonia del Nord",
    "ml MLI": "Mali",
    "mq MTQ": "Martinica",
    "mx MEX": "Messico",
    "ng NGA": "Nigeria",
    "nl NED": "Olanda",
    "no NOR": "Norvegia",
    "nz NZL": "Nuova Zelanda",
    "pe PER": "Perù",
    "pl POL": "Polonia",
    "pt POR": "Portogallo",
    "py PAR": "Paraguay",
    "ro ROU": "Romania",
    "rs SRB": "Serbia",
    "ru RUS": "Russia",
    "sa KSA": "Arabia Saudita",
    "sc SEY": "Seychelles",
    "sct SCO": "Scozia",
    "se SWE": "Svezia",
    "si SVN": "Slovenia",
    "sk SVK": "Slovacchia",
    "sl SLE": "Sierra Leone",
    "sn SEN": "Senegal",
    "sr SUR": "Suriname",
    "tn TUN": "Tunisia",
    "tr TUR": "Turchia",
    "ua UKR": "Ucraina",
    "ug UGA": "Uganda",
    "uy URU": "Uruguay",
    "us USA": "Stati Uniti",
    "uz UZB": "Uzbekistan",
    "ve VEN": "Venezuela",
    "xk KVX": "Kosovo",
    "za RSA": "Sudafrica",
    "zm ZAM": "Zambia",
    "zw ZIM": "Zimbabwe"
}

def format_nation(df, nation_map):
    df = df.copy()

    # Crea la nuova colonna "Nation2" mappando la "Nation"
    df["Nation2"] = df["Nation"].map(nation_map)

    # Sposta "Nation2" subito dopo "Nation"
    cols = list(df.columns)
    if "Nation" in cols and "Nation2" in cols:
        idx = cols.index("Nation")
        cols.insert(idx + 1, cols.pop(cols.index("Nation2")))
        df = df[cols]

    # Log informativo
    converted = df["Nation2"].notna().sum()
    not_converted = df["Nation2"].isna().sum()
    print(f"[LOG] Conversioni riuscite: {converted}")
    print(f"[LOG] Conversioni non trovate: {not_converted}")

    # Riempio i valori non convertiti con la versione originale
    df["Nation2"] = df["Nation2"].fillna(df["Nation"])

    return df

df_final_2425 = format_nation(df_final_2425, nation_map)
df_final_2324 = format_nation(df_final_2324, nation_map)
df_final_2223 = format_nation(df_final_2223, nation_map)
df_final_2122 = format_nation(df_final_2122, nation_map)
df_final_2021 = format_nation(df_final_2021, nation_map)
df_final_1920 = format_nation(df_final_1920, nation_map)
df_final_1819 = format_nation(df_final_1819, nation_map)

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 5 - SCARICARE I VALORI DI MERCATO DA TRANSFERMARKT
#Compilo le liste con gli urls dei link di Transfermarkt per avere i vdm dei calciatori.
#Transfermarkt rende complesso scaricare i dati dal sito quindi ho dovuto simulare il comportamento
#di un umano per ovviare al riconoscimento anti-bot del sito.
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

market_values_2425=["https://www.transfermarkt.it/inter/kader/verein/46/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/juventus-fc/kader/verein/506/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/ac-milan/kader/verein/5/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/atalanta/kader/verein/800/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/ssc-napoli/kader/verein/6195/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/as-roma/kader/verein/12/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/acf-fiorentina/kader/verein/430/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/bologna-fc/kader/verein/1025/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/ss-lazio/kader/verein/398/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/torino-fc/kader/verein/416/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/udinese-calcio/kader/verein/410/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/genoa-cfc/kader/verein/252/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/us-lecce/kader/verein/1005/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/hellas-verona/kader/verein/276/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/empoli-fc/kader/verein/749/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/ac-monza/kader/verein/2919/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/cagliari-calcio/kader/verein/1390/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/como-1907/kader/verein/1047/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/parma-calcio/kader/verein/130/saison_id/2024/plus/1",
                    "https://www.transfermarkt.it/venezia-fc/kader/verein/607/saison_id/2024/plus/1"]

market_values_2324=["https://www.transfermarkt.it/inter/kader/verein/46/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/juventus-fc/kader/verein/506/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/ac-milan/kader/verein/5/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/atalanta/kader/verein/800/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/ssc-napoli/kader/verein/6195/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/as-roma/kader/verein/12/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/acf-fiorentina/kader/verein/430/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/bologna-fc/kader/verein/1025/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/ss-lazio/kader/verein/398/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/torino-fc/kader/verein/416/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/udinese-calcio/kader/verein/410/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/genoa-cfc/kader/verein/252/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/us-lecce/kader/verein/1005/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/hellas-verona/kader/verein/276/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/empoli-fc/kader/verein/749/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/ac-monza/kader/verein/2919/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/cagliari-calcio/kader/verein/1390/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/frosinone-calcio/kader/verein/8970/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/us-sassuolo/kader/verein/6574/saison_id/2023/plus/1",
                    "https://www.transfermarkt.it/us-salernitana/kader/verein/380/saison_id/2023/plus/1"]

market_values_2223=["https://www.transfermarkt.it/inter/kader/verein/46/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/juventus-fc/kader/verein/506/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/ac-milan/kader/verein/5/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/atalanta/kader/verein/800/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/ssc-napoli/kader/verein/6195/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/as-roma/kader/verein/12/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/acf-fiorentina/kader/verein/430/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/bologna-fc/kader/verein/1025/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/ss-lazio/kader/verein/398/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/torino-fc/kader/verein/416/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/udinese-calcio/kader/verein/410/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/us-lecce/kader/verein/1005/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/us-sassuolo/kader/verein/6574/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/hellas-verona/kader/verein/276/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/empoli-fc/kader/verein/749/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/ac-monza/kader/verein/2919/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/us-salernitana/kader/verein/380/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/spezia-calcio/kader/verein/3522/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/us-cremonese/kader/verein/2239/saison_id/2022/plus/1",
                    "https://www.transfermarkt.it/uc-sampdoria/kader/verein/1038/saison_id/2022/plus/1"]

market_values_2122=["https://www.transfermarkt.it/inter/kader/verein/46/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/juventus-fc/kader/verein/506/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/ac-milan/kader/verein/5/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/atalanta/kader/verein/800/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/ssc-napoli/kader/verein/6195/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/as-roma/kader/verein/12/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/acf-fiorentina/kader/verein/430/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/bologna-fc/kader/verein/1025/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/ss-lazio/kader/verein/398/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/torino-fc/kader/verein/416/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/udinese-calcio/kader/verein/410/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/genoa-cfc/kader/verein/252/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/venezia-fc/kader/verein/607/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/hellas-verona/kader/verein/276/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/empoli-fc/kader/verein/749/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/us-sassuolo/kader/verein/6574/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/cagliari-calcio/kader/verein/1390/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/spezia-calcio/kader/verein/3522/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/uc-sampdoria/kader/verein/1038/saison_id/2021/plus/1",
                    "https://www.transfermarkt.it/us-salernitana/kader/verein/380/saison_id/2021/plus/1"]

market_values_2021=["https://www.transfermarkt.it/inter/kader/verein/46/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/juventus-fc/kader/verein/506/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/ac-milan/kader/verein/5/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/atalanta/kader/verein/800/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/ssc-napoli/kader/verein/6195/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/as-roma/kader/verein/12/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/acf-fiorentina/kader/verein/430/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/bologna-fc/kader/verein/1025/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/ss-lazio/kader/verein/398/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/torino-fc/kader/verein/416/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/udinese-calcio/kader/verein/410/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/genoa-cfc/kader/verein/252/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/parma-calcio/kader/verein/130/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/hellas-verona/kader/verein/276/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/fc-crotone/kader/verein/4083/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/benevento-calcio/kader/verein/4171/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/cagliari-calcio/kader/verein/1390/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/spezia-calcio/kader/verein/3522/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/us-sassuolo/kader/verein/6574/saison_id/2020/plus/1",
                    "https://www.transfermarkt.it/uc-sampdoria/kader/verein/1038/saison_id/2020/plus/1"]

market_values_1920=["https://www.transfermarkt.it/inter/kader/verein/46/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/juventus-fc/kader/verein/506/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/ac-milan/kader/verein/5/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/atalanta/kader/verein/800/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/ssc-napoli/kader/verein/6195/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/as-roma/kader/verein/12/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/acf-fiorentina/kader/verein/430/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/bologna-fc/kader/verein/1025/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/ss-lazio/kader/verein/398/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/torino-fc/kader/verein/416/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/udinese-calcio/kader/verein/410/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/genoa-cfc/kader/verein/252/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/us-lecce/kader/verein/1005/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/hellas-verona/kader/verein/276/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/spal/kader/verein/2722/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/brescia-calcio/kader/verein/19/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/cagliari-calcio/kader/verein/1390/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/parma-calcio/kader/verein/130/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/us-sassuolo/kader/verein/6574/saison_id/2019/plus/1",
                    "https://www.transfermarkt.it/uc-sampdoria/kader/verein/1038/saison_id/2019/plus/1"]

market_values_1819=["https://www.transfermarkt.it/inter/kader/verein/46/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/juventus-fc/kader/verein/506/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/ac-milan/kader/verein/5/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/atalanta/kader/verein/800/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/ssc-napoli/kader/verein/6195/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/as-roma/kader/verein/12/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/acf-fiorentina/kader/verein/430/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/bologna-fc/kader/verein/1025/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/ss-lazio/kader/verein/398/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/torino-fc/kader/verein/416/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/udinese-calcio/kader/verein/410/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/genoa-cfc/kader/verein/252/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/uc-sampdoria/kader/verein/1038/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/us-sassuolo/kader/verein/6574/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/parma-calcio/kader/verein/130/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/empoli-fc/kader/verein/749/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/cagliari-calcio/kader/verein/1390/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/ac-chievo-verona/kader/verein/862/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/frosinone-calcio/kader/verein/8970/saison_id/2018/plus/1",
                    "https://www.transfermarkt.it/spal/kader/verein/2722/saison_id/2018/plus/1"]

driver_path = r"C:/Users/ludov/OneDrive/Desktop/msedgedriver.exe"


#FUNZIONE CHE ESEGUE LO SCRAPING COMPLETO DEGLI URLS DI TRANSFERMARKT
def scrape_transfermarkt_squads(urls, season, driver_path):
    # Configurazione Edge (browser visibile)
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0"
    )

    service = Service(driver_path)
    driver = webdriver.Edge(service=service, options=options)

    all_players = []

    for url in urls:
        driver.get(url)

        #Simulo comportamento umano inserendo uno sleep casuale
        time.sleep(random.uniform(3, 7))

        #Eseguo uno scroll graduale per caricare tutta la pagina e fingere di essere un umano
        for step in range(1, 6):
            driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight*{step}/5);")
            time.sleep(1.5)

        try:
            # Attesa esplicita fino al caricamento della tabella
            WebDriverWait(driver, 70).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table.items > tbody > tr"))
            )
            soup = BeautifulSoup(driver.page_source, "html.parser")
            rows = soup.select("table.items > tbody > tr")
            print(f"[DEBUG] {url} -> trovate {len(rows)} righe nella tabella")

        except:
            print(f"[DEBUG] Timeout nel caricamento della tabella per {url}")
            continue

        # Eseguo parsing dei dati
        #Sono costretto a usare questo codice "strano" dato che la struttura di transfermarkt è particolare
        #e il sito è in tedesco
        
        #Trovo il nome dal titolo dell'immagine del calciatore
        names = [img.get("title") for img in soup.find_all("img", {"class": "bilderrahmen-fixed"})]

        stats = soup.find_all("td", {"class": "zentriert"})
        numbers = [
            td.find("div", class_="rn_nummer").text.strip() if td.find("div", class_="rn_nummer") else None
            for td in stats[0::8]
        ]
        ages = [td.text.strip() for td in stats[1::8]]
        dob = [a.split(" (")[0] for a in ages]
        age = [a.split(" (")[1].replace(")", "") if "(" in a else None for a in ages]

        countries = [
            td.find("img").get("title") if td.find("img") else None
            for td in stats[2::8]
        ]
        current_clubs = [
            td.find("a").get("title") if td.find("a") else None
            for td in stats[3::8]
        ]
        heights = [td.text.strip() for td in stats[4::8]]
        foots = [td.text.strip() for td in stats[5::8]]
        joined_date = [td.text.strip() for td in stats[6::8]]

        signing_info = [td for td in stats[7::8]]
        signing_fee = [
            td.find("a").get("title").split(": Ablöse ")[1] if td.find("a") else None
            for td in signing_info
        ]
        signed_from = [
            td.find("a").get("title").split(": Ablöse ")[0] if td.find("a") else None
            for td in signing_info
        ]

        values = [
            td.find("a").text.strip() if td.find("a") else "€0"
            for td in soup.find_all("td", {"class": "rechts hauptlink"})
        ]

        positions = [
            element.find_all("tr")[1].find("td").text.strip()
            if len(element.find_all("tr")) > 1 else None
            for element in soup.find_all("td", {"class": "posrela"})
        ]

        # Creo DataFrame ---
        data = {
            "player": names,
            "number": numbers,
            "date of birth": dob,
            "age": age,
            "nation": countries,
            "current_club": current_clubs,
            "height": heights,
            "foot": foots,
            "joined_date": joined_date,
            "signing_fee": signing_fee,
            "signed_from": signed_from,
            "value": values,
            "position": positions,
            "season": [season] * len(names),
        }

        df = pd.DataFrame(data)

        #Aggiungo colonna con l'anno di nascita ("born")
        df["born"] = df["date of birth"].str.extract(r"(\d{4})")

        all_players.append(df)

        #Sleep casuale tra un URL e l’altro
        time.sleep(random.uniform(4, 8))

    driver.quit()
    return pd.concat(all_players, ignore_index=True)

vdm_1819 = scrape_transfermarkt_squads(market_values_1819, "2018-2019", driver_path)
vdm_1920 = scrape_transfermarkt_squads(market_values_1920, "2019-2020", driver_path)
vdm_2021 = scrape_transfermarkt_squads(market_values_2021, "2020-2021", driver_path)
vdm_2122 = scrape_transfermarkt_squads(market_values_2122, "2021-2022", driver_path)
vdm_2223 = scrape_transfermarkt_squads(market_values_2223, "2022-2023", driver_path)
vdm_2324 = scrape_transfermarkt_squads(market_values_2324, "2023-2024", driver_path)
vdm_2425 = scrape_transfermarkt_squads(market_values_2425, "2024-2025", driver_path)

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 6 - PULIZIA DEI DATASET VDM PRIMA DEL MERGE DEI DATAFRAME
#Cambio il format dei valori di mercato passando dal format testuale al format numerico
#Cambio il format dell'altezza passando dal format testuale al format numerico
#Pulisco i dataframe vdm rimuovendo:
#    - i giocatori con altezza mancante (NaN)
#    - i giocatori con piede preferito vuoto
#    - le colonne non necessarie ovvero 'number' (numero di maglia), 'current_club', 'joined_date', 'signing_fee', 'signed_from'
#    - elimino i doppioni se coincidono 'player','date of birth' e 'height'   

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#Cambio il format dei valori di mercato trasformandoli in numeri assoluti
def change_format_vdm(df: pd.DataFrame) -> pd.DataFrame:
    def convert_value(v):
        if pd.isna(v):
            return None
        
        # Se è già numerico, restituiscilo come int
        if isinstance(v, (int, float)):
            return int(v)
        
        # Se è stringa, pulisci e converti
        v = str(v).strip().lower().replace("€", "").replace(",", ".")
        
        if "mln" in v:
            # esempio: "15.00 mln" -> 15.00 * 1_000_000
            num = float(v.replace("mln", "").strip())
            return int(num * 1_000_000)
        
        elif "mila" in v:
            # esempio: "200 mila" -> 200 * 1_000
            num = float(v.replace("mila", "").strip())
            return int(num * 1_000)
        
        else:
            # fallback: numero nudo tipo "0"
            try:
                return int(float(v))
            except:
                return None

    df = df.copy()
    df["value"] = df["value"].apply(convert_value)
    return df

vdm_1819 = change_format_vdm(vdm_1819)
vdm_1920 = change_format_vdm(vdm_1920)
vdm_2021 = change_format_vdm(vdm_2021)
vdm_2122 = change_format_vdm(vdm_2122)
vdm_2223 = change_format_vdm(vdm_2223)
vdm_2324 = change_format_vdm(vdm_2324)
vdm_2425 = change_format_vdm(vdm_2425)

#Converto la colonna 'height' nei dataset vdm da formato stringa (es. '1,85 m') a float (es. 1.85),
#gestendo anche valori anomali come '-' (assenza del dato)
def change_format_height(df):
    df = df.copy()

    df["height"] = (
        df["height"]
        .astype(str)                               # assicura che tutti i valori siano stringhe
        .str.replace("m", "", regex=False)         # rimuove la 'm'
        .str.replace(",", ".", regex=False)        # converte la virgola in punto
        .str.strip()                               # rimuove spazi bianchi
        .replace({"": None, "-": None})            # gestisce valori anomali
    )

    # Converte in float ignorando gli errori
    df["height"] = pd.to_numeric(df["height"], errors="coerce")

    return df

vdm_1819 = change_format_height(vdm_1819)
vdm_1920 = change_format_height(vdm_1920)
vdm_2021 = change_format_height(vdm_2021)
vdm_2122 = change_format_height(vdm_2122)
vdm_2223 = change_format_height(vdm_2223)
vdm_2324 = change_format_height(vdm_2324)
vdm_2425 = change_format_height(vdm_2425)

#Pulisco i dataframe vdm rimuovendo:
#    - i giocatori con altezza mancante (NaN)
#    - i giocatori con piede preferito vuoto
#    - le colonne non necessarie ovvero 'number' (numero di maglia), 'current_club', 'joined_date', 'signing_fee', 'signed_from'
#    - elimino i doppioni se coincidono 'player','date of birth' e 'height'    
def clean_vdm(df):

    df = df.copy()

    #Elimino le colonne non necessarie
    drop_cols = ["number", "current_club", "joined_date", "signing_fee", "signed_from"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    #Normalizzo 'foot' (rimuovo spazi e converto NaN in stringa vuota)
    df["foot"] = df["foot"].astype(str).str.strip()

    #Mantengo solo righe con altezza valida e piede non vuoto
    df = df[df["height"].notna() & (df["foot"] != "")]

    #Elimino i giocatori duplicati (stesso nome, data di nascita e altezza)
    if all(col in df.columns for col in ["player", "date of birth", "height"]):
        before = len(df)
        df = df.drop_duplicates(subset=["player", "date of birth", "height"], keep="first").reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            print(f"Rimossi {removed} duplicati basati su player + date of birth + height.")
    else:
        print("Colonne necessarie per il controllo duplicati non trovate.")

    return df.reset_index(drop=True)

vdm_1819 = clean_vdm(vdm_1819)
vdm_1920 = clean_vdm(vdm_1920)
vdm_2021 = clean_vdm(vdm_2021)
vdm_2122 = clean_vdm(vdm_2122)
vdm_2223 = clean_vdm(vdm_2223)
vdm_2324 = clean_vdm(vdm_2324)
vdm_2425 = clean_vdm(vdm_2425)

#Estraggo i nomi delle colonne che contengono '%' oppure 'per90'
#da un DataFrame df_final (basta passarne uno, tutti hanno le stesse colonne).
#Questo mi serve per avere i nomi delle colonne da dover calcolare nuovamente dopo aver
#aggregato le entries duplicate dei singoli giocatori.

def cols_per90(df):
    cols_to_calc = [col for col in df.columns if ('%' in col) or ('per90' in col) or ('/' in col) or ("+" in col) or ("-" in col)]

    # Log utile
    print(f"Trovate {len(cols_to_calc)} colonne con '%','per90','/','+','-'.")

    return cols_to_calc

cols_to_calc=cols_per90(df_final_1819)

#Creo una nuova colonna detta "pos" con la nuova suddivisione dei ruoli in GK,DF,WB,MF,FW
#e riordino le colonne come desiderate.
#La suddivisione dei ruoli è basata sui ruoli di Transfermarkt suddivisi in 5 gruppi:
    #1)GK
    #2)DF
    #3)WB
    #4)MF
    #5)FW
    
def format_positions(vdm):
    pos_map = {
        "Portiere": "GK",
        "Difensore centrale": "DF",
        "Esterno di destra": "WB",
        "Esterno di sinistra": "WB",
        "Terzino destro": "WB",
        "Terzino sinistro": "WB",
        "Mediano": "MF",
        "Centrocampista": "MF",
        "Trequartista": "MF",
        "Ala destra": "FW",
        "Ala sinistra": "FW",
        "Punta centrale": "FW",
        "Seconda punta": "FW"
    }

    vdm["pos"] = vdm["position"].map(pos_map).fillna("UNK")  #in caso di valori non previsti

    desired_order = [
        "player", "pos", "position", "age", "born",
        "date of birth", "foot", "height", "nation", "value", "season"
    ]

    cols = list(vdm.columns)
    if "pos" in cols:
        cols.remove("pos")
    insert_index = cols.index("player") + 1 if "player" in cols else 0
    cols.insert(insert_index, "pos")
    vdm = vdm[cols]

    other_cols = [c for c in vdm.columns if c not in desired_order]
    vdm = vdm[[c for c in desired_order if c in vdm.columns] + other_cols]

    print("Colonna 'pos' creata e colonne riordinate correttamente.")
    return vdm

vdm_1819 = format_positions(vdm_1819)
vdm_1920 = format_positions(vdm_1920)
vdm_2021 = format_positions(vdm_2021)
vdm_2122 = format_positions(vdm_2122)
vdm_2223 = format_positions(vdm_2223)
vdm_2324 = format_positions(vdm_2324)
vdm_2425 = format_positions(vdm_2425)

#Ricalcolo di tutte le variabili composte ovvero le variabili che derivano da operazioni
#aritmetiche tra due o più variabili

def recalculate_composite_metrics(df):
    df = df.copy()

    # Funzione di sicurezza per evitare divisioni per zero
    def safe_div(a, b):
        return np.where(b != 0, a / b, 0)

    recalculated_vars = []  # per contare quante variabili vengono ricalcolate
    aaa= df["%GK_launched_40yds"]/100 #la variabile ha il format sbagliato e quindi safe_div mostra risultati sballati
    df["def_actions"] = df["tackles_won"]+df["blocks"]+df["interceptions"]+df["clearance"]

    try:
        df["90s"] = safe_div(df["Min"], 90); recalculated_vars.append("90s")
        df["G+A"] = df["Goals"] + df["Ast"]; recalculated_vars.append("G+A")
        df["G-PK"] = df["Goals"] - df["PK"]; recalculated_vars.append("G-PK")
        df["npxG + xAG"] = df["npxG"] + df["xAG"]; recalculated_vars.append("npxG + xAG")

        df["Gls_per90"] = safe_div(df["Goals"], df["90s"]); recalculated_vars.append("Gls_per90")
        df["Ast_per90"] = safe_div(df["Ast"], df["90s"]); recalculated_vars.append("Ast_per90")
        df["G+A_per90"] = safe_div(df["Goals"] + df["Ast"], df["90s"]); recalculated_vars.append("G+A_per90")
        df["G-PK_per90"] = safe_div(df["Goals"] - df["PK"], df["90s"]); recalculated_vars.append("G-PK_per90")
        df["G+A-PK_per90"] = safe_div(df["Goals"] + df["Ast"] - df["PK"], df["90s"]); recalculated_vars.append("G+A-PK_per90")
        df["xG_per90"] = safe_div(df["xG"], df["90s"]); recalculated_vars.append("xG_per90")
        df["xAG_per90"] = safe_div(df["xAG"], df["90s"]); recalculated_vars.append("xAG_per90")
        df["xG+xAG_per90"] = safe_div(df["xG"] + df["xAG"], df["90s"]); recalculated_vars.append("xG+xAG_per90")
        df["npxG_per90"] = safe_div(df["npxG"], df["90s"]); recalculated_vars.append("npxG_per90")
        df["npxG+xAG_per90"] = safe_div(df["npxG"] + df["xAG"], df["90s"]); recalculated_vars.append("npxG+xAG_per90")
        df["Gag_per90"] = safe_div(df["Gag"], df["90s"]); recalculated_vars.append("Gag_per90")

        df["Save%"] = safe_div(df["Saves"], df["SoTA"]); recalculated_vars.append("Save%")
        df["CS%"] = safe_div(df["CS"], df["W"] + df["D"] + df["L"]); recalculated_vars.append("CS%")
        df["PK_Save%"] = safe_div(df["PKsaved"], df["PK_att_ag"] - df["PKmiss"]); recalculated_vars.append("PK_Save%")
        df["PSxG/SoTA"] = safe_div(df["PSxG"], df["SoTA"]); recalculated_vars.append("PSxG/SoTA")
        df["PSxG-GA"] = df["PSxG"] - df["Gag"]; recalculated_vars.append("PSxG-GA")
        df["PSxG-GA_per90"] = safe_div(df["PSxG"] - df["Gag"], df["90s"]); recalculated_vars.append("PSxG-GA_per90")

        df["pass_40yds_Cmp%"] = safe_div(df["pass_40yds_Cmp"], df["pass_40yds_Att"]); recalculated_vars.append("pass_40yds_Cmp%")
        df["pass_launched_%"] = safe_div((df["pass_40yds_Att"] - (df["GK_Att"] * aaa)),df["pass_att(no GK)"]); recalculated_vars.append("pass_launched_%")
        df["cross_stp%"] = safe_div(df["cross_stp"], df["cross_faced"]); recalculated_vars.append("cross_stp%")
        df["OPA_per90"] = safe_div(df["OPA"], df["90s"]); recalculated_vars.append("OPA_per90")

        df["SoT%"] = safe_div(df["SoT"], df["Sh"]); recalculated_vars.append("SoT%")
        df["Sh_per90"] = safe_div(df["Sh"], df["90s"]); recalculated_vars.append("Sh_per90")
        df["SoT_per90"] = safe_div(df["SoT"], df["90s"]); recalculated_vars.append("SoT_per90")
        df["G/Sh"] = safe_div(df["Goals"], df["Sh"]); recalculated_vars.append("G/Sh")
        df["G/SoT"] = safe_div(df["Goals"], df["SoT"]); recalculated_vars.append("G/SoT")
        df["npxG/Sh"] = safe_div(df["npxG"], df["Sh"]); recalculated_vars.append("npxG/Sh")
        df["G-xG"] = df["Goals"] - df["xG"]; recalculated_vars.append("G-xG")
        df["npG-npxG"] = (df["Goals"] - df["PK"]) - df["npxG"]; recalculated_vars.append("npG-npxG")

        df["pass_cmp%"] = safe_div(df["pass_cmp"], df["pass_att"]); recalculated_vars.append("pass_cmp%")
        df["short_pass_cmp%"] = safe_div(df["short_pass_cmp"], df["short_pass_att"]); recalculated_vars.append("short_pass_cmp%")
        df["med_pass_cmp%"] = safe_div(df["med_pass_cmp"], df["med_pass_att"]); recalculated_vars.append("med_pass_cmp%")
        df["long_pass_cmp%"] = safe_div(df["long_pass_cmp"], df["long_pass_att"]); recalculated_vars.append("long_pass_cmp%")

        df["A-xAG"] = df["Ast"] - df["xAG"]; recalculated_vars.append("A-xAG")
        df["SCA_per90"] = safe_div(df["SCA"], df["90s"]); recalculated_vars.append("SCA_per90")
        df["GCA_per90"] = safe_div(df["GCA"], df["90s"]); recalculated_vars.append("GCA_per90")

        df["%dribblers_tackled"] = safe_div(df["dribblers_tackled"], df["dribblers_challenged"]); recalculated_vars.append("%dribblers_tackled")
        df["tackles + int"] = df["tackles"] + df["interceptions"]; recalculated_vars.append("tackles + int")
        df["dribbling_succ%"] = safe_div(df["dribbling_succ"], df["dribbling_att"]); recalculated_vars.append("dribbling_succ%")
        df["dribbling_Tkld%"] = safe_div(df["dribbling_tkld"], df["dribbling_att"]); recalculated_vars.append("dribbling_Tkld%")
        df["aerial_duels%"] = safe_div(df["aerial_duels_won"], df["aerial_duels_won"] + df["aerial_duels_lost"]); recalculated_vars.append("aerial_duels%")
        
        #Variabili aggiunte in seguito alla definizione delle variabile proxy rispetto ai KPIs
        #NORMALIZZATE SU 90 minuti
        df["Saves_per90"] = safe_div(df["Saves"], df["90s"]); recalculated_vars.append("Saves_per90")
     
        df["errors_leading_shot_per90"] = safe_div(df["errors_leading_shot"], df["90s"]); recalculated_vars.append("errors_leading_shot_per90")
        df["miscontrols_per90"] = safe_div(df["miscontrols"], df["90s"]); recalculated_vars.append("miscontrols_per90")
        df["ball_recoveries_per90"] = safe_div(df["ball_recoveries"], df["90s"]); recalculated_vars.append("ball_recoveries_per90")
        df["dispossessed_per90"] = safe_div(df["dispossessed"], df["90s"]); recalculated_vars.append("dispossessed_per90")    
        df["miscontrols_per90"] = safe_div(df["miscontrols"], df["90s"]); recalculated_vars.append("miscontrols_per90")
        df["offsides_per90"] = safe_div(df["offsides"], df["90s"]); recalculated_vars.append("offsides_per90")        

        df["PK_con_per90"] = safe_div(df["PK_con"], df["90s"]); recalculated_vars.append("PK_con_per90")
        df["2CrdY_per90"] = safe_div(df["2CrdY"], df["90s"]); recalculated_vars.append("2CrdY_per90")
        df["fouls_com_per90"] = safe_div(df["fouls_com"], df["90s"]); recalculated_vars.append("fouls_com_per90")       
        df["fouls_drawn_per90"] = safe_div(df["fouls_drawn"], df["90s"]); recalculated_vars.append("fouls_drawn_per90")
        
        df["interceptions_per90"] = safe_div(df["interceptions"], df["90s"]); recalculated_vars.append("interceptions_per90")
        df["blocks_per90"] = safe_div(df["blocks"], df["90s"]); recalculated_vars.append("blocks_per90")
        df["clearances_per90"] = safe_div(df["clearance"], df["90s"]); recalculated_vars.append("clearances_per90")
        
        df["PrgC_per90"] = safe_div(df["PrgC"], df["90s"]); recalculated_vars.append("PrgC_per90")
        df["carries_into_att_3rd_per90"] = safe_div(df["carries_into_att_3rd"], df["90s"]); recalculated_vars.append("carries_into_att_3rd_per90")
        df["carries_into_PA_per90"] = safe_div(df["carries_into_PA"], df["90s"]); recalculated_vars.append("carries_into_PA_per90")
        
        df["pass_into_att_3rd_per90"] = safe_div(df["pass_into_att_3rd"], df["90s"]); recalculated_vars.append("pass_into_att_3rd_per90")
        df["live_passes_per90"] = safe_div(df["live_passes"], df["90s"]); recalculated_vars.append("live_passes_per90")
        df["dead_passes_per90"] = safe_div(df["dead_passes"], df["90s"]); recalculated_vars.append("dead_passes_per90")
        df["key_pass_per90"] = safe_div(df["key_pass"], df["90s"]); recalculated_vars.append("key_pass_per90")
        df["PrgP_per90"] = safe_div(df["PrgP"], df["90s"]); recalculated_vars.append("PrgP_per90")
        df["PrgR_per90"] = safe_div(df["PrgR"], df["90s"]); recalculated_vars.append("PrgR_per90")
        df["PPA_per90"] = safe_div(df["PPA"], df["90s"]); recalculated_vars.append("PPA_per90")
        
        df["touches_def_3rd_per90"] = safe_div(df["touches_def_3rd"], df["90s"]); recalculated_vars.append("touches_def_3rd_per90")
        df["touches_mid_3rd_per90"] = safe_div(df["touches_mid_3rd"], df["90s"]); recalculated_vars.append("touches_mid_3rd_per90")
        df["touches_att_3rd_per90"] = safe_div(df["touches_att_3rd"], df["90s"]); recalculated_vars.append("touches_att_3rd_per90")
        df["touches_att_PA_per90"] = safe_div(df["touches_att_PA"], df["90s"]); recalculated_vars.append("touches_att_PA_per90")
        
        df["tackle_att_3rd_per90"] = safe_div(df["tackle_att_3rd"], df["90s"]); recalculated_vars.append("tackle_att_3rd_per90")
        df["tackle_mid_3rd_per90"] = safe_div(df["tackle_mid_3rd"], df["90s"]); recalculated_vars.append("tackle_mid_3rd_per90")
        df["tackles_def_3rd_per90"] = safe_div(df["tackles_def_3rd"], df["90s"]); recalculated_vars.append("tackles_def_3rd_per90")
        df["tackles_won_per90"] = safe_div(df["tackles_won"], df["90s"]); recalculated_vars.append("tackles_won_per90")
        df["tackles_won%"] = safe_div(df["tackles_won"], df["tackles"]); recalculated_vars.append("tackles_won%")
        df["def_actions_per90"] = safe_div(df["def_actions"], df["90s"]); recalculated_vars.append("def_actions_per90%")
        
        df["crosses_per90"] = safe_div(df["crosses"], df["90s"]); recalculated_vars.append("crosses_per90")
        df["cross_into_PA_per90"] = safe_div(df["cross_into_PA"], df["90s"]); recalculated_vars.append("cross_into_PA_per90")
        df["cross_stp_per90"] = safe_div(df["cross_stp"], df["90s"]); recalculated_vars.append("cross_stp_per90")
        
        #Creazione feature random con valori tra 0 e 1 per i modelli di controllo
        np.random.seed(42) # Per riproducibilità
        df["random"] = np.random.rand(len(df))
        
    except KeyError as e:
        print(f"Variabile mancante nel DataFrame: {e}")

    # Conversione numerica e arrotondamento
    cols_to_round = [col for col in df.columns if df[col].dtype in [float, int]]
    df[cols_to_round] = df[cols_to_round].astype(float).round(2)

    # 🔹 Log finale
    print(f"Ricostruite {len(recalculated_vars)} variabili composte.")
    print(f"Variabili ricalcolate: {', '.join(recalculated_vars)}")

    return df

df_final_1819=recalculate_composite_metrics(df_final_1819)
df_final_1920=recalculate_composite_metrics(df_final_1920)
df_final_2021=recalculate_composite_metrics(df_final_2021)
df_final_2122=recalculate_composite_metrics(df_final_2122)
df_final_2223=recalculate_composite_metrics(df_final_2223)
df_final_2324=recalculate_composite_metrics(df_final_2324)
df_final_2425=recalculate_composite_metrics(df_final_2425)

#Funzione che gestisce i casi particolari di nazionalità in df_final
#CASO GREGOIRE DEFREL:
    #Defrel tecnicamente è francese ma è nato nella Martinica francese.
    #FBRef lo considera come nativo della Martinica quindi il codice nazione è
    #uguale a "mq MTQ" nonostante per vdm sia francese.
    #Lo modifico in df_final modificando la sua nazionalità da "mq MTQ"-"Martinica"
    #a "fr FRA"-"Francia"
    
#CASO ADRIEN TAMEZE
    #Tameze è un giocatore francese che ha giocato con le nazionali giovanili francesi.
    #FBRef lo considera del camerun mentre vdm lo considera francese.
    #Modifico la dicitura "cm CMR" in "fr FRA" per la variabile "Nation" e inserisco "Francia"
    #per la variabile "Nation2" nei df_final.

#CASO ABDOU HARROUI
    #Harroui è un calciatore marocchino con nazionalità olandese che ha giocato
    #con le nazionali giovanili olandesi.
    #FBRef lo considera olandese mentre vdm lo considera marocchino.
    #Modifico la dicitura da "Marocco" a "Olanda" per la variabile "nation" in vdm.

#CASO EMMANUEL RIVIERE
    #Riviere è un calciatore della nazionale della Martinica con cui ha esordito.
    #FBref lo considerava della Martinica ma io ho modificato
    #i giocatori della martinica in francesi per far combaciare Defrel francese.
    
def nationality_particular_cases(df, nome_df="DataFrame"):

    df = df.copy()
    changes = []  # per log interni

    # -------------------------------
    # CASO 1 — Grégoire Defrel
    # -------------------------------
    mask_defrel = df["Player"].str.contains("Defrel", case=False, na=False) if "Player" in df.columns else df["player"].str.contains("Defrel", case=False, na=False)
    if mask_defrel.any():
        if "Nation" in df.columns and "Nation2" in df.columns:
            df.loc[mask_defrel, "Nation"] = "fr FRA"
            df.loc[mask_defrel, "Nation2"] = "Francia"
            changes.append("Grégoire Defrel aggiornato: Martinica → Francia")
        else:
            changes.append(" Colonne Nation/Nation2 non trovate, impossibile modificare Grégoire Defrel")
    else:
        print(f"Grégoire Defrel non è presente nel {nome_df} in esame.")

    # -------------------------------
    # CASO 2 — Adrien Tameze
    # -------------------------------
    mask_tameze = df["Player"].str.contains("Tameze", case=False, na=False) if "Player" in df.columns else df["player"].str.contains("Tameze", case=False, na=False)
    if mask_tameze.any():
        if "Nation" in df.columns and "Nation2" in df.columns:
            df.loc[mask_tameze, "Nation"] = "fr FRA"
            df.loc[mask_tameze, "Nation2"] = "Francia"
            changes.append("Adrien Tameze aggiornato: Camerun → Francia")
        else:
            changes.append("Colonne Nation/Nation2 non trovate, impossibile modificare Adrien Tameze")
    else:
        print(f" Adrien Tameze non è presente nel {nome_df} in esame.")

    # -------------------------------
    # CASO 3 — Abdou Harroui
    # -------------------------------
    mask_harroui = df["Player"].str.contains("Harroui", case=False, na=False) if "Player" in df.columns else df["player"].str.contains("Harroui", case=False, na=False)
    if mask_harroui.any():
        if "nation" in df.columns:
            df.loc[mask_harroui, "nation"] = "Olanda"
            changes.append("Abdou Harroui aggiornato: Marocco → Olanda")
        else:
            changes.append("Colonna 'nation' non trovata, impossibile modificare Abdou Harroui")
    else:
        print(f"Abdou Harroui non è presente nel {nome_df} in esame.")

    # -------------------------------
    # 🔹 LOG FINALE
    # -------------------------------
    if changes:
        print(f"\n Correzioni applicate in {nome_df}:")
        for c in changes:
            print("  -", c)
    else:
        print(f"Nessuna correzione necessaria in {nome_df}.")

    return df

df_final_1819 = nationality_particular_cases(df_final_1819, nome_df="df_final_1819")
df_final_1920 = nationality_particular_cases(df_final_1920, nome_df="df_final_1920")
df_final_2021 = nationality_particular_cases(df_final_2021, nome_df="df_final_2021")
df_final_2122 = nationality_particular_cases(df_final_2122, nome_df="df_final_2122")
df_final_2223 = nationality_particular_cases(df_final_2223, nome_df="df_final_2223")
df_final_2324 = nationality_particular_cases(df_final_2324, nome_df="df_final_2324")
df_final_2425 = nationality_particular_cases(df_final_2425, nome_df="df_final_2425")

vdm_1819 = nationality_particular_cases(vdm_1819, nome_df="vdm_1819")
vdm_1920 = nationality_particular_cases(vdm_1920, nome_df="vdm_1920")
vdm_2021 = nationality_particular_cases(vdm_2021, nome_df="vdm_2021")
vdm_2122 = nationality_particular_cases(vdm_2122, nome_df="vdm_2122")
vdm_2223 = nationality_particular_cases(vdm_2223, nome_df="vdm_2223")
vdm_2324 = nationality_particular_cases(vdm_2324, nome_df="vdm_2324")
vdm_2425 = nationality_particular_cases(vdm_2425, nome_df="vdm_2425")


#Funzione che prende in input i dataframe e elimina i calciatori che hanno giocato
#meno di un tot di minuti in campionato. 
#Il valore standard è 360 minuti ovvero 4 partite complete che equivale circa al 10%
#del campionato di Serie A (38 partite totali)
def delete_min(df, min_threshold=360):

    if "Min" not in df.columns:
        print("Colonna 'Min' non trovata nel dataframe.")
        return df

    df = df.copy()

    # Convertiamo 'Min' in numerico nel caso non lo fosse
    df["Min"] = pd.to_numeric(df["Min"], errors="coerce")

    # Filtriamo i giocatori
    initial_count = len(df)
    df_filtered = df[df["Min"] >= min_threshold].reset_index(drop=True)
    removed_count = initial_count - len(df_filtered)

    # Log finale
    print(f"Filtrati i giocatori con meno di {min_threshold} minuti giocati.")
    print(f"Rimasti {len(df_filtered)} su {initial_count} giocatori totali ({removed_count} rimossi).")

    return df_filtered

df_final_1819=delete_min(df_final_1819)
df_final_1920=delete_min(df_final_1920)
df_final_2021=delete_min(df_final_2021)
df_final_2122=delete_min(df_final_2122)
df_final_2223=delete_min(df_final_2223)
df_final_2324=delete_min(df_final_2324)
df_final_2425=delete_min(df_final_2425)
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 7 - ESEGUIRE IL MERGE DEI DATAFRAME VDM E DF_FINAL
#eseguo il merge dei dataframe scaricati da FBref e quelli scaricati da TransferMarkt con i valori di mercato.
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
def merge_final_vdm(df_final, vdm):

    df_final = df_final.copy()
    vdm = vdm.copy()

    #Uniformo i nomi delle colonne nel vdm per poter fare il merge
    vdm = vdm.rename(columns={
        "player": "Player",
        "nation": "Nation2",
        "born": "Born",
        "season": "Season"
    })

    #Funzione di normalizzazione per rimuovere accenti, caratteri speciali e eseguire sostituzioni mirate
    #per far combaciare i giocatori
    def normalize_text(s):
        if pd.isna(s):
            return ""
        s = str(s).strip().lower()

        # Normalizza accenti e caratteri Unicode
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

        # Rimozione apostrofi (es. n'koulou -> nkoulou)
        s = s.replace("'", "").replace("’", "")

        # Sostituzioni specifiche richieste
        s = s.replace("ł", "l")
        s = s.replace("đ", "dj")
        s = s.replace("ð", "d")
        s = s.replace("æ", "ae")
        s = s.replace("ø","o") 
        s = s.replace("í", "i") 
        s = s.replace("ì", "i") 
        s = s.replace("þ","th")


        # Sostituzioni per casi specifici di giocatori
        s = s.replace("kostas manolas", "konstantinos manolas")
        s = s.replace("samir santos", "samir caetano")
        s = s.replace("freddie veseli", "frederic veseli")
        s = s.replace("hamed junior traore", "hamed traore")
        s = s.replace("dalbert henrique", "dalbert")
        s = s.replace("felipe dal belo", "felipe")
        s = s.replace("fabian ruiz pena", "fabian ruiz")
        s = s.replace("danilo larangeira", "danilo")
        s = s.replace("elif elmas", "eljif elmas")
        s = s.replace("giorgos kyriakopoulos","georgios kyriakopoulos")
        s = s.replace("gleison bremer","bremer")
        s = s.replace("igor julio","igor")
        s = s.replace("julian chabot","jeff chabot")
        s = s.replace("yevhen shakhov","yevgen shakhov")
        s = s.replace("angelo da costa junior","angelo da costa")
        s = s.replace("aleksei miranchuk","aleksey miranchuk")
        s = s.replace("jean-daniel akpa-akpro","jean-daniel akpa akpro")
        s = s.replace("lennart-marten czyborra","lennart czyborra")
        s = s.replace("salvador ferrer","salva ferrer")
        s = s.replace("simeon nwankwo","simy")
        s = s.replace("yıldırım mert cetin","mert cetin")
        s = s.replace("andre-frank zambo anguissa","frank anguissa")
        s = s.replace("antonio vacca","antonio junior vacca")
        s = s.replace("dimitris nikolaou","dimitrios nikolaou")
        s = s.replace("leo skiri ostigard","leo ostigard")
        s = s.replace("mickael cuisance","michael cuisance")
        s = s.replace("nicolas gonzalez","nico gonzalez")
        s = s.replace("pablo galdames millan","pablo galdames")
        s = s.replace("emanuel aiwum","emanuel aiwu")
        s = s.replace("emil ceide","emil konradsen ceide")
        s = s.replace("juan david cabal","juan cabal")
        s = s.replace("kim min-jae","min-jae kim")
        s = s.replace("mohamed camara","mady camara")
        s = s.replace("pepin machin","pepin")
        s = s.replace("daniel bandeira","dani silva")
        s = s.replace("lautaro gianetti","lautaro giannetti")
        s = s.replace("obite ndicka","evan ndicka")
        s = s.replace("valentin castellanos", "taty castellanos")
        s = s.replace("steven shpendi","stiven shpendi")
        s = s.replace("pasaridis","pasalidis")
        s = s.replace("victor bernth kristiansen","victor kristiansen")
        s = s.replace("yann aurel bisseck","yann bisseck")
        s = s.replace("benjamin dominguez","benja dominguez")
        s = s.replace("emerson royal", "emerson")
        s = s.replace("del prato","delprato")
        s = s.replace("faustino anjorin","tino anjorin")
        s = s.replace("kialonda gaspar","kialonda")
        s = s.replace("marc-oliver kempf","marc oliver kempf")
        s = s.replace("michel ndary adopo","michel adopo")
        s = s.replace("mikael ellertsson","mikael egill ellertsson")
        s = s.replace("mohamed haj","anas haj mohamed")
        s = s.replace("nicolas paz","nico paz")
        s = s.replace("souleymane toure","isaak toure")
        
        return s

    #Applico la normalizzazione a Player, Nation2 e Season
    for col in ["Player", "Nation2", "Season"]:
        df_final[col] = df_final[col].apply(normalize_text)
        vdm[col] = vdm[col].apply(normalize_text)

    #Converto la colonna Born in numerica
    df_final["Born"] = pd.to_numeric(df_final["Born"], errors="coerce")
    vdm["Born"] = pd.to_numeric(vdm["Born"], errors="coerce")

    #Imposto il multi-indice identificativo
    keys = ["Player", "Nation2", "Born", "Season"]

    df_final = df_final.set_index(keys)
    vdm = vdm.set_index(keys)

    #Eseguo un merge mantenendo solo i giocatori presenti in df_final
    merge_df = df_final.join(vdm, how="left", rsuffix="_vdm")

    #Reset dell’indice per tornare a DataFrame standard
    merge_df = merge_df.reset_index()

    #Log finale
    print(f"Merge completato: {merge_df.shape[0]} righe unite su {df_final.shape[0]} giocatori totali ({vdm.shape[0]} presenti in vdm).")

    missing = merge_df["value"].isna().sum() if "value" in merge_df.columns else 0
    print(f"Giocatori in df_merge senza match in vdm: {missing}")

    return merge_df

df_merge_1819=merge_final_vdm(df_final_1819, vdm_1819)
df_merge_1920=merge_final_vdm(df_final_1920, vdm_1920)
df_merge_2021=merge_final_vdm(df_final_2021, vdm_2021)
df_merge_2122=merge_final_vdm(df_final_2122, vdm_2122) 
df_merge_2223=merge_final_vdm(df_final_2223, vdm_2223) 
df_merge_2324=merge_final_vdm(df_final_2324, vdm_2324)
df_merge_2425=merge_final_vdm(df_final_2425, vdm_2425)

#Funzione che elimina i giocatori con value=nan.
#Si tratta dell'ultima pulizia dei dataframe merge
def delete_value_nan(df_merge: pd.DataFrame) -> pd.DataFrame:
    df_merge = df_merge.copy()

    if "value" not in df_merge.columns:
        print("Colonna 'value' non trovata nel dataframe.")
        return df_merge

    initial_rows = len(df_merge)
    df_merge = df_merge[df_merge["value"].notna()].reset_index(drop=True)
    removed_rows = initial_rows - len(df_merge)

    print(f"Rimossi {removed_rows} giocatori con valore NaN. Righe finali: {len(df_merge)}")

    return df_merge

df_merge_1819 = delete_value_nan(df_merge_1819)
df_merge_1920 = delete_value_nan(df_merge_1920)
df_merge_2021 = delete_value_nan(df_merge_2021)
df_merge_2122 = delete_value_nan(df_merge_2122)
df_merge_2223 = delete_value_nan(df_merge_2223)
df_merge_2324 = delete_value_nan(df_merge_2324)
df_merge_2425 = delete_value_nan(df_merge_2425)

#Ordino le colonne in modo tale da avere le informazioni sui calciatori (variabili qualitative)
#nella parte iniziale del dataframe per comodità di fruizione
def change_format_merge(df_merge: pd.DataFrame) -> pd.DataFrame:
    df_merge = df_merge.copy()

    #Ordine desiderato per le prime colonne
    ordered_cols = [
        "Player", "Pos", "pos", "position", "Squad",
        "Nation", "Nation2", "date of birth", "Born",
        "age", "Age", "foot", "height", "value", "Season"]

    #Colonne effettivamente presenti nel dataframe in quell’ordine
    existing_ordered = [col for col in ordered_cols if col in df_merge.columns]

    #Tutte le altre colonne non ancora incluse
    remaining_cols = [col for col in df_merge.columns if col not in existing_ordered]

    #Riordina il dataframe
    df_merge = df_merge[existing_ordered + remaining_cols]

    print(f"Colonne riordinate ({len(existing_ordered)} principali + {len(remaining_cols)} rimanenti).")

    return df_merge

df_merge_1819 = change_format_merge(df_merge_1819)
df_merge_1920 = change_format_merge(df_merge_1920)
df_merge_2021 = change_format_merge(df_merge_2021)
df_merge_2122 = change_format_merge(df_merge_2122)
df_merge_2223 = change_format_merge(df_merge_2223)
df_merge_2324 = change_format_merge(df_merge_2324)
df_merge_2425 = change_format_merge(df_merge_2425)

#Elimina le colonne 'Age' e 'Pos' dal dataframe, se presenti.
#"Pos" è la posizione nel vecchio formato di FBRef mentre da adesso uso le 5 
#posizioni create (GK,DF,WB,MF,FW)
#Elimino anche alcune variabili ripetute scappate ai controlli precedenti
def drop_columns_merge(df_merge: pd.DataFrame) -> pd.DataFrame:
    df_merge = df_merge.copy()
    cols_to_drop = ["Age", "Pos", "npxG + xAG", "prg_passes_rec", "prog_pass", "prg_carries", "cross", "passes_cmp"]

    existing_cols = [col for col in cols_to_drop if col in df_merge.columns]
    df_merge = df_merge.drop(columns=existing_cols, errors="ignore")

    print(f"Colonne eliminate: {', '.join(existing_cols) if existing_cols else 'nessuna trovata.'}")

    return df_merge

df_merge_1819 = drop_columns_merge(df_merge_1819)
df_merge_1920 = drop_columns_merge(df_merge_1920)
df_merge_2021 = drop_columns_merge(df_merge_2021)
df_merge_2122 = drop_columns_merge(df_merge_2122)
df_merge_2223 = drop_columns_merge(df_merge_2223)
df_merge_2324 = drop_columns_merge(df_merge_2324)
df_merge_2425 = drop_columns_merge(df_merge_2425)

#Modifico il format della colonna "age" in numerico
def change_format_age(df_merge: pd.DataFrame) -> pd.DataFrame:    
    df_merge = df_merge.copy()

    if "age" in df_merge.columns:
        df_merge["age"] = pd.to_numeric(df_merge["age"], errors="coerce")
        print("Colonna 'age' convertita in formato numerico.")
    else:
        print("La colonna 'age' non è presente nel dataframe. Nessuna modifica eseguita.")

    return df_merge

df_merge_1819 = change_format_age(df_merge_1819)
df_merge_1920 = change_format_age(df_merge_1920)
df_merge_2021 = change_format_age(df_merge_2021)
df_merge_2122 = change_format_age(df_merge_2122)
df_merge_2223 = change_format_age(df_merge_2223)
df_merge_2324 = change_format_age(df_merge_2324)
df_merge_2425 = change_format_age(df_merge_2425)

#Elimina i nan dai dataframe
df_merge_1819 = df_merge_1819.fillna(0)
df_merge_1920 = df_merge_1920.fillna(0)
df_merge_2021 = df_merge_2021.fillna(0)
df_merge_2122 = df_merge_2122.fillna(0)
df_merge_2223 = df_merge_2223.fillna(0)
df_merge_2324 = df_merge_2324.fillna(0)
df_merge_2425 = df_merge_2425.fillna(0)
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 8 - DIVIDERE I DF_MERGE NEI DF PER OGNI RUOLO
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
def create_dataset_training(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    df = df.copy()
    pos = pos.strip().upper()

    # Controllo di validità sulla posizione
    valid_positions = ["GK", "DF", "WB", "MF", "FW"]
    if pos not in valid_positions:
        raise ValueError(f"Posizione '{pos}' non valida. Scegli tra: {', '.join(valid_positions)}")

    # Controllo presenza della colonna 'pos'
    if "pos" not in df.columns:
        raise KeyError("La colonna 'pos' non è presente nel dataframe.")

    # Filtraggio per posizione
    subset_df = df[df["pos"].str.upper() == pos]

    # Log informativo
    print(f"Estratti {subset_df.shape[0]} giocatori con posizione '{pos}' su {df.shape[0]} totali.")

    return subset_df

df_gk_1819 = create_dataset_training(df_merge_1819, pos = "GK")
df_df_1819 = create_dataset_training(df_merge_1819, pos = "DF")
df_wb_1819 = create_dataset_training(df_merge_1819, pos = "WB")
df_mf_1819 = create_dataset_training(df_merge_1819, pos = "MF")
df_fw_1819 = create_dataset_training(df_merge_1819, pos = "FW")

df_gk_1920 = create_dataset_training(df_merge_1920, pos = "GK")
df_df_1920 = create_dataset_training(df_merge_1920, pos = "DF")
df_wb_1920 = create_dataset_training(df_merge_1920, pos = "WB")
df_mf_1920 = create_dataset_training(df_merge_1920, pos = "MF")
df_fw_1920 = create_dataset_training(df_merge_1920, pos = "FW")

df_gk_2021 = create_dataset_training(df_merge_2021, pos = "GK")
df_df_2021 = create_dataset_training(df_merge_2021, pos = "DF")
df_wb_2021 = create_dataset_training(df_merge_2021, pos = "WB")
df_mf_2021 = create_dataset_training(df_merge_2021, pos = "MF")
df_fw_2021 = create_dataset_training(df_merge_2021, pos = "FW")

df_gk_2122 = create_dataset_training(df_merge_2122, pos = "GK")
df_df_2122 = create_dataset_training(df_merge_2122, pos = "DF")
df_wb_2122 = create_dataset_training(df_merge_2122, pos = "WB")
df_mf_2122 = create_dataset_training(df_merge_2122, pos = "MF")
df_fw_2122 = create_dataset_training(df_merge_2122, pos = "FW")

df_gk_2223 = create_dataset_training(df_merge_2223, pos = "GK")
df_df_2223 = create_dataset_training(df_merge_2223, pos = "DF")
df_wb_2223 = create_dataset_training(df_merge_2223, pos = "WB")
df_mf_2223 = create_dataset_training(df_merge_2223, pos = "MF")
df_fw_2223 = create_dataset_training(df_merge_2223, pos = "FW")

df_gk_2324 = create_dataset_training(df_merge_2324, pos = "GK")
df_df_2324 = create_dataset_training(df_merge_2324, pos = "DF")
df_wb_2324 = create_dataset_training(df_merge_2324, pos = "WB")
df_mf_2324 = create_dataset_training(df_merge_2324, pos = "MF")
df_fw_2324 = create_dataset_training(df_merge_2324, pos = "FW")

df_gk_2425 = create_dataset_training(df_merge_2425, pos = "GK")
df_df_2425 = create_dataset_training(df_merge_2425, pos = "DF")
df_wb_2425 = create_dataset_training(df_merge_2425, pos = "WB")
df_mf_2425 = create_dataset_training(df_merge_2425, pos = "MF")
df_fw_2425 = create_dataset_training(df_merge_2425, pos = "FW")


def concat_df(dfs: list) -> pd.DataFrame:

    # Controllo: lista vuota
    if not dfs:
        print("La lista dei dataframe è vuota. Ritorno un dataframe vuoto.")
        return pd.DataFrame()

    # Controllo: tutti gli elementi devono essere DataFrame
    if not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise TypeError("Tutti gli elementi della lista devono essere di tipo dataframe.")

    # Rimuove eventuali None
    dfs = [df for df in dfs if df is not None and not df.empty]

    if not dfs:
        print("Tutti i dataframe erano vuoti o None. Ritorno un dataframe vuoto.")
        return pd.DataFrame()

    # Concatenazione
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merge completato: {merged_df.shape[0]} righe totali, da {len(dfs)} dataframe uniti.")

    return merged_df

df_gk_1820 = concat_df([df_gk_1819, df_gk_1920])
df_gk_1821 = concat_df([df_gk_1819, df_gk_1920, df_gk_2021 ])
df_gk_1822 = concat_df([df_gk_1819, df_gk_1920, df_gk_2021, df_gk_2122])
df_gk_1823 = concat_df([df_gk_1819, df_gk_1920, df_gk_2021, df_gk_2122, df_gk_2223])
df_gk_1824 = concat_df([df_gk_1819, df_gk_1920, df_gk_2021, df_gk_2122, df_gk_2223, df_gk_2324])

df_wb_1820 = concat_df([df_wb_1819, df_wb_1920])
df_wb_1821 = concat_df([df_wb_1819, df_wb_1920, df_wb_2021])
df_wb_1822 = concat_df([df_wb_1819, df_wb_1920, df_wb_2021, df_wb_2122])
df_wb_1823 = concat_df([df_wb_1819, df_wb_1920, df_wb_2021, df_wb_2122, df_wb_2223])
df_wb_1824 = concat_df([df_wb_1819, df_wb_1920, df_wb_2021, df_wb_2122, df_wb_2223, df_wb_2324])

df_df_1820 = concat_df([df_df_1819, df_df_1920])
df_df_1821 = concat_df([df_df_1819, df_df_1920, df_df_2021])
df_df_1822 = concat_df([df_df_1819, df_df_1920, df_df_2021, df_df_2122])
df_df_1823 = concat_df([df_df_1819, df_df_1920, df_df_2021, df_df_2122, df_df_2223])
df_df_1824 = concat_df([df_df_1819, df_df_1920, df_df_2021, df_df_2122, df_df_2223, df_df_2324])

df_mf_1820 = concat_df([df_mf_1819, df_mf_1920])
df_mf_1821 = concat_df([df_mf_1819, df_mf_1920, df_mf_2021])
df_mf_1822 = concat_df([df_mf_1819, df_mf_1920, df_mf_2021, df_mf_2122])
df_mf_1823 = concat_df([df_mf_1819, df_mf_1920, df_mf_2021, df_mf_2122, df_mf_2223])
df_mf_1824 = concat_df([df_mf_1819, df_mf_1920, df_mf_2021, df_mf_2122, df_mf_2223, df_mf_2324])

df_fw_1820 = concat_df([df_fw_1819, df_fw_1920])
df_fw_1821 = concat_df([df_fw_1819, df_fw_1920, df_fw_2021])
df_fw_1822 = concat_df([df_fw_1819, df_fw_1920, df_fw_2021, df_fw_2122])
df_fw_1823 = concat_df([df_fw_1819, df_fw_1920, df_fw_2021, df_fw_2122, df_fw_2223])
df_fw_1824 = concat_df([df_fw_1819, df_fw_1920, df_fw_2021, df_fw_2122, df_fw_2223, df_fw_2324])
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 9 - GRAFICI ESPLORATIVI DEI DATASET
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#Andiamo a tracciare qualche grafico in modo tale da poter delineare a occhio se i dati hanno delle 
#particolari caratteristiche.
#Tramite la data visualization posso andare a compiere una piccola analisi preliminare 
#prima ancora della fase di definizione dei modelli

#Per la parte grafica uso la libreria SEABORN(sns) immaginata per la visualizzazione statistica di alto Livello.
#Questa libreria è costruita sopra Matplotlib, ma fornisce un'interfaccia di livello superiore specificamente orientata 
#alla visualizzazione di dati statistici ed è integrata con i DataFrame di Pandas, consentendo di creare grafici complessi 
#con poche righe di codice, gestendo automaticamente le etichette e le aggregazioni statistiche.
#--------------------------------------------------------------------------------------------------------------------------
#DIFENSORI 2018-2024
#--------------------------------------------------------------------------------------------------------------------------
graph = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["age", "height", "Min"], height = 4)
#age = vdm sembra diminuire con l'età
#altezza = vdm sembra aumentare con l'altezza
#min = vdm sembra crescere con i min giocati

graph2 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["aerial_duels%", "tackles_won%", "tackles_def_3rd_per90"], height = 4)
#non si vedono relazioni

graph3 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["interceptions_per90", "blocks_per90", "clearances_per90"], height = 4)
#non si vedono relazioni con i vdm

graph4 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["key_pass_per90", "PrgP_per90", "PrgC_per90"], height = 4)
#PrgC e PrgP sembrano avere effetto sul vdm, sembra che i difensori centrale che verticalizzano mediamente 
#di più abbiano vdm più alti. 

graph5 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["carries_into_att_3rd_per90", "touches_att_3rd_per90", "touches_mid_3rd_per90"], height = 4)
#touches_mid_3rd ha chiara influenza sul valore di mercato dei difensori, sembra che i difensori che toccano spesso la palla nella
#parte centrale del campo abbiano vdm più alti. Questo può essere influenzato dal fatto che squadre forti tendono a giocare con la difesa alta
#schiacciando l'avversario nella metà campo avversaria

graph5 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["crosses_per90", "cross_into_PA_per90", "carries_into_att_3rd_per90"], height = 4)
#non si vedono relazioni con i vdm

graph6 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["SCA_per90", "GCA_per90", "%dribblers_tackled"], height = 4)
#SCA e %dribblers_tackled hanno una debole relazione con il vdm

graph7 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["ball_recoveries_per90", "errors_leading_shot_per90", "dispossessed_per90"], height = 4)
#tutte e tre sembrano avere relazione con il vdm

graph8 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["pass_cmp%", "short_pass_cmp%", "med_pass_cmp%", "long_pass_cmp%"], height = 4)
#tutte le var sembrano avere forte relazione con il vdm

graph9 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["tackles_won_per90", "tackles_won%", "tackles_def_3rd_per90","tackle_mid_3rd_per90"], height = 4)
#non si vedono relazioni con i vdm

graph10 = sns.pairplot(data=df_df_1824, y_vars=['value'],x_vars=["def_actions", "def_actions_per90"], height = 4)


#VARIABILI CATEGORICHE
#Provo adesso a fare qualche bar-plot per le variabili categoriche
var_qual = ["Season", "foot", "Squad", "Nation2"]

sns.set(rc={'figure.figsize':(8,4)})    #ingrandisco bordi grafico

for var in var_qual:
    ax = sns.barplot(x=var, y="value", data=df_df_1824, errorbar=('ci', False)) #, hue = 'Model')
    for container in ax.containers:
        ax.bar_label(container)
    plt.title(var)
    plt.show()

#VARIABILI QUANTITATIVE
var_quant = ["age", "height", "Min", "PrgC_per90", "touches_mid_3rd_per90", "PrgP_per90", "pass_cmp%"]
sns.set(rc={'figure.figsize':(8,5)})

for var in var_quant:
    x = df_df_1824[var].values
    sns.displot(x, color = 'blue');

    # calcola la media
    mean = df_df_1824[var].mean()

    #inserisci nel grafico la media
    plt.axvline(mean, 0,1, color = 'red')
    plt.title(var)
    plt.show()

#Traccio anche i box-plot delle mie variabili quantitative almeno posso andare a 
#valutare se ci sono degli outlier all'interno dei dati in esame
sns.set(rc={'figure.figsize':(8,5)})

for var in var_quant:    
    x = df_df_1824[var].values
    ax = sns.boxplot(x, color = '#D1EC46')
    print('The meadian is: ', df_df_1824[var].median())
    plt.title(var)
    plt.show()
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10 - DELINEAZIONE DEI MODELLI DI MACHINE LEARNING
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#FASE 10.1 - LASSO REGRESSION 
#La Lasso Regression penalizza la dimensione dei coefficienti ed elimina le variabili non rilevanti.
#Se le feature non sono sulla stessa scala (es. altezza in metri vs. Save% in percentuali), 
#Lasso penalizzerà maggiormente i coefficienti delle feature con valori numerici piccoli 
#(come le percentuali) e ignorerà le altre. 
#Lo StandardScaler porta tutte le feature ad avere media zero e deviazione standard uno, 
#garantendo che la penalità di Lasso sia applicata in modo imparziale in base all'importanza predittiva, 
#non alla scala dei valori.
numeric_cols = df_df_1824.select_dtypes(include=np.number).columns.tolist() #CORRETTO, 217 colonne numeriche
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.1.1 - LASSO PORTIERI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column= "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]

x_train_gk = df_gk_1824[all_features]
y_train_gk = df_gk_1824[y_column]
x_test_gk = df_gk_2425[all_features]
y_test_gk = df_gk_2425[y_column]

#Momento di inizio del modello
start_time_gk_lasso = time.time()

try:
    age_index = all_features.index('age')
except ValueError:
    print("ERRORE: La colonna 'age' non è presente nelle tue features filtrate.")

age_column_index = [age_index]
other_columns_indices = [i for i, col in enumerate(all_features) if col != 'age']

# Definisci il trasformatore
feature_transformer = ColumnTransformer(
    transformers=[
        ('age_poly', PolynomialFeatures(degree=2, include_bias=False), age_column_index)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)), #RIDONDANTE
    ('transformer', feature_transformer),
    ('scaler', StandardScaler()),
    ('lasso', Lasso(random_state=42, max_iter=10000)) # Il nostro modello da ottimizzare
])

#???????????????????????????????????????????????????????????????????????????????????????
#FAI UN CHECK SU QUESTA GRIDSEARCH
#???????????????????????????????????????????????????????????????????????????????????????
#SELEZIONE AUTOMATICA (GridSearchCV)??????????
#L'idea è semplice: useremo l'ultima stagione del training set (2023-2024) come "set di validazione".
#Training per GridSearch: Dati 2018-2023
#Validazione per GridSearch: Dati 2023-2024
#In questo modo, GridSearchCV sarà costretto a scegliere l'alpha che meglio prevede la stagione successiva, 
#che è esattamente il mio obiettivo.
x_train_gk_with_season = df_gk_1824[all_features + ['Season']]

# Identifica l'ultima stagione nel set di training
last_season = x_train_gk_with_season['Season'].max() # '2023/2024'

print(f"Creazione split di validazione: validazione su stagione {last_season}")
validation_fold = np.where(x_train_gk_with_season['Season'] == last_season, 0, -1)

# Crea lo splitter
tscv = TimeSeriesSplit(n_splits=5) 

param_grid = {
    'lasso__alpha': np.logspace(5, 8, 1000) #100 valori di alpha tra 10^5 e 10^7
}

print("Avvio della Grid Search per trovare l'alpha ottimale di Lasso...")
grid_search_gk = GridSearchCV(
    pipeline, 
    param_grid, 
    cv= tscv, #predefined split
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=1  # Mostra i progressi
)

# Avvia l'addestramento e la ricerca
grid_search_gk.fit(x_train_gk, y_train_gk)

#4 RISULTATI DELLA SELEZIONE 
print("\n--- Risultati Grid Search ---")
best_alpha_gk = grid_search_gk.best_params_['lasso__alpha']
print(f"Alpha ottimale trovato: {best_alpha_gk:.6f}")
print(f"Miglior MAE in Cross-Validation: {-grid_search_gk.best_score_:.2f}")

other_columns_names = [col for col in all_features if col != 'age']

best_model_gk = grid_search_gk.best_estimator_ #'best_model' = modello finale con alpha migliore

#VALUTAZIONE FINALE SUL TEST SET ---
y_pred_train_gk = best_model_gk.predict(x_train_gk)
y_pred_test_gk = best_model_gk.predict(x_test_gk)

mae_test_gk_lasso = mean_absolute_error(y_test_gk, y_pred_test_gk)
mae_train_gk_lasso = mean_absolute_error(y_train_gk, y_pred_train_gk)
r2_test_gk_lasso = r2_score(y_test_gk, y_pred_test_gk)
r2_train_gk_lasso = r2_score(y_train_gk, y_pred_train_gk)

print("\n--- Performance Modello Finale Ottimizzato ---")
print(f"MAE sul set di TRAINING: {mae_train_gk_lasso:.2f}")
print(f"MAE sul set di TEST: {mae_test_gk_lasso:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_gk_lasso:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_gk_lasso:.4f}")

#ANALISI DEI COEFFICIENTI 
# Ricrea l'elenco dei nomi delle feature nell'ordine corretto
other_columns_names = [col for col in all_features if col != 'age']
feature_names_transformed = ['age', 'age^2'] + other_columns_names

# Estrai i coefficienti dal modello ottimale
lasso_regressor_gk = best_model_gk.named_steps['lasso']
lasso_coef_gk = pd.Series(lasso_regressor_gk.coef_, index=feature_names_transformed)

# Filtra quelle azzerate
zero_features_gk = lasso_coef_gk[np.abs(lasso_coef_gk) < 1e-6]
relevant_features_gk = lasso_coef_gk[np.abs(lasso_coef_gk) >= 1e-6]

print("\n--- Feature Selection Automatica Eseguita ---")
print(f"Feature totali iniziali: {len(all_features)}")
print(f"Feature eliminate da Lasso (coefficiente ≈ 0): {len(zero_features_gk)}")
print(f"Feature mantenute (significative): {len(relevant_features_gk)}")

print("\nFeature più importanti (coefficienti più alti):")
print(relevant_features_gk.abs().sort_values(ascending=False).head(25))

#Momento di fine del modello
end_time_gk_lasso = time.time()
execution_time_gk_lasso = end_time_gk_lasso - start_time_gk_lasso
print(f"--- Tempo di esecuzione Modello LASSO sui Portieri: {execution_time_gk_lasso:.2f} secondi ---")

# =============================================================================
# MODELLO DI CONTROLLO PORTIERI: REGRESSIONE LINEARE SEMPLICE CON VARIABILE RANDOM
# =============================================================================
feature_control = ["random"]
y_column = "value"
print("\n--- Inizio Addestramento Regressione Lineare di Controllo per i GK---")

# 1. Preparazione Dati
x_train_gk = df_gk_1824[['random']]
y_train_gk = df_gk_1824['value']
x_test_gk = df_gk_2425[['random']]
y_test_gk = df_gk_2425['value']

# 2. Definizione del Modello (Senza Pipeline perché non servono iperparametri)
# Usiamo comunque lo scaler per coerenza con gli altri modelli
scaler_x = StandardScaler()
x_train_scaled_gk_ctrl = scaler_x.fit_transform(x_train_gk)
x_test_scaled_gk_ctrl = scaler_x.transform(x_test_gk)

model_linear_gk_ctrl = LinearRegression()

# 3. Fit del modello
model_linear_gk_ctrl.fit(x_train_scaled_gk_ctrl, y_train_gk)

# 4. Valutazione
y_pred_train_gk_ctrl= model_linear_gk_ctrl.predict(x_train_scaled_gk_ctrl)
y_pred_test_gk_ctrl= model_linear_gk_ctrl.predict(x_test_scaled_gk_ctrl)

mae_test_gk_lasso_ctrl  = mean_absolute_error(y_test_gk, y_pred_test_gk_ctrl)
mae_train_gk_lasso_ctrl  = mean_absolute_error(y_train_gk, y_pred_train_gk_ctrl)

r2_test_gk_lasso_ctrl  = r2_score(y_test_gk, y_pred_test_gk_ctrl)
r2_train_gk_lasso_ctrl = r2_score(y_train_gk, y_pred_train_gk_ctrl)

# 5. Estrazione del coefficiente (molto importante per la tesi!)
coeff_gk_random = model_linear_gk_ctrl.coef_[0]

print("\n--- RISULTATI REGRESSIONE LINEARE (RANDOM) ---")
print(f"Coefficiente assegnato a 'random': {coeff_gk_random:.4f}")
print(f"MAE Test: {mae_test_gk_lasso_ctrl:.2f} M€")
print(f"MAE Train: {mae_train_gk_lasso_ctrl:.2f} M€")
print(f"R2 Train: {r2_train_gk_lasso_ctrl:.4f}")
print(f"R2 Test:  {r2_test_gk_lasso_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.1.2 - LASSO DIFENSORI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column= "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]

x_train_df = df_df_1824[all_features]
y_train_df = df_df_1824[y_column]
x_test_df = df_df_2425[all_features]
y_test_df = df_df_2425[y_column]

#Momento di inizio del modello
start_time_df_lasso = time.time()

try:
    age_index = all_features.index('age')
except ValueError:
    print("ERRORE: La colonna 'age' non è presente nelle tue features filtrate.")
    # Interrompi o gestisci l'errore qui

age_column_index = [age_index]
other_columns_indices = [i for i, col in enumerate(all_features) if col != 'age']

# Definisco il trasformatore
feature_transformer = ColumnTransformer(
    transformers=[
        ('age_poly', PolynomialFeatures(degree=2, include_bias=False), age_column_index)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)), #RIDONDANTE
    ('transformer', feature_transformer),
    ('scaler', StandardScaler()),
    ('lasso', Lasso(random_state=42, max_iter=10000)) # Il nostro modello da ottimizzare
])

#SELEZIONE AUTOMATICA (GridSearchCV) ---
x_train_df_with_season = df_df_1824[all_features + ['Season']]

# Identifica l'ultima stagione nel set di training
last_season = x_train_df_with_season['Season'].max() # '2023/2024'

print(f"Creazione split di validazione: validazione su stagione {last_season}")
# Creo l'array per PredefinedSplit
# -1 significa "usa questa riga per l'addestramento"
#  0 significa "usa questa riga per la validazione (test_fold)"
validation_fold = np.where(x_train_df_with_season['Season'] == last_season, 0, -1)

# Creo lo splitter
tscv = TimeSeriesSplit(n_splits=5) 

param_grid = {
    'lasso__alpha': np.logspace(5, 8, 1000) #100 valori di alpha tra 10^5 e 10^7
}

print("Avvio della Grid Search per trovare l'alpha ottimale di Lasso...")
# Usiamo 'neg_mean_absolute_error' perché GridSearchCV cerca di massimizzare
# (massimizzare un MAE negativo = minimizzare il MAE)
grid_search_df = GridSearchCV(
    pipeline, 
    param_grid, 
    cv= tscv, #predefined split
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=1  # Mostra i progressi
)

# Avvia l'addestramento e la ricerca
grid_search_df.fit(x_train_df, y_train_df)

#RISULTATI DELLA SELEZIONE
print("\n--- Risultati Grid Search ---")
best_alpha_df = grid_search_df.best_params_['lasso__alpha']
print(f"Alpha ottimale trovato: {best_alpha_df:.6f}")
print(f"Miglior MAE in Cross-Validation: {-grid_search_df.best_score_:.2f}")

other_columns_names = [col for col in all_features if col != 'age']

best_model_df = grid_search_df.best_estimator_ #'best_model' = modello finale con alpha migliore

#VALUTAZIONE FINALE SUL TEST SET
y_pred_train_df = best_model_df.predict(x_train_df)
y_pred_test_df = best_model_df.predict(x_test_df)

mae_test_df_lasso = mean_absolute_error(y_test_df, y_pred_test_df)
mae_train_df_lasso = mean_absolute_error(y_train_df, y_pred_train_df)
r2_test_df_lasso = r2_score(y_test_df, y_pred_test_df)
r2_train_df_lasso = r2_score(y_train_df, y_pred_train_df)

print("\n--- Performance Modello Finale Ottimizzato ---")
print(f"MAE sul set di TRAINING: {mae_train_df_lasso:.2f}")
print(f"MAE sul set di TEST: {mae_test_df_lasso:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_df_lasso:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_df_lasso:.4f}")

#ANALISI DEI COEFFICIENTI 
# Ricrea l'elenco dei nomi delle feature nell'ordine corretto
other_columns_names = [col for col in all_features if col != 'age']
feature_names_transformed = ['age', 'age^2'] + other_columns_names

# Estrai i coefficienti dal modello ottimale
lasso_regressor_df = best_model_df.named_steps['lasso']
lasso_coef_df = pd.Series(lasso_regressor_df.coef_, index=feature_names_transformed)

# Filtra quelle azzerate
zero_features_df = lasso_coef_df[np.abs(lasso_coef_df) < 1e-6]
relevant_features_df = lasso_coef_df[np.abs(lasso_coef_df) >= 1e-6]

print("\n--- Feature Selection Automatica Eseguita ---")
print(f"Feature totali iniziali: {len(all_features)}")
print(f"Feature eliminate da Lasso (coefficiente ≈ 0): {len(zero_features_df)}")
print(f"Feature mantenute (significative): {len(relevant_features_df)}")

print("\nFeature più importanti (coefficienti più alti):")
print(relevant_features_df.abs().sort_values(ascending=False).head(15))

#Momento di fine del modello
end_time_df_lasso = time.time()
execution_time_df_lasso = end_time_df_lasso - start_time_df_lasso
print(f"--- Tempo di esecuzione Modello LASSO sui Difensori: {execution_time_df_lasso:.2f} secondi ---")

# =============================================================================
# MODELLO DI CONTROLLO DIFENSORI: REGRESSIONE LINEARE SEMPLICE CON VARIABILE RANDOM
# =============================================================================
feature_control = ["random"]
y_column = "value"
print("\n---Inizio Addestramento Regressione Lineare di Controllo per i DF---")

# 1. Preparazione Dati
x_train_df = df_df_1824[feature_control]
y_train_df = df_df_1824[y_column]
x_test_df = df_df_2425[feature_control]
y_test_df = df_df_2425[y_column]

# 2. Definizione del Modello (Senza Pipeline perché non servono iperparametri)
# Usiamo comunque lo scaler per coerenza con gli altri modelli
scaler_x = StandardScaler()
x_train_scaled_df_ctrl = scaler_x.fit_transform(x_train_df)
x_test_scaled_df_ctrl = scaler_x.transform(x_test_df)

model_linear_df_ctrl = LinearRegression()

# 3. Fit del modello
model_linear_df_ctrl.fit(x_train_scaled_df_ctrl, y_train_df)

# 4. Valutazione
y_pred_train_df_ctrl= model_linear_df_ctrl.predict(x_train_scaled_df_ctrl)
y_pred_test_df_ctrl= model_linear_df_ctrl.predict(x_test_scaled_df_ctrl)

mae_test_df_lasso_ctrl  = mean_absolute_error(y_test_df, y_pred_test_df_ctrl)
mae_train_df_lasso_ctrl  = mean_absolute_error(y_train_df, y_pred_train_df_ctrl)

r2_test_df_lasso_ctrl  = r2_score(y_test_df, y_pred_test_df_ctrl)
r2_train_df_lasso_ctrl = r2_score(y_train_df, y_pred_train_df_ctrl)

# 5. Estrazione del coefficiente (molto importante per la tesi!)
coeff_df_random = model_linear_df_ctrl.coef_[0]

print("\n--- RISULTATI REGRESSIONE LINEARE (RANDOM) SUI DF---")
print(f"Coefficiente assegnato a 'random': {coeff_df_random:.4f}")
print(f"MAE Test: {mae_test_df_lasso_ctrl:.2f} M€")
print(f"MAE Train: {mae_train_df_lasso_ctrl:.2f} M€")
print(f"R2 Train: {r2_train_df_lasso_ctrl:.4f}")
print(f"R2 Test:  {r2_test_df_lasso_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.1.3 - LASSO WINGBACK
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column= "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]

x_train_wb = df_wb_1824[all_features]
y_train_wb = df_wb_1824[y_column]
x_test_wb = df_wb_2425[all_features]
y_test_wb = df_wb_2425[y_column]

#Momento di inizio del modello
start_time_wb_lasso = time.time()

try:
    age_index = all_features.index('age')
except ValueError:
    print("ERRORE: La colonna 'age' non è presente nelle tue features filtrate.")

age_column_index = [age_index]
other_columns_indices = [i for i, col in enumerate(all_features) if col != 'age']

# Definisci il trasformatore
feature_transformer = ColumnTransformer(
    transformers=[
        ('age_poly', PolynomialFeatures(degree=2, include_bias=False), age_column_index)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)), #RIDONDANTE
    ('transformer', feature_transformer),
    ('scaler', StandardScaler()),
    ('lasso', Lasso(random_state=42, max_iter=10000)) # Il nostro modello da ottimizzare
])

#SELEZIONE AUTOMATICA (GridSearchCV) ---
x_train_wb_with_season = df_wb_1824[all_features + ['Season']]

# Identifica l'ultima stagione nel set di training
last_season = x_train_wb_with_season['Season'].max() # '2023/2024'

print(f"Creazione split di validazione: validazione su stagione {last_season}")
validation_fold = np.where(x_train_wb_with_season['Season'] == last_season, 0, -1)

# Crea lo splitter
tscv = TimeSeriesSplit(n_splits=5) 

param_grid = {
    'lasso__alpha': np.logspace(5, 8, 1000) #100 valori di alpha tra 10^5 e 10^7
}

print("Avvio della Grid Search per trovare l'alpha ottimale di Lasso...")
grid_search_wb = GridSearchCV(
    pipeline, 
    param_grid, 
    cv= tscv, #predefined split
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=1  # Mostra i progressi
)

# Avvia l'addestramento e la ricerca
grid_search_wb.fit(x_train_wb, y_train_wb)

#RISULTATI DELLA SELEZIONE
print("\n--- Risultati Grid Search ---")
best_alpha_wb = grid_search_wb.best_params_['lasso__alpha']
print(f"Alpha ottimale trovato: {best_alpha_wb:.6f}")
print(f"Miglior MAE in Cross-Validation: {-grid_search_wb.best_score_:.2f}")

other_columns_names = [col for col in all_features if col != 'age']

best_model_wb = grid_search_wb.best_estimator_ #'best_model' = modello finale con alpha migliore

#VALUTAZIONE FINALE SUL TEST SET
y_pred_train_wb = best_model_wb.predict(x_train_wb)
y_pred_test_wb = best_model_wb.predict(x_test_wb)

mae_test_wb_lasso = mean_absolute_error(y_test_wb, y_pred_test_wb)
mae_train_wb_lasso = mean_absolute_error(y_train_wb, y_pred_train_wb)
r2_test_wb_lasso = r2_score(y_test_wb, y_pred_test_wb)
r2_train_wb_lasso = r2_score(y_train_wb, y_pred_train_wb)

print("\n--- Performance Modello Finale Ottimizzato ---")
print(f"MAE sul set di TRAINING: {mae_train_wb_lasso:.2f}")
print(f"MAE sul set di TEST: {mae_test_wb_lasso:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_wb_lasso:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_wb_lasso:.4f}")

#ANALISI DEI COEFFICIENTI 
# Ricrea l'elenco dei nomi delle feature nell'ordine corretto
other_columns_names = [col for col in all_features if col != 'age']
feature_names_transformed = ['age', 'age^2'] + other_columns_names

# Estraggo i coefficienti dal modello ottimale
lasso_regressor_wb = best_model_wb.named_steps['lasso']
lasso_coef_wb = pd.Series(lasso_regressor_wb.coef_, index=feature_names_transformed)

# Filtro variabili azzerate
zero_features_wb = lasso_coef_wb[np.abs(lasso_coef_wb) < 1e-6]
relevant_features_wb = lasso_coef_wb[np.abs(lasso_coef_wb) >= 1e-6]

print("\n--- Feature Selection Automatica Eseguita ---")
print(f"Feature totali iniziali: {len(all_features)}")
print(f"Feature eliminate da Lasso (coefficiente ≈ 0): {len(zero_features_wb)}")
print(f"Feature mantenute (significative): {len(relevant_features_wb)}")

print("\nFeature più importanti (coefficienti più alti):")
print(relevant_features_wb.abs().sort_values(ascending=False).head(20))

#Momento di fine del modello
end_time_wb_lasso = time.time()
execution_time_wb_lasso = end_time_wb_lasso - start_time_wb_lasso
print(f"--- Tempo di esecuzione Modello LASSO sui Wingback: {execution_time_wb_lasso:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO LASSO WINGBACK(USO REGRESSIONE LINEARE)
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

x_train_wb = df_wb_1824[feature_control]
y_train_wb = df_wb_1824[y_column]
x_test_wb = df_wb_2425[feature_control]
y_test_wb = df_wb_2425[y_column]

x_train_with_season = df_wb_1824[feature_control + ['Season']]
tscv = TimeSeriesSplit(n_splits=5) 

pipeline_control_wb = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(random_state=42, max_iter=10000)) 
])

param_grid = {
    'lasso__alpha': np.logspace(5, 7, 20) 
}

grid_search_wb_ctrl = GridSearchCV(
    pipeline_control_wb, 
    param_grid, 
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1 
)

grid_search_wb_ctrl.fit(x_train_wb, y_train_wb)

#VALUTAZIONE
best_model_wb_ctrl = grid_search_wb_ctrl.best_estimator_

y_pred_train_wb_ctrl = best_model_wb_ctrl.predict(x_train_wb)
y_pred_test_wb_ctrl = best_model_wb_ctrl.predict(x_test_wb)

r2_test_wb_lasso_ctrl = r2_score(y_test_wb, y_pred_test_wb_ctrl)
r2_train_wb_lasso_ctrl = r2_score(y_train_wb, y_pred_train_wb_ctrl)
mae_train_wb_lasso_ctrl = mean_absolute_error(y_train_wb, y_pred_train_wb_ctrl)
mae_test_wb_lasso_ctrl = mean_absolute_error(y_test_wb, y_pred_test_wb_ctrl)

print("RISULTATI MODELLO DI CONTROLLO (RANDOM) - WINGBACK")
print(f"Alpha ottimale scelto: {grid_search_wb_ctrl.best_params_['lasso__alpha']:.2f}")
print(f"MAE Train: {mae_train_wb_lasso_ctrl:.2f}")
print(f"MAE Test: {mae_test_wb_lasso_ctrl:.2f}")
print(f"R2 Train:  {r2_train_wb_lasso_ctrl:.4f}")
print(f"R2 Test:  {r2_test_wb_lasso_ctrl:.4f}")

# =============================================================================
# MODELLO DI CONTROLLO WINGBACK: REGRESSIONE LINEARE SEMPLICE CON VARIABILE RANDOM
# =============================================================================
feature_control = ["random"]
y_column = "value"
print("\n---Inizio Addestramento Regressione Lineare di Controllo per i WB---")

# 1. Preparazione Dati
x_train_wb = df_wb_1824[feature_control]
y_train_wb = df_wb_1824[y_column]
x_test_wb = df_wb_2425[feature_control]
y_test_wb = df_wb_2425[y_column]

# 2. Definizione del Modello (Senza Pipeline perché non servono iperparametri)
# Usiamo comunque lo scaler per coerenza con gli altri modelli
scaler_x = StandardScaler()
x_train_scaled_wb_ctrl = scaler_x.fit_transform(x_train_wb)
x_test_scaled_wb_ctrl = scaler_x.transform(x_test_wb)

model_linear_wb_ctrl = LinearRegression()

# 3. Fit del modello
model_linear_wb_ctrl.fit(x_train_scaled_wb_ctrl, y_train_wb)

# 4. Valutazione
y_pred_train_wb_ctrl= model_linear_wb_ctrl.predict(x_train_scaled_wb_ctrl)
y_pred_test_wb_ctrl= model_linear_wb_ctrl.predict(x_test_scaled_wb_ctrl)

mae_test_wb_lasso_ctrl  = mean_absolute_error(y_test_wb, y_pred_test_wb_ctrl)
mae_train_wb_lasso_ctrl  = mean_absolute_error(y_train_wb, y_pred_train_wb_ctrl)

r2_test_wb_lasso_ctrl  = r2_score(y_test_wb, y_pred_test_wb_ctrl)
r2_train_wb_lasso_ctrl = r2_score(y_train_wb, y_pred_train_wb_ctrl)

# 5. Estrazione del coefficiente (molto importante per la tesi!)
coeff_wb_random = model_linear_wb_ctrl.coef_[0]

print("\n--- RISULTATI REGRESSIONE LINEARE (RANDOM) SUI WB---")
print(f"Coefficiente assegnato a 'random': {coeff_wb_random:.4f}")
print(f"MAE Test: {mae_test_wb_lasso_ctrl:.2f} M€")
print(f"MAE Train: {mae_train_wb_lasso_ctrl:.2f} M€")
print(f"R2 Train: {r2_train_wb_lasso_ctrl:.4f}")
print(f"R2 Test:  {r2_test_wb_lasso_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.1.4 - LASSO CENTROCAMPISTI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column= "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]

x_train_mf = df_mf_1824[all_features]
y_train_mf = df_mf_1824[y_column]
x_test_mf = df_mf_2425[all_features]
y_test_mf = df_mf_2425[y_column]

#Momento di inizio del modello
start_time_mf_lasso = time.time()

try:
    age_index = all_features.index('age')
except ValueError:
    print("ERRORE: La colonna 'age' non è presente nelle tue features filtrate.")

age_column_index = [age_index]
other_columns_indices = [i for i, col in enumerate(all_features) if col != 'age']

# Definisci il trasformatore
feature_transformer = ColumnTransformer(
    transformers=[
        ('age_poly', PolynomialFeatures(degree=2, include_bias=False), age_column_index)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)), #RIDONDANTE
    ('transformer', feature_transformer),
    ('scaler', StandardScaler()),
    ('lasso', Lasso(random_state=42, max_iter=10000)) # Il nostro modello da ottimizzare
])

#SELEZIONE AUTOMATICA (GridSearchCV)
x_train_mf_with_season = df_mf_1824[all_features + ['Season']]

# Identifico l'ultima stagione nel set di training
last_season = x_train_mf_with_season['Season'].max() # '2023/2024'

print(f"Creazione split di validazione: validazione su stagione {last_season}")
validation_fold = np.where(x_train_mf_with_season['Season'] == last_season, 0, -1)

# Creo lo splitter
tscv = TimeSeriesSplit(n_splits=5) 

param_grid = {
    'lasso__alpha': np.logspace(5, 8, 1000) #1000 valori di alpha tra 10^5 e 10^8
}

print("Avvio della Grid Search per trovare l'alpha ottimale di Lasso...")
grid_search_mf = GridSearchCV(
    pipeline, 
    param_grid, 
    cv= tscv, #predefined split
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=1  # Mostra i progressi
)

# Avvio l'addestramento e la ricerca
grid_search_mf.fit(x_train_mf, y_train_mf)

#RISULTATI DELLA SELEZIONE
print("\n--- Risultati Grid Search ---")
best_alpha_mf = grid_search_mf.best_params_['lasso__alpha']
print(f"Alpha ottimale trovato: {best_alpha_mf:.6f}")
print(f"Miglior MAE in Cross-Validation: {-grid_search_mf.best_score_:.2f}")

other_columns_names = [col for col in all_features if col != 'age']

best_model_mf = grid_search_mf.best_estimator_ #'best_model' = modello finale con alpha migliore

#VALUTAZIONE FINALE SUL TEST SET
y_pred_train_mf = best_model_mf.predict(x_train_mf)
y_pred_test_mf = best_model_mf.predict(x_test_mf)

mae_test_mf_lasso = mean_absolute_error(y_test_mf, y_pred_test_mf)
mae_train_mf_lasso = mean_absolute_error(y_train_mf, y_pred_train_mf)
r2_test_mf_lasso = r2_score(y_test_mf, y_pred_test_mf)
r2_train_mf_lasso = r2_score(y_train_mf, y_pred_train_mf)

print("\n--- Performance Modello Finale Ottimizzato ---")
print(f"MAE sul set di TRAINING: {mae_train_mf_lasso:.2f}")
print(f"MAE sul set di TEST: {mae_test_mf_lasso:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_mf_lasso:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_mf_lasso:.4f}")

#ANALISI DEI COEFFICIENTI 
# Ricreo l'elenco dei nomi delle feature nell'ordine corretto
other_columns_names = [col for col in all_features if col != 'age']
feature_names_transformed = ['age', 'age^2'] + other_columns_names

# Estraggo i coefficienti dal modello ottimale
lasso_regressor_mf = best_model_mf.named_steps['lasso']
lasso_coef_mf = pd.Series(lasso_regressor_mf.coef_, index=feature_names_transformed)

# Filtro variabili azzerate
zero_features_mf = lasso_coef_mf[np.abs(lasso_coef_mf) < 1e-6]
relevant_features_mf = lasso_coef_mf[np.abs(lasso_coef_mf) >= 1e-6]

print("\n--- Feature Selection Automatica Eseguita ---")
print(f"Feature totali iniziali: {len(all_features)}")
print(f"Feature eliminate da Lasso (coefficiente ≈ 0): {len(zero_features_mf)}")
print(f"Feature mantenute (significative): {len(relevant_features_mf)}")

print("\nFeature più importanti (coefficienti più alti):")
print(relevant_features_mf.abs().sort_values(ascending=False).head(20))

#Momento di fine del modello
end_time_mf_lasso = time.time()
execution_time_mf_lasso = end_time_mf_lasso - start_time_mf_lasso
print(f"--- Tempo di esecuzione Modello LASSO sui Centrocampisti: {execution_time_mf_lasso:.2f} secondi ---")
# =============================================================================
# MODELLO DI CONTROLLO CENTROCAMPISTI: REGRESSIONE LINEARE SEMPLICE CON VARIABILE RANDOM
# =============================================================================
feature_control = ["random"]
y_column = "value"
print("\n---Inizio Addestramento Regressione Lineare di Controllo per i MF---")

# 1. Preparazione Dati
x_train_mf = df_mf_1824[feature_control]
y_train_mf = df_mf_1824[y_column]
x_test_mf = df_mf_2425[feature_control]
y_test_mf = df_mf_2425[y_column]

# 2. Definizione del Modello (Senza Pipeline perché non servono iperparametri)
# Usiamo comunque lo scaler per coerenza con gli altri modelli
scaler_x = StandardScaler()
x_train_scaled_mf_ctrl = scaler_x.fit_transform(x_train_mf)
x_test_scaled_mf_ctrl = scaler_x.transform(x_test_mf)

model_linear_mf_ctrl = LinearRegression()

# 3. Fit del modello
model_linear_mf_ctrl.fit(x_train_scaled_mf_ctrl, y_train_mf)

# 4. Valutazione
y_pred_train_mf_ctrl= model_linear_mf_ctrl.predict(x_train_scaled_mf_ctrl)
y_pred_test_mf_ctrl= model_linear_mf_ctrl.predict(x_test_scaled_mf_ctrl)

mae_test_mf_lasso_ctrl  = mean_absolute_error(y_test_mf, y_pred_test_mf_ctrl)
mae_train_mf_lasso_ctrl  = mean_absolute_error(y_train_mf, y_pred_train_mf_ctrl)

r2_test_mf_lasso_ctrl  = r2_score(y_test_mf, y_pred_test_mf_ctrl)
r2_train_mf_lasso_ctrl = r2_score(y_train_mf, y_pred_train_mf_ctrl)

# 5. Estrazione del coefficiente (molto importante per la tesi!)
coeff_mf_random = model_linear_mf_ctrl.coef_[0]

print("\n--- RISULTATI REGRESSIONE LINEARE (RANDOM) SUI MF---")
print(f"Coefficiente assegnato a 'random': {coeff_mf_random:.4f}")
print(f"MAE Test: {mae_test_mf_lasso_ctrl:.2f} M€")
print(f"MAE Train: {mae_train_mf_lasso_ctrl:.2f} M€")
print(f"R2 Train: {r2_train_mf_lasso_ctrl:.4f}")
print(f"R2 Test:  {r2_test_mf_lasso_ctrl:.4f}")

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.1.5 - LASSO ATTACCANTI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column= "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]

x_train_fw = df_fw_1824[all_features]
y_train_fw = df_fw_1824[y_column]
x_test_fw = df_fw_2425[all_features]
y_test_fw = df_fw_2425[y_column]

#Momento di inizio del modello
start_time_fw_lasso = time.time()

try:
    age_index = all_features.index('age')
except ValueError:
    print("ERRORE: La colonna 'age' non è presente nelle tue features filtrate.")

age_column_index = [age_index]
other_columns_indices = [i for i, col in enumerate(all_features) if col != 'age']

# Definisci il trasformatore
feature_transformer = ColumnTransformer(
    transformers=[
        ('age_poly', PolynomialFeatures(degree=2, include_bias=False), age_column_index)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)), #RIDONDANTE
    ('transformer', feature_transformer),
    ('scaler', StandardScaler()),
    ('lasso', Lasso(random_state=42, max_iter=10000)) # Il nostro modello da ottimizzare
])

#SELEZIONE AUTOMATICA (GridSearchCV)
x_train_fw_with_season = df_fw_1824[all_features + ['Season']]

# Identifico l'ultima stagione nel set di training
last_season = x_train_fw_with_season['Season'].max() # '2023/2024'

print(f"Creazione split di validazione: validazione su stagione {last_season}")
validation_fold = np.where(x_train_fw_with_season['Season'] == last_season, 0, -1)

# Creo lo splitter
tscv = TimeSeriesSplit(n_splits=5) 

param_grid = {
    'lasso__alpha': np.logspace(5, 8, 1000) #100 valori di alpha tra 10^5 e 10^7
}

print("Avvio della Grid Search per trovare l'alpha ottimale di Lasso...")
grid_search_fw = GridSearchCV(
    pipeline, 
    param_grid, 
    cv= tscv, #predefined split
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=1  # Mostra i progressi
)

# Avvia l'addestramento e la ricerca
grid_search_fw.fit(x_train_fw, y_train_fw)

#RISULTATI DELLA SELEZIONE ---
print("\n--- Risultati Grid Search ---")
best_alpha_fw = grid_search_fw.best_params_['lasso__alpha']
print(f"Alpha ottimale trovato: {best_alpha_fw:.6f}")
print(f"Miglior MAE in Cross-Validation: {-grid_search_fw.best_score_:.2f}")

other_columns_names = [col for col in all_features if col != 'age']

best_model_fw = grid_search_fw.best_estimator_ #'best_model' = modello finale con alpha migliore

#VALUTAZIONE FINALE SUL TEST SET
y_pred_train_fw = best_model_fw.predict(x_train_fw)
y_pred_test_fw = best_model_fw.predict(x_test_fw)

mae_test_fw_lasso = mean_absolute_error(y_test_fw, y_pred_test_fw)
mae_train_fw_lasso = mean_absolute_error(y_train_fw, y_pred_train_fw)
r2_test_fw_lasso = r2_score(y_test_fw, y_pred_test_fw)
r2_train_fw_lasso = r2_score(y_train_fw, y_pred_train_fw)

print("\n--- Performance Modello Finale Ottimizzato ---")
print(f"MAE sul set di TRAINING: {mae_train_fw_lasso:.2f}")
print(f"MAE sul set di TEST: {mae_test_fw_lasso:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_fw_lasso:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_fw_lasso:.4f}")

#ANALISI DEI COEFFICIENTI 
# Ricreo l'elenco dei nomi delle feature nell'ordine corretto
other_columns_names = [col for col in all_features if col != 'age']
feature_names_transformed = ['age', 'age^2'] + other_columns_names

# Estraggo i coefficienti dal modello ottimale
lasso_regressor_fw = best_model_fw.named_steps['lasso']
lasso_coef_fw = pd.Series(lasso_regressor_fw.coef_, index=feature_names_transformed)

# Filtro quelle azzerate
zero_features_fw = lasso_coef_fw[np.abs(lasso_coef_fw) < 1e-6]
relevant_features_fw = lasso_coef_fw[np.abs(lasso_coef_fw) >= 1e-6]

print("\n--- Feature Selection Automatica Eseguita ---")
print(f"Feature totali iniziali: {len(all_features)}")
print(f"Feature eliminate da Lasso (coefficiente ≈ 0): {len(zero_features_fw)}")
print(f"Feature mantenute (significative): {len(relevant_features_fw)}")

print("\nFeature più importanti (coefficienti più alti):")
print(relevant_features_fw.abs().sort_values(ascending=False).head(25))

#Momento di fine del modello
end_time_fw_lasso = time.time()
execution_time_fw_lasso = end_time_fw_lasso - start_time_fw_lasso
print(f"--- Tempo di esecuzione Modello LASSO sugli Attaccanti: {execution_time_fw_lasso:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO LASSO ATTACCANTI(USO REGRESSIONE LINEARE)
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

x_train_fw = df_fw_1824[feature_control]
y_train_fw = df_fw_1824[y_column]
x_test_fw = df_fw_2425[feature_control]
y_test_fw = df_fw_2425[y_column]

x_train_with_season = df_fw_1824[feature_control + ['Season']]
tscv = TimeSeriesSplit(n_splits=5) 

pipeline_control_fw = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(random_state=42, max_iter=10000)) 
])

param_grid = {
    'lasso__alpha': np.logspace(5, 7, 20) 
}

grid_search_fw_ctrl = GridSearchCV(
    pipeline_control_fw, 
    param_grid, 
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1 
)

grid_search_fw_ctrl.fit(x_train_fw, y_train_fw)

#VALUTAZIONE ---
best_model_fw_ctrl = grid_search_fw_ctrl.best_estimator_

y_pred_train_fw_ctrl = best_model_fw_ctrl.predict(x_train_fw)
y_pred_test_fw_ctrl = best_model_fw_ctrl.predict(x_test_fw)

r2_test_fw_lasso_ctrl = r2_score(y_test_fw, y_pred_test_fw_ctrl)
r2_train_fw_lasso_ctrl = r2_score(y_train_fw, y_pred_train_fw_ctrl)
mae_train_fw_lasso_ctrl = mean_absolute_error(y_train_fw, y_pred_train_fw_ctrl)
mae_test_fw_lasso_ctrl = mean_absolute_error(y_test_fw, y_pred_test_fw_ctrl)

print("RISULTATI MODELLO DI CONTROLLO (RANDOM) - ATTACCANTI")
print(f"Alpha ottimale scelto: {grid_search_fw_ctrl.best_params_['lasso__alpha']:.2f}")
print(f"MAE Train: {mae_train_fw_lasso_ctrl:.2f}")
print(f"MAE Test: {mae_test_fw_lasso_ctrl:.2f}")
print(f"R2 Train:  {r2_train_fw_lasso_ctrl:.4f}")
print(f"R2 Test:  {r2_test_fw_lasso_ctrl:.4f}")
# =============================================================================
# MODELLO DI CONTROLLO ATTACCANTI: REGRESSIONE LINEARE SEMPLICE CON VARIABILE RANDOM
# =============================================================================
feature_control = ["random"]
y_column = "value"
print("\n---Inizio Addestramento Regressione Lineare di Controllo per i FW---")

# 1. Preparazione Dati
x_train_fw = df_fw_1824[feature_control]
y_train_fw = df_fw_1824[y_column]
x_test_fw = df_fw_2425[feature_control]
y_test_fw = df_fw_2425[y_column]

# 2. Definizione del Modello (Senza Pipeline perché non servono iperparametri)
# Usiamo comunque lo scaler per coerenza con gli altri modelli
scaler_x = StandardScaler()
x_train_scaled_fw_ctrl = scaler_x.fit_transform(x_train_fw)
x_test_scaled_fw_ctrl = scaler_x.transform(x_test_fw)

model_linear_fw_ctrl = LinearRegression()

# 3. Fit del modello
model_linear_fw_ctrl.fit(x_train_scaled_fw_ctrl, y_train_fw)

# 4. Valutazione
y_pred_train_fw_ctrl= model_linear_fw_ctrl.predict(x_train_scaled_fw_ctrl)
y_pred_test_fw_ctrl= model_linear_fw_ctrl.predict(x_test_scaled_fw_ctrl)

mae_test_fw_lasso_ctrl  = mean_absolute_error(y_test_fw, y_pred_test_fw_ctrl)
mae_train_fw_lasso_ctrl  = mean_absolute_error(y_train_fw, y_pred_train_fw_ctrl)

r2_test_fw_lasso_ctrl  = r2_score(y_test_fw, y_pred_test_fw_ctrl)
r2_train_fw_lasso_ctrl = r2_score(y_train_fw, y_pred_train_fw_ctrl)

# 5. Estrazione del coefficiente (molto importante per la tesi!)
coeff_fw_random = model_linear_fw_ctrl.coef_[0]

print("\n--- RISULTATI REGRESSIONE LINEARE (RANDOM) SUI FW---")
print(f"Coefficiente assegnato a 'random': {coeff_fw_random:.4f}")
print(f"MAE Test: {mae_test_fw_lasso_ctrl:.2f} M€")
print(f"MAE Train: {mae_train_fw_lasso_ctrl:.2f} M€")
print(f"R2 Train: {r2_train_fw_lasso_ctrl:.4f}")
print(f"R2 Test:  {r2_test_fw_lasso_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.2 - SUPPORT VECTOR REGRESSION (SVR) - KERNEL RBF
#La SVR (Regressione ai vettori di supporto) è un algoritmo utilizzato per le 
#regression tasks che cerca di adattarsi al maggior numero possibile di punti
#all'interno di un margine di tolleranza epsilon.
#Nel mio caso ho utilizzato il kernel RBF.

#Funzione di scoring personalizzata per la Permutation Importance
def unscaled_mae_scorer_func(model, X, y_true_unscaled):
    """
    Scorer personalizzato:
    1. Prende le previsioni SCALATE dal modello
    2. Le de-scala (inverse_transform)
    3. Calcola il MAE contro la y REALE (non scalata)
    4. Restituisce il MAE NEGATIVO (perché permutation_importance massimizza)
    """
    y_pred_scaled = model.predict(X)
    
    # 1. Riporta le previsioni in euro (milioni)
    # Dobbiamo usare lo scaler 'y_scaler_fw' che è nello scope globale
    y_pred_unscaled = y_scaler_fw.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # 2. Calcola il MAE in euro
    mae_in_euros = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    
    # 3. Restituisci il NEGATIVO del MAE
    return -mae_in_euros 
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.2.1 - SVR PORTIERI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento SVR con GridSearchCV su {len(all_features)} feature.")

x_train_gk = df_gk_1824[all_features]
y_train_gk = df_gk_1824[y_column]
x_test_gk = df_gk_2425[all_features]
y_test_gk = df_gk_2425[y_column]

#Momento di inizio del modello
start_time_gk_svr = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Obbligatorio per SVR
y_scaler_gk = StandardScaler()
y_train_scaled_gk_svr = y_scaler_gk.fit_transform(y_train_gk.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Semplificata) ---
# Il kernel RBF gestirà la non-linearità di 'age' automaticamente
pipeline_svr = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), # Scaler ora scala tutte le 216 feature
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---

# Definiamo la GRIGLIA di iperparametri da testare
# Questa è una griglia relativamente piccola per contenere i tempi.
# Se i risultati sono ai bordi della griglia, dovrai espanderla.

#param_grid = {
#    'svr__C': [10, 11, 12, 100, 1000],          # 6 valori
#    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
#    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7]      # 6 valori
#}

#Solo per dataset coin solo "age", no "born"
param_grid = {
   'svr__C': [6, 7, 8, 9, 10, 11],         
    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],      
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6]     
}
#Combinazioni totali: 7 * 5 * 5 = 175

# Usiamo lo stesso TimeSeriesSplit per correttezza metodologica
tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV rigorosa per SVR (kernel RBF)...")
print(f"Totale combinazioni: {len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])}")
print(f"Totale modelli da addestrare: {(len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])) * tscv.get_n_splits()}")

grid_search_gk_svr = GridSearchCV(
    pipeline_svr,
    param_grid=param_grid, # Usa la griglia definita
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=2  # Mostra i progressi
)

# FIT su X_train e Y_TRAIN_SCALED
grid_search_gk_svr.fit(x_train_gk, y_train_scaled_gk_svr)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV SVR ---")
print(f"Migliori iperparametri trovati: {grid_search_gk_svr.best_params_}")
#print(f"Miglior MAE Scalato in CV: {-grid_search_gk_svr.best_score_:.4f}")

best_model_gk_svr = grid_search_gk_svr.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---

y_pred_train_scaled_gk_svr = best_model_gk_svr.predict(x_train_gk)
y_pred_test_scaled_gk_svr = best_model_gk_svr.predict(x_test_gk)

y_pred_train_final_gk_svr = y_scaler_gk.inverse_transform(y_pred_train_scaled_gk_svr.reshape(-1, 1))
y_pred_test_final_gk_svr = y_scaler_gk.inverse_transform(y_pred_test_scaled_gk_svr.reshape(-1, 1))

mae_test_gk_svr = mean_absolute_error(y_test_gk, y_pred_test_final_gk_svr)
mae_train_gk_svr = mean_absolute_error(y_train_gk, y_pred_train_final_gk_svr)
r2_test_gk_svr = r2_score(y_test_gk, y_pred_test_final_gk_svr)
r2_train_gk_svr = r2_score(y_train_gk, y_pred_train_final_gk_svr)

print("\n--- Performance Modello SVR Ottimizzato (Portieri) ---")
print(f"MAE sul set di TRAINING: {mae_train_gk_svr:.2f}")
print(f"MAE sul set di TEST: {mae_test_gk_svr:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_gk_svr:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_gk_svr:.4f}")

print("\n--- Confronto Lasso vs SVR RBF per Portieri  ---")
print(f"Lasso (Portieri):           R2 Test = {r2_test_gk_lasso:.4f}")
print(f"SVR RBF (Portieri):         R2 Test = {r2_test_gk_svr:.4f}")

#PERMUTATION IMPORTANCE GK
print("\nAvvio calcolo Permutation Importance per i GK")

# n_repeats=10 rende il calcolo più robusto (esegue 10 shuffle per feature)
perm_imp_gk_svr = permutation_importance(
    best_model_gk_svr, 
    x_test_gk, 
    y_test_gk, 
    n_repeats=10,
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_func # Puoi anche usare 'neg_mean_absolute error'
)

# --- 3. Organizza e Stampa i Risultati ---
# 'perm_imp_mf_svr.importances_mean' è il calo medio di R² per ogni feature
importance_gk_svr = pd.DataFrame({
    'feature': x_train_gk.columns, # Assumendo che x_train sia un DataFrame
    'importance_mean': perm_imp_gk_svr.importances_mean,
    'importance_std': perm_imp_gk_svr.importances_std
})

# Ordina per importanza
importance_gk_svr = importance_gk_svr.sort_values('importance_mean', ascending=False)

#SE USO MAE: l'output "importance_mean" è è l'aumento (in milioni di euro) 
#dell'errore MAE che subisci se quella feature viene rimossa o rotta.
#SE USO R2: output "importance_mean" è il calo medio di r2 nel caso la feature 
#venga "rotta" e rimescolata
print("\n--- Feature Importance dei GK per SVR ---")
print(importance_gk_svr.head(10))

#Momento di fine del modello
end_time_gk_svr = time.time()
execution_time_gk_svr = end_time_gk_svr - start_time_gk_svr
print(f"--- Tempo di esecuzione Modello SVR sui Portieri: {execution_time_gk_svr:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO SVR-RBF PORTIERI(USO SVR)
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento SVR di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_gk = df_gk_1824[feature_control]
y_train_gk = df_gk_1824[y_column] 
x_test_gk = df_gk_2425[feature_control]
y_test_gk = df_gk_2425[y_column] 

# --- 2. SCALARE LA Y (TARGET) ---
y_scaler_gk_svr_ctrl = StandardScaler()
y_train_scaled_gk_svr_ctrl = y_scaler_gk_svr_ctrl.fit_transform(y_train_gk.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Identica per coerenza) ---
pipeline_gk_svr_ctrl = Pipeline([
    ('scaler', StandardScaler()), 
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (Identica per coerenza) ---
# Usiamo la stessa identica griglia 6x6x6 per un confronto 1:1
param_grid_gk_svr_ctrl = {
    'svr__C': [10, 11, 12, 100, 1000],          # 6 valori
    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7]      # 6 valori 
} 
# Combinazioni totali: 216

tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV (Benchmark SVR Random)...")

grid_search_gk_svr_ctrl = GridSearchCV(
    pipeline_gk_svr_ctrl,
    param_grid=param_grid_gk_svr_ctrl, 
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1 # Meno verboso del modello reale
)

# FIT su X_train (solo random) e Y_TRAIN_SCALED
grid_search_gk_svr_ctrl.fit(x_train_gk, y_train_scaled_gk_svr_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random) ---")
print(f"Migliori iperparametri trovati: {grid_search_gk_svr_ctrl.best_params_}")

best_model_gk_svr_ctrl = grid_search_gk_svr_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
y_pred_train_scaled_gk_svr_ctrl = best_model_gk_svr_ctrl.predict(x_train_gk)
y_pred_test_scaled_gk_svr_ctrl = best_model_gk_svr_ctrl.predict(x_test_gk)

y_pred_train_gk_svr_ctrl = y_scaler_gk_svr_ctrl.inverse_transform(y_pred_train_scaled_gk_svr_ctrl.reshape(-1, 1))
y_pred_test_gk_svr_ctrl = y_scaler_gk_svr_ctrl.inverse_transform(y_pred_test_scaled_gk_svr_ctrl.reshape(-1, 1))

mae_test_gk_svr_ctrl = mean_absolute_error(y_test_gk, y_pred_test_gk_svr_ctrl)
mae_train_gk_svr_ctrl = mean_absolute_error(y_train_gk, y_pred_train_gk_svr_ctrl)
r2_test_gk_svr_ctrl = r2_score(y_test_gk, y_pred_test_gk_svr_ctrl)
r2_train_gk_svr_ctrl = r2_score(y_train_gk, y_pred_train_gk_svr_ctrl)

print("RISULTATI MODELLO DI CONTROLLO SVR (RANDOM) - PORTIERI")
print(f"MAE sul set di TRAINING: {mae_train_gk_svr_ctrl:.2f}")
print(f"MAE sul set di TEST: {mae_test_gk_svr_ctrl:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_gk_svr_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_gk_svr_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.2.2 - SVR DIFENSORI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# Definisci X e Y (usando tutte le feature tranne 'value' e 'Season')
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento SVR con GridSearchCV su {len(all_features)} feature.")

x_train_df = df_df_1824[all_features]
y_train_df = df_df_1824[y_column]
x_test_df = df_df_2425[all_features]
y_test_df = df_df_2425[y_column]

#Momento di inizio del modello
start_time_df_svr = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Obbligatorio per SVR
y_scaler_df = StandardScaler()
y_train_scaled_df_svr = y_scaler_df.fit_transform(y_train_df.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Semplificata) ---
# Il kernel RBF gestirà la non-linearità di 'age' automaticamente
pipeline_svr = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), # Scaler ora scala tutte le 216 feature
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---

# Definiamo la GRIGLIA di iperparametri da testare
# Questa è una griglia relativamente piccola per contenere i tempi.
# Se i risultati sono ai bordi della griglia, dovrai espanderla.
param_grid = {
    'svr__C': [10, 11, 12],          # 6 valori
    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7]      # 6 valori
}


# Usiamo lo stesso TimeSeriesSplit per correttezza metodologica
tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV rigorosa per SVR (kernel RBF)...")
print(f"Totale combinazioni: {len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])}")
print(f"Totale modelli da addestrare: {(len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])) * tscv.get_n_splits()}")

grid_search_df_svr = GridSearchCV(
    pipeline_svr,
    param_grid=param_grid, # Usa la griglia definita
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=2  # Mostra i progressi
)

# FIT su X_train e Y_TRAIN_SCALED
grid_search_df_svr.fit(x_train_df, y_train_scaled_df_svr)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV SVR ---")
print(f"Migliori iperparametri trovati: {grid_search_df_svr.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_df_svr.best_score_:.4f}")

best_model_df_svr = grid_search_df_svr.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---

y_pred_train_scaled_df_svr = best_model_df_svr.predict(x_train_df)
y_pred_test_scaled_df_svr = best_model_df_svr.predict(x_test_df)

y_pred_train_final_df_svr = y_scaler_df.inverse_transform(y_pred_train_scaled_df_svr.reshape(-1, 1))
y_pred_test_final_df_svr = y_scaler_df.inverse_transform(y_pred_test_scaled_df_svr.reshape(-1, 1))

mae_test_df_svr = mean_absolute_error(y_test_df, y_pred_test_final_df_svr)
mae_train_df_svr = mean_absolute_error(y_train_df, y_pred_train_final_df_svr)
r2_test_df_svr = r2_score(y_test_df, y_pred_test_final_df_svr)
r2_train_df_svr = r2_score(y_train_df, y_pred_train_final_df_svr)

print("\n--- Performance Modello SVR Ottimizzato (Difensori) ---")
print(f"MAE sul set di TRAINING: {mae_train_df_svr:.2f}")
print(f"MAE sul set di TEST: {mae_test_df_svr:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_df_svr:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_df_svr:.4f}")

print("\n--- Confronto Lasso vs SVR ---")
print(f"Lasso (Difensori):           R2 Test = {r2_test_df_lasso:.4f}")
print(f"SVR RBF (Difensori):         R2 Test = {r2_test_df_svr:.4f}")

print("\nAvvio calcolo Permutation Importance per i DF")
# Usiamo R² come metrica di scoring
# n_repeats=10 rende il calcolo più robusto (esegue 10 shuffle per feature)
perm_imp_df_svr = permutation_importance(
    best_model_df_svr, 
    x_test_df, 
    y_test_df, 
    n_repeats=20,
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_func # Puoi anche usare 'neg_mean_absolute_error'
)

# --- 3. Organizza e Stampa i Risultati ---
# 'perm_imp_mf_svr.importances_mean' è il calo medio di R² per ogni feature
importance_df_svr = pd.DataFrame({
    'feature': x_train_df.columns, # Assumendo che x_train sia un DataFrame
    'importance_mean': perm_imp_df_svr.importances_mean,
    'importance_std': perm_imp_df_svr.importances_std
})

# Ordina per importanza
importance_df_svr = importance_df_svr.sort_values('importance_mean', ascending=False)

#L'output "importance_mean" è la diminuizione media del mio R2 se "rompo" quella
#features
print("\n--- Feature Importance dei DF per SVR ---")
print(importance_df_svr.head(10))

#Momento di fine del modello
end_time_df_svr = time.time()
execution_time_df_svr = end_time_df_svr - start_time_df_svr
print(f"--- Tempo di esecuzione Modello SVR sui Difensori: {execution_time_df_svr:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO SVR-RBF DIFENSORI(USO SVR)
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento SVR di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_df = df_df_1824[feature_control]
y_train_df = df_df_1824[y_column] 
x_test_df = df_df_2425[feature_control]
y_test_df = df_df_2425[y_column] 

# --- 2. SCALARE LA Y (TARGET) ---
y_scaler_df_svr_ctrl = StandardScaler()
y_train_scaled_df_svr_ctrl = y_scaler_df_svr_ctrl.fit_transform(y_train_df.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Identica per coerenza) ---
pipeline_df_svr_ctrl = Pipeline([
    ('scaler', StandardScaler()), 
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (Identica per coerenza) ---
# Usiamo la stessa identica griglia 6x6x6 per un confronto 1:1
param_grid_df_svr_ctrl = {
    'svr__C': [10, 11, 12, 100, 1000],          # 6 valori
    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7]      # 6 valori
} 
# Combinazioni totali: 216

tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV (Benchmark SVR Random)...")

grid_search_df_svr_ctrl = GridSearchCV(
    pipeline_df_svr_ctrl,
    param_grid=param_grid_df_svr_ctrl, 
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1 # Meno verboso del modello reale
)

# FIT su X_train (solo random) e Y_TRAIN_SCALED
grid_search_df_svr_ctrl.fit(x_train_df, y_train_scaled_df_svr_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random) ---")
print(f"Migliori iperparametri trovati: {grid_search_df_svr_ctrl.best_params_}")

best_model_df_svr_ctrl = grid_search_df_svr_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
y_pred_train_scaled_df_svr_ctrl = best_model_df_svr_ctrl.predict(x_train_df)
y_pred_test_scaled_df_svr_ctrl = best_model_df_svr_ctrl.predict(x_test_df)

y_pred_train_df_svr_ctrl = y_scaler_df_svr_ctrl.inverse_transform(y_pred_train_scaled_df_svr_ctrl.reshape(-1, 1))
y_pred_test_df_svr_ctrl = y_scaler_df_svr_ctrl.inverse_transform(y_pred_test_scaled_df_svr_ctrl.reshape(-1, 1))

mae_test_df_svr_ctrl = mean_absolute_error(y_test_df, y_pred_test_df_svr_ctrl)
mae_train_df_svr_ctrl = mean_absolute_error(y_train_df, y_pred_train_df_svr_ctrl)
r2_test_df_svr_ctrl = r2_score(y_test_df, y_pred_test_df_svr_ctrl)
r2_train_df_svr_ctrl = r2_score(y_train_df, y_pred_train_df_svr_ctrl)

print("RISULTATI MODELLO DI CONTROLLO SVR (RANDOM) - DIFENSORI")
print(f"MAE sul set di TRAINING: {mae_train_df_svr_ctrl:.2f}")
print(f"MAE sul set di TEST: {mae_test_df_svr_ctrl:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_df_svr_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_df_svr_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.2.3 - SVR WINGBACK
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento SVR con GridSearchCV su {len(all_features)} feature.")

x_train_wb = df_wb_1824[all_features]
y_train_wb = df_wb_1824[y_column]
x_test_wb = df_wb_2425[all_features]
y_test_wb = df_wb_2425[y_column]

#Momento di inizio del modello
start_time_wb_svr = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Obbligatorio per SVR
y_scaler_wb = StandardScaler()
y_train_scaled_wb_svr = y_scaler_wb.fit_transform(y_train_wb.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Semplificata) ---
# Il kernel RBF gestirà la non-linearità di 'age' automaticamente
pipeline_svr = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), # Scaler ora scala tutte le 216 feature
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---

# Definiamo la GRIGLIA di iperparametri da testare
# Questa è una griglia relativamente piccola per contenere i tempi.
# Se i risultati sono ai bordi della griglia, dovrai espanderla.
param_grid = {
    'svr__C': [10, 11, 12, 100],          # 6 valori
    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7]      # 6 valori
}

# Usiamo lo stesso TimeSeriesSplit per correttezza metodologica
tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV rigorosa per SVR (kernel RBF)...")
print(f"Totale combinazioni: {len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])}")
print(f"Totale modelli da addestrare: {(len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])) * tscv.get_n_splits()}")

grid_search_wb_svr = GridSearchCV(
    pipeline_svr,
    param_grid=param_grid, # Usa la griglia definita
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=2  # Mostra i progressi
)

# FIT su X_train e Y_TRAIN_SCALED
grid_search_wb_svr.fit(x_train_wb, y_train_scaled_wb_svr)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV SVR ---")
print(f"Migliori iperparametri trovati: {grid_search_wb_svr.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_wb_svr.best_score_:.4f}")

best_model_wb_svr = grid_search_wb_svr.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---

y_pred_train_scaled_wb_svr = best_model_wb_svr.predict(x_train_wb)
y_pred_test_scaled_wb_svr = best_model_wb_svr.predict(x_test_wb)

y_pred_train_final_wb_svr = y_scaler_wb.inverse_transform(y_pred_train_scaled_wb_svr.reshape(-1, 1))
y_pred_test_final_wb_svr = y_scaler_wb.inverse_transform(y_pred_test_scaled_wb_svr.reshape(-1, 1))

mae_test_wb_svr = mean_absolute_error(y_test_wb, y_pred_test_final_wb_svr)
mae_train_wb_svr = mean_absolute_error(y_train_wb, y_pred_train_final_wb_svr)
r2_test_wb_svr = r2_score(y_test_wb, y_pred_test_final_wb_svr)
r2_train_wb_svr = r2_score(y_train_wb, y_pred_train_final_wb_svr)

print("\n--- Performance Modello SVR Ottimizzato (Wingback) ---")
print(f"MAE sul set di TRAINING: {mae_train_wb_svr:.2f}")
print(f"MAE sul set di TEST: {mae_test_wb_svr:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_wb_svr:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_wb_svr:.4f}")

print("\n--- Confronto Lasso vs SVR RBF per Wingback---")
print(f"Lasso (Wingback):           R2 Test = {r2_test_wb_lasso:.4f}")
print(f"SVR RBF (Wingback):         R2 Test = {r2_test_wb_svr:.4f}")

print("\nAvvio calcolo Permutation Importance per i WB")

# Usiamo R² come metrica di scoring
# n_repeats=10 rende il calcolo più robusto (esegue 10 shuffle per feature)
perm_imp_wb_svr = permutation_importance(
    best_model_wb_svr, 
    x_test_wb, 
    y_test_wb, 
    n_repeats=20,
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_func # Puoi anche usare 'neg_mean_absolute_error'
)

# --- 3. Organizza e Stampa i Risultati ---
# 'perm_imp_mf_svr.importances_mean' è il calo medio di R² per ogni feature
importance_wb_svr = pd.DataFrame({
    'feature': x_train_wb.columns, # Assumendo che x_train sia un DataFrame
    'importance_mean': perm_imp_wb_svr.importances_mean,
    'importance_std': perm_imp_wb_svr.importances_std
})

# Ordina per importanza
importance_wb_svr = importance_wb_svr.sort_values('importance_mean', ascending=False)

#L'output "importance_mean" è la diminuizione media del mio R2 se "rompo" quella
#features
print("\n--- Feature Importance dei WB per SVR ---")
print(importance_wb_svr.head(18))

#Momento di fine del modello
end_time_wb_svr = time.time()
execution_time_wb_svr = end_time_wb_svr - start_time_wb_svr
print(f"--- Tempo di esecuzione Modello SVR sui Wingback: {execution_time_wb_svr:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO SVR-RBF WINGBACK(USO SVR)
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento SVR di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_wb = df_wb_1824[feature_control]
y_train_wb = df_wb_1824[y_column] 
x_test_wb = df_wb_2425[feature_control]
y_test_wb = df_wb_2425[y_column] 

# --- 2. SCALARE LA Y (TARGET) ---
y_scaler_wb_svr_ctrl = StandardScaler()
y_train_scaled_wb_svr_ctrl = y_scaler_wb_svr_ctrl.fit_transform(y_train_wb.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Identica per coerenza) ---
pipeline_wb_svr_ctrl = Pipeline([
    ('scaler', StandardScaler()), 
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (Identica per coerenza) ---
# Usiamo la stessa identica griglia 6x6x6 per un confronto 1:1
param_grid_wb_svr_ctrl = {
    'svr__C': [10, 11, 12, 100, 1000],          # 6 valori
    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7]  
} 
# Combinazioni totali: 216

tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV (Benchmark SVR Random)...")

grid_search_wb_svr_ctrl = GridSearchCV(
    pipeline_wb_svr_ctrl,
    param_grid=param_grid_wb_svr_ctrl, 
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1 # Meno verboso del modello reale
)

# FIT su X_train (solo random) e Y_TRAIN_SCALED
grid_search_wb_svr_ctrl.fit(x_train_wb, y_train_scaled_wb_svr_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random) ---")
print(f"Migliori iperparametri trovati: {grid_search_wb_svr_ctrl.best_params_}")

best_model_wb_svr_ctrl = grid_search_wb_svr_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
y_pred_train_scaled_wb_svr_ctrl = best_model_wb_svr_ctrl.predict(x_train_wb)
y_pred_test_scaled_wb_svr_ctrl = best_model_wb_svr_ctrl.predict(x_test_wb)

y_pred_train_wb_svr_ctrl = y_scaler_wb_svr_ctrl.inverse_transform(y_pred_train_scaled_wb_svr_ctrl.reshape(-1, 1))
y_pred_test_wb_svr_ctrl = y_scaler_wb_svr_ctrl.inverse_transform(y_pred_test_scaled_wb_svr_ctrl.reshape(-1, 1))

mae_test_wb_svr_ctrl = mean_absolute_error(y_test_wb, y_pred_test_wb_svr_ctrl)
mae_train_wb_svr_ctrl = mean_absolute_error(y_train_wb, y_pred_train_wb_svr_ctrl)
r2_test_wb_svr_ctrl = r2_score(y_test_wb, y_pred_test_wb_svr_ctrl)
r2_train_wb_svr_ctrl = r2_score(y_train_wb, y_pred_train_wb_svr_ctrl)

print("RISULTATI MODELLO DI CONTROLLO SVR (RANDOM) - WINGBACK")
print(f"MAE sul set di TRAINING: {mae_train_wb_svr_ctrl:.2f}")
print(f"MAE sul set di TEST: {mae_test_wb_svr_ctrl:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_wb_svr_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_wb_svr_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.2.4 - SVR CENTROCAMPISTI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento SVR con GridSearchCV su {len(all_features)} feature.")

x_train_mf = df_mf_1824[all_features]
y_train_mf = df_mf_1824[y_column]
x_test_mf = df_mf_2425[all_features]
y_test_mf = df_mf_2425[y_column]

#Momento di inizio del modello
start_time_mf_svr = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Obbligatorio per SVR
y_scaler_mf = StandardScaler()
y_train_scaled_mf_svr = y_scaler_mf.fit_transform(y_train_mf.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Semplificata) ---
# Il kernel RBF gestirà la non-linearità di 'age' automaticamente
pipeline_svr = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), # Scaler ora scala tutte le 216 feature
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---

# Definiamo la GRIGLIA di iperparametri da testare
# Questa è una griglia relativamente piccola per contenere i tempi.
# Se i risultati sono ai bordi della griglia, dovrò espanderla.
param_grid = {
    'svr__C': [7, 8, 9, 10],        
    'svr__gamma': [0.01, 0.1, 1, 1.1],       
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7],
}

# Usiamo lo stesso TimeSeriesSplit per correttezza metodologica
tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV rigorosa per SVR (kernel RBF)...")
print(f"Totale combinazioni: {len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])}")
print(f"Totale modelli da addestrare: {(len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])) * tscv.get_n_splits()}")

grid_search_mf_svr = GridSearchCV(
    pipeline_svr,
    param_grid=param_grid, # Usa la griglia definita
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=2  # Mostra i progressi
)

# FIT su X_train e Y_TRAIN_SCALED
grid_search_mf_svr.fit(x_train_mf, y_train_scaled_mf_svr)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV SVR ---")
print(f"Migliori iperparametri trovati: {grid_search_mf_svr.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_mf_svr.best_score_:.4f}")

best_model_mf_svr = grid_search_mf_svr.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---

y_pred_train_scaled_mf_svr = best_model_mf_svr.predict(x_train_mf)
y_pred_test_scaled_mf_svr = best_model_mf_svr.predict(x_test_mf)

y_pred_train_final_mf_svr = y_scaler_mf.inverse_transform(y_pred_train_scaled_mf_svr.reshape(-1, 1))
y_pred_test_final_mf_svr = y_scaler_mf.inverse_transform(y_pred_test_scaled_mf_svr.reshape(-1, 1))

mae_test_mf_svr = mean_absolute_error(y_test_mf, y_pred_test_final_mf_svr)
mae_train_mf_svr = mean_absolute_error(y_train_mf, y_pred_train_final_mf_svr)
r2_test_mf_svr = r2_score(y_test_mf, y_pred_test_final_mf_svr)
r2_train_mf_svr = r2_score(y_train_mf, y_pred_train_final_mf_svr)

print("\n--- Performance Modello SVR Ottimizzato (Centrocampisti) ---")
print(f"MAE sul set di TRAINING: {mae_train_mf_svr:.2f}")
print(f"MAE sul set di TEST: {mae_test_mf_svr:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_mf_svr:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_mf_svr:.4f}")

print("\n--- Confronto Lasso vs SVR RBF per Centrocampisti ---")
print(f"Lasso (Centrocampisti):           R2 Test = {r2_test_mf_lasso:.4f}")
print(f"SVR RBF (Centrocampisti):         R2 Test = {r2_test_mf_svr:.4f}")

print("\nAvvio calcolo Permutation Importance")

# Usiamo R² come metrica di scoring
# n_repeats=10 rende il calcolo più robusto (esegue 10 shuffle per feature)
perm_imp_mf_svr = permutation_importance(
    best_model_mf_svr, 
    x_test_mf, 
    y_test_mf, 
    n_repeats=20,
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_func 
)

# --- 3. Organizza e Stampa i Risultati ---
# 'perm_imp_mf_svr.importances_mean' è il calo medio di R² per ogni feature
importance_mf_svr = pd.DataFrame({
    'feature': x_train_mf.columns, # Assumendo che x_train sia un DataFrame
    'importance_mean': perm_imp_mf_svr.importances_mean,
    'importance_std': perm_imp_mf_svr.importances_std
})

# Ordina per importanza
importance_mf_svr = importance_mf_svr.sort_values('importance_mean', ascending=False)

#L'output "importance_mean" è la diminuizione media del mio R2 se "rompo" quella
#features
print("\n--- Feature Importance dei MF per SVR ---")
print(importance_mf_svr.head(14))

#Momento di fine del modello
end_time_mf_svr = time.time()
execution_time_mf_svr = end_time_mf_svr - start_time_mf_svr
print(f"--- Tempo di esecuzione Modello SVR sui Centrocampisti: {execution_time_mf_svr:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO SVR-RBF CENTROCAMPISTI(USO SVR)
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento SVR di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_mf = df_mf_1824[feature_control]
y_train_mf = df_mf_1824[y_column] 
x_test_mf = df_mf_2425[feature_control]
y_test_mf = df_mf_2425[y_column] 

# --- 2. SCALARE LA Y (TARGET) ---
y_scaler_mf_svr_ctrl = StandardScaler()
y_train_scaled_mf_svr_ctrl = y_scaler_mf_svr_ctrl.fit_transform(y_train_mf.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Identica per coerenza) ---
pipeline_mf_svr_ctrl = Pipeline([
    ('scaler', StandardScaler()), 
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (Identica per coerenza) ---
# Usiamo la stessa identica griglia 6x6x6 per un confronto 1:1
param_grid_mf_svr_ctrl = {
    'svr__C': [10, 11, 12, 100, 1000],          # 6 valori
    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7] 
} 
# Combinazioni totali: 216

tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV (Benchmark SVR Random)...")

grid_search_mf_svr_ctrl = GridSearchCV(
    pipeline_mf_svr_ctrl,
    param_grid=param_grid_mf_svr_ctrl, 
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1 # Meno verboso del modello reale
)

# FIT su X_train (solo random) e Y_TRAIN_SCALED
grid_search_mf_svr_ctrl.fit(x_train_mf, y_train_scaled_mf_svr_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random) ---")
print(f"Migliori iperparametri trovati: {grid_search_mf_svr_ctrl.best_params_}")

best_model_mf_svr_ctrl = grid_search_mf_svr_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
y_pred_train_scaled_mf_svr_ctrl = best_model_mf_svr_ctrl.predict(x_train_mf)
y_pred_test_scaled_mf_svr_ctrl = best_model_mf_svr_ctrl.predict(x_test_mf)

y_pred_train_mf_svr_ctrl = y_scaler_mf_svr_ctrl.inverse_transform(y_pred_train_scaled_mf_svr_ctrl.reshape(-1, 1))
y_pred_test_mf_svr_ctrl = y_scaler_mf_svr_ctrl.inverse_transform(y_pred_test_scaled_mf_svr_ctrl.reshape(-1, 1))

mae_test_mf_svr_ctrl = mean_absolute_error(y_test_mf, y_pred_test_mf_svr_ctrl)
mae_train_mf_svr_ctrl = mean_absolute_error(y_train_mf, y_pred_train_mf_svr_ctrl)
r2_test_mf_svr_ctrl = r2_score(y_test_mf, y_pred_test_mf_svr_ctrl)
r2_train_mf_svr_ctrl = r2_score(y_train_mf, y_pred_train_mf_svr_ctrl)

print("RISULTATI MODELLO DI CONTROLLO SVR (RANDOM) - CENTROCAMPISTI")
print(f"MAE sul set di TRAINING: {mae_train_mf_svr_ctrl:.2f}")
print(f"MAE sul set di TEST: {mae_test_mf_svr_ctrl:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_mf_svr_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_mf_svr_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.2.5 - SVR ATTACCANTI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento SVR con GridSearchCV su {len(all_features)} feature.")

x_train_fw = df_fw_1824[all_features]
y_train_fw = df_fw_1824[y_column]
x_test_fw = df_fw_2425[all_features]
y_test_fw = df_fw_2425[y_column]

#Momento di inizio del modello
start_time_fw_svr = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Obbligatorio per SVR
y_scaler_fw = StandardScaler()
y_train_scaled_fw_svr = y_scaler_fw.fit_transform(y_train_fw.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Semplificata) ---
# Il kernel RBF gestirà la non-linearità di 'age' automaticamente
pipeline_svr = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), # Scaler ora scala tutte le 216 feature
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---

# Definiamo la GRIGLIA di iperparametri da testare
# Questa è una griglia relativamente piccola per contenere i tempi.
# Se i risultati sono ai bordi della griglia, dovrai espanderla.
param_grid = {
    'svr__C': [8, 10, 11, 12, 100, 1000],          # 6 valori
    'svr__gamma': [0.0001, 0.001, 0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
    'svr__epsilon': [0.001, 0.01, 0.1, 0.2, 0.5, 0.6, 0.7]      # 6 valori
}
# Usiamo lo stesso TimeSeriesSplit per correttezza metodologica
tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV rigorosa per SVR (kernel RBF)...")
print(f"Totale combinazioni: {len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])}")
print(f"Totale modelli da addestrare: {(len(param_grid['svr__C']) * len(param_grid['svr__gamma']) * len(param_grid['svr__epsilon'])) * tscv.get_n_splits()}")

grid_search_fw_svr = GridSearchCV(
    pipeline_svr,
    param_grid=param_grid, # Usa la griglia definita
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, # Usa tutti i processori
    verbose=2  # Mostra i progressi
)

# FIT su X_train e Y_TRAIN_SCALED
grid_search_fw_svr.fit(x_train_fw, y_train_scaled_fw_svr)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV SVR ---")
print(f"Migliori iperparametri trovati: {grid_search_fw_svr.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_fw_svr.best_score_:.4f}")

best_model_fw_svr = grid_search_fw_svr.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---

y_pred_train_scaled_fw_svr = best_model_fw_svr.predict(x_train_fw)
y_pred_test_scaled_fw_svr = best_model_fw_svr.predict(x_test_fw)

y_pred_train_final_fw_svr = y_scaler_fw.inverse_transform(y_pred_train_scaled_fw_svr.reshape(-1, 1))
y_pred_test_final_fw_svr = y_scaler_fw.inverse_transform(y_pred_test_scaled_fw_svr.reshape(-1, 1))

mae_test_fw_svr = mean_absolute_error(y_test_fw, y_pred_test_final_fw_svr)
mae_train_fw_svr = mean_absolute_error(y_train_fw, y_pred_train_final_fw_svr)
r2_test_fw_svr = r2_score(y_test_fw, y_pred_test_final_fw_svr)
r2_train_fw_svr = r2_score(y_train_fw, y_pred_train_final_fw_svr)

print("\n--- Performance Modello SVR Ottimizzato (Attaccanti) ---")
print(f"MAE sul set di TRAINING: {mae_train_fw_svr:.2f}")
print(f"MAE sul set di TEST: {mae_test_fw_svr:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_fw_svr:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_fw_svr:.4f}")

print("\n--- Confronto Lasso vs SVR RBF per Attaccanti ---")
print(f"Lasso (Attaccanti):           R2 Test = {r2_test_fw_lasso:.4f}")
print(f"SVR RBF (Attaccanti):         R2 Test = {r2_test_fw_svr:.4f}")

#PERMUTATION IMPORTANCE ATTACCANTI
print("\nAvvio calcolo Permutation Importance")

perm_imp_fw_svr = permutation_importance(
    best_model_fw_svr, 
    x_test_fw, 
    y_test_fw,       # Passiamo la Y REALE (non scalata)
    n_repeats=20,
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_func 
)

# 3. Organizza e Stampa i Risultati
# 'perm_imp_fw_svr.importances_mean' è ora l'AUMENTO medio del MAE in MILIONI
importance_fw_svr = pd.DataFrame({
    'feature': x_train_fw.columns, 
    'importance_mean': perm_imp_fw_svr.importances_mean,
    'importance_std': perm_imp_fw_svr.importances_std
})

# Ordina per importanza (i valori più alti sono i peggiori)
importance_fw_svr = importance_fw_svr.sort_values('importance_mean', ascending=False)

print("\n--- Feature Importance dei FW per SVR (misurata in € MAE) ---")
print("L'output 'importance_mean' è l'aumento medio del MAE (in Milioni)")
print("se 'rompiamo' quella feature.")
print(importance_fw_svr.head(25))

#Momento di fine del modello
end_time_fw_svr = time.time()
execution_time_fw_svr = end_time_fw_svr - start_time_fw_svr
print(f"--- Tempo di esecuzione Modello SVR sugli Attaccanti: {execution_time_fw_svr:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO SVR-RBF ATTACCANTI(USO SVR)
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento SVR di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_fw = df_fw_1824[feature_control]
y_train_fw = df_fw_1824[y_column] 
x_test_fw = df_fw_2425[feature_control]
y_test_fw = df_fw_2425[y_column] 

# --- 2. SCALARE LA Y (TARGET) ---
y_scaler_fw_svr_ctrl = StandardScaler()
y_train_scaled_fw_svr_ctrl = y_scaler_fw_svr_ctrl.fit_transform(y_train_fw.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE (Identica per coerenza) ---
pipeline_fw_svr_ctrl = Pipeline([
    ('scaler', StandardScaler()), 
    ('svr', SVR(kernel='rbf')) 
])

# --- 4. SELEZIONE IPERPARAMETRI (Identica per coerenza) ---
# Usiamo la stessa identica griglia 6x6x6 per un confronto 1:1
param_grid_fw_svr_ctrl = {
    'svr__C': [10, 11, 12, 100, 1000],          # 6 valori
    'svr__gamma': [0.001, 0.01, 0.1, 1, 1.1],       # 6 valori
    'svr__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7] 
} 
# Combinazioni totali: 216

tscv = TimeSeriesSplit(n_splits=5) 
print("Avvio della GridSearchCV (Benchmark SVR Random)...")

grid_search_fw_svr_ctrl = GridSearchCV(
    pipeline_fw_svr_ctrl,
    param_grid=param_grid_fw_svr_ctrl, 
    cv=tscv, 
    scoring='neg_mean_absolute_error', 
    n_jobs=-1, 
    verbose=1 # Meno verboso del modello reale
)

# FIT su X_train (solo random) e Y_TRAIN_SCALED
grid_search_fw_svr_ctrl.fit(x_train_fw, y_train_scaled_fw_svr_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random) ---")
print(f"Migliori iperparametri trovati: {grid_search_fw_svr_ctrl.best_params_}")

best_model_fw_svr_ctrl = grid_search_fw_svr_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
y_pred_train_scaled_fw_svr_ctrl = best_model_fw_svr_ctrl.predict(x_train_fw)
y_pred_test_scaled_fw_svr_ctrl = best_model_fw_svr_ctrl.predict(x_test_fw)

y_pred_train_fw_svr_ctrl = y_scaler_fw_svr_ctrl.inverse_transform(y_pred_train_scaled_fw_svr_ctrl.reshape(-1, 1))
y_pred_test_fw_svr_ctrl = y_scaler_fw_svr_ctrl.inverse_transform(y_pred_test_scaled_fw_svr_ctrl.reshape(-1, 1))

mae_test_fw_svr_ctrl = mean_absolute_error(y_test_fw, y_pred_test_fw_svr_ctrl)
mae_train_fw_svr_ctrl = mean_absolute_error(y_train_fw, y_pred_train_fw_svr_ctrl)
r2_test_fw_svr_ctrl = r2_score(y_test_fw, y_pred_test_fw_svr_ctrl)
r2_train_fw_svr_ctrl = r2_score(y_train_fw, y_pred_train_fw_svr_ctrl)

print("RISULTATI MODELLO DI CONTROLLO SVR (RANDOM) - ATTACCANTI")
print(f"MAE sul set di TRAINING: {mae_train_fw_svr_ctrl:.2f}")
print(f"MAE sul set di TEST: {mae_test_fw_svr_ctrl:.2f}")
print(f"R2 Score sul set di TRAINING: {r2_train_fw_svr_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_fw_svr_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.3 - XGBoost
#XGBoost (versione breve di eXtreme Gradient Boosting) è un algoritmo avanzato di machine learning
#creato per garantire efficienza, velocità e ottime prestazioni. 
#Si tratta di un modello di ensemble learning ovvero un metodo basato sull'unione di molteplici modelli deboli
#allo scopo di creare un modello forte.
#XGBoost usa alberi decisionali come base learners e li combina allo scopo di migliorare le performance del modello.
#Ogni nuovo albero è allenato per correggere gli errori fatti dal modello precedente (idea del "boosting").

#L'idea alla base del XGBoost è quella che ogni albero cerca di correggere gli errori dell'albero precedente.
#Le fasi sono le seguenti:
#1)Inizia con un base learner: Il primo modello è trainato sui dati. Nei problemi di regressione semplicemente prevede la media della variabile target
#2)Calcola gli errori: calcola gli errori tra valori previsti e valori veri
#3)Training del prossimo albero: Il prossimo albero è trainato sugli errori dell'albero precedente. Questo step permette di correggere gli errori 
#fatti dal primo albero
#4)Ripeti il processo: Questo processo continua con ogni nuovo albero cercando di correggere gli errori dell'albero precedente fino a quando 
#un criterio di stop non viene raggiunto
#5) Combina le previsioni: La previsione finale è una combinazione di tutte le previsioni fatte

"""Per garantire la riproducibilità dei risultati e la confrontabilità dei modelli, 
è stato fissato un seme di casualità (random_state=42) per tutte le procedure stocastiche 
dell'algoritmo XGBoost"""
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#FASE 10.3.1 - XGBoost PORTIERI
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento XGBoost con GridSearchCV su {len(all_features)} feature.")

x_train_gk = df_gk_1824[all_features]
y_train_gk = df_gk_1824[y_column]
x_test_gk = df_gk_2425[all_features]
y_test_gk = df_gk_2425[y_column]

# Inizio cronometro
start_time_gk_xgb = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Anche se XGBoost è robusto, scaliamo la Y per confrontare i MAE scalati 
# durante la GridSearch con quelli dell'SVR.
y_scaler_gk = StandardScaler()
y_train_scaled_gk_xgb = y_scaler_gk.fit_transform(y_train_gk.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Usiamo XGBRegressor. n_jobs=1 all'interno del modello perché la parallelizzazione
# è già gestita esternamente dalla GridSearchCV (n_jobs=-1).
pipeline_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# n_estimators: numero di alberi utilizzati per l'ensemble learning
# learning_rate: "passo" del modello ad ogni correzione degli errori
    #value finale= value base + eta*modello1 + eta*modello2 +...
# max_depth: profondità degli alberi (controllo complessità)
    #indica il numero di nodi di ogni albero
#subsample:percentuale di giocatori (righe) da usare per addestrare ogni albero
#colsample_bytree: percentuale di statistiche (feature) da usare per ogni albero

#param_grid = {
#    'xgb__n_estimators': [50, 70, 80, 90],  #50
#    'xgb__learning_rate': [0.1, 0.2, 0.3],  #0.2
#    'xgb__max_depth': [3, 5, 7, 9, 11],    #9
#    'xgb__subsample': [0.7, 0.8, 1.0],  #0.8
#    'xgb__colsample_bytree': [0.7, 0.8, 0.9] #0.7
#}

#Solo per dataset con "age" e non "Born"
param_grid = {
    'xgb__n_estimators': [30, 40, 50, 70, 80],  
    'xgb__learning_rate': [0.05, 0.1, 0.2],
    'xgb__max_depth': [2, 3, 5, 7, 9],
    'xgb__subsample': [0.7, 0.8, 1.0],  
    'xgb__colsample_bytree': [0.7, 0.8, 0.9] 
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV per XGBoost")

grid_search_gk_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# FIT sui dati scalati
grid_search_gk_xgb.fit(x_train_gk, y_train_scaled_gk_xgb)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV XGBoost ---")
print(f"Migliori iperparametri: {grid_search_gk_xgb.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_gk_xgb.best_score_:.4f}")

best_model_gk_xgb = grid_search_gk_xgb.best_estimator_

# --- 6. VALUTAZIONE FINALE (DE-SCALATA) ---
y_pred_train_scaled_gk_xgb = best_model_gk_xgb.predict(x_train_gk)
y_pred_test_scaled_gk_xgb = best_model_gk_xgb.predict(x_test_gk)

# Riportiamo le previsioni in Milioni di Euro
y_pred_train_final_gk_xgb = y_scaler_gk.inverse_transform(y_pred_train_scaled_gk_xgb.reshape(-1, 1))
y_pred_test_final_gk_xgb = y_scaler_gk.inverse_transform(y_pred_test_scaled_gk_xgb.reshape(-1, 1))

mae_test_gk_xgb = mean_absolute_error(y_test_gk, y_pred_test_final_gk_xgb)
mae_train_gk_xgb = mean_absolute_error(y_train_gk, y_pred_train_final_gk_xgb)
r2_test_gk_xgb = r2_score(y_test_gk, y_pred_test_final_gk_xgb)
r2_train_gk_xgb = r2_score(y_train_gk, y_pred_train_final_gk_xgb)

print("\n--- Performance Modello XGBoost Ottimizzato (Portieri) ---")
print(f"MAE Training Set: {mae_train_gk_xgb:.2f} M€")
print(f"MAE Test Set: {mae_test_gk_xgb:.2f} M€")
print(f"R2 Score Training: {r2_train_gk_xgb:.4f}")
print(f"R2 Score Test: {r2_test_gk_xgb:.4f}")

print("\n--- Confronto Finale ---")
print(f"Lasso (Portieri):    R2 Test = {r2_test_gk_lasso:.4f}")
print(f"SVR RBF (portieri):  R2 Test = {r2_test_gk_svr:.4f}")
print(f"XGBoost (Portieri):  R2 Test = {r2_test_gk_xgb:.4f}")

# --- 7. PERMUTATION IMPORTANCE (Basata su MAE Reale) ---
print("\nAvvio calcolo Permutation Importance (XGBoost)...")

# Riutilizziamo la funzione di scoring personalizzata definita per l'SVR
# che de-scala y e calcola il MAE in euro reali.
def unscaled_mae_scorer_xgb(model, X, y_true_unscaled):
    y_pred_scaled = model.predict(X)
    y_pred_unscaled = y_scaler_gk.inverse_transform(y_pred_scaled.reshape(-1, 1))
    return -mean_absolute_error(y_true_unscaled, y_pred_unscaled)

perm_imp_gk_xgb = permutation_importance(
    best_model_gk_xgb, 
    x_test_gk, 
    y_test_gk, 
    n_repeats=20, # Numero di ripetizioni per stabilità
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_xgb
)

importance_gk_xgb = pd.DataFrame({
    'feature': x_train_gk.columns, 
    'importance_mean': perm_imp_gk_xgb.importances_mean,
    'importance_std': perm_imp_gk_xgb.importances_std
})

importance_gk_xgb = importance_gk_xgb.sort_values('importance_mean', ascending=False)

print("\n--- Feature Importance dei GK per XGBoost (In € MAE) ---")
print(importance_gk_xgb.head(25))

# Fine cronometro
end_time_gk_xgb = time.time()
execution_time_gk_xgb = end_time_gk_xgb - start_time_gk_xgb
print(f"\n--- Tempo di esecuzione XGBoost: {execution_time_gk_xgb:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO XGBoost (BENCHMARCK RANDOM) - PORTIERI
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento XGBoost di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_gk = df_gk_1824[feature_control]
y_train_gk = df_gk_1824[y_column]
x_test_gk = df_gk_2425[feature_control]
y_test_gk = df_gk_2425[y_column]

# --- 2. SCALARE LA Y (TARGET) ---
# Necessario per mantenere il confronto coerente con il modello reale
y_scaler_gk_xgb_ctrl = StandardScaler()
y_train_scaled_gk_xgb_ctrl = y_scaler_gk_xgb_ctrl.fit_transform(y_train_gk.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Utilizziamo la stessa struttura del modello principale per coerenza scientifica
pipeline_gk_xgb_ctrl = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# Usiamo la stessa griglia del modello reale per vedere se l'algoritmo
# riesce "accidentalmente" a trovare pattern nel rumore.
param_grid_gk_xgb_ctrl = {
    'xgb__n_estimators': [50, 70, 90],
    'xgb__learning_rate': [0.1, 0.2, 0.3],
    'xgb__max_depth': [5, 7, 9],
    'xgb__subsample': [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV (Benchmark XGBoost Random)...")

grid_search_gk_xgb_ctrl = GridSearchCV(
    pipeline_gk_xgb_ctrl,
    param_grid=param_grid_gk_xgb_ctrl,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1 # Meno verboso rispetto al modello reale
)

# FIT sul rumore (X_train_ctrl contiene solo la variabile random)
grid_search_gk_xgb_ctrl.fit(x_train_gk, y_train_scaled_gk_xgb_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random XGBoost) ---")
print(f"Migliori iperparametri trovati: {grid_search_gk_xgb_ctrl.best_params_}")

best_model_gk_xgb_ctrl = grid_search_gk_xgb_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
# Predizioni sui dati scalati
y_pred_train_scaled_gk_xgb_ctrl = best_model_gk_xgb_ctrl.predict(x_train_gk)
y_pred_test_scaled_gk_xgb_ctrl = best_model_gk_xgb_ctrl.predict(x_test_gk)

# Riportiamo i valori in milioni di euro (de-scaling)
y_pred_train_gk_xgb_ctrl = y_scaler_gk_xgb_ctrl.inverse_transform(y_pred_train_scaled_gk_xgb_ctrl.reshape(-1, 1))
y_pred_test_gk_xgb_ctrl = y_scaler_gk_xgb_ctrl.inverse_transform(y_pred_test_scaled_gk_xgb_ctrl.reshape(-1, 1))

# Calcolo metriche reali
mae_test_gk_xgb_ctrl = mean_absolute_error(y_test_gk, y_pred_test_gk_xgb_ctrl)
mae_train_gk_xgb_ctrl = mean_absolute_error(y_train_gk, y_pred_train_gk_xgb_ctrl)
r2_test_gk_xgb_ctrl = r2_score(y_test_gk, y_pred_test_gk_xgb_ctrl)
r2_train_gk_xgb_ctrl = r2_score(y_train_gk, y_pred_train_gk_xgb_ctrl)

print("\n--- PERFORMANCE MODELLO DI CONTROLLO XGBOOST (RANDOM) - PORTIERI ---")
print(f"MAE sul set di TRAINING: {mae_train_gk_xgb_ctrl:.2f} M€")
print(f"MAE sul set di TEST: {mae_test_gk_xgb_ctrl:.2f} M€")
print(f"R2 Score sul set di TRAINING: {r2_train_gk_xgb_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_gk_xgb_ctrl:.4f}")

# --- CONFRONTO FINALE PER LA TESI ---
print("\n--- Verifica Potere Predittivo (R2 Test) ---")
print(f"XGBoost Reale (Portieri): {r2_test_gk_xgb:.4f}")
print(f"XGBoost Random (Controllo): {r2_test_gk_xgb_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.3.2 - XGBoost DIFENSORI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento XGBoost con GridSearchCV su {len(all_features)} feature.")

x_train_df = df_df_1824[all_features]
y_train_df = df_df_1824[y_column]
x_test_df = df_df_2425[all_features]
y_test_df = df_df_2425[y_column]

# Inizio cronometro
start_time_df_xgb = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Anche se XGBoost è robusto, scaliamo la Y per confrontare i MAE scalati 
# durante la GridSearch con quelli dell'SVR.
y_scaler_df = StandardScaler()
y_train_scaled_df_xgb = y_scaler_df.fit_transform(y_train_df.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Usiamo XGBRegressor. n_jobs=1 all'interno del modello perché la parallelizzazione
# è già gestita esternamente dalla GridSearchCV (n_jobs=-1).
pipeline_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# n_estimators: numero di alberi utilizzati per l'ensemble learning
# learning_rate: "passo" del modello ad ogni correzione degli errori
    #value finale= value base + eta*model1 + eta*model2 +...
# max_depth: profondità degli alberi (controllo complessità)
    #indica il numero di nodi di ogni albero
#subsample:percentuale di giocatori (righe) da usare per addestrare ogni albero
#colsample_bytree: percentuale di statistiche (feature) da usare per ogni albero

#param_grid = {
#    'xgb__n_estimators': [60, 80, 90], 
#    'xgb__learning_rate': [0.05, 0.1, 0.15],
#    'xgb__max_depth': [3, 4, 5],
#    'xgb__subsample': [0.8, 0.9, 1.0],
#    'xgb__colsample_bytree': [0.6, 0.8, 0.9] 
#}

#Solo per dataset con "age" e non "Born"
param_grid = {
    'xgb__n_estimators': [60, 80, 90], 
    'xgb__learning_rate': [0.05, 0.1, 0.15],
    'xgb__max_depth': [2, 3, 4, 5],
    'xgb__subsample': [0.8, 0.9, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 0.9, 0.95] 
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV per XGBoost")

grid_search_df_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# FIT sui dati scalati
grid_search_df_xgb.fit(x_train_df, y_train_scaled_df_xgb)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV XGBoost ---")
print(f"Migliori iperparametri: {grid_search_df_xgb.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_df_xgb.best_score_:.4f}")

best_model_df_xgb = grid_search_df_xgb.best_estimator_

# --- 6. VALUTAZIONE FINALE (DE-SCALATA) ---
y_pred_train_scaled_df_xgb = best_model_df_xgb.predict(x_train_df)
y_pred_test_scaled_df_xgb = best_model_df_xgb.predict(x_test_df)

# Riportiamo le previsioni in Milioni di Euro
y_pred_train_final_df_xgb = y_scaler_df.inverse_transform(y_pred_train_scaled_df_xgb.reshape(-1, 1))
y_pred_test_final_df_xgb = y_scaler_df.inverse_transform(y_pred_test_scaled_df_xgb.reshape(-1, 1))

mae_test_df_xgb = mean_absolute_error(y_test_df, y_pred_test_final_df_xgb)
mae_train_df_xgb = mean_absolute_error(y_train_df, y_pred_train_final_df_xgb)
r2_test_df_xgb = r2_score(y_test_df, y_pred_test_final_df_xgb)
r2_train_df_xgb = r2_score(y_train_df, y_pred_train_final_df_xgb)

print("\n--- Performance Modello XGBoost Ottimizzato (Difensori) ---")
print(f"MAE Training Set: {mae_train_df_xgb:.2f} M€")
print(f"MAE Test Set: {mae_test_df_xgb:.2f} M€")
print(f"R2 Score Training: {r2_train_df_xgb:.4f}")
print(f"R2 Score Test: {r2_test_df_xgb:.4f}")

print("\n--- Confronto Finale ---")
print(f"Lasso (Difensori):    R2 Test = {r2_test_df_lasso:.4f}")
print(f"SVR RBF (Difensori):  R2 Test = {r2_test_df_svr:.4f}")
print(f"XGBoost (Difensori):  R2 Test = {r2_test_df_xgb:.4f}")

# --- 7. PERMUTATION IMPORTANCE (Basata su MAE Reale) ---
print("\nAvvio calcolo Permutation Importance (XGBoost)...")

# Riutilizziamo la funzione di scoring personalizzata definita per l'SVR
# che de-scala y e calcola il MAE in euro reali.
def unscaled_mae_scorer_xgb(model, X, y_true_unscaled):
    y_pred_scaled = model.predict(X)
    y_pred_unscaled = y_scaler_df.inverse_transform(y_pred_scaled.reshape(-1, 1))
    return -mean_absolute_error(y_true_unscaled, y_pred_unscaled)

perm_imp_df_xgb = permutation_importance(
    best_model_df_xgb, 
    x_test_df, 
    y_test_df, 
    n_repeats=20, # Numero di ripetizioni per stabilità
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_xgb
)

importance_df_xgb = pd.DataFrame({
    'feature': x_train_df.columns, 
    'importance_mean': perm_imp_df_xgb.importances_mean,
    'importance_std': perm_imp_df_xgb.importances_std
})

importance_df_xgb = importance_df_xgb.sort_values('importance_mean', ascending=False)

print("\n--- Feature Importance dei DF per XGBoost (In € MAE) ---")
print(importance_df_xgb.head(25))

# Fine cronometro
end_time_df_xgb = time.time()
execution_time_df_xgb = end_time_df_xgb - start_time_df_xgb
print(f"\n--- Tempo di esecuzione XGBoost: {execution_time_df_xgb:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO XGBoost (BENCHMARCK RANDOM) - DIFENSORI
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento XGBoost di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_df = df_df_1824[feature_control]
y_train_df = df_df_1824[y_column]
x_test_df = df_df_2425[feature_control]
y_test_df = df_df_2425[y_column]

# --- 2. SCALARE LA Y (TARGET) ---
# Necessario per mantenere il confronto coerente con il modello reale
y_scaler_df_xgb_ctrl = StandardScaler()
y_train_scaled_df_xgb_ctrl = y_scaler_df_xgb_ctrl.fit_transform(y_train_df.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Utilizziamo la stessa struttura del modello principale per coerenza scientifica
pipeline_df_xgb_ctrl = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# Usiamo la stessa griglia del modello reale per vedere se l'algoritmo
# riesce "accidentalmente" a trovare pattern nel rumore.
param_grid_df_xgb_ctrl = {
    'xgb__n_estimators': [50, 70, 90],
    'xgb__learning_rate': [0.1, 0.2, 0.3],
    'xgb__max_depth': [5, 7, 9],
    'xgb__subsample': [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV (Benchmark XGBoost Random)...")

grid_search_df_xgb_ctrl = GridSearchCV(
    pipeline_df_xgb_ctrl,
    param_grid=param_grid_df_xgb_ctrl,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1 # Meno verboso rispetto al modello reale
)

# FIT sul rumore (X_train_ctrl contiene solo la variabile random)
grid_search_df_xgb_ctrl.fit(x_train_df, y_train_scaled_df_xgb_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random XGBoost) ---")
print(f"Migliori iperparametri trovati: {grid_search_df_xgb_ctrl.best_params_}")

best_model_df_xgb_ctrl = grid_search_df_xgb_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
# Predizioni sui dati scalati
y_pred_train_scaled_df_xgb_ctrl = best_model_df_xgb_ctrl.predict(x_train_df)
y_pred_test_scaled_df_xgb_ctrl = best_model_df_xgb_ctrl.predict(x_test_df)

# Riportiamo i valori in milioni di euro (de-scaling)
y_pred_train_df_xgb_ctrl = y_scaler_df_xgb_ctrl.inverse_transform(y_pred_train_scaled_df_xgb_ctrl.reshape(-1, 1))
y_pred_test_df_xgb_ctrl = y_scaler_df_xgb_ctrl.inverse_transform(y_pred_test_scaled_df_xgb_ctrl.reshape(-1, 1))

# Calcolo metriche reali
mae_test_df_xgb_ctrl = mean_absolute_error(y_test_df, y_pred_test_df_xgb_ctrl)
mae_train_df_xgb_ctrl = mean_absolute_error(y_train_df, y_pred_train_df_xgb_ctrl)
r2_test_df_xgb_ctrl = r2_score(y_test_df, y_pred_test_df_xgb_ctrl)
r2_train_df_xgb_ctrl = r2_score(y_train_df, y_pred_train_df_xgb_ctrl)

print("\n--- PERFORMANCE MODELLO DI CONTROLLO XGBOOST (RANDOM) - DIFENSORI ---")
print(f"MAE sul set di TRAINING: {mae_train_df_xgb_ctrl:.2f} M€")
print(f"MAE sul set di TEST: {mae_test_df_xgb_ctrl:.2f} M€")
print(f"R2 Score sul set di TRAINING: {r2_train_df_xgb_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_df_xgb_ctrl:.4f}")

# --- CONFRONTO FINALE PER LA TESI ---
print("\n--- Verifica Potere Predittivo (R2 Test) ---")
print(f"XGBoost Reale (Difensori): {r2_test_df_xgb:.4f}")
print(f"XGBoost Random (Controllo): {r2_test_df_xgb_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.3.3 - XGBoost WINGBACK
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento XGBoost con GridSearchCV su {len(all_features)} feature.")

x_train_wb = df_wb_1824[all_features]
y_train_wb = df_wb_1824[y_column]
x_test_wb = df_wb_2425[all_features]
y_test_wb = df_wb_2425[y_column]

# Inizio cronometro
start_time_wb_xgb = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Anche se XGBoost è robusto, scaliamo la Y per confrontare i MAE scalati 
# durante la GridSearch con quelli dell'SVR.
y_scaler_wb = StandardScaler()
y_train_scaled_wb_xgb = y_scaler_wb.fit_transform(y_train_wb.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Usiamo XGBRegressor. n_jobs=1 all'interno del modello perché la parallelizzazione
# è già gestita esternamente dalla GridSearchCV (n_jobs=-1).
pipeline_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# n_estimators: numero di alberi utilizzati per l'ensemble learning
# max_depth: profondità degli alberi (controllo complessità)
# learning_rate: "passo" del modello ad ogni correzione degli errori
# subsample: % di righe campionate per ogni albero

#param_grid = {
#    'xgb__n_estimators': [50, 70, 90, 100],  
#    'xgb__learning_rate': [0.05, 0.1, 0.2],  
#    'xgb__max_depth': [3, 4, 5], #4
#    'xgb__subsample': [0.7, 0.8, 1.0],  
#    'xgb__colsample_bytree': [0.6, 0.7, 0.8] 
#}

#Solo per datset con "age" e non "born"
param_grid = {
    'xgb__n_estimators': [70, 90, 100, 110],  
    'xgb__learning_rate': [0.05, 0.1, 0.2],  
    'xgb__max_depth': [3, 4, 5, 6], 
    'xgb__subsample': [0.6, 0.7, 0.8],  
    'xgb__colsample_bytree': [0.7, 0.8, 0.9] 
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV per XGBoost")

grid_search_wb_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# FIT sui dati scalati
grid_search_wb_xgb.fit(x_train_wb, y_train_scaled_wb_xgb)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV XGBoost ---")
print(f"Migliori iperparametri: {grid_search_wb_xgb.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_wb_xgb.best_score_:.4f}")

best_model_wb_xgb = grid_search_wb_xgb.best_estimator_

# --- 6. VALUTAZIONE FINALE (DE-SCALATA) ---
y_pred_train_scaled_wb_xgb = best_model_wb_xgb.predict(x_train_wb)
y_pred_test_scaled_wb_xgb = best_model_wb_xgb.predict(x_test_wb)

# Riportiamo le previsioni in Milioni di Euro
y_pred_train_final_wb_xgb = y_scaler_wb.inverse_transform(y_pred_train_scaled_wb_xgb.reshape(-1, 1))
y_pred_test_final_wb_xgb = y_scaler_wb.inverse_transform(y_pred_test_scaled_wb_xgb.reshape(-1, 1))

mae_test_wb_xgb = mean_absolute_error(y_test_wb, y_pred_test_final_wb_xgb)
mae_train_wb_xgb = mean_absolute_error(y_train_wb, y_pred_train_final_wb_xgb)
r2_test_wb_xgb = r2_score(y_test_wb, y_pred_test_final_wb_xgb)
r2_train_wb_xgb = r2_score(y_train_wb, y_pred_train_final_wb_xgb)

print("\n--- Performance Modello XGBoost Ottimizzato (Wingback) ---")
print(f"MAE Training Set: {mae_train_wb_xgb:.2f} M€")
print(f"MAE Test Set: {mae_test_wb_xgb:.2f} M€")
print(f"R2 Score Training: {r2_train_wb_xgb:.4f}")
print(f"R2 Score Test: {r2_test_wb_xgb:.4f}")

print("\n--- Confronto Finale ---")
print(f"Lasso (Wingback):    R2 Test = {r2_test_wb_lasso:.4f}")
print(f"SVR RBF (Wingback):  R2 Test = {r2_test_wb_svr:.4f}")
print(f"XGBoost (Wingback):  R2 Test = {r2_test_wb_xgb:.4f}")

# --- 7. PERMUTATION IMPORTANCE (Basata su MAE Reale) ---
print("\nAvvio calcolo Permutation Importance (XGBoost)...")

# Riutilizziamo la funzione di scoring personalizzata definita per l'SVR
# che de-scala y e calcola il MAE in euro reali.
def unscaled_mae_scorer_xgb(model, X, y_true_unscaled):
    y_pred_scaled = model.predict(X)
    y_pred_unscaled = y_scaler_wb.inverse_transform(y_pred_scaled.reshape(-1, 1))
    return -mean_absolute_error(y_true_unscaled, y_pred_unscaled)

perm_imp_wb_xgb = permutation_importance(
    best_model_wb_xgb, 
    x_test_wb, 
    y_test_wb, 
    n_repeats=20, # Numero di ripetizioni per stabilità
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_xgb
)

importance_wb_xgb = pd.DataFrame({
    'feature': x_train_wb.columns, 
    'importance_mean': perm_imp_wb_xgb.importances_mean,
    'importance_std': perm_imp_wb_xgb.importances_std
})

importance_wb_xgb = importance_wb_xgb.sort_values('importance_mean', ascending=False)

print("\n--- Feature Importance dei WB per XGBoost (In € MAE) ---")
print(importance_wb_xgb.head(25))

# Fine cronometro
end_time_wb_xgb = time.time()
execution_time_wb_xgb = end_time_wb_xgb - start_time_wb_xgb
print(f"\n--- Tempo di esecuzione XGBoost: {execution_time_wb_xgb:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO XGBoost (BENCHMARCK RANDOM) - WINGBACK
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento XGBoost di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_wb = df_wb_1824[feature_control]
y_train_wb = df_wb_1824[y_column]
x_test_wb = df_wb_2425[feature_control]
y_test_wb = df_wb_2425[y_column]

# --- 2. SCALARE LA Y (TARGET) ---
# Necessario per mantenere il confronto coerente con il modello reale
y_scaler_wb_xgb_ctrl = StandardScaler()
y_train_scaled_wb_xgb_ctrl = y_scaler_wb_xgb_ctrl.fit_transform(y_train_wb.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Utilizziamo la stessa struttura del modello principale per coerenza scientifica
pipeline_wb_xgb_ctrl = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# Usiamo la stessa griglia del modello reale per vedere se l'algoritmo
# riesce "accidentalmente" a trovare pattern nel rumore.
param_grid_wb_xgb_ctrl = {
    'xgb__n_estimators': [50, 70, 90],
    'xgb__learning_rate': [0.1, 0.2, 0.3],
    'xgb__max_depth': [5, 7, 9],
    'xgb__subsample': [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV (Benchmark XGBoost Random)...")

grid_search_wb_xgb_ctrl = GridSearchCV(
    pipeline_wb_xgb_ctrl,
    param_grid=param_grid_wb_xgb_ctrl,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1 # Meno verboso rispetto al modello reale
)

# FIT sul rumore (X_train_ctrl contiene solo la variabile random)
grid_search_wb_xgb_ctrl.fit(x_train_wb, y_train_scaled_wb_xgb_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random XGBoost) ---")
print(f"Migliori iperparametri trovati: {grid_search_wb_xgb_ctrl.best_params_}")

best_model_wb_xgb_ctrl = grid_search_wb_xgb_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
# Predizioni sui dati scalati
y_pred_train_scaled_wb_xgb_ctrl = best_model_wb_xgb_ctrl.predict(x_train_wb)
y_pred_test_scaled_wb_xgb_ctrl = best_model_wb_xgb_ctrl.predict(x_test_wb)

# Riportiamo i valori in milioni di euro (de-scaling)
y_pred_train_wb_xgb_ctrl = y_scaler_wb_xgb_ctrl.inverse_transform(y_pred_train_scaled_wb_xgb_ctrl.reshape(-1, 1))
y_pred_test_wb_xgb_ctrl = y_scaler_wb_xgb_ctrl.inverse_transform(y_pred_test_scaled_wb_xgb_ctrl.reshape(-1, 1))

# Calcolo metriche reali
mae_test_wb_xgb_ctrl = mean_absolute_error(y_test_wb, y_pred_test_wb_xgb_ctrl)
mae_train_wb_xgb_ctrl = mean_absolute_error(y_train_wb, y_pred_train_wb_xgb_ctrl)
r2_test_wb_xgb_ctrl = r2_score(y_test_wb, y_pred_test_wb_xgb_ctrl)
r2_train_wb_xgb_ctrl = r2_score(y_train_wb, y_pred_train_wb_xgb_ctrl)

print("\n--- PERFORMANCE MODELLO DI CONTROLLO XGBOOST (RANDOM) - CENTROCAMPISTI ---")
print(f"MAE sul set di TRAINING: {mae_train_wb_xgb_ctrl:.2f} M€")
print(f"MAE sul set di TEST: {mae_test_wb_xgb_ctrl:.2f} M€")
print(f"R2 Score sul set di TRAINING: {r2_train_wb_xgb_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_wb_xgb_ctrl:.4f}")

# --- CONFRONTO FINALE PER LA TESI ---
print("\n--- Verifica Potere Predittivo (R2 Test) ---")
print(f"XGBoost Reale (Wingback): {r2_test_wb_xgb:.4f}")
print(f"XGBoost Random (Controllo): {r2_test_wb_xgb_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.3.4 - XGBoost CENTROCAMPISTI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento XGBoost con GridSearchCV su {len(all_features)} feature.")

x_train_mf = df_mf_1824[all_features]
y_train_mf = df_mf_1824[y_column]
x_test_mf = df_mf_2425[all_features]
y_test_mf = df_mf_2425[y_column]

# Inizio cronometro
start_time_mf_xgb = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Anche se XGBoost è robusto, scaliamo la Y per confrontare i MAE scalati 
# durante la GridSearch con quelli dell'SVR.
y_scaler_mf = StandardScaler()
y_train_scaled_mf_xgb = y_scaler_mf.fit_transform(y_train_mf.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Usiamo XGBRegressor. n_jobs=1 all'interno del modello perché la parallelizzazione
# è già gestita esternamente dalla GridSearchCV (n_jobs=-1).
pipeline_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# n_estimators: numero di alberi utilizzati per l'ensemble learning
# max_depth: profondità degli alberi (controllo complessità)
# learning_rate: "passo" del modello ad ogni correzione degli errori
# subsample: % di righe campionate per ogni albero

#param_grid = {
#    'xgb__n_estimators': [70, 90, 100, 110], #100
#    'xgb__learning_rate': [0.03, 0.05, 0.1], #0.05
#    'xgb__max_depth': [3, 4, 5], #4
#    'xgb__subsample': [0.5, 0.6, 0.75], #0.6
#    'xgb__colsample_bytree': [0.5, 0.6, 0.7, 0.8] #0.6
#}

#Solo per dataset con "age" e no "Born"
param_grid = {
    'xgb__n_estimators': [100,110, 120, 130], 
    'xgb__learning_rate': [0.05, 0.1, 0.2], 
    'xgb__max_depth': [2, 3, 4, 5],
    'xgb__subsample': [0.5, 0.6, 0.75], 
    'xgb__colsample_bytree': [0.6, 0.7, 0.8] 
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV per XGBoost")

grid_search_mf_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# FIT sui dati scalati
grid_search_mf_xgb.fit(x_train_mf, y_train_scaled_mf_xgb)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV XGBoost ---")
print(f"Migliori iperparametri: {grid_search_mf_xgb.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_mf_xgb.best_score_:.4f}")

best_model_mf_xgb = grid_search_mf_xgb.best_estimator_

# --- 6. VALUTAZIONE FINALE (DE-SCALATA) ---
y_pred_train_scaled_mf_xgb = best_model_mf_xgb.predict(x_train_mf)
y_pred_test_scaled_mf_xgb = best_model_mf_xgb.predict(x_test_mf)

# Riportiamo le previsioni in Milioni di Euro
y_pred_train_final_mf_xgb = y_scaler_mf.inverse_transform(y_pred_train_scaled_mf_xgb.reshape(-1, 1))
y_pred_test_final_mf_xgb = y_scaler_mf.inverse_transform(y_pred_test_scaled_mf_xgb.reshape(-1, 1))

mae_test_mf_xgb = mean_absolute_error(y_test_mf, y_pred_test_final_mf_xgb)
mae_train_mf_xgb = mean_absolute_error(y_train_mf, y_pred_train_final_mf_xgb)
r2_test_mf_xgb = r2_score(y_test_mf, y_pred_test_final_mf_xgb)
r2_train_mf_xgb = r2_score(y_train_mf, y_pred_train_final_mf_xgb)

print("\n--- Performance Modello XGBoost Ottimizzato (Centrocampisti) ---")
print(f"MAE Training Set: {mae_train_mf_xgb:.2f} M€")
print(f"MAE Test Set: {mae_test_mf_xgb:.2f} M€")
print(f"R2 Score Training: {r2_train_mf_xgb:.4f}")
print(f"R2 Score Test: {r2_test_mf_xgb:.4f}")

print("\n--- Confronto Finale ---")
print(f"Lasso (Centrocampisti):    R2 Test = {r2_test_mf_lasso:.4f}")
print(f"SVR RBF (Centrocampisti):  R2 Test = {r2_test_mf_svr:.4f}")
print(f"XGBoost (Centrocampisti):  R2 Test = {r2_test_mf_xgb:.4f}")

# --- 7. PERMUTATION IMPORTANCE (Basata su MAE Reale) ---
print("\nAvvio calcolo Permutation Importance (XGBoost)...")

# Riutilizziamo la funzione di scoring personalizzata definita per l'SVR
# che de-scala y e calcola il MAE in euro reali.
def unscaled_mae_scorer_xgb(model, X, y_true_unscaled):
    y_pred_scaled = model.predict(X)
    y_pred_unscaled = y_scaler_mf.inverse_transform(y_pred_scaled.reshape(-1, 1))
    return -mean_absolute_error(y_true_unscaled, y_pred_unscaled)

perm_imp_mf_xgb = permutation_importance(
    best_model_mf_xgb, 
    x_test_mf, 
    y_test_mf, 
    n_repeats=20, # Numero di ripetizioni per stabilità
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_xgb
)

importance_mf_xgb = pd.DataFrame({
    'feature': x_train_mf.columns, 
    'importance_mean': perm_imp_mf_xgb.importances_mean,
    'importance_std': perm_imp_mf_xgb.importances_std
})

importance_mf_xgb = importance_mf_xgb.sort_values('importance_mean', ascending=False)

print("\n--- Feature Importance dei MF per XGBoost (In € MAE) ---")
print(importance_mf_xgb.head(25))

# Fine cronometro
end_time_mf_xgb = time.time()
execution_time_mf_xgb = end_time_mf_xgb - start_time_mf_xgb
print(f"\n--- Tempo di esecuzione XGBoost MF: {execution_time_mf_xgb:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO XGBoost (BENCHMARCK RANDOM) - CENTROCAMPISTI
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento XGBoost di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_mf = df_mf_1824[feature_control]
y_train_mf = df_mf_1824[y_column]
x_test_mf = df_mf_2425[feature_control]
y_test_mf = df_mf_2425[y_column]

# --- 2. SCALARE LA Y (TARGET) ---
# Necessario per mantenere il confronto coerente con il modello reale
y_scaler_mf_xgb_ctrl = StandardScaler()
y_train_scaled_mf_xgb_ctrl = y_scaler_mf_xgb_ctrl.fit_transform(y_train_mf.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Utilizziamo la stessa struttura del modello principale per coerenza scientifica
pipeline_mf_xgb_ctrl = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# Usiamo la stessa griglia del modello reale per vedere se l'algoritmo
# riesce "accidentalmente" a trovare pattern nel rumore.
param_grid_mf_xgb_ctrl = {
    'xgb__n_estimators': [50, 70, 90],
    'xgb__learning_rate': [0.1, 0.2, 0.3],
    'xgb__max_depth': [5, 7, 9],
    'xgb__subsample': [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV (Benchmark XGBoost Random)...")

grid_search_mf_xgb_ctrl = GridSearchCV(
    pipeline_mf_xgb_ctrl,
    param_grid=param_grid_mf_xgb_ctrl,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1 # Meno verboso rispetto al modello reale
)

# FIT sul rumore (X_train_ctrl contiene solo la variabile random)
grid_search_mf_xgb_ctrl.fit(x_train_mf, y_train_scaled_mf_xgb_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random XGBoost) ---")
print(f"Migliori iperparametri trovati: {grid_search_mf_xgb_ctrl.best_params_}")

best_model_mf_xgb_ctrl = grid_search_mf_xgb_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
# Predizioni sui dati scalati
y_pred_train_scaled_mf_xgb_ctrl = best_model_mf_xgb_ctrl.predict(x_train_mf)
y_pred_test_scaled_mf_xgb_ctrl = best_model_mf_xgb_ctrl.predict(x_test_mf)

# Riportiamo i valori in milioni di euro (de-scaling)
y_pred_train_mf_xgb_ctrl = y_scaler_mf_xgb_ctrl.inverse_transform(y_pred_train_scaled_mf_xgb_ctrl.reshape(-1, 1))
y_pred_test_mf_xgb_ctrl = y_scaler_mf_xgb_ctrl.inverse_transform(y_pred_test_scaled_mf_xgb_ctrl.reshape(-1, 1))

# Calcolo metriche reali
mae_test_mf_xgb_ctrl = mean_absolute_error(y_test_mf, y_pred_test_mf_xgb_ctrl)
mae_train_mf_xgb_ctrl = mean_absolute_error(y_train_mf, y_pred_train_mf_xgb_ctrl)
r2_test_mf_xgb_ctrl = r2_score(y_test_mf, y_pred_test_mf_xgb_ctrl)
r2_train_mf_xgb_ctrl = r2_score(y_train_mf, y_pred_train_mf_xgb_ctrl)

print("\n--- PERFORMANCE MODELLO DI CONTROLLO XGBOOST (RANDOM) - CENTROCAMPISTI ---")
print(f"MAE sul set di TRAINING: {mae_train_mf_xgb_ctrl:.2f} M€")
print(f"MAE sul set di TEST: {mae_test_mf_xgb_ctrl:.2f} M€")
print(f"R2 Score sul set di TRAINING: {r2_train_mf_xgb_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_mf_xgb_ctrl:.4f}")

# --- CONFRONTO FINALE PER LA TESI ---
print("\n--- Verifica Potere Predittivo (R2 Test) ---")
print(f"XGBoost Reale (Centrocampisti): {r2_test_mf_xgb:.4f}")
print(f"XGBoost Random (Controllo): {r2_test_mf_xgb_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FASE 10.3.5 - XGBoost ATTACCANTI
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
y_column = "value"
all_features = [col for col in numeric_cols if col != "value" and col != "random" and col!="Born"]
print(f"Inizio addestramento XGBoost con GridSearchCV su {len(all_features)} feature.")

x_train_fw = df_fw_1824[all_features]
y_train_fw = df_fw_1824[y_column]
x_test_fw = df_fw_2425[all_features]
y_test_fw = df_fw_2425[y_column]

# Inizio cronometro
start_time_fw_xgb = time.time()

# --- 2. SCALARE LA Y (TARGET) ---
# Anche se XGBoost è robusto, scaliamo la Y per confrontare i MAE scalati 
# durante la GridSearch con quelli dell'SVR.
y_scaler_fw = StandardScaler()
y_train_scaled_fw_xgb = y_scaler_fw.fit_transform(y_train_fw.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Usiamo XGBRegressor. n_jobs=1 all'interno del modello perché la parallelizzazione
# è già gestita esternamente dalla GridSearchCV (n_jobs=-1).
pipeline_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# n_estimators: numero di alberi utilizzati per l'ensemble learning
# learning_rate: "passo" del modello ad ogni correzione degli errori
    #value finale= value base + eta*model1 + eta*model2 +...
# max_depth: profondità degli alberi (controllo complessità)
    #indica il numero di nodi di ogni albero
#subsample:percentuale di giocatori (righe) da usare per addestrare ogni albero
#colsample_bytree: percentuale di statistiche (feature) da usare per ogni albero

# subsample: % di righe campionate per ogni albero
#param_grid = {
#    'xgb__n_estimators': [90, 100, 110],   #
#    'xgb__learning_rate': [0.1, 0.2, 0.3],  #
#    'xgb__max_depth': [3, 4, 5],    #
#    'xgb__subsample': [0.6, 0.7, 0.8],  #
#    'xgb__colsample_bytree': [0.7, 0.8, 0.9]    #
#}

#Solo per dataset con "age" e non "Born"
param_grid = {
    'xgb__n_estimators': [130, 140, 160],   
    'xgb__learning_rate': [0.05, 0.1, 0.2],  
    'xgb__max_depth': [2, 3, 4, 5],    
    'xgb__subsample': [0.6, 0.7, 0.8],  
    'xgb__colsample_bytree': [0.7, 0.8, 0.9]    
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV per XGBoost")

grid_search_fw_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

# FIT sui dati scalati
grid_search_fw_xgb.fit(x_train_fw, y_train_scaled_fw_xgb)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV XGBoost ---")
print(f"Migliori iperparametri: {grid_search_fw_xgb.best_params_}")
print(f"Miglior MAE Scalato in CV: {-grid_search_fw_xgb.best_score_:.4f}")

best_model_fw_xgb = grid_search_fw_xgb.best_estimator_

# --- 6. VALUTAZIONE FINALE (DE-SCALATA) ---
y_pred_train_scaled_fw_xgb = best_model_fw_xgb.predict(x_train_fw)
y_pred_test_scaled_fw_xgb = best_model_fw_xgb.predict(x_test_fw)

# Riportiamo le previsioni in Milioni di Euro
y_pred_train_final_fw_xgb = y_scaler_fw.inverse_transform(y_pred_train_scaled_fw_xgb.reshape(-1, 1))
y_pred_test_final_fw_xgb = y_scaler_fw.inverse_transform(y_pred_test_scaled_fw_xgb.reshape(-1, 1))

mae_test_fw_xgb = mean_absolute_error(y_test_fw, y_pred_test_final_fw_xgb)
mae_train_fw_xgb = mean_absolute_error(y_train_fw, y_pred_train_final_fw_xgb)
r2_test_fw_xgb = r2_score(y_test_fw, y_pred_test_final_fw_xgb)
r2_train_fw_xgb = r2_score(y_train_fw, y_pred_train_final_fw_xgb)

print("\n--- Performance Modello XGBoost Ottimizzato (Attaccanti) ---")
print(f"MAE Training Set: {mae_train_fw_xgb:.2f} M€")
print(f"MAE Test Set: {mae_test_fw_xgb:.2f} M€")
print(f"R2 Score Training: {r2_train_fw_xgb:.4f}")
print(f"R2 Score Test: {r2_test_fw_xgb:.4f}")

print("\n--- Confronto Finale ---")
print(f"Lasso (Attaccanti):    R2 Test = {r2_test_fw_lasso:.4f}")
print(f"SVR RBF (Attaccanti):  R2 Test = {r2_test_fw_svr:.4f}")
print(f"XGBoost (Attaccanti):  R2 Test = {r2_test_fw_xgb:.4f}")

# --- 7. PERMUTATION IMPORTANCE (Basata su MAE Reale) ---
print("\nAvvio calcolo Permutation Importance (XGBoost)...")

# Riutilizziamo la funzione di scoring personalizzata definita per l'SVR
# che de-scala y e calcola il MAE in euro reali.
def unscaled_mae_scorer_xgb(model, X, y_true_unscaled):
    y_pred_scaled = model.predict(X)
    y_pred_unscaled = y_scaler_fw.inverse_transform(y_pred_scaled.reshape(-1, 1))
    return -mean_absolute_error(y_true_unscaled, y_pred_unscaled)

perm_imp_fw_xgb = permutation_importance(
    best_model_fw_xgb, 
    x_test_fw, 
    y_test_fw, 
    n_repeats=20, # Numero di ripetizioni per stabilità
    random_state=42, 
    n_jobs=-1,
    scoring=unscaled_mae_scorer_xgb
)

importance_fw_xgb = pd.DataFrame({
    'feature': x_train_fw.columns, 
    'importance_mean': perm_imp_fw_xgb.importances_mean,
    'importance_std': perm_imp_fw_xgb.importances_std
})

importance_fw_xgb = importance_fw_xgb.sort_values('importance_mean', ascending=False)

print("\n--- Feature Importance dei FW per XGBoost (In € MAE) ---")
print(importance_fw_xgb.head(25))

# Fine cronometro
end_time_fw_xgb = time.time()
execution_time_fw_xgb = end_time_fw_xgb - start_time_fw_xgb
print(f"\n--- Tempo di esecuzione XGBoost: {execution_time_fw_xgb:.2f} secondi ---")
#-------------------------------------------------------------------------------------------------------
#MODELLO DI CONTROLLO XGBoost (BENCHMARCK RANDOM) - ATTACCANTI
#-------------------------------------------------------------------------------------------------------
feature_control = ["random"]
y_column = "value"

print(f"Inizio addestramento XGBoost di CONTROLLO (Benchmark Random) su {len(feature_control)} feature.")

x_train_fw = df_fw_1824[feature_control]
y_train_fw = df_fw_1824[y_column]
x_test_fw = df_fw_2425[feature_control]
y_test_fw = df_fw_2425[y_column]

# --- 2. SCALARE LA Y (TARGET) ---
# Necessario per mantenere il confronto coerente con il modello reale
y_scaler_fw_xgb_ctrl = StandardScaler()
y_train_scaled_fw_xgb_ctrl = y_scaler_fw_xgb_ctrl.fit_transform(y_train_fw.values.reshape(-1, 1)).ravel()

# --- 3. DEFINIZIONE DELLA PIPELINE ---
# Utilizziamo la stessa struttura del modello principale per coerenza scientifica
pipeline_fw_xgb_ctrl = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# --- 4. SELEZIONE IPERPARAMETRI (GridSearchCV) ---
# Usiamo la stessa griglia del modello reale per vedere se l'algoritmo
# riesce "accidentalmente" a trovare pattern nel rumore.
param_grid_fw_xgb_ctrl = {
    'xgb__n_estimators': [50, 70, 90],
    'xgb__learning_rate': [0.1, 0.2, 0.3],
    'xgb__max_depth': [5, 7, 9],
    'xgb__subsample': [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8]
}

tscv = TimeSeriesSplit(n_splits=5)
print("Avvio della GridSearchCV (Benchmark XGBoost Random)...")

grid_search_fw_xgb_ctrl = GridSearchCV(
    pipeline_fw_xgb_ctrl,
    param_grid=param_grid_fw_xgb_ctrl,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1 # Meno verboso rispetto al modello reale
)

# FIT sul rumore (X_train_ctrl contiene solo la variabile random)
grid_search_fw_xgb_ctrl.fit(x_train_fw, y_train_scaled_fw_xgb_ctrl)

# --- 5. RISULTATI ---
print("\n--- Risultati GridSearchCV (Benchmark Random XGBoost) ---")
print(f"Migliori iperparametri trovati: {grid_search_fw_xgb_ctrl.best_params_}")

best_model_fw_xgb_ctrl = grid_search_fw_xgb_ctrl.best_estimator_

# --- 6. VALUTAZIONE FINALE (CON INVERSE_TRANSFORM) ---
# Predizioni sui dati scalati
y_pred_train_scaled_fw_xgb_ctrl = best_model_fw_xgb_ctrl.predict(x_train_fw)
y_pred_test_scaled_fw_xgb_ctrl = best_model_fw_xgb_ctrl.predict(x_test_fw)

# Riportiamo i valori in milioni di euro (de-scaling)
y_pred_train_fw_xgb_ctrl = y_scaler_fw_xgb_ctrl.inverse_transform(y_pred_train_scaled_fw_xgb_ctrl.reshape(-1, 1))
y_pred_test_fw_xgb_ctrl = y_scaler_fw_xgb_ctrl.inverse_transform(y_pred_test_scaled_fw_xgb_ctrl.reshape(-1, 1))

# Calcolo metriche reali
mae_test_fw_xgb_ctrl = mean_absolute_error(y_test_fw, y_pred_test_fw_xgb_ctrl)
mae_train_fw_xgb_ctrl = mean_absolute_error(y_train_fw, y_pred_train_fw_xgb_ctrl)
r2_test_fw_xgb_ctrl = r2_score(y_test_fw, y_pred_test_fw_xgb_ctrl)
r2_train_fw_xgb_ctrl = r2_score(y_train_fw, y_pred_train_fw_xgb_ctrl)

print("\n--- PERFORMANCE MODELLO DI CONTROLLO XGBOOST (RANDOM) - ATTACCANTI ---")
print(f"MAE sul set di TRAINING: {mae_train_fw_xgb_ctrl:.2f} M€")
print(f"MAE sul set di TEST: {mae_test_fw_xgb_ctrl:.2f} M€")
print(f"R2 Score sul set di TRAINING: {r2_train_fw_xgb_ctrl:.4f}")
print(f"R2 Score sul set di TEST: {r2_test_fw_xgb_ctrl:.4f}")

# --- CONFRONTO FINALE PER LA TESI ---
print("\n--- Verifica Potere Predittivo (R2 Test) ---")
print(f"XGBoost Reale (Attaccanti): {r2_test_fw_xgb:.4f}")
print(f"XGBoost Random (Controllo): {r2_test_fw_xgb_ctrl:.4f}")
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#FASE 11 - GRAFICI
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#GRAFICI R2 PER I MODELLI LASSO
#-------------------------------------------------------------------------------------------------------------
# Etichette per l'asse X (i gruppi)
labels_plot = ['Portieri (GK)', 'Difensori (DF)', 'Wingback (WB)', 'Centrocampisti (MF)', 'Attaccanti (FW)']

# Liste di valori R²
test_scores_lasso = [r2_test_gk_lasso, r2_test_df_lasso, r2_test_wb_lasso, r2_test_mf_lasso, r2_test_fw_lasso]
train_scores_lasso = [r2_train_gk_lasso, r2_train_df_lasso, r2_train_wb_lasso, r2_train_mf_lasso, r2_train_fw_lasso]

# Posizionamento sull'asse X
x = np.arange(len(labels_plot))  # le posizioni dei gruppi: [0, 1, 2, 3, 4]
width = 0.35  # Larghezza delle singole barre

#CREAZIONE DEL GRAFICO A BARRE RAGGRUPPATO ---
# Aumenta la dimensione del grafico per una migliore leggibilità
fig, ax = plt.subplots(figsize=(12, 7))

# Crea le barre R² TRAIN
# Spostiamo le barre a DESTRA di 'x' di mezza larghezza (width/2)
rects3 = ax.bar(x - width/2, train_scores_lasso, width, 
                label='R² Train', color='darkorange')
# Crea le barre R² TEST
# Spostiamo le barre a SINISTRA di 'x' di mezza larghezza (width/2)
rects4 = ax.bar(x + width/2, test_scores_lasso, width, 
                label='R² Test', color='royalblue')

#PULIZIA E ETICHETTE ---
# Aggiungi etichette, titolo e legenda
ax.set_ylabel('R² Score')
ax.set_title('Confronto R² Score (Training set vs. Test set) per Lasso Regression', fontsize=16)
# Imposta le etichette dell'asse X al centro dei gruppi di barre
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
# Imposta i limiti dell'asse Y (R² è solitamente tra 0 e 1, ma può essere negativo)
# Aggiusta questo se i tuoi R² sono molto bassi
ax.set_ylim([0, 1])
# Aggiungi una linea a R²=0
ax.axhline(0, color='grey', linewidth=0.8)
# Aggiungi la legenda
ax.legend()
# Aggiungi i valori esatti sopra ogni barra (opzionale, ma utile)
def add_bar_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 punti di offset verticale
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
add_bar_labels(rects3)
add_bar_labels(rects4)
# Ottimizza lo spazio
fig.tight_layout()
# Mostra il grafico
plt.show()

# -------------------------------------------------------------------------------------------------
# Performance sul Training Set (df_df_1824)
# -------------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
# Creo lo scatterplot dei valori reali vs. previsti
plt.scatter(y_train_df, y_pred_train_df, alpha=0.5, label='Predizioni Training')

# Aggiungo la linea a 45 gradi (Predizione Perfetta)
# Troviamo i limiti min/max per tracciare la linea
all_train_values = np.concatenate([y_train_df, y_pred_train_df])
min_val = all_train_values.min()
max_val = all_train_values.max()

plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predizione Perfetta (Reale = Previsto)')

# Etichette e Titolo
plt.xlabel("Valore reale (Milioni di €)")
plt.ylabel("Valore previsto (Milioni di €)")
plt.title("Performance Modello su Training Set (df_df_1824)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------------------------------
# Performance sul Test Set (df_df_2425)
# -------------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(y_test_df, y_pred_test_df, alpha=0.5, color='green', label='Predizioni Test')

# Aggiungo la linea a 45 gradi
all_test_values = np.concatenate([y_test_df, y_pred_test_df])
min_val_test = all_test_values.min()
max_val_test = all_test_values.max()

plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2, label='Predizione Perfetta (Reale = Previsto)')

# Etichette e Titolo
plt.xlabel("Valore Reale (Milioni di €)")
plt.ylabel("Valore Previsto (Milioni di €)")
plt.title("Performance Modello su Test Set (df_df_2425)")
plt.legend()
plt.grid(True)
plt.show()
# -------------------------------------------------------------------------------------
# Grafico 1: Performance sul Training Set (df_wb_1824)
# -------------------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
# Creo lo scatterplot dei valori reali vs. previsti
plt.scatter(y_train_wb, y_pred_train_wb, alpha=0.5, label='Predizioni Training')

# Aggiungo la linea a 45 gradi (Predizione Perfetta)
# Troviamo i limiti min/max per tracciare la linea
all_train_values = np.concatenate([y_train_wb, y_pred_train_wb])
min_val = all_train_values.min()
max_val = all_train_values.max()

plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predizione Perfetta (Reale = Previsto)')

# Etichette e Titolo
plt.xlabel("Valore reale (Milioni di €)")
plt.ylabel("Valore previsto (Milioni di €)")
plt.title("Performance Modello su Training Set (df_wb_1824)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# Grafico 2: Performance sul Test Set (df_wb_2425)
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(y_test_wb, y_pred_test_wb, alpha=0.5, color='green', label='Predizioni Test')

# Aggiungo la linea a 45 gradi
all_test_values = np.concatenate([y_test_wb, y_pred_test_wb])
min_val_test = all_test_values.min()
max_val_test = all_test_values.max()

plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2, label='Predizione Perfetta (Reale = Previsto)')

# Etichette e Titolo
plt.xlabel("Valore Reale (Milioni di €)")
plt.ylabel("Valore Previsto (Milioni di €)")
plt.title("Performance Modello su Test Set (df_wb_2425)")
plt.legend()
plt.grid(True)
plt.show()
# -------------------------------------------------------------------
# Grafico 1: Performance sul Training Set (df_mf_1824)
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
# Creo lo scatterplot dei valori reali vs. previsti
plt.scatter(y_train_mf, y_pred_train_mf, alpha=0.5, label='Predizioni Training')

# Aggiungo la linea a 45 gradi (Predizione Perfetta)
# Trovo i limiti min/max per tracciare la linea
all_train_values = np.concatenate([y_train_mf, y_pred_train_mf])
min_val = all_train_values.min()
max_val = all_train_values.max()

plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predizione Perfetta (Reale = Previsto)')

# Etichette e Titolo
plt.xlabel("Valore reale (Milioni di €)")
plt.ylabel("Valore previsto (Milioni di €)")
plt.title("Performance Modello su Training Set (df_mf_1824)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# Grafico 2: Performance sul Test Set (df_mf_2425)
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(y_test_mf, y_pred_test_mf, alpha=0.5, color='green', label='Predizioni Test')

# Aggiungo la linea a 45 gradi
all_test_values = np.concatenate([y_test_mf, y_pred_test_mf])
min_val_test = all_test_values.min()
max_val_test = all_test_values.max()

plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2, label='Predizione Perfetta (Reale = Previsto)')

# Etichette e Titolo
plt.xlabel("Valore Reale (Milioni di €)")
plt.ylabel("Valore Previsto (Milioni di €)")
plt.title("Performance Modello su Test Set (df_mf_2425)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# Grafico 1: Performance sul Training Set (df_fw_1824)
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
# Creo lo scatterplot dei valori reali vs. previsti
plt.scatter(y_train_fw, y_pred_train_fw, alpha=0.5, label='Predizioni Training')

# Aggiungo la linea a 45 gradi (Predizione Perfetta)
# Trovo i limiti min/max per tracciare la linea
all_train_values = np.concatenate([y_train_fw, y_pred_train_fw])
min_val = all_train_values.min()
max_val = all_train_values.max()

plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predizione Perfetta (Reale = Previsto)')

# Etichette e Titolo
plt.xlabel("Valore reale (Milioni di €)")
plt.ylabel("Valore previsto (Milioni di €)")
plt.title("Performance Modello su Training Set (df_fw_1824)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# Grafico 2: Performance sul Test Set (df_fw_2425)
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(y_test_fw, y_pred_test_fw, alpha=0.5, color='green', label='Predizioni Test')

# Aggiungo la linea a 45 gradi
all_test_values = np.concatenate([y_test_fw, y_pred_test_fw])
min_val_test = all_test_values.min()
max_val_test = all_test_values.max()

plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2, label='Predizione Perfetta (Reale = Previsto)')

# Etichette e Titolo
plt.xlabel("Valore Reale (Milioni di €)")
plt.ylabel("Valore Previsto (Milioni di €)")
plt.title("Performance Modello su Test Set (df_fw_2425)")
plt.legend()
plt.grid(True)
plt.show()
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#GRAFICI R2 PER I MODELLI LASSO
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# Etichette per l'asse X (i gruppi)
labels_plot = ['Portieri (GK)', 'Difensori (DF)', 'Wingback (WB)', 'Centrocampisti (MF)', 'Attaccanti (FW)']

# Liste di valori R²
test_scores_lasso = [r2_test_gk_lasso, r2_test_df_lasso, r2_test_wb_lasso, r2_test_mf_lasso, r2_test_fw_lasso]
train_scores_lasso = [r2_train_gk_lasso, r2_train_df_lasso, r2_train_wb_lasso, r2_train_mf_lasso, r2_train_fw_lasso]

# Posizionamento sull'asse X
x = np.arange(len(labels_plot))  # le posizioni dei gruppi: [0, 1, 2, 3, 4]
width = 0.35  # Larghezza delle singole barre

#CREAZIONE DEL GRAFICO A BARRE RAGGRUPPATO ---
# Aumento la dimensione del grafico per una migliore leggibilità
fig, ax = plt.subplots(figsize=(12, 7))

# Creo le barre R² TRAIN
# Sposto le barre a DESTRA di 'x' di mezza larghezza (width/2)
rects1 = ax.bar(x - width/2, train_scores_lasso, width, 
                label='R² Train', color='darkorange')
# Crea le barre R² TEST
# Sposto le barre a SINISTRA di 'x' di mezza larghezza (width/2)
rects2 = ax.bar(x + width/2, test_scores_lasso, width, 
                label='R² Test', color='royalblue')

#PULIZIA E ETICHETTE ---
# Aggiungo etichette, titolo e legenda
ax.set_ylabel('R² Score')
ax.set_title('Confronto R² Score (Training set vs. Test set) per LASSO', fontsize=16)
# Imposto le etichette dell'asse X al centro dei gruppi di barre
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
# Imposto i limiti dell'asse Y (R² è solitamente tra 0 e 1, ma può essere negativo)
# Aggiusto questo se i tuoi R² sono molto bassi
ax.set_ylim([0, 1])
# Aggiungo una linea a R²=0
ax.axhline(0, color='grey', linewidth=0.8)
# Aggiungo la legenda
ax.legend()
# Aggiungo i valori esatti sopra ogni barra (opzionale, ma utile)
def add_bar_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 punti di offset verticale
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
add_bar_labels(rects1)
add_bar_labels(rects2)
# Ottimizzo lo spazio
fig.tight_layout()
# Mostro il grafico
plt.show()
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#GRAFICI R2 PER I MODELLI SVR
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# Etichette per l'asse X (i gruppi)
labels_plot = ['Portieri (GK)', 'Difensori (DF)', 'Wingback (WB)', 'Centrocampisti (MF)', 'Attaccanti (FW)']

# Liste di valori R²
test_scores_svr = [r2_test_gk_svr, r2_test_df_svr, r2_test_wb_svr, r2_test_mf_svr, r2_test_fw_svr]
train_scores_svr = [r2_train_gk_svr, r2_train_df_svr, r2_train_wb_svr, r2_train_mf_svr, r2_train_fw_svr]

# Posizionamento sull'asse X
x = np.arange(len(labels_plot))  # le posizioni dei gruppi: [0, 1, 2, 3, 4]
width = 0.35  # Larghezza delle singole barre

#CREAZIONE DEL GRAFICO A BARRE RAGGRUPPATO ---
# Aumenta la dimensione del grafico per una migliore leggibilità
fig, ax = plt.subplots(figsize=(12, 7))

# Crea le barre R² TRAIN
# Spostiamo le barre a DESTRA di 'x' di mezza larghezza (width/2)
rects3 = ax.bar(x - width/2, train_scores_svr, width, 
                label='R² Train', color='darkorange')
# Crea le barre R² TEST
# Spostiamo le barre a SINISTRA di 'x' di mezza larghezza (width/2)
rects4 = ax.bar(x + width/2, test_scores_svr, width, 
                label='R² Test', color='royalblue')

#PULIZIA E ETICHETTE ---
# Aggiungi etichette, titolo e legenda
ax.set_ylabel('R² Score')
ax.set_title('Confronto R² Score (Training set vs. Test set) per SVR', fontsize=16)
# Imposta le etichette dell'asse X al centro dei gruppi di barre
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
# Imposta i limiti dell'asse Y (R² è solitamente tra 0 e 1, ma può essere negativo)
# Aggiusta questo se i tuoi R² sono molto bassi
ax.set_ylim([0, 1])
# Aggiungi una linea a R²=0
ax.axhline(0, color='grey', linewidth=0.8)
# Aggiungi la legenda
ax.legend()
# Aggiungi i valori esatti sopra ogni barra (opzionale, ma utile)
def add_bar_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 punti di offset verticale
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
add_bar_labels(rects3)
add_bar_labels(rects4)
# Ottimizza lo spazio
fig.tight_layout()
# Mostra il grafico
plt.show()
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#GRAFICI R2 PER I MODELLI XGBoost
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# Etichette per l'asse X (i gruppi)
labels_plot = ['Portieri (GK)', 'Difensori (DF)', 'Wingback (WB)', 'Centrocampisti (MF)', 'Attaccanti (FW)']

# Liste di valori R²
test_scores_xgb = [r2_test_gk_xgb, r2_test_df_xgb, r2_test_wb_xgb, r2_test_mf_xgb, r2_test_fw_xgb]
train_scores_xgb = [r2_train_gk_xgb, r2_train_df_xgb, r2_train_wb_xgb, r2_train_mf_xgb, r2_train_fw_xgb]

# Posizionamento sull'asse X
x = np.arange(len(labels_plot))  # le posizioni dei gruppi: [0, 1, 2, 3, 4]
width = 0.35  # Larghezza delle singole barre

#CREAZIONE DEL GRAFICO A BARRE RAGGRUPPATO ---
# Aumenta la dimensione del grafico per una migliore leggibilità
fig, ax = plt.subplots(figsize=(12, 7))

# Crea le barre R² TRAIN
# Spostiamo le barre a DESTRA di 'x' di mezza larghezza (width/2)
rects3 = ax.bar(x - width/2, train_scores_xgb, width, 
                label='R² Train', color='darkorange')
# Crea le barre R² TEST
# Spostiamo le barre a SINISTRA di 'x' di mezza larghezza (width/2)
rects4 = ax.bar(x + width/2, test_scores_xgb, width, 
                label='R² Test', color='royalblue')

#PULIZIA E ETICHETTE ---
# Aggiungi etichette, titolo e legenda
ax.set_ylabel('R² Score')
ax.set_title('Confronto R² Score (Training set vs. Test set) per XGBoost', fontsize=16)
# Imposta le etichette dell'asse X al centro dei gruppi di barre
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
# Imposta i limiti dell'asse Y (R² è solitamente tra 0 e 1, ma può essere negativo)
# Aggiusta questo se i tuoi R² sono molto bassi
ax.set_ylim([0, 1])
# Aggiungi una linea a R²=0
ax.axhline(0, color='grey', linewidth=0.8)
# Aggiungi la legenda
ax.legend()
# Aggiungi i valori esatti sopra ogni barra (opzionale, ma utile)
def add_bar_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 punti di offset verticale
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
add_bar_labels(rects3)
add_bar_labels(rects4)
# Ottimizza lo spazio
fig.tight_layout()
# Mostra il grafico
plt.show()
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#GRAFICI R2 PER I MODELLI A CONFRONTO
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# 1. Preparazione dei dati
labels_plot = ['Portieri (GK)', 'Difensori (DF)', 'Wingback (WB)', 'Centrocampisti (MF)', 'Attaccanti (FW)']


test_scores_lasso = [r2_test_gk_lasso, r2_test_df_lasso, r2_test_wb_lasso, r2_test_mf_lasso, r2_test_fw_lasso]
test_scores_svr  = [r2_test_gk_svr, r2_test_df_svr, r2_test_wb_svr, r2_test_mf_svr, r2_test_fw_svr]
test_scores_xgb  = [r2_test_gk_xgb, r2_test_df_xgb, r2_test_wb_xgb, r2_test_mf_xgb, r2_test_fw_xgb]

x = np.arange(len(labels_plot))  # Posizione delle etichette
width = 0.25  # Larghezza delle singole barre

# 2. Creazione del grafico
fig, ax = plt.subplots(figsize=(12, 7))

# Disegno delle tre serie di barre
rects1 = ax.bar(x - width, test_scores_lasso, width, label='Lasso Regression', color='#FFFF00') # Giallo
rects2 = ax.bar(x, test_scores_svr , width, label='SVR (RBF)', color='#FFA500')                # Arancione
rects3 = ax.bar(x + width, test_scores_xgb, width, label='XGBoost', color='#FF0000')         # Rosso

# 3. Personalizzazione estetica
ax.set_ylabel('R2 Score (Test Set)')
ax.set_title('Confronto della performance dei modelli per Ruolo (R2 Score sul test set)')
ax.set_xticks(x)
ax.set_xticklabels(labels_plot)
ax.set_ylim(0, 1.0) # L'R2 solitamente va da 0 a 1 (se positivo)
ax.legend()

# Aggiungiamo una griglia orizzontale per facilitare la lettura
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Funzione per aggiungere il valore numerico sopra ogni barra
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

# 4. Mostra e salva il grafico
plt.show()
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#EXTRA - TRASFORMA I DATAFRAME IN FILE CSV

#Tutti i ruoli dal 2018 al 2024
df_gk_1824.to_csv('dati_portieri_1824.csv', index=False, sep=';', encoding='utf-8-sig')
df_df_1824.to_csv('dati_difensori_1824.csv', index=False, sep=';', encoding='utf-8-sig')
df_wb_1824.to_csv('dati_wingback_1824.csv', index=False, sep=';', encoding='utf-8-sig')
df_mf_1824.to_csv('dati_centrocampisti_1824.csv', index=False, sep=';', encoding='utf-8-sig')
df_fw_1824.to_csv('dati_attaccanti_1824.csv', index=False, sep=';', encoding='utf-8-sig')

#Portieri per ogni stagione
df_gk_1819.to_csv('dati_portieri_1819.csv', index=False, sep=';', encoding='utf-8-sig')
df_gk_1920.to_csv('dati_portieri_1920.csv', index=False, sep=';', encoding='utf-8-sig')
df_gk_2021.to_csv('dati_portieri_2021.csv', index=False, sep=';', encoding='utf-8-sig')
df_gk_2122.to_csv('dati_portieri_2122.csv', index=False, sep=';', encoding='utf-8-sig')
df_gk_2223.to_csv('dati_portieri_2223.csv', index=False, sep=';', encoding='utf-8-sig')
df_gk_2324.to_csv('dati_portieri_2324.csv', index=False, sep=';', encoding='utf-8-sig')
df_gk_2425.to_csv('dati_portieri_2425.csv', index=False, sep=';', encoding='utf-8-sig')

#Difensori per ogni stagione
df_df_1819.to_csv('dati_difensori_1819.csv', index=False, sep=';', encoding='utf-8-sig')
df_df_1920.to_csv('dati_difensori_1920.csv', index=False, sep=';', encoding='utf-8-sig')
df_df_2021.to_csv('dati_difensori_2021.csv', index=False, sep=';', encoding='utf-8-sig')
df_df_2122.to_csv('dati_difensori_2122.csv', index=False, sep=';', encoding='utf-8-sig')
df_df_2223.to_csv('dati_difensori_2223.csv', index=False, sep=';', encoding='utf-8-sig')
df_df_2324.to_csv('dati_difensori_2324.csv', index=False, sep=';', encoding='utf-8-sig')
df_df_2425.to_csv('dati_difensori_2425.csv', index=False, sep=';', encoding='utf-8-sig')

#Wingback per ogni stagione
df_wb_1819.to_csv('dati_wingback_1819.csv', index=False, sep=';', encoding='utf-8-sig')
df_wb_1920.to_csv('dati_wingback_1920.csv', index=False, sep=';', encoding='utf-8-sig')
df_wb_2021.to_csv('dati_wingback_2021.csv', index=False, sep=';', encoding='utf-8-sig')
df_wb_2122.to_csv('dati_wingback_2122.csv', index=False, sep=';', encoding='utf-8-sig')
df_wb_2223.to_csv('dati_wingback_2223.csv', index=False, sep=';', encoding='utf-8-sig')
df_wb_2324.to_csv('dati_wingback_2324.csv', index=False, sep=';', encoding='utf-8-sig')
df_wb_2425.to_csv('dati_wingback_2425.csv', index=False, sep=';', encoding='utf-8-sig')

#Centrocampisti per ogni stagione
df_mf_1819.to_csv('dati_centrocampisti_1819.csv', index=False, sep=';', encoding='utf-8-sig')
df_mf_1920.to_csv('dati_centrocampisti_1920.csv', index=False, sep=';', encoding='utf-8-sig')
df_mf_2021.to_csv('dati_centrocampisti_2021.csv', index=False, sep=';', encoding='utf-8-sig')
df_mf_2122.to_csv('dati_centrocampisti_2122.csv', index=False, sep=';', encoding='utf-8-sig')
df_mf_2223.to_csv('dati_centrocampisti_2223.csv', index=False, sep=';', encoding='utf-8-sig')
df_mf_2324.to_csv('dati_centrocampisti_2324.csv', index=False, sep=';', encoding='utf-8-sig')
df_mf_2425.to_csv('dati_centrocampisti_2425.csv', index=False, sep=';', encoding='utf-8-sig')

#Attaccanti per ogni stagione
df_fw_1819.to_csv('dati_attaccanti_1819.csv', index=False, sep=';', encoding='utf-8-sig')
df_fw_1920.to_csv('dati_attaccanti_1920.csv', index=False, sep=';', encoding='utf-8-sig')
df_fw_2021.to_csv('dati_attaccanti_2021.csv', index=False, sep=';', encoding='utf-8-sig')
df_fw_2122.to_csv('dati_attaccanti_2122.csv', index=False, sep=';', encoding='utf-8-sig')
df_fw_2223.to_csv('dati_attaccanti_2223.csv', index=False, sep=';', encoding='utf-8-sig')
df_fw_2324.to_csv('dati_attaccanti_2324.csv', index=False, sep=';', encoding='utf-8-sig')
df_fw_2425.to_csv('dati_attaccanti_2425.csv', index=False, sep=';', encoding='utf-8-sig')
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#FEATURE IMPORTANCE
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#La Feature Importance (Importanza delle Caratteristiche) è una tecnica utilizzata nel Machine Learning 
#per assegnare un punteggio a ciascuna feature di input in base a quanto quella feature è stata utile o 
#significativa per il modello nel fare previsioni accurate.
#In pratica è la tecnica per capire quali feature sono ridondanti e quali sono utili a fini previsionali. 

#PER I MODELLI BASATI SU ALBERI (RANDOM FOREST, GRADIENT BOOSTING, XGBoost) 
# La PERMUTATION FEATURE IMPORTANCE (PFI) misura l'incremento nell'errore di previsione del modello dopo che
#abbiamo permutato casualmente i valori della feature, in modo da farci capire se la feature ha una relazione
#forte con la variabile target.
#Il concetto alla base è semplice.
#Una feature è "importante" se eseguendo una permutazione casuale l'errore del modello aumenta perchè in questo caso
#il modello si basava sulla feature per eseguire le previsioni.
#Una feature non è importangte se eseguendo una permutazione dei suoi valori l'errore del modello non cambia
#perchè in questo caso il modello sta ignorando la feature a scopi predittivi.
#PER INFO: (https://christophm.github.io/interpretable-ml-book/feature-importance.html)