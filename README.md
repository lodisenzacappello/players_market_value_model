MODELLAZIONE DEI VALORI DI MERCATO DEI CALCIATORI DI SERIE A

Nel mondo del calcio moderno sta iniziando lentamente una rivoluzione basata sui numeri. 
Ispirandosi al modello “Moneyball”, reso famoso al grande pubblico dal film con Brad Pitt nel 2011, sempre più squadre europee hanno capito l’importanza dell’utilizzo dei dati per prendere decisioni informate e guadagnare un vantaggio competitivo rispetto alle dirette concorrenti. 
Sono ormai molte le società professionistiche che si affidano ad aziende esterne dedite all’analisi dati in campo sportivo per studiare approfonditamente le statistiche dei propri atleti. 
I dati vengono utilizzati sia per studiare strategie di campo efficaci sia per valutare le prestazioni dei propri calciatori. 
Nonostante questo nuovo approccio allo sport, l’utilizzo dei dati nel mondo del calciomercato e specialmente nel calciomercato italiano è ancora in fase embrionale e solo pochi club hanno cercato di implementare queste nuove tecnologie.

Nel progetto vado a modellare i valori di mercato di Transfermarkt tramite 214 variabili di campo.
Lo scopo è individuare un modello efficace nel prevedere il valore di mercato di un calciatore e definire quali variabili influenzano maggiormente il vdm per i diversi ruoli.

Il dataset in analisi è composto da XXXX calciatori che hanno giocato in Serie A tra la stagione 2018-2019 e la stagione 2024-2025. 
Le statistiche di campo provengono da FBRef (provider Opta) mentre i valori di mercato provengono da TransferMarkt.

#------------------------------------------------------------------------------------------------------------------------------------------------------------


Ho definito 5 ruoli in base ai ruoli possibili su Transfermarkt. I ruoli vanno a definire 5 gruppi di calciatori.

1)GK = GOALKEEPER = Portiere

2)DF = DEFENDER = Difensore centrale

3)WB = WINGBACK = Esterno di destra/Esterno di sinistra/Terzino destro/Terzino sinistro

4)MF = MIDFIELDER = Mediano/Centrocampista/Trequartista

5)FW = ATTACCANTE = Ala destra/Ala sinistra/Punta centrale/Seconda Punta

#------------------------------------------------------------------------------------------------------------------------------------------------------------

I modelli utilizzati sono:


1)LASSO REGRESSION

2)SUPPORT VECTOR REGRESSION (SVR) CON KERNEL RBF

3)XGBoost 

#------------------------------------------------------------------------------------------------------------------------------------------------------------

I valori dei coefficienti di determinazione sono:

1) PORTIERI (GK)

		R2 Lasso =   0.5326
		R2 SVR =     0.3268
		R2 XGBoost = 0.7679*

2) DIFENSORI (DF)

		R2 Lasso =   0.3587
		R2 SVR =     0.5213*
		R2 XGBoost = 0.5056

3) ESTERNI (WB)

		R2 Lasso =   0.5920
		R2 SVR =     0.6410
		R2 XGBoost = 0.6500*

4) CENTROCAMPISTI (MF)

		R2 Lasso =   0.4987
		R2 SVR =     0.5093
		R2 XGBoost = 0.6011*

5) ATTACCANTI (FW)

		R2 Lasso =   0.5409
		R2 SVR =     0.5429
		R2 XGBoost = 0.6195*
#------------------------------------------------------------------------------------------------------------------------------------------------------------
Le fasi del progetto sono le seguenti:

FASE 0- IMPORTAZIONE DELLE LIBRERIE

FASE 1- CREAZIONE DELLE LISTE PER IL CORRETTO DOWNLOAD DEI DATI DA FBREF

FASE 2- PROGRAMMA PER SCARICARE LE TABELLE DA FBREF

FASE 3 - DATA CLEANING DEI DATASET DI FBREF

FASE 4 - CREAZIONE DI UN UNICO DATASET CON LE STATISTICHE DI CAMPO PER OGNI STAGIONE (DF_FINAL)

FASE 5 - SCARICARE I VALORI DI MERCATO

FASE 6 - PULIZIA DEI VDM PRIMA DEL MERGE DEI DATAFRAME

FASE 7 - ESEGUIRE IL MERGE DEI DATAFRAME VDM E DF_FINAL

FASE 8 - DIVIDERE I DF_MERGE IN DATAFRAME DI TRAINING E VALIDATION

FASE 9 - GRAFICI ESPLORATIVI DEI DATASET

FASE 10 - DELINEAZIONE DEI MODELLI DI MACHINE LEARNING

FASE 10.1 - LASSO REGRESSION 

	FASE 10.1.1 - LASSO PORTIERI	
	FASE 10.1.2 - LASSO DIFENSORI
	FASE 10.1.3 - LASSO WINGBACK
	FASE 10.1.4 - LASSO CENTROCAMPISTI
	FASE 10.1.5 - LASSO ATTACCANTI

FASE 10.2 - SUPPORT VECTOR REGRESSION (SVR) - KERNEL RBF

	FASE 10.2.1 - SVR PORTIERI
	FASE 10.2.2 - SVR DIFENSORI
	FASE 10.2.3 - SVR WINGBACK
	FASE 10.2.4 - SVR CENTROCAMPISTI
	FASE 10.2.5 - SVR ATTACCANTI

FASE 10.3 - XGBoost

	FASE 10.3.1 - XGBoost PORTIERI
	FASE 10.3.2 - XGBoost DIFENSORI
	FASE 10.3.3 - XGBoost WINGBACK
	FASE 10.3.4 - XGBoost CENTROCAMPISTI
	FASE 10.3.5 - XGBoost ATTACCANTI

FASE 11 - GRAFICI
