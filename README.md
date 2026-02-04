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


Nella fase di merge dei dataset ho definito 5 ruoli che vanno a definire 5 gruppi di calciatori.

1)GK = GOALKEEPER = Portiere

2)DF = DEFENDER = Difensore centrale

3)WB = WINGBACK = Esterno di destra/Esterno di sinistra/Terzino destro/Terzino sinistro

4)MF = MIDFIELDER = Mediano/Centrocampista/Trequartista

5)FW = ATTACCANTE = Ala destra/Ala sinistra/Punta centrale/Seconda Punta

#------------------------------------------------------------------------------------------------------------------------------------------------------------

I modelli utilizzati sono:


1)LASSO REGRESSION

2)SUPPORT VECTOR REGRESSION (SVR)

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
