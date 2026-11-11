# Verifica dell'esperimento di recommendation

## Esito

Sono state ricostruite le liste Top-30 dei cinque modelli: **601.800 righe**, cinque modelli per 4.012 buyer per trenta posizioni. Sono inclusi codice eseguibile, pesi dei modelli latenti, metriche per buyer, catalogo, profili, audit dei target e dati di esposizione per tutte le categorie e tutti i prodotti.

Il confronto comprende 239 controlli numerici: 229 valori puntuali e dieci estremi di intervalli bootstrap. Dei valori puntuali, 214 coincidono alla precisione del riferimento; 15 controlli differiscono e riguardano soltanto NCF. Alcuni controlli ripetono una metrica presente sia nella tabella principale sia nell'appendice: non sono confronti statisticamente indipendenti.

**Global quantity, Category quantity, ItemKNN e BPR-MF corrispondono a tutti i riferimenti disponibili controllati. NCF non corrisponde esattamente in tutto.** Non sarebbe corretto presentare l'intera ricostruzione come identica ai risultati riportati.

## Differenze NCF a Top-20

| Metrica | Riferimento nella tesi | Ricostruzione |
|---|---:|---:|
| Precision | 0,070% | 0.069791% |
| Recall | 0,129% | 0.128784% |
| NDCG | 0,101% | 0.101159% |
| MRR | 0,251% | 0.248883% |
| Head | 16,99% | 16.964108% |
| Mid | 23,26% | 23.263958% |
| Tail | 59,76% | 59.771934% |
| Tail basato sul rango | 77,07% | 77.089980% |
| Coverage | 4,88% | 4.858366% |

Per NCF i 56 hit Top-20, Precision, Recall, NDCG e hit rate coincidono alla precisione della tabella. Restano differenze di ordinamento e di allocazione degli spazi, oltre ad alcuni indicatori di diversita e ai controlli Top-10/Top-30. Le differenze percentuali riportate arrivano a circa 0,0263 punti percentuali. Sono piccole, ma non nulle: tutte sono conservate in `compatibility_report.csv`, senza correggere a mano i dati.

## Controlli superati

- 1.726.977 righe pulite; 1.381.836 training e 345.141 validation.
- Fine training 18 aprile 2022; inizio validation 19 aprile 2022; date interamente separate.
- Core di training: 4.507 buyer, 30.360 prodotti, 308.854 coppie distinte.
- Stessi 4.012 buyer valutati per ogni modello; nessun campionamento di buyer o candidati.
- Soglie HTB di training 15,1975296443 e 35,5883339786.
- Liste senza ripetizioni e senza prodotti gia acquistati in training.
- Stesse impostazioni dei metodi: seed 42, due epoche per i modelli latenti, architetture, learning rate, batch size e campionamento negativo documentati.
- Gradienti BPR e NCF confrontati con differenze finite.
- Perdite di training delle due epoche coincidenti con i valori documentati alla precisione disponibile.
- Tutti i dieci estremi degli intervalli Recall@20 coincidono al terzo decimale percentuale. Bootstrap: seed 42; modelli in ordine alfabetico; metriche Precision, Recall, NDCG, MRR, hit rate, end-to-end Recall, in questo ordine, con 1.000 estrazioni per metrica.
- Le 80.240 posizioni Global quantity Top-20 coincidono esattamente con il controllo indipendente disponibile.
- Le celle visualizzate nella heatmap delle categorie coincidono alla precisione di una cifra decimale.

## Limite della verifica

Le liste individuali originali dei cinque modelli e il codice NumPy originale non sono disponibili. Tabelle e figure conservano valori arrotondati, non l'intera informazione dei fit. Non e quindi possibile certificare l'identita record per record con gli output originali, anche per i quattro modelli i cui indicatori coincidono.

Sono state controllate anche diverse realizzazioni numeriche della stessa rete NCF. Questo non costituisce una ricerca di iperparametri o un benchmark ottimizzato: dati, architettura, seed, epoche e impostazioni statistiche restano fissi. I dettagli aritmetici e il backend possono cambiare gli ultimi ordinamenti. La causa esatta della discrepanza rispetto all'esecuzione originale non e stata identificata in modo univoco. Il pacchetto fissa e documenta una realizzazione ripetibile, con training float32 e un thread BLAS; i pesi finali sono stati ricreati e confrontati byte per byte a livello di array in una seconda esecuzione dello stesso fit.

## Conservazione e utilizzo

I CSV contengono valori naturali tra zero e uno, salvo conteggi e colonne esplicitamente in punti percentuali. Il confronto applica la conversione in percentuale dichiarata dalla tesi. Il file `published_reference.json` conserva il riferimento numerico; `compatibility_report.csv` conserva ogni confronto.

La tesi LaTeX e il suo PDF non sono stati modificati. Prima di combinare nuove analisi di NCF con valori gia riportati, occorre mantenere esplicita questa discrepanza oppure adottare in modo coerente i valori di questa esecuzione.
