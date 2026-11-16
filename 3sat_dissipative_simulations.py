import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import h5py #per salvare gli ouput dati

    
L=9
m=1+(L-3)/2
#m=L-2

#quante istanze vuoi simulare?
N_hamiltoniane=70

master_seed_H=16 #unico generatore per tutti gli rng


#parametri per le simulazioni di hamiltoniana singola
N_simulazioni=300
N_passi=100
master_seed=17 #non viene usato (non togliere comunque) 

#quante variabili sono CONDIVISE? lasciare non commentato quella che si vuole usare
var_cond=1
#var_cond=2

#seleziona ARCHIVIO
#nome_archivio="simul_3sat.h5"   #primo archivio, solo due medie_E per L=5
#nome_archivio="simul2_3sat.h5"  #secondo archivio, una variabile condivisa, codice corretto
#nome_archivio="simul3_3sat.h5"   #terzo archivio, due variabili condivise
#nome_archivio="simul4_3sat.h5"
nome_archivio="simul5_3sat.h5"

#operatori di Pauli
I=np.eye(2)
Sx=np.array([[0.,1.],[1.,0.]])
Sy=np.array([[0.,-1.0j],[1.0j, 0.]])
Sz=np.array([[1.,0.],[0.,-1.]])
op_pauli=[I, Sx, Sy, Sz]

#proiettori up e down
P_up=np.array([[1., 0.],[0., 0.]])
P_down=np.array([[0., 0.],[0., 1.]])

#tolleranza per arrivare nel gs
epsilon_gs=0.01  

#creo una classe per contenere la forma degli operatori e i siti su cui agiscono
class Proiettore:
    def __init__(self, forma, sito):
        self.forma=np.asarray(forma)

        arr=np.asarray(sito, dtype=int) #così siti può essere sia scalare che intero; il primo qubit è il qubit 0
        if arr.ndim==0:
            self.sito=int(arr.item())
        else:
            self.sito=list(sito)

    def __repr__(self):
        return(f"Proiettore: \n"
               f"Forma={self.forma} \n"
               f"Sito={self.sito} \n")



def generatore_istanze(rng, L):  #usalo solo con L dispari
    #genero le sequenze di siti su cui agisce l'istanza (sono già le istanze negate)
    siti=[]
    num_clausole=1+(L-3)//2  #tranne la prima clausola, tutte le altre aggiungono due variabili ogni volta;
                            #// restituisce l'intero minore più vicino (senza resto quindi)

    if L%2==1:  #L dispari = unica combinazione possibile di siti
        for i in range(0, num_clausole*2, 2):  #la prima variabile di ogni nuova clausola avanza di due unità rispetto alla prima variabile della clausola precedente
            sito=np.array([i, i+1, i+2])
            siti.append(sito)
    elif L%2==0:  #L pari
        print("Si è inserito una L pari!")


    #genero le vere e proprie clausole
    num_clausole=len(siti)
    istanza=[]
    for i in range (num_clausole):
        verita=rng.integers(0,2, 3)
        clausola=[verita, siti[i]]
        istanza.append(clausola)

    return istanza


def generatore_istanze2(rng, L): #due variabili condivise, sia per L pari che dispari
    siti=[]
    num_clausole=L-2  #tranne la prima clausola, le altre richiedono solo una variabile in più rispetto alla precedente

    #elenco dei siti
    for i in range(0, num_clausole):
        sito=np.array([i, i+1, i+2])
        siti.append(sito)

    #genero clausole
    num_clausole=len(siti)
    istanza=[]
    for i in range(num_clausole):
        verita=rng.integers(0,2, 3)
        clausola=[verita, siti[i]]
        istanza.append(clausola)

    return istanza


def frustration_rate(istanza):
    k=0
    if var_cond==1:
        for i in range(len(istanza)-1):
            shared_i=istanza[i][0][2]
            shared_i2=istanza[i+1][0][0]
            if shared_i!=shared_i2: k+=1
    if var_cond==2:
        for i in range(len(istanza)-1):
            shared1_i=istanza[i][0][1]
            shared2_i=istanza[i][0][2]
            shared1_i2=istanza[i+1][0][0]
            shared2_i2=istanza[i+1][0][1]
            if (shared1_i!=shared1_i2): k+=1
            if (shared2_i!=shared2_i2): k+=1
    return k






def istanza_to_hamiltoniana(istanza):
    #istanza è un oggetto del tipo
    #istanza_1=[[(0,0,1),(1,2,3)], [(0,1,1),(3,4,5)]]  (lista di liste di due array da 3 numeri)

    #trasformo ciascuna clausola in un termine dell'hamiltoniana
    H=[] #lista di liste di "Proiettore"
    for clausola in istanza:
        termine=[]
        for i in range(3):
            if (clausola[0][i])==0: P=P_down
            else: P=P_up
            qubit=clausola[1][i]
            proiettore=Proiettore(P, qubit)
            termine.append(proiettore)
        H.append(termine)

    return H  #lista dei termini dell'hamiltoniana




def projection_singole(proiettori, stato): #proiettori deve essere una lista di oggetti Proiettore a singolo qubit; la funzione è pensata per quando ho vari proiettori singoli associati al proprio qubit
    dim=len(stato)
    #"stato" deve essere già in forma di tensore multiindice

    for proiettore in proiettori:
        qubit=proiettore.sito
        forma=proiettore.forma

        stato=np.tensordot(forma, stato, axes=([1], [qubit]))
        stato=np.moveaxis(stato, 0, qubit)
        #(vedi depolarizing_channel_Lq per dettagli su queste operazioni)
            

    return stato



def depolarizing_channel_Lq(rng, stato, siti): 
    #applico ciascun operatore di pauli (estratto casualmente) al qubit corrispondente
    #"stato" deve già essere in forma tensore multindice


    for qubit in siti:
        pauli=rng.integers(0, len(op_pauli))
        if pauli==0:  continue  #se l'operatore di pauli scelto è l'identità salto l'operazione, lo stato non cambia

        matrice=op_pauli[pauli]

        #voglio che la matrice di Pauli sia moltiplicata solo per gli stati del qubit corrispondente
        stato=np.tensordot(matrice, stato, axes=([1], [qubit]))
        #devo riportare l'asse contratto (che è passato in poizione 0) alla sua posizione originela
        stato=np.moveaxis(stato, 0, qubit)
                                  
        
    return stato




def measure_and_projection_singole(rng, proiettori, stato): #proiettori è una lista di oggetti Proiettore; in particolare è il singolo termine dell'hamiltoniano
    dim=len(stato)

    #calcolo la probabilità di ricadere nello stato eccitato o nel gs
    p_ecc=np.real(np.vdot(stato, projection_singole(proiettori, stato)))   #ho dovuto mettere vdot perché ora lo stato non è più in forma vettoriale, ma tensore multiindice; vdot coniuga in automatico il primo termine
    if(p_ecc<1e-10): p_ecc=0. #ho provato a mettere una soglia per evitare le divergenze che vedo comparire
    p_gs=1.-p_ecc

    #raggruppo in un array tutti i siti su cui agisce il termine dell'hamiltoniano
    siti=np.array(list({proiettore.sito for proiettore in proiettori}), dtype=int)


    #eseguo la misura
    risultato_misura=0.
    p=rng.uniform(0.0,1.0)

    if (p<p_ecc): #misuro primo eccitato --> proietto con P --> depolarizing channel
        norma_ecc=np.sqrt(p_ecc)
        stato_ecc=projection_singole(proiettori, stato)/norma_ecc
        stato_1=depolarizing_channel_Lq(rng, stato_ecc, siti)
        risultato_misura=1.

    else: #misuro g.s-->proietto con P ortogonale
        norma_gs=np.sqrt(p_gs)
        stato_1=(stato - projection_singole(proiettori, stato))/norma_gs  #così non devo calcolare il proiettore ortogonale
        risultato_misura=0.
    
    return stato_1, risultato_misura



def esp_neg(t, E0, tau):
    return E0*2**(-t/tau)


def esp_neg_ub_H(x,m):  #m è il numero massimo di clausole
    return m*(1-(7/8)**m)**(x/m)

def simul_traiettoria(rng, hamiltoniana, stato_init):
    H=hamiltoniana
    num_termini=len(H)
    psi_0=stato_init

    trovato_gs=False #per trovare il t_gs di questa hammiltoniana

    storico_en, storico_mis=np.zeros(N_passi), np.zeros(N_passi)

    #misura e proiezione: scelgo a caso uno dei termini dell'hamiltoniano e applico con quello measure_and_projection N_passi volte
    for k in range(N_passi):

        #energia stato
        E_tot=0.
        for j in range(num_termini):
            E_j=np.real(np.vdot(psi_0, projection_singole(H[j], psi_0)))  #vdot coniuga in automatico
            E_tot+=E_j
                    
        storico_en[k]=E_tot

        idx=rng.integers(0, num_termini)
        psi_0, risultato_misura=measure_and_projection_singole(rng, H[idx], psi_0)
        storico_mis[k]=risultato_misura


        #condizione per salvare il tempo di arrivo al gs
        if not trovato_gs and E_tot<epsilon_gs:
            t_gs_singolo=k
            trovato_gs=True

    stato_fin=psi_0

    return storico_en, storico_mis, t_gs_singolo, stato_fin
 


def tempo_en_H_singola(istanza, L, rng): #provo a mettere lo stesso rng dell'hamiltoniana
    dim=2**L    

    
    #stato di partenza
    psi_sovr=np.ones(dim)/np.sqrt(dim) #sovrapposizione quantistica tutti stati base
    psi_init=psi_sovr.reshape((2,)*L) #trasformo già lo stato in tensore multindice


    #gli dico di prendere l'hamiltoniana dell'argomento "istanza"
    hamiltoniana=istanza_to_hamiltoniana(istanza)



    #lista di liste per mettere in memoria le energie e i risultati delle misure
    E_lista_arrays=[]
    misure_lista_arrays=[]

    #array dei tempi di convergenza
    t_gs_array=np.full(N_simulazioni, np.nan)


    #inizio il ciclo di simulazioni
    for i in range(N_simulazioni):
        #rng è un argomento della funzione
        

        #eseguo la simulazione
        storico_en, storico_mis, t_gs_singolo, stato_fin=simul_traiettoria(rng, hamiltoniana, psi_init)

        E_lista_arrays.append(storico_en)
        misure_lista_arrays.append(storico_mis)
        t_gs_array[i]=t_gs_singolo



    #calcolo la media delle energie per ciascun passo di tutte le simulazioni
    matrice_simul=np.array(E_lista_arrays)
    media_E=np.mean(matrice_simul, axis=0) #media colonna per colonna, lungo l'asse delle simulazioni



    
    #GRAFICO le traiettorie e la media (prime 5 hamitloniane)
    
    fig, ax=plt.subplots()

    #creo la lista di colori per graficare gli outputs
    N_colori=N_simulazioni
    cmap=plt.colormaps["turbo"]
    colori=[cmap(i/(N_colori-1)) for i in range(N_colori)]

    #calcolo l'upper bound
    x=np.arange(N_passi)
    E_ub=esp_neg_ub_H(x, m)


    #singole traiettorie
    for k in range(N_simulazioni):
        #spessore=1.+0.1*k
        spessore=1.2
        ordine_visivo = N_simulazioni - k  # es. k=0 ha zorder massimo
        ax.plot(E_lista_arrays[k], alpha=0.5, color='#4A90E2', linewidth=spessore, zorder=ordine_visivo)

    #media energie
    ax.plot(media_E, color='r', linewidth=1.8,  zorder=N_simulazioni+1, label="Average energy")

    #upper bound
    #ax.plot(E_ub, linewidth=2., zorder=N_simulazioni+1, label="upper bound energy")

    #aggiungo anche la soglia
    plt.axhline(y=epsilon_gs, linestyle='--', color='black', label='Threshold')

    ax.set_xlabel("t [step]", fontsize=20)
    ax.set_ylabel("E", fontsize=20)
    ax.set_title("Trajectories and average energy", fontsize=22)
    ax.set_xlim(0,60)
    ax.set_ylim(0,3.2)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=20)
    ax.set_yscale('log')
    
    # Coordinate in frazione dell'asse (transform=ax.transAxes): (0,0) basso-sinistra, (1,1) alto-destra
    '''
    testo_box = (
        f"Istanza: {istanza}  \n"
        f"Hamiltoniana: {hamiltoniana}"
    )

    # Inserimento con box semitrasparente
    ax.text(
        0.95, 0.75, testo_box,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )
    '''

    #plt.close(fig)
    plt.show()
    



    #CALCOLO DI t_gs
    #1. estraggo una prima misura di tempo di convergenza prendendo il primo step tale che media_E è sotto la soglia
    indici_zeri=np.flatnonzero(media_E<epsilon_gs)  #flatnonzero restituisce gli indici della condizione fra ()
    if indici_zeri.size>0: 
        t_gs=indici_zeri[0]

        stdev_t_gs=np.std(matrice_simul[:, t_gs], ddof=1)/np.sqrt(N_simulazioni)
    else:
        t_gs=N_passi
        print("Nessun valore nullo presente nell'array, non siamo arrivati al gs")

    #print("Tempo di arrivo al ground state:   ")  #print di t_gs
    #print(t_gs)
    
    
    '''
    #2. voglio ora provare a estrarre il tempo caratteristico dell'hamiltoniana studiata tramite fit
    #per questo voglio fare un fit con un esponenziale negativo delle energie medie
    x=np.arange(t_gs)
    y=media_E[:t_gs]
    

    #deviazione standard della media
    stdev_media_E=np.std(matrice_simul, axis=0, ddof=1)/np.sqrt(N_simulazioni)
    s_y=stdev_media_E[:t_gs]
    
    #guess iniziali per l'algoritmo
    a_guess=media_E[0]  #E0
    b_guess=2.  #tau
    p0=[a_guess, b_guess]

    #fit di Levenberg-Marquardt
    parametri_ottimali, matrice_covarianza = curve_fit(esp_neg, x, y, p0=p0, sigma=s_y, absolute_sigma=True)  #con incertezze
    #parametri_ottimali, matrice_covarianza = curve_fit(esp_neg, x, y, p0=p0, absolute_sigma=True)  #senza incertezze
    incertezze_par=np.sqrt(np.diag(matrice_covarianza))
    a_opt, b_opt = parametri_ottimali
    a_err, b_err=incertezze_par

    #chi2
    valori_attesi=esp_neg(x, a_opt, b_opt)
    residui=y-valori_attesi
    chi2=np.sum(((residui)/s_y)**2)
    gdl=len(x)-len(parametri_ottimali)
    chi2_rid=chi2/gdl

    print('parametri ottimali con incertezze: E0, tau (E=E0*exp(-t/tau))')
    print(f"{a_opt}\t{a_err}")
    print(f"{b_opt}\t{b_err}")

    print('chi2 ridotto:  ')
    print(chi2_rid)


    #plotto il fit

    funz=r'Fit: $y = E_0e^(-\frac{t}{\tau})$'
    fig, ax2=plt.subplots(2,1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax2[0].errorbar(x, y, marker='o', yerr=s_y, markersize=3,capsize=3, label='Dati misurati', zorder=2)
    ax2[0].plot([], [], ' ', label=funz)
    ax2[0].plot(x, esp_neg(x, a_opt, b_opt), color='red', label=f'Fit\n$E_0$={a_opt:.1f}, $tau$={b_opt:.1f}',  zorder=1)
    ax2[0].set_title('Fit esponenziale negativo')
    ax2[0].set_xlabel('tempo (# iterazione)')
    ax2[0].set_ylabel('eneriga media')
    ax2[0].legend(fontsize=13)

    ax2[1].errorbar(x, residui, yerr=s_y, fmt='o',markersize=3,capsize=3, label='residui')
    ax2[1].set_title('residui')
    ax2[1].set_ylabel('residui')
    # Aggiunge la linea orizzontale a y=0
    ax2[1].axhline(y=0, color='blue', linestyle='--', linewidth=1.5, zorder=1, alpha=0.7)

    plt.tight_layout()
    plt.show()
    '''
    
    '''
    #3. un terzo metodo è quello di fittare sull'istogramma di tutti i t_gs
    #assocerò un'incertezza poissoniana ai conteggi
    #t_gs_array ha salvato tutti i tempi di arrivo al gs delle singole traiettorie

    #creo istogramma dei t_gs_i
    conteggi, bordi=np.histogram(
        t_gs_array,
        bins=N_passi,
        density=False,              # False = conteggi assoluti; True = normalizzato a densità di probabilità (area = 1)
    )

    centri=0.5*(bordi[:-1]+bordi[1:])
    s_conteggi=np.sqrt(conteggi)

    #plotto l'istogramma
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    x=centri
    y=conteggi
    s_y=s_conteggi
    ax.errorbar(x, y, yerr=s_y, fmt='o', markersize=3, capsize=3, label=' histo t_gs', zorder=2)

    ax.set_xlabel(r"Valore ($t_gs$)")
    ax.set_ylabel("Conteggi")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()
    '''


    

    return t_gs, stdev_t_gs, media_E


#salvataggio dei dati
def salva_dati_h5(
        media_E_lista,
        t_gs_array,
        t_gs_medio,
        stdev_t_gs,
        nome_archivio,
        L=L, 
        N_hamiltoniane=N_hamiltoniane, 
        master_seed_H=master_seed_H, 
        N_simul=N_simulazioni,
        master_seed=master_seed
        ):
    with h5py.File(nome_archivio, "a") as f:
        nome_gruppo=f"L_{L}"

        #se il gruppo esiste già lo sovrascrive
        if nome_gruppo in f:
            del f[nome_gruppo]

        grp=f.create_group(nome_gruppo)
        grp.attrs["L"]=L
        grp.attrs["N_hamiltoniane"]=N_hamiltoniane
        grp.attrs["master_seed_H"]=master_seed_H
        grp.attrs["N_simulazioni"]=N_simul
        grp.attrs["N_passi"]=N_passi
        grp.attrs["master_seed"]=master_seed
        grp.create_dataset("media_E_lista", data=media_E_lista, compression="gzip")
        grp.create_dataset("t_gs_array", data=t_gs_array, compression="gzip")
        grp.attrs["t_gs_medio"]=t_gs_medio
        grp.attrs["stdev_t_gs"]=stdev_t_gs

    print(f"Gruppo {nome_gruppo} di archivio {nome_archivio} aggiornato con successo")









def main():


    #seed per rng di hamiltoniane e simulazioni
    ss2=np.random.SeedSequence(entropy=master_seed_H)
    child_seeds2=ss2.spawn(N_hamiltoniane)

    #lista per inglobare i tempi di convergenza e le media_E
    t_gs_array=np.zeros(N_hamiltoniane)
    stdev_t_gs_array=np.zeros(N_hamiltoniane)
    media_E_lista=[]
    k_array=np.zeros(N_hamiltoniane)  #racchiude gli indici di frustrazione dell'hamiltoniana

    #simulazioni sulle hamiltoniane
    for i in range(N_hamiltoniane):
        #definisco il generatore di numeri casuali
        child=child_seeds2[i]
        rng=np.random.default_rng(child)

        #genero istanza:
        if(var_cond==1):
            istanza=generatore_istanze(rng, L)  #una sola variabile condivisa tra clausole
        elif(var_cond==2):
            istanza=generatore_istanze2(rng, L)  #due variabili condivise tra clausole

        #calcolo il tempo di convergenza associato a quell'istanza
        t_gs_array[i], stdev_t_gs_array[i], media_E=tempo_en_H_singola(istanza, L, rng)
        media_E_lista.append(media_E)

        #calcolo il frustration rate dell'hamiltoniana
        k_array[i]=frustration_rate(istanza)

    #media e stdev dei tempi di convergenza
    t_gs_medio=np.mean(t_gs_array)
    stdev_t_gs=np.std(t_gs_array, ddof=1)

    print(f"Media e stdev dei tempi di convergenza (step) per {L} qubit:  {t_gs_medio}  {stdev_t_gs}")



    #salvo i dati e sovrascrivo se già ci sono; alcuni parametri sono dati di default (vedi definizione)
    salva_dati_h5(media_E_lista, t_gs_array, t_gs_medio, stdev_t_gs, nome_archivio=nome_archivio)


    #plotto i t_gs contro i k
    fig2, ax2=plt.subplots()

    ax2.errorbar(k_array, t_gs_array, fmt='o', yerr=stdev_t_gs_array, alpha=0.5, label=r'$t_{gs,i}$')

    ax2.set_xlabel("k [-]", fontsize=20)
    ax2.set_ylabel("t [steps]", fontsize=20)
    ax2.set_title("Convergence time over frustration rate", fontsize=22)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.legend(fontsize=20)
    ax2.set_xticks(np.arange(3, ax2.get_xlim()[1], 1))

    plt.close(fig2)
    plt.show()


if __name__ == "__main__":
    main()




'''
COMMENTI
1.Con L=5 vengono fuori esattamente due evoluzioni temporali uguali, anche a con hamiltoniane diverse:
Nel senso che ogni singola traiettoria è ordinatamente uguale a un'altra, ma solo in due modi e con ordine di modo misto.
Il problema non viene risolto né con Nsimul=1000 né cambiando master_seed.
Non è un problema di ss.spawn.
I due modi vengono fuori se l'unica variabile condivisa (qubit 2) ha "segno" concorde o discorde nelle due clausole.

Ho risolto mettendo rng anche tra gli argomenti di tempo_en_singola; 
in questo modo simulo con lo stesso rng che ha generato l'hamiltoniana, così ho ogni volta generatori legati alla singola hamiltoniana.
Ho scoperto anche dov'era il problema: quando usavo i due ss.spawn.
Infatti ho visto che impostando una sequenza uguale di child_seeds (entrambi arange) ritorna il problema di sopra. 
Probabilmente anche due ss.spawn diversi generavano un qualche tipo di correlazione tra le due sequenze di child_seeds (anche se il master seed e la lunghezza erano diversi).
Deduco che il modo giusto di usare ss.spawn sia di generare subito tutti i seed che ti servono
(nel mio caso N_simulazioni+N_hamiltoniane)
Non implemento comunque perché probabilmente è superfluo.

COSE DA FARE
1. Rifare tutte le simulazioni ora che gli rng sono a posto  FATTO

2. Implementare istanze a 2 variabili condivise FATTO

3. Capire quali livelli enrgetici siano effettivamente accesibili, ho il dubbio che E=2 non lo sia

4. nei grafici, fare anche media delle medie per singolo L (altrimenti ci sono forse troppe linee)
'''
