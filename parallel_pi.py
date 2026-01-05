from mpi4py import MPI
import random
import time
import math

#Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constantes
SERVER = 0
PACKET_SIZE = 10000000 #10 millions de tirages
MAX_TIME = 10.0
PRINT_EVERY = 1.0

# Vérification du nombre de processus
if size != 5:
    if rank == SERVER:
        print("Ce programme doit être lancé avec exactement 5 processus MPI")
    exit()
    
# Fonction monte-carlo
def monte_carlo_pi(nb_tirages):
    """
    Effectue nb_tirages tirages aléatoires dans le carré
    et compte ceux dans le quart de cercle.
    """
    points_dans_cercle = 0
    for _ in range(nb_tirages):
        x = random.random()
        y = random.random()
        if x*x + y*y < 1.0:
            points_dans_cercle += 1
    return points_dans_cercle, nb_tirages

#Serveur
if rank == SERVER:
    total_in = 0
    total_samples = 0

    t_start = time.time()
    last_print = t_start
    running = True

    print("Server démarrage avec 4 clients")

    while running:
        # Création de l'objet Status
        status = MPI.Status()

        # Réception depuis n'importe quel client
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                       

        src = status.Get_source()
        tag = status.Get_tag()

        Nin_local = data["Nin_local"]
        Ntotal_local = data["Ntotal_local"]

        # Accumulation globale
        total_in += Nin_local
        total_samples += Ntotal_local

        pi_est = 4.0 * total_in / total_samples
        err_est = 1.0 / math.sqrt(total_samples)

        now = time.time()
        if now - last_print >= PRINT_EVERY:
            elapsed = now - t_start
            print(f"[SERVER] t={elapsed:5.2f}s | src={src} | "
                  f"N={total_samples} | pi≈{pi_est:.9f} | err≈{err_est:.2e}")
            last_print = now

        # Critère d'arrêt
        if (now - t_start) > MAX_TIME or err_est < 1e-9:
            cmd = "STOP"
            running = False
        else:
            cmd = "CONTINUE"

        comm.send(cmd, dest=src, tag=1)

    print(f"[SERVER] FIN : pi≈{pi_est:.9f}, N={total_samples}")

#Clients
else:
    packets_done = 0
    while True:
        Nin_local, Ntotal_local = monte_carlo_pi(PACKET_SIZE)
        packets_done += 1

        msg = {
            "Nin_local": Nin_local,
            "Ntotal_local": Ntotal_local
        }

        comm.send(msg, dest=SERVER, tag=0)

        cmd = comm.recv(source=SERVER, tag=1)
        if cmd == "STOP":
            print(f"[CLIENT {rank}] Arrêt après {packets_done} paquets")
            break
