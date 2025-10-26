This proyect explores tabular Reinforcement Learning methods through the implementation of a custom auction environment. The agent learns optimal bidding strategies through Q-Learning, competing against multiple opponent strategies. 
Key achievements include the development of a Markov Decision Process-compliant environment, implementation of enhanced Q-Learning with state discretization, and advanced training techniques including multi-opponent training and curriculum learning.

To run the training, you can run trainAuction.py directly, but it's recommended to run the compete.py script after changing the tournament/live match rendering code (in the main file of compete.py).
To run it correctly, you must specify the script name, followed by --method (and the training method) --episodes (and the total number of episodes you want in the training). Ej:
python3 compete.py --method curriculum --episodes 300000

