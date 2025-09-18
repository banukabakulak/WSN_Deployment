from CentralizedNAD import *
from DecomposedNAD import DecomposedNAD
from GreedyNAD import GreedyNAD
from Instance import *
import sys

# Override with command-line arguments if provided
if len(sys.argv) == 7:
    size = sys.argv[3]                         # string
    N, S, L, seedId = map(int, sys.argv[4:7])  # integers
else: # Set default values
    size, N, S, L = 'demo', 15, 3, 'long'  # (N, S, L) = (problem size, #poses, #sensors, #periods)

isCNAD, isDeNAD, isGreedy = True, False, False
# isCNAD, isDeNAD, isGreedy = False, True, False
# isCNAD, isDeNAD, isGreedy = False, False, True
isRelease = False

#for seedId in range(10):
seedId = 2
instance = Instance(size, N, S, L, seedId, isCNAD)

if isCNAD:
    CNAD = CentralizedNAD(0.1, 5, isRelease)
    if CNAD.solve(instance):
        CNAD.reportOutput(f"CNAD{instance.name}")
        if not isRelease:
            CNAD.visualizeSolution(instance, False)
    else:
        print(f"CNAD{instance.name} failed to solve with seedId {seedId}")

if isDeNAD:
    DeNAD = DecomposedNAD(0.1, 5, isRelease)
    if DeNAD.solve(instance):
        DeNAD.reportOutput(f"DeNAD{instance.name}")
        if not isRelease:
            DeNAD.visualizeSolution(instance, False)
    else:
        print(f"DeNAD{instance.name} failed to solve with seedId {seedId}")

if isGreedy:
    Greedy = GreedyNAD(0.1, 5, isRelease)
    if Greedy.solve(instance):
        Greedy.reportOutput(f"Greedy{instance.name}")
        if not isRelease:
            Greedy.visualizeSolution(instance, False)
    else:
        print(f"Greedy{instance.name} failed to solve with seedId {seedId}")



