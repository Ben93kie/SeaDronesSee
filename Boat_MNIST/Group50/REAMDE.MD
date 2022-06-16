Here are the pdf file describing our skript as well as the python file we created:

We submitted 3 Models to the leaderboard. All use an Adam optimiser.

CrystalBall makes use of a split ratio of 6/1 and MSE loss as well as additional grayscale pictures in 30 Epochs of training.
    python3 ./AR_CH_Boat.py -k 6 --nsplits 1 --lr 0.001 --lr-redux 0.00005 --dropout 0.3 --epochs 30 --inflate-train

Tarotkarten uses a 5/1 split, BCE loss and additional grayscale pictures in 30 Epochs of training.
    python3 ./AR_CH_Boat.py -k 5 --nsplits 1 --lr 0.001 --lr-redux 0.00005 --dropout 0.3 --epochs 30 --inflate-train
 
Wuenschelrute uses a 7/1 split, BCE loss and 50 Epochs and slower lr decrease, but no grayscale images.
Since this small validation group often leads to tied accuracies, we inplemented train loss as a tiebreaker.
    python3 ./AR_CH_Boat.py -k 7 --nsplits 1 --lr 0.001 --lr-redux 0.00003 --dropout 0.3 --epochs 50
