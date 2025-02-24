#!bin/bash

for s in 0.2 0.4 0.6 0.8 1.0 1.2 1.4; do
    python main.py --N 4 --D 16 --s $s --prefix "lyapunov_s${s}" --plot-lyapunov --save-data
done