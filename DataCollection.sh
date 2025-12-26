#!/bin/bash

python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=strobfl_learn --gar=strobfl --k=10 --C=0.8 --B=10 --ndrift=0 --dmode=independent --ifactor=0.3 --arate=1.0
python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=adam --gar=avg --k=10 --C=0.8 --B=10 --ndrift=0 --dmode=independent --ifactor=0.3 --arate=1.0

python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=strobfl_learn --gar=strobfl --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=independent --ifactor=0.3 --arate=1.0
python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=adam --gar=avg --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=independent --ifactor=0.3 --arate=1.0

python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=strobfl_learn --gar=strobfl --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=shared --ifactor=0.3 --arate=1.0
python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=adam --gar=avg --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=shared --ifactor=0.3 --arate=1.0

python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=strobfl_learn --gar=strobfl --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=independent --ifactor=1.0 --arate=1.0
python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=adam --gar=avg --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=independent --ifactor=1.0 --arate=1.0

python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=strobfl_learn --gar=strobfl --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=independent --ifactor=0.3 --arate=2.0
python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=adam --gar=avg --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=independent --ifactor=0.3 --arate=2.0

python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=strobfl_learn --gar=strobfl --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=independent --ifactor=0.3 --arate=0.6
python STROBFLMain.py --dataset=synthetic-class1 --T=50 --optimizer=adam --gar=avg --k=10 --C=0.8 --B=10 --ndrift=4 --dmode=independent --ifactor=0.3 --arate=0.6