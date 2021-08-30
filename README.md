# mlscripts

Scripts to be used in conjunction with [FyPy](https://github.com/gokhalen/fypy) to generate machine learning data, train CNNs, and post_process.

# Data generation

python3.8 mlsetup.py --ntrain=10 --nhomo=0 --generate=True --solve=True --nelemx=64 --nelemy=96 --shift=0 --length=1.0 --breadth=1.5 --eltype='linelas2dnumbasri'

See mlsetup.py for explnations of arguments

# Running the ML code

python3.8 ml.py --mltype=field --iptype=strainyy --nepochs=64 --mode=train --nimg=2000 --noise=1 --noisetype=none --activation=twisted_tanh --inputscale=global  --boundshift=0.25 --featurescale=True