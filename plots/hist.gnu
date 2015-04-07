set terminal png truecolor size 350, 300
set output 'hist.png'

set xtics ("Caffe" 0, "Pylearn2" 1, "Torch" 2, "Theano" 3)

set boxwidth 0.7 relative
set style fill solid 0.5

set ylabel "minutes" font ", 20"
set yrange [0:70]

set title "MLP" font ", 20"

plot 'hist.txt' using 2 notitle with boxes, \
     '' using 0:2:2 notitle with labels offset 0,1
