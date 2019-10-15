

for i in {0..10}; do python active_learn.py active_learning_data $i models qbc; done;
for i in {0..10}; do python active_learn.py active_learning_data $i models random; done;
for i in {0..10}; do python active_learn.py active_learning_data $i models umin; done;
for i in {0..10}; do python active_learn.py active_learning_data $i models all; done;

for i in {0..10}; do python reptile.py active_learning_data reptile-weights $i umin; done;
for i in {0..10}; do python reptile.py active_learning_data reptile-weights $i random; done;
for i in {0..10}; do python reptile.py active_learning_data reptile-weights $i qbc; done;
