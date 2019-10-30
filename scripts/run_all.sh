for i in {0..11}; do env TF_XLA_FLAGS=--tf_xla_cpu_global_jit python active_learn.py active_learning_data models $i qbc; done;
for i in {0..11}; do env TF_XLA_FLAGS=--tf_xla_cpu_global_jit python active_learn.py active_learning_data models $i  random; done;
for i in {0..11}; do env TF_XLA_FLAGS=--tf_xla_cpu_global_jit python active_learn.py active_learning_data models $i  umin; done;
for i in {0..11}; do env TF_XLA_FLAGS=--tf_xla_cpu_global_jit python active_learn.py active_learning_data models $i all; done;

for i in {0..11}; do env TF_XLA_FLAGS=--tf_xla_cpu_global_jit python reptile.py active_learning_data reptile-models $i umin; done;
for i in {0..11}; do env TF_XLA_FLAGS=--tf_xla_cpu_global_jit python reptile.py active_learning_data reptile-models $i random; done;
for i in {0..11}; do env TF_XLA_FLAGS=--tf_xla_cpu_global_jit python reptile.py active_learning_data reptile-models $i qbc; done;
