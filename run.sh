data_dir=data/SEM
model_dir=joint_nonbirnn
max_sequence_length=80  # max length for train/valid/test sequence
task=joint  # available options: intent; tagging; joint
bidirectional_rnn=False  # available options: True; False

CUDA_VISIBLE_DEVICES=2 python run_multi-task_rnn.py --data_dir $data_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --task $task \
      --bidirectional_rnn $bidirectional_rnn
