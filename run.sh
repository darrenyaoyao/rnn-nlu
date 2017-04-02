data_dir=data/ATIS_samples
model_dir=model_tmp
max_sequence_length=100  # max length for train/valid/test sequence
task=joint  # available options: intent; tagging; joint
bidirectional_rnn=False  # available options: True; False

CUDA_VISIBLE_DEVICES=2 python run_multi-task_rnn.py --data_dir $data_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --task $task \
      --bidirectional_rnn $bidirectional_rnn
