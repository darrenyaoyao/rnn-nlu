data_dir=data/fake_data
model_dir=music_model_tmp
max_sequence_length=20  # max length for train/valid/test sequence
task=joint  # available options: intent; tagging; joint
bidirectional_rnn=False  # available options: True; False

CUDA_VISIBLE_DEVICES=2 python run_multi-task_rnn.py --data_dir $data_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --task $task \
      --bidirectional_rnn $bidirectional_rnn
