from random import shuffle

datas = open('templates/data').readlines()
intent_templates = open('templates/intent_template').readlines()
seq_templates = open('templates/seq_template').readlines()

class Template:
  def __init__(self, intent, sentence):
    self.intent = intent
    self.sentence = sentence

class Singer:
  def __init__(self, singer_name):
    self.name = singer_name
    self.albums = {}

  def add_album(self, album_name, songs_list):
    if album_name not in self.albums:
      album = Album(album_name)
      self.albums[album_name] = album
    else:
      album = self.albums[album_name]
    album.add_songs(songs_list)

class Album:
  def __init__(self, album_name):
    self.name = album_name
    self.songs = []

  def add_songs(self, songs_list):
    self.songs = self.songs + songs_list

templates = []
for i, seq_template in enumerate(seq_templates):
  template = Template(intent_templates[i], seq_template)
  templates.append(templates)

singers = {}
for i in range(0, len(datas), 4):
  singer_name = datas[i].replace('\n', '')
  if singer_name not in singers:
    singer = Singer(singer_name)
    singers[singer_name] = singer
  else:
    singer = singers[singer_name]
  singer.add_album(datas[i+1].replace('\n', ''), datas[i+2].replace('\n', '').split(', '))

label_datas = []
seq_in_datas = []
seq_out_datas = []
for i, seq_template in enumerate(seq_templates):
  for singer in singers:
    for album in singers[singer].albums:
      for song in singers[singer].albums[album].songs:
        seq_in_data = ''
        seq_out_data = ''
        seq_template_tokens = seq_template.split()
        for token in seq_template_tokens:
          if token == '<singer>':
            seq_in_data = seq_in_data + singer + ' '
            for j, t in enumerate(singer.split()):
              if j == 0:
                seq_out_data = seq_out_data + 'B-singer' + ' '
              else:
                seq_out_data = seq_out_data + 'I-singer' + ' '
          elif token == '<song>':
            seq_in_data = seq_in_data + song + ' '
            for j, t in enumerate(song.split()):
              if j == 0:
                seq_out_data = seq_out_data + 'B-song' + ' '
              else:
                seq_out_data = seq_out_data + 'I-song' + ' '
          elif token == '<album>':
            seq_in_data = seq_in_data + album + ' '
            for j, t in enumerate(album.split()):
              if j == 0:
                seq_out_data = seq_out_data + 'B-album' + ' '
              else:
                seq_out_data = seq_out_data + 'I-album' + ' '
          else:
            seq_in_data = seq_in_data + token + ' '
            seq_out_data = seq_out_data + 'O' + ' '
        seq_in_datas.append(seq_in_data)
        seq_out_datas.append(seq_out_data)
        label_datas.append(intent_templates[i].replace('\n', ''))

label_datas_shuffle = []
seq_in_datas_shuffle = []
seq_out_datas_shuffle = []
index_shuf = range(len(label_datas))
shuffle(index_shuf)
for i in index_shuf:
  label_datas_shuffle.append(label_datas[i])
  seq_in_datas_shuffle.append(seq_in_datas[i])
  seq_out_datas_shuffle.append(seq_out_datas[i])


train_label = open('data/fake_data/train/train.label', 'w')
train_seq_in = open('data/fake_data/train/train.seq.in', 'w')
train_seq_out = open('data/fake_data/train/train.seq.out', 'w')
valid_label = open('data/fake_data/valid/valid.label', 'w')
valid_seq_in = open('data/fake_data/valid/valid.seq.in', 'w')
valid_seq_out = open('data/fake_data/valid/valid.seq.out', 'w')
test_label = open('data/fake_data/test/test.label', 'w')
test_seq_in = open('data/fake_data/test/test.seq.in', 'w')
test_seq_out = open('data/fake_data/test/test.seq.out', 'w')

for i, label in enumerate(label_datas_shuffle[:250]):
  test_label.write(label+'\n')
  test_seq_in.write(seq_in_datas_shuffle[i]+'\n')
  test_seq_out.write(seq_out_datas_shuffle[i]+'\n')

for i, label in enumerate(label_datas_shuffle[250:500]):
  valid_label.write(label+'\n')
  valid_seq_in.write(seq_in_datas_shuffle[i+250]+'\n')
  valid_seq_out.write(seq_out_datas_shuffle[i+250]+'\n')

for i, label in enumerate(label_datas_shuffle[500:]):
  train_label.write(label+'\n')
  train_seq_in.write(seq_in_datas_shuffle[i+500]+'\n')
  train_seq_out.write(seq_out_datas_shuffle[i+500]+'\n')



