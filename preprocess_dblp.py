
all_keyword_list = []
with open(p_path, 'r', encoding='utf-8') as f:
  for line in f.readlines():
    line = line.lower()
    line = line.strip('\n').split('\t')
    # print(line)
    line[1] = re.findall('[a-zA-Z0-9]+', line[1])

    all_keyword_list.extend(
        [tmp_key for tmp_key in line[1] if tmp_key not in sp_word])
cnt = collections.Counter(all_keyword_list)

selected_keyword = []
for k, v in cnt.items():
  if v > 50:
    selected_keyword.append(k)
# 最终选择 334 个selected_keyword

with open(p_path, 'r', encoding='utf-8') as f:
  for line in f.readlines():
    line = line.lower()
    line = line.strip('\n').split('\t')
    # print(line)
    line[1] = re.findall('[a-zA-Z0-9]+', line[1])
    paper2key[str('P' + line[0])] = [tmp_key for tmp_key in line[1] if
                                     tmp_key in selected_keyword]
#

author_word = []
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
for author in selected_author:
  tmp = []
  for pi in adj_dict_ap[author]:
    tmp.extend(paper2key[pi])
  author_word.append(tmp)
ohe = MultiLabelBinarizer()
author2feature = ohe.fit_transform(author_word)


  

idx2term = {}
with open(t_path, 'r') as f:
  for line in f.readlines():
    line = line.strip('\n').split('\t')
    idx2term[line[0]] = line[1]
term2idx = {v: k for k, v in idx2term.items()}
idx2sp = {}
for k, v in idx2term.items():
  if v in stopwords.words('english'):
    idx2sp[k] = v



def split_idx(author_label, train_size, val_size):

  train_per_cls = int(train_size / 4)
  val_per_cls = int(val_size / 4)
  y = np.argmax(author_label, axis=1)
  train, val, test = [], [], []
  k0, k1, k2, k3 = 0, 0, 0, 0
  for i in range(y.shape[0]):
    if y[i] == 0 and k0 < train_per_cls:
      train.append(i)
      k0 += 1
    elif y[i] == 0 and train_per_cls <= k0 < train_per_cls + val_per_cls:
      val.append(i)
      k0 += 1
    elif y[i] == 1 and k1 < train_per_cls:
      train.append(i)
      k1 += 1
    elif y[i] == 1 and train_per_cls <= k1 < train_per_cls + val_per_cls:
      val.append(i)
      k1 += 1
    elif y[i] == 2 and k2 < train_per_cls:
      train.append(i)
      k2 += 1
    elif y[i] == 2 and train_per_cls <= k2 < train_per_cls + val_per_cls:
      val.append(i)
      k2 += 1
    elif y[i] == 3 and k3 < train_per_cls:
      train.append(i)
      k3 += 1
    elif y[i] == 3 and train_per_cls <= k3 < train_per_cls + val_per_cls:
      val.append(i)
      k3 += 1
    else:
      test.append(i)
  print('train_size: {}, val_szie: {}, test_size: {}'.format(
      len(train), len(val), len(test)))
  return train, val, test
