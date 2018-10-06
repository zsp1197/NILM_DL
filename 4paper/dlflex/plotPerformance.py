# Created by zhai at 2018/10/2
# Email: zsp1197@163.com
# 文件通过运行test_seq2seq/test_predict_SeqAttn_multiple获得
import matplotlib.pyplot as plt
import sys

sys.path.append('...')
import Tools

infer_dict = Tools.deserialize_object('infer_dict.dict')
seq_lens = list(infer_dict.keys())
seq_lens = [10, 15, 20, 25, 27, 30]
odMeans = {}
f1s = {}
for seq_len in seq_lens:
    f1s.update({seq_len: [val['micro'] for key, val in infer_dict[seq_len].items()]})
    odMeans.update({seq_len: [val['odMean'] for key, val in infer_dict[seq_len].items()]})

plt.subplot(1,2,1)
for seq_len in f1s:
    plt.plot(list(range(1, len(f1s[seq_len]) + 1)), f1s[seq_len], '-o', ms=7, lw=2, alpha=0.7,
             label=f'win_size={seq_len}')
plt.grid()
plt.xlabel('pre')
plt.ylabel('F1-micro')
plt.xlim(xmin = 1)
plt.legend(fontsize='x-small')
# plt.show()
plt.subplot(1,2,2)
for seq_len in odMeans:
    plt.plot(list(range(1, len(f1s[seq_len]) + 1)), odMeans[seq_len], '-o', ms=7, lw=2, alpha=0.7,
             label=f'win_size={seq_len}')
plt.grid()
plt.xlabel('pre')
plt.ylabel('MAE (s)')
plt.xlim(xmin = 1)
plt.legend(fontsize='x-small')
plt.show()