import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

BET = 0.5       # Bar edge thickness
VAD = -12       # Vertical annotation displacement

# Experiemnts:
#
# 'autoplacement' (Fig. 6: collocated, Cachew, FastFlow, PecanAP)
# 'autoorder' (Fig. 7: Cachew, PecanAO)
# 'final' (Fig. 8: collocated, Cachew, PecanAP, PecanAP & AO)

parser.add_argument('-e', '--experiment', type=str, help='the experiment to plot', default='final') # Fig 1: 'intro' Fig 6: 'autoplacement' Fig 7: autoorder Fig 8: 'final'
parser.add_argument('-r', '--repeats', type=int, help='number of repeats (not currently used)', default=1)
parser.add_argument('-m', '--model', type=str, help='model used', default='ResNet50_v2-8')
parser.add_argument('-t', '--tpu_costs', help='tpu costs', nargs='+')
parser.add_argument('-c', '--cpu_costs', help='cpu costs', nargs='+')

args = parser.parse_args()

exp = args.experiment
repeats = args.repeats
model = args.model

AE_TPU_cost = [float(cost) for cost in args.tpu_costs] # collocated, Cachew, Pecan
AE_worker_cost = [float(cost) for cost in args.cpu_costs] # collocated, Cachew, Pecan

print(AE_TPU_cost)
print(AE_worker_cost)

plt.rcParams.update({'font.size': 12})
#matplotlib.use('TkAgg')

# Set up figure
fig = plt.figure(figsize=(8, 4))

# Evaluation experiments (have data used in the actual paper)
#'''
# ["ResNet50_v2-8", "SimCLR", "RetinaNet", "ASRTrans", "ResNet50_v3-8"]
df0 = pd.DataFrame({ # No service
    "TPU cost":[float(AE_TPU_cost[0])], # 4.5466666
    "Worker cost":[float(AE_worker_cost[0])] # 0.0
    }, index=[model]
).round(2)
df1 = pd.DataFrame({ # Cachew
    "TPU cost":[float(AE_TPU_cost[1])], # 0.5584464
    "Worker cost":[float(AE_worker_cost[1])] # 0.914125078
    }, index=[model]
).round(2)
df2 = pd.DataFrame({ # AutoPlacement
    "TPU cost":[0.571304456, 1.274444444, 0.8225333333, 1.029111111, 0.59598],
    "Worker cost":[0.2559419544, 0.5709456639, 0.01417274683, 0.0, 0.7813848404]
    }, index=["ResNet50_v2-8", "SimCLR", "RetinaNet", "ASRTrans", "ResNet50_v3-8"]
).round(2)
df3 = pd.DataFrame({ # AutoOrder
    "TPU cost":[0.5446576, 1.129777778, 0.8211555556, 0.998962963, 0.5967177333],
    "Worker cost":[0.5724715726, 0.7397366689, 0.1839370896, 0.1455258594, 0.6954240684]
    }, index=["ResNet50_v2-8", "SimCLR", "RetinaNet", "ASRTrans", "ResNet50_v3-8"]
).round(2)
df4 = pd.DataFrame({ # Pecan (AutoPlacement + AutoOrder)
    "TPU cost":[float(AE_TPU_cost[2])], # 0.5446576
    "Worker cost":[float(AE_worker_cost[2])]
    }, index=[model]
).round(2)
df5 = pd.DataFrame({ # FastFlow
    "TPU cost":[2.893333333, 2.0407424, 0.9675306667, 1.156222222, 4.44498648],
    "Worker cost":[0.4985388333, 0.3516322587, 0.1667113859, 0.1122899372, 0.4316879949]
    }, index=["ResNet50_v2-8", "SimCLR", "RetinaNet", "ASRTrans", "ResNet50_v3-8"]
).round(2)
#'''

bars=[model]
x = np.arange(len(bars))  # the label locations

#fig, ax = plt.subplots()
plt.grid(axis='y', linewidth=0.5)

###### Final eval graph
if exp == 'final':
   outputFile = 'plots/fig8_' + model
   width = 0.23  # the width of the bars
   fs = 10

   TPU0a = plt.bar(x - 3*width/2, df0['TPU cost'], width=width, label='No Service)', linewidth=BET, edgecolor='#000000', color='#1f77b4', hatch="")
   for b in TPU0a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, VAD), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU1a = plt.bar(x - 1*width/2, df1['TPU cost'], width=width, label='Cachew', linewidth=BET, edgecolor='#000000', color='#ff7f0e', hatch="")
   for b in TPU1a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, VAD), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU1b = plt.bar(x - 1*width/2, df1['Worker cost'], bottom=df1['TPU cost'], width=width, linewidth=BET, edgecolor='#000000', color='#ffaf18', hatch="//")
   for b in TPU1b:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height+b.get_y()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   #TPU2a = plt.bar(x + 1*width/2, df2['TPU cost'], width=width, label='Cachew + AutoPlacement', linewidth=BET, edgecolor='#000000', color='#2ca02c', hatch="")
   #for b in TPU2a:
   #   height = b.get_height()
   #   plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, VAD), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   #TPU2b = plt.bar(x + 1*width/2, df2['Worker cost'], bottom=df2['TPU cost'], width=width, linewidth=BET, edgecolor='#000000', color='#52b04c', hatch="//")
   #for b in TPU2b:
   #   height = b.get_height()
   #   plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height+b.get_y()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU4a = plt.bar(x + 1*width/2, df4['TPU cost'], width=width, label='Pecan', linewidth=BET, edgecolor='#000000', color='#8c564b', hatch="")
   for b in TPU4a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, VAD), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU4b = plt.bar(x + 1*width/2, df4['Worker cost'], bottom=df4['TPU cost'], width=width, linewidth=BET, edgecolor='#000000', color='#bca68b', hatch="//")
   for b in TPU4b:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height+b.get_y()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=fs)

   plt.yticks(np.arange(0, 9, 2), fontsize=10)

   no_service = mpatches.Patch(edgecolor='#000000', facecolor='#1f77b4', label='tf.data collocated')
   cachew = mpatches.Patch(edgecolor='#000000', facecolor='#ff7f0e', label='Cachew')
   cachew_AP = mpatches.Patch(edgecolor='#000000', facecolor='#2ca02c', label='Pecan AutoPlacement')
   pecan = mpatches.Patch(edgecolor='#000000', facecolor='#8c564b', label='Pecan AutoPlacement\n+ AutoOrder')
   tpu_cost = mpatches.Patch(edgecolor='#000000', facecolor='#ffffff', hatch='', label='TPU cost')
   work_cost = mpatches.Patch(edgecolor='#000000', facecolor='#ffffff', hatch='//', label='Remote worker cost')
   #plt.legend(handles=[no_service, cachew_AP, tpu_cost, cachew, pecan, work_cost], loc='upper left', ncol=2)
   plt.legend(handles=[no_service, cachew, tpu_cost, pecan, work_cost], loc='upper left', ncol=2)

###### JUST AUTOORDER OR NOT
if exp == 'autoorder':
   outputFile = 'AutoOrder_Cost'
   width = 0.33  # the width of the bars
   fs = 12
   vad = -15

   TPU2a = plt.bar(x - 1*width/2, df1['TPU cost'], width=width, label='Cachew', linewidth=BET, edgecolor='#000000', color='#ff7f0e', hatch="")
   for b in TPU2a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, vad), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU2b = plt.bar(x - 1*width/2, df1['Worker cost'], bottom=df1['TPU cost'], width=width, linewidth=BET, edgecolor='#000000', color='#ffaf18', hatch="//")
   for b in TPU2b:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height+b.get_y()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU4a = plt.bar(x + 1*width/2, df3['TPU cost'], width=width, label='Cachew + AutoOrder', linewidth=BET, edgecolor='#000000', color='#9467bd', hatch="")
   for b in TPU4a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, vad), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU4b = plt.bar(x + 1*width/2, df3['Worker cost'], bottom=df3['TPU cost'], width=width, linewidth=BET, edgecolor='#000000', color='#c9b0ed', hatch="//")
   for b in TPU4b:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height+b.get_y()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=fs)

   plt.yticks(np.arange(0, 3.1, 1), fontsize=10)

   cachew = mpatches.Patch(edgecolor='#000000', facecolor='#ff7f0e', label='Cachew')
   cachew_AO = mpatches.Patch(edgecolor='#000000', facecolor='#9467bd', label='Pecan AutoOrder')
   tpu_cost = mpatches.Patch(edgecolor='#000000', facecolor='#ffffff', hatch='', label='TPU cost')
   work_cost = mpatches.Patch(edgecolor='#000000', facecolor='#ffffff', hatch='//', label='Remote worker cost')
   plt.legend(handles=[cachew, tpu_cost, cachew_AO, work_cost], loc='upper right', ncol=2)

###### NO SERVICE, CACHEW, FF, AP
if exp == 'autoplacement':
   outputFile = 'Cachew_AP_FF'
   width = 0.23  # the width of the bars
   fs = 10

   TPU0a = plt.bar(x - 3*width/2, df0['TPU cost'], width=width, label='No Service)', linewidth=BET, edgecolor='#000000', color='#1f77b4', hatch="")
   for b in TPU0a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, VAD), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU1a = plt.bar(x - 1*width/2, df1['TPU cost'], width=width, label='Cachew', linewidth=BET, edgecolor='#000000', color='#ff7f0e', hatch="")
   for b in TPU1a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, VAD), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU1b = plt.bar(x - 1*width/2, df1['Worker cost'], bottom=df1['TPU cost'], width=width, linewidth=BET, edgecolor='#000000', color='#ffaf18', hatch="//")
   for b in TPU1b:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height+b.get_y()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU3a = plt.bar(x + 1*width/2, df5['TPU cost'], width=width, label='FastFlow', linewidth=BET, edgecolor='#000000', color='#7f7f7f', hatch="")
   for b in TPU3a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, VAD), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU3b = plt.bar(x + 1*width/2, df5['Worker cost'], bottom=df5['TPU cost'], width=width, linewidth=BET, edgecolor='#000000', color='#cfcfcf', hatch="//")
   for b in TPU3b:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height+b.get_y()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU2a = plt.bar(x + 3*width/2, df2['TPU cost'], width=width, label='Cachew + AutoPlacement', linewidth=BET, edgecolor='#000000', color='#2ca02c', hatch="")
   for b in TPU2a:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height), xytext=(0, VAD), textcoords="offset points", ha='center', va='bottom', fontsize=fs)
   TPU2b = plt.bar(x + 3*width/2, df2['Worker cost'], bottom=df2['TPU cost'], width=width, linewidth=BET, edgecolor='#000000', color='#52b04c', hatch="//")
   for b in TPU2b:
      height = b.get_height()
      plt.annotate('{}'.format(round(height,2)), xy=(b.get_x() + b.get_width() / 2, height+b.get_y()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=fs)

   plt.yticks(np.arange(0, 9, 2), fontsize=10)

   no_service = mpatches.Patch(edgecolor='#000000', facecolor='#1f77b4', label='tf.data collocated')
   cachew = mpatches.Patch(edgecolor='#000000', facecolor='#ff7f0e', label='Cachew')
   ff = mpatches.Patch(edgecolor='#000000', facecolor='#7f7f7f', label='FastFlow')
   cachew_AP = mpatches.Patch(edgecolor='#000000', facecolor='#2ca02c', label='Pecan AutoPlacement')
   tpu_cost = mpatches.Patch(edgecolor='#000000', facecolor='#ffffff', hatch='', label='TPU cost')
   work_cost = mpatches.Patch(edgecolor='#000000', facecolor='#ffffff', hatch='//', label='Remote worker cost')
   plt.legend(handles=[no_service, cachew, tpu_cost, ff, cachew_AP, work_cost], loc='upper left', ncol=2)

plt.ylabel('Total training cost per epoch ($)')
plt.xlabel('Model')
plt.xticks(x, bars)

plt.xlim(right=len(df1)-0.5, left=-0.5)

fig.tight_layout()
plt.subplots_adjust(left=0.1, right=0.99, top=0.975, bottom=0.1)

plt.savefig(outputFile+'.jpg', dpi=2000, bbox_inches='tight')
plt.savefig(outputFile+'.pdf', bbox_inches='tight')

#plt.show()
print("DONE")
