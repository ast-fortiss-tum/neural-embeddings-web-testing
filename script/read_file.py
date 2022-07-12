import numpy as np
import pandas as pd

addressbook = np.load('addressbook' + '_' + 'all.npy')
petclinic = np.load('petclinic' + '_' + 'all.npy')
claroline = np.load('claroline' + '_' + 'all.npy')
dimeshift = np.load('dimeshift' + '_' + 'all.npy')
mrbs = np.load('mrbs' + '_' + 'all.npy')
phoenix = np.load('phoenix' + '_' + 'all.npy')
ppma = np.load('ppma' + '_' + 'all.npy')
mantisbt = np.load('mantisbt' + '_' + 'all.npy')
pagekit = np.load('pagekit' + '_' + 'all.npy')

# all = np.append(addressbook, petclinic)
# all = np.append(all, claroline)
# all = np.append(all, dimeshift)
# all = np.append(all, mrbs)
# all = np.append(all, phoenix)
# all = np.append(all, ppma)
# all = np.append(all, mantisbt)
# all = np.append(all, pagekit)
#
# # print(len(all))
# df = pd.read_csv('D:\\doc2vec\\dataset\\training_sets\\SS_threshold_set.csv')
#
# df['doc2vec_distance_' + 'all'] = all
# df.to_csv('D:\\doc2vec\\dataset\\training_sets\\SS_threshold_set.csv')
