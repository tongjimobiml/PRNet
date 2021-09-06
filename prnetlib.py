
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical

from attnLSTM import AttentionLSTM
from numpy import *
import numpy as np
import pandas as pd
import math as Math



from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Activation, Dropout, Conv2D, Reshape, Flatten,Input

from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import regularizers


# In[3]:


rc = 6378137
rj = 6356725
from math import atan, cos, asin, sqrt, pow, pi, sin
def rad(d):
    return d * Math.pi / 180.0

def azimuth(pt_a, pt_b):
    lon_a, lat_a = pt_a
    lon_b, lat_b = pt_b
    rlon_a, rlat_a = rad(lon_a), rad(lat_a)
    rlon_b, rlat_b = rad(lon_b), rad(lat_b)
    ec=rj+(rc-rj)*(90.-lat_a)/90.
    ed=ec*cos(rlat_a)

    dx = (rlon_b - rlon_a) * ec
    dy = (rlat_b - rlat_a) * ed
    if dy == 0:
        angle = 90. 
    else:
        angle = atan(abs(dx / dy)) * 180.0 / pi
    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    if dlon > 0 and dlat <= 0:
        angle = (90. - angle) + 90
    elif dlon <= 0 and dlat < 0:
        angle = angle + 180 
    elif dlon < 0 and dlat >= 0:
        angle = (90. - angle) + 270 
    return angle

def distance(true_pt, pred_pt):
    lat1 = float(true_pt[1])
    lng1 = float(true_pt[0])
    lat2 = float(pred_pt[1])
    lng2 = float(pred_pt[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) +
    Math.cos(radLat1)*Math.cos(radLat2)*Math.pow(Math.sin(b/2),2)))
    s = s * 6378.137
    s = round(s * 10000) / 10
    return s


import time
import datetime

def compute_time_interval(start, end):
   
    start = datetime.datetime.fromtimestamp(start / 1000.0)
    end = datetime.datetime.fromtimestamp(end / 1000.0)
    
    # 相减得到秒数
    seconds = (end- start).seconds
    return seconds


def make_seq_input(df):
    groups = df.groupby(['TrajID_1'])
    mr_seq = []
    t_inter_seq = []
    for n, g in groups:
        g = g.sort_values(by=['MRTime_1'],ascending=True)
        g = g.reset_index(drop = True)
        bsid = g[['RNCID_1', 'CellID_1']].values
        flag = bsid[0, :]
        flag_idx = 0
        t_time = g['MRTime_1'].values
        for i in range(0, bsid.shape[0]):
            end = bsid[i, :]
            if flag[0] != end[0] or flag[1] != end[1]:
                flag = end
                if flag_idx == i-1:
                    mr_seq.append(g.iloc[flag_idx: flag_idx+1, :])
                else:
                    mr_seq.append(g.iloc[flag_idx: i, :])
                flag_idx = i
                
    return mr_seq



def interval_list(df):
   
    if df.shape[0] == 1:
        return [-1]
    else:
        #print (df.shape)
        t_time = list(df['MRTime_1'])
        #print (t_time)
        time_list = []
        time_list.append(-1)
        for i in range(df.shape[0]-1):
            #print (i, i+1)
            time_list.append(compute_time_interval(t_time[i], t_time[i + 1]))
        return time_list

import numpy as np
lonStep_1m = 0.0000105
latStep_1m = 0.0000090201

class RoadGrid:
    def __init__(self, label, grid_size):
        length = grid_size*latStep_1m
        width = grid_size*lonStep_1m
        self.length = length
        self.width = width
        def orginal_plot(label):
            tr = np.max(label,axis=0)
            tr[0]+=25*lonStep_1m
            tr[1]+=25*latStep_1m
            # plot(label[:,0], label[:,1], 'b,')
            bl = np.min(label,axis=0)
            bl[0]-=25*lonStep_1m
            bl[1]-=25*latStep_1m

            return bl[0], tr[0], bl[1], tr[1]

        xl,xr,yb,yt = orginal_plot(label)
        self.xl = xl
        self.xr = xr
        self.yb = yb
        self.yt = yt
        gridSet = set()
        grid_dict = {}
        self.grid_dict = {}
        for pos in label:
            lon = pos[0]
            lat = pos[1]

            m = int((lon-xl)/width)
            n = int((lat-yb)/length)
            if (m,n) not in grid_dict:
                grid_dict[(m,n)] = []
            grid_dict[(m,n)].append((lon, lat))
            gridSet.add((m,n))
        # print len(gridSet)
        gridlist = list(gridSet)
            
        grid_center = [tuple(np.mean(np.array(grid_dict[grid]),axis=0)) for grid in gridlist]

        self.gridlist = gridlist

        self.grids = [(xl+i[0]*width,yb + i[1]*length) for i in grid_dict.keys()]
        self.grid_center = grid_center
        self.n_grid = len(self.grid_center)
        self.grid_dict = grid_dict

    def transform(self, label, sparse=True):
        def one_hot(idx, n):
            a = [0] * n
            a[idx] = 1
            return a
        grid_pos = [self.gridlist.index((int((i[0]-self.xl)/self.width),int((i[1]-self.yb)/self.length))) for i in label]
        if sparse:
            grid_pos = np.array([one_hot(x, len(self.gridlist)) for x in grid_pos], dtype=np.int32)
        return grid_pos



def merge_2g_engpara(data):
    if data=='jd2':
        eng_para = pd.read_csv('BS_ALL.csv', encoding='gbk')
        eng_para = eng_para[['RNCID_1', 'CellID_1', 'Lon','Lat']]
    elif data=='sp2':
        eng_para = pd.read_csv('siping_2g_new_gongcan.csv', encoding='gbk')
        eng_para = eng_para[['RNCID', 'CellID', 'Lon','Lat']]
    elif data=='jd4':
        eng_para = pd.read_csv('jiading_4g_new_gongcan.csv', encoding='gbk')
        eng_para = eng_para[['RNCID', 'CellID', 'Lon','Lat']]
    else:
        eng_para = pd.read_csv('siping_4g_new_gongcan.csv', encoding='gbk')
        eng_para = eng_para[['RNCID', 'CellID', 'Lon','Lat']]
    #eng_para = eng_para[eng_para.LAC.notnull() & eng_para[u'经度'].notnull()]
    eng_para = eng_para.drop_duplicates()
    #eng_para.rename(columns={u'经度': 'lon', u'纬度': 'lat'}, inplace=True)
    return eng_para

def make_rf_dataset(data, eng_para, ds):
    for i in range(1, 8):
        data['MRTime_%d'% i]=data['MRTime']
        data['TrajID_%d'% i]=data['TrajID']
        if ds=='jd2':
            data = data.merge(eng_para, left_on=['RNCID_%d' % i, 'CellID_%d' % i], right_on=['RNCID_1','CellID_1'], how='left', suffixes=('', '%d' % i))
        else:
            data = data.merge(eng_para, left_on=['RNCID_%d' % i, 'CellID_%d' % i], right_on=['RNCID','CellID'], how='left', suffixes=('', '%d' % i))
        
        temp=data['CellID_%d'% i].tolist()
    data = data.fillna(-999.)
    #print data.columns
    if ds=='jd2' or ds=='sp2':
        feature = data[['TrajID_1','MRTime_1','Lon','Lat','Dbm_1','RNCID_1','CellID_1','AsuLevel_1',
                        'TrajID_2','MRTime_2','Lon2','Lat2','Dbm_2','RNCID_2','CellID_2','AsuLevel_2',
                        'TrajID_3','MRTime_3','Lon3','Lat3','Dbm_3','RNCID_3','CellID_3','AsuLevel_3',
                        'TrajID_4','MRTime_4','Lon4','Lat4','Dbm_4','RNCID_4','CellID_4','AsuLevel_4',
                        'TrajID_5','MRTime_5','Lon5','Lat5','Dbm_5','RNCID_5','CellID_5','AsuLevel_5',
                        'TrajID_6','MRTime_6','Lon6','Lat6','Dbm_6','RNCID_6','CellID_6','AsuLevel_6',
                        'TrajID_7','MRTime_7','Lon7','Lat7','Dbm_7','RNCID_7','CellID_7','AsuLevel_7']] #5*5 
    else:
        feature = data[['TrajID_1','MRTime_1','Lon','Lat','Dbm_1','RNCID_1','CellID_1','AsuLevel_1', 'Basic_psc_pci_1',
                        'TrajID_2','MRTime_2','Lon2','Lat2','Dbm_2','RNCID_2','CellID_2','AsuLevel_2', 'Basic_psc_pci_2',
                        'TrajID_3','MRTime_3','Lon3','Lat3','Dbm_3','RNCID_3','CellID_3','AsuLevel_3', 'Basic_psc_pci_3',
                        'TrajID_4','MRTime_4','Lon4','Lat4','Dbm_4','RNCID_4','CellID_4','AsuLevel_4', 'Basic_psc_pci_4',
                        'TrajID_5','MRTime_5','Lon5','Lat5','Dbm_5','RNCID_5','CellID_5','AsuLevel_5', 'Basic_psc_pci_5',
                        'TrajID_6','MRTime_6','Lon6','Lat6','Dbm_6','RNCID_6','CellID_6','AsuLevel_6', 'Basic_psc_pci_6',
                        'TrajID_7','MRTime_7','Lon7','Lat7','Dbm_7','RNCID_7','CellID_7','AsuLevel_7', 'Basic_psc_pci_7',]]
   
    label = data[['Longitude', 'Latitude']]

    return feature, label

from keras.layers import LSTM as AttentionLSTM 

def spare_label(tr_label_, te_label_):
    rg = RoadGrid(np.vstack((tr_label_.values,te_label_.values)),50)
    tr_label_g = rg.transform(tr_label_.values, False)
    te_label_g = rg.transform(te_label_.values, False)
    
    local_g_n = len(set(tr_label_g)|set(te_label_g))
    return tr_label_g, te_label_g, local_g_n, rg


def local_feature(tr_feature_r, te_feature_r, tr_label_g, te_label_g, local_g_n, data_dim):
    Y_c = to_categorical(tr_label_g, num_classes = local_g_n)
    Y_c_te = to_categorical(te_label_g, num_classes = local_g_n)
    l_e = te_feature_r.shape[0]
    l = tr_feature_r.shape[0]

    X= preprocessing.scale(tr_feature_r)
    X_te = preprocessing.scale(te_feature_r)
    X = X.reshape(l,data_dim,7,1)
    X_te = X_te.reshape(l_e,data_dim,7,1)
    return X, X_te, Y_c, Y_c_te

def getpridict(X_ter, model, rg):
    dense1_output_ter = model.predict(X_ter,batch_size=24) 
    result_max_ter=[]
    for i in dense1_output_ter:
        i=list(i)
        result_max_ter.append(i.index(max(i)))
    ter_pred_ = np.array([rg.grid_center[idx] for idx in result_max_ter])
    return result_max_ter,ter_pred_

def localpred(local_g_n, training_epoch, X, Y, X_te, rg, data_dim):
    model=Sequential()
    model.add(Conv2D(filters=64,
                 kernel_size=(data_dim,1),
                 padding='valid',
                 input_shape=(data_dim,7,1),
                 activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Reshape((7,64)))
    model.add(LSTM(200, return_sequences=False, stateful=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(local_g_n, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
    
    for i in range(training_epoch):
        model.fit(X, Y,
                  epochs=10,
                  batch_size=24,
                  verbose=0)
        print ("pre_localtrain", i)
       
    result_max_tr,tr_pred_ = getpridict(X, model, rg)
    result_max_te,te_pred_ = getpridict(X_te, model, rg)
    #model.save("prnet_local_jd2g_100.h5")
    return tr_pred_, te_pred_


def global_feature(tr_feature_r, te_feature_r, tr_label_, te_label_, tr_pred_, te_pred_,batch_size, ds, data_dim):
    te_feature_r['Longitude'] = te_label_.iloc[:, 0].values
    te_feature_r['Latitude'] = te_label_.iloc[:, 1].values

    tr_feature_r['Longitude'] = tr_label_.iloc[:, 0].values
    tr_feature_r['Latitude'] = tr_label_.iloc[:, 1].values

    te_feature_r['local_lon'] = te_pred_[:, 0]
    te_feature_r['local_lat'] = te_pred_[:, 1]

    tr_feature_r['local_lon'] = tr_pred_[:, 0]
    tr_feature_r['local_lat'] = tr_pred_[:, 1]
    
    for i in range(1,8):
        tr_feature_r['Lon_first_%d' % i]=tr_feature_r['local_lon']
        tr_feature_r['Lat_first_%d' % i]=tr_feature_r['local_lat']
        te_feature_r['Lon_first_%d' % i]=te_feature_r['local_lon']
        te_feature_r['Lat_first_%d' % i]=te_feature_r['local_lat']
    if ds=='jd2' or ds=='sp2':
        feature_second=['MRTime_1','Lon','Lat','Dbm_1','AsuLevel_1','RNCID_1', 'CellID_1','Lon_first_1','Lat_first_1',
                    'MRTime_2','Lon2','Lat2','Dbm_2','AsuLevel_2','RNCID_2', 'CellID_2','Lon_first_2','Lat_first_2',
                    'MRTime_3','Lon3','Lat3','Dbm_3','AsuLevel_3','RNCID_3', 'CellID_3','Lon_first_3','Lat_first_3',
                    'MRTime_4','Lon4','Lat4','Dbm_4','AsuLevel_4','RNCID_4', 'CellID_4','Lon_first_4','Lat_first_4',
                    'MRTime_5','Lon5','Lat5','Dbm_5','AsuLevel_5','RNCID_5', 'CellID_5','Lon_first_5','Lat_first_5',
                    'MRTime_6','Lon6','Lat6','Dbm_6','AsuLevel_6','RNCID_6', 'CellID_6','Lon_first_6','Lat_first_6',
                    'MRTime_7','Lon7','Lat7','Dbm_7','AsuLevel_7','RNCID_7','CellID_7', 'Lon_first_7','Lat_first_7']
    else:
        feature_second=['MRTime_1','Lon','Lat','Dbm_1','AsuLevel_1','RNCID_1', 'CellID_1','Lon_first_1','Lat_first_1','Basic_psc_pci_1',
                'MRTime_2','Lon2','Lat2','Dbm_2','AsuLevel_2','RNCID_2', 'CellID_2','Lon_first_2','Lat_first_2','Basic_psc_pci_2',
                'MRTime_3','Lon3','Lat3','Dbm_3','AsuLevel_3','RNCID_3', 'CellID_3','Lon_first_3','Lat_first_3','Basic_psc_pci_3',
                'MRTime_4','Lon4','Lat4','Dbm_4','AsuLevel_4','RNCID_4', 'CellID_4','Lon_first_4','Lat_first_4','Basic_psc_pci_4',
                'MRTime_5','Lon5','Lat5','Dbm_5','AsuLevel_5','RNCID_5', 'CellID_5','Lon_first_5','Lat_first_5','Basic_psc_pci_5',
                'MRTime_6','Lon6','Lat6','Dbm_6','AsuLevel_6','RNCID_6', 'CellID_6','Lon_first_6','Lat_first_6','Basic_psc_pci_6',
                'MRTime_7','Lon7','Lat7','Dbm_7','AsuLevel_7','RNCID_7','CellID_7', 'Lon_first_7','Lat_first_7','Basic_psc_pci_7',]

    tr_feature_r_2=tr_feature_r[feature_second]
    te_feature_r_2=te_feature_r[feature_second]
    
    l2 = int(tr_feature_r.shape[0]/batch_size)*batch_size
    l2_te = int(te_feature_r.shape[0]/batch_size)*batch_size
    X_l_lstm= preprocessing.scale(tr_feature_r_2.iloc[0:l2, :])
    X_l_lstm_te = preprocessing.scale(te_feature_r_2.iloc[0:l2_te, :])
    X_lstm = X_l_lstm.reshape(l2,7,data_dim)
    X_lstm_te = X_l_lstm_te.reshape(l2_te,7,data_dim)
    min_max_scaler = preprocessing.MinMaxScaler()
    Y_s = min_max_scaler.fit_transform(tr_label_.iloc[0:l2, :])
    Y_s_te = min_max_scaler.transform(te_label_.iloc[0:l2_te, :])
    
    return X_lstm, X_lstm_te, Y_s, Y_s_te, min_max_scaler

def generate_alstm(batch_size, timesteps, data_dim):
    model_sec = Sequential()
    model_sec.add(AttentionLSTM(200, return_sequences=False, stateful=True,
                   batch_input_shape=(batch_size, timesteps, data_dim)))
    
    model_sec.add(Dense(64, activation='relu',name="Dense_1"))
    model_sec.add(Dense(2, activation='sigmoid',name="Dense_2"))
    return model_sec



def globalpred(batch_size, X_lstm, X_lstm_te, Y_s, Y_s_te, min_max_scaler, train_epoch, data_dim):
    #data_dim = 9
    timesteps = 7
    model_sec = generate_alstm(batch_size, timesteps, data_dim)
   
    model_sec.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    
    for i in range(train_epoch):
        model_sec.fit(X_lstm, Y_s,
                  epochs =10,
                  batch_size = batch_size,
                  verbose = 2)
        print (i)
    result_te = model_sec.predict(X_lstm_te,batch_size=batch_size)
    result_te_lat_lon= min_max_scaler.inverse_transform(result_te)
    re_label = min_max_scaler.inverse_transform(Y_s_te)
    
    return result_te_lat_lon, re_label


