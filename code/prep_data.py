__author__ = 'kalpit'

import glob
import fnmatch
import os
import json


def get_available_tracks(base):
    file_locs = glob.glob(base + '*')
    file_names = [x[x.rfind('/') + 1:-4] for x in file_locs]
    return file_locs, file_names


def write_similar_available_tracks(base, a_file_locs, a_file_names, fname, X):
    fw = open(fname, 'w')
    # Pairs = [] # list of lists. [track_id_1, track_id_2, similarity_score]
    d = 0
    e = 0
    # iterate through all files in a_file_names
    for i in range(len(a_file_names)):
        if i%(len(a_file_names)/1000)==0:
            print i, '/', len(a_file_names)
        f_loc = base+'/'+a_file_names[i][2]+'/'+a_file_names[i][3]+'/'+a_file_names[i][4]+'/'+a_file_names[i]+'.json'


        if not os.path.exists(f_loc): 
            continue

        with open(f_loc) as f:
            data = json.load(f)
            data = data['similars']
            d += len(data)
            for x in data:
                try:
                    _ = X[x[0]]                    
                except KeyError:
                    continue
                fw.write(a_file_names[i] + ',' + x[0] + ',' + str(x[1]) + '\n')
                e += 1

    print 'total files similar to the ones available               : ' + str(d)
    print 'total files similar and available to the ones available : ' + str(e)
    fw.close()
    return


if __name__ == '__main__':
    ##### Available Tracks #####
    BaseDir_Available_Tracks = '../data/lyrics/train/'
    a_file_locs, a_file_names = get_available_tracks(BaseDir_Available_Tracks)
    print 'available files : ', len(a_file_names)

    ##### DICT #####
    x = {x:1 for x in a_file_names}

    ##### Similar (and available) Tracks #####
    BaseDir_Similar_Tracks = '../lastfm_train'
    similar_tracks_fname = '../lastfm_train_mappings.txt'
    write_similar_available_tracks(BaseDir_Similar_Tracks, a_file_locs, a_file_names, similar_tracks_fname, x)
    # print len(Pairs)
