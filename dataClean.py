import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from sklearn.preprocessing import LabelEncoder


import gc


def selectM(U, list1, list2 , value = -1):
    row = [[i] * len(list2) for i in list1]
    col = np.array(list2 * len(list1)).reshape(len(list1), len(list2))
    if value == -1:
        return U[row, col]
    else:
        Value = value*np.ones([len(list1),len(list2)])
        U[row,col] +=  Value



def findNetwork(tableName,colName,split , id ):

    #tableName = song; colName = "genre_ids"; split = "|" ; id = 'song_id'
    if split != 0:

        # select useful columns
        temp = tableName.ix[:,[id,colName]]

        # split colName based on var split , due to orignal data has sth like "a|b"
        temp['split'+colName] = temp[colName].map(lambda x: str(x).split(split))
        temp['lenOfType'] = temp['split'+colName].map(lambda  x : len(x))
        colNameSpread = [ j  for i in temp['split'+colName] for j in i]
        idSpread = [ [temp[id][i]]*j for i,j in enumerate(temp['lenOfType'])]
        idSpread = [j for i in idSpread for j in i]

        # id_colNameDF has the rows which contain the id-tag pair
        id_colNameDF = pd.DataFrame({id:idSpread,colName:colNameSpread})
    else :
        id_colNameDF = tableName.ix[:,[id,colName]]
    id_colNameDF ,colNameList = changeNameToID(id_colNameDF,colName ,plan="B" )

    id_colNameDF[id] = id_colNameDF[id].astype(int)
    id_colNameDF[colName] = id_colNameDF[colName].astype(int)

    flag = id_colNameDF.ix[colNameList.astype(int) == -1,'genre_ids'].values[0]


    idNumber = tableName[id].shape[0]
    # this method is abolished for the reason of my memmory is not enough
    # PPmatrix = csc_matrix((idNumber,idNumber))
    #
    # id_colNameDFGoupbyColName = id_colNameDF.groupby(colName).apply(lambda x: x[id].tolist() )
    #
    # id_colNameDFGoupbyColName  = id_colNameDFGoupbyColName[id_colNameDFGoupbyColName.apply(lambda x: len(x)).sort_values(ascending=True).index]
    #
    #
    # for i in range(1,110):#(id_colNameDFGoupbyColName.index.shape[0]-7)
    #
    #     trying = id_colNameDFGoupbyColName[(i-1):i].get_values()[0]
    #
    #     selectM(PPmatrix,trying ,trying , 1 )

    # now we decide to return a USER-TAG matrix and culculate the Sii' during loop

    id_colNameDF['target'] = 1
    id_colNameDF.ix[colNameList.astype(int) == -1,'target'] = -1

    ObjectTagmatrix = csc_matrix((id_colNameDF['target'], (id_colNameDF[id], id_colNameDF[colName])))

    return(ObjectTagmatrix)


def changeNameToID(tableName,id , plan = 'A'):
    if plan == 'A':

        originalName = tableName[id]

        originalName_index = originalName.index

        originalNameUnique = originalName.unique()

        originalNameCategory = originalName.astype("category", categories=originalNameUnique).cat.codes

        tableName[id] = originalNameCategory

        del originalName_index,originalNameUnique,originalNameCategory

        return(tableName,originalName)
    elif plan == 'B':
        le = LabelEncoder()
        le.fit(tableName[id])
        originalName = tableName[id]
        tableName[id] = le.transform(tableName[id])
        return (tableName,originalName)



def fillNAN(tableName,values):
    # values = {'genre_ids' : 'unkown' }
    tableName = tableName.fillna(value = values)
    return  tableName
if __name__ == "__name__":
    os.chdir("/home/xuan/桌面/recommand Sys/")

    #train = pd.read_csv("train.csv")

    members = pd.read_csv("members.csv",encoding="UTF-8")
    song_extra_info = pd.read_csv("song_extra_info.csv",encoding="UTF-8")



    song = pd.read_csv("songsCSV.csv",encoding="UTF-8" )
    song = fillNAN(song, {'genre_ids' : -1 })
    song , song_id = changeNameToID(song, 'song_id' , plan = "B")



