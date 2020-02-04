
# coding: utf-8

# In[15]:


import numpy as np
from sklearn import cross_validation
import csv as csv
import pandas as pd

import sklearn


# In[53]:


with open('avalanche_accidents_switzerland_since_1995.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    taille_data = 'tamere'
    avalanches = [ [_ for _ in range(17)] for j in range(402) ]
    i = 0
    
    for x in data:
        for k in range(17):
            word = x[k]
            avalanches[i][k] = word
        i += 1


# In[49]:


avalanches


# In[57]:


data = np.array(avalanches)


# In[62]:


data[0][16]


# In[3]:





# In[60]:


data.shape


# In[63]:


datapd = pd.DataFrame(data=data[1:, :], columns=data[0,:])


# In[81]:


datapd


# In[73]:


#On supprime la colonne datequality

del datapd['date.quality']


# In[80]:


datapd.describe()


# In[79]:


#On supprime les colonnes start.zone.coordinates, coordinates.quality, canton

del datapd['canton'], datapd['start.zone.coordinates.x'], datapd['start.zone.coordinates.y'], datapd['coordinates.quality']


# In[84]:


X = datapd['forecasted.dangerlevel']

def details_uniq(X):
    dicX={}
    for x in X:
        if x not in dicX:
            dicX[x] = 1
        else: 
            dicX[x] += 1
    return dicX

details_uniq(X)


# In[85]:


dead = datapd['number.dead']
caught = datapd['number.caught']
fb = datapd['number.fully.buried']

print(details_uniq(dead), details_uniq(caught), details_uniq(fb))


# In[86]:


del datapd['avalanche.id']


# In[87]:


datapd


# In[109]:


#suite du preprocess:

#transformation de la date en quelque chose d'exploitable:
def transfo_annee(str_date):
    annee = int(str_date[:4])
    return annee-1995

#on fait le mois +6 mod 12 pour transférer août 
def transfo_date(str_date):
    mois = int(str_date[5:7])
    jour = int(str_date[8:10])
    transf = jour + 31*((mois+6)%12) #on attribue un nombre unique à chaque jour de l'année, en gardant une continuité entre
    #décembre et janvier.
    return transf


# In[112]:


datapd['date'] = datapd['date'].apply(transfo_date)
datapd['hydrological.year'] = datapd['hydrological.year'].apply(transfo_annee)


# In[121]:


datapd = datapd.rename(index=str, columns={'date':'day_and_month', 'hydrological.year':'year'})


# In[122]:


datapd


# In[123]:


details_uniq(datapd['start.zone.slope.aspect']) 


# In[124]:


dic_directions = {'NW':2,'NNE':15,'E':4,'NNW':1,'SE':10,'N':0,'W':12,'SW':6,'ESE':11,'NE':14,'WNW':3,'S':8,'ENE':13,'WSW':5,'SSE':9,'NA':-1,'SSW':7}


# In[125]:


#Modification de l'orientation de la zone, pour avoir des valeurs continues

datapd['start.zone.slope.aspect'] = datapd['start.zone.slope.aspect'].apply(lambda x: dic_directions[x])


# In[141]:


datapd = datapd.rename(index=str, columns={'start.zone.slope.aspect':'zone_orientation'})


# In[126]:


datapd


# In[127]:


dangerlevels = details_uniq(datapd['forecasted.dangerlevel'])
inclinations = details_uniq(datapd['start.zone.inclination'])
print(dangerlevels, inclinations)


# In[136]:


def mean_withoutNA(dico_counts):
    nb_useful = 0
    tot_count = 0
    for occurence in dico_counts:
        if occurence != 'NA':

            current_count = int(dico_counts[occurence])
            nb_useful += current_count
            tot_count += current_count*int(occurence)
    return tot_count/nb_useful


# In[137]:


mean_danger = mean_withoutNA(dangerlevels)
mean_inclinations = mean_withoutNA(inclinations)
print(mean_danger, mean_inclinations)


# In[138]:


#ON remplace les NA par les moyennes des autres valeurs pour l'inclinaison et le risque d'avalanche
def replace(value_to_check, string_to_replace, mean):
    if value_to_check == string_to_replace:
        return mean
    return value_to_check        


# In[139]:


datapd['forecasted.dangerlevel'] = datapd['forecasted.dangerlevel'].apply(lambda x: replace(x, 'NA', 2.69))
datapd['start.zone.inclination'] = datapd['start.zone.inclination'].apply(lambda x: replace(x, 'NA', 40.27))


# In[142]:


datapd


# In[143]:


activities = details_uniq(datapd['activity'])
activities


# In[144]:


#On enlève la colonne local.name et on remplace le unknown par la moyenne
del datapd['local.name']


# In[145]:


dico_activities = {'offpiste': 2, 'tour': 3, 'transportation.corridor': 1, 'building': 0, 'other, mixed or unknown': 'NA'}
datapd['activity'] = datapd['activity'].apply(lambda x: dico_activities[x])

activities = details_uniq(datapd['activity'])
mean_activites = mean_withoutNA(activities)

datapd['activity'] = datapd['activity'].apply(lambda x: replace(x, 'NA', mean_activites))


# In[205]:


datapd


# In[147]:


# on a plusieurs objectifs: prédire d'abord si une avalanche mortelle aura lieu suivant les conditions données.

# Puis on peut prédire suivant les conditions quel est le risque d'avalanche selon le type d'activité pratiquée.

#Enfin on peut calculer notre risque et le comparer au risque initial calculé donné dans la bdd


# Prédire si une avalanche mortelle aura lieu suivant certaines conditions:
# 
#     On n'a pas de données sur les moments où il n'y a pas d'avalanches, ce qu'on peut faire:
# - pour un point contenant des conditions trouver les distances à tous les autres points, 
# en pondérant pour certaines features, puis calculer un risque suivant la distance cumulée
# - pour un point, on calcule les plus proches voisins sans la date, puis on considère la date des voisins pour trouver une fréquence 
# de déclenchement d'avalanches
# - Clustering des points, en déduire suivants les clusters des risques?
# 
# 

# In[206]:


#on pondère pas parce qu'on sait pas comment trouver les poids, sachant qu'on peut pas évaluer le modèle comme on n'a pas de risque "type"

data = datapd.values

train_risk = data[:, :6]


# In[155]:


#Pour calculer les poids de la pondération, on fait un algo génétique, et on calcule le risque type avec une fonction 
#Fitness, qui calcule les proches voisins (distance avec les poids =1 ) et en déduit suivant la fréquence d'une avalanche
#dans ces conditions un risque potentiel.
datapd = datapd.apply(pd.to_numeric)


# In[157]:


datapd.describe()


# In[ ]:


dic_std = {'2':530, '3':5.3, '4':4.17}


def risk_fitness(vecteur, other_points, precision_distance):
    
    proches = []
    for point in other_points:
        est_proche = True
        for k in dic_std:
            if k == 3 and not -1 <= abs(point[k]-vecteur[k])%16 <= 1:
                est_proche = False
            distance_max = dic_std[k]*precision_distance
            elif not abs(point[k]-vecteur[k]) <= distance_max:
                est_proche = False
        if est_proche: proches.append(point)
    
    #on prend le nombre d'avalanches proches, ca nous donne un risque potentiel

    #ON ABANDONNE l'IDEE, TROP COMPLIQUE ET CA MARCHE PAS


# In[166]:


#création des données avec la normalisation des colonnes day_and_month, start zone elevation, zone orientation, startzone inclination
#et forecasted dangerlevel

data[:, 0] = (data[:, 0]-250.69)/59
data[:, 1], data[:, 5] = data[:, 5], data[:, 1]
data[:, 1] = (data[:, 1]-2.69)/0.55
data[:, 2] = (data[:, 2]-2517)/530
data[:, 3] = (data[:, 3]-6.94)/5.3
data[:, 4] = (data[:, 4]-40.3)/4.17


# In[207]:


data


# In[180]:


#on utilise les poids égaux à 1, on calcule les distances cumulées aux nb_proches plus proches voisins, on note combien de morts
#cela a causé au total -> on en déduit un risque si la distance totale est grande

dic_mean_std = {'0':[250.69, 59], '2': [2517,530], '3': [6.94, 5.3], '4': [40.3, 4.17], '5': [2.69, 0.55]}

def distance_cum(point, other_point, nb_proches): #on s'intéresse aux 5 premieres colonnes
    nb_other_points = other_point.shape[0]
    distances = [0 for i in range(nb_other_points)]
    
    for i in range(nb_other_points):
        dist = 0
        current_pt = other_point[i]
        for k in [0, 2, 4, 5]:
            dist += abs(current_pt[k]-point[k]) / dic_mean_std[str(k)][1]
        #zone orientation traitée à part à cause du modulo 16
        dist += abs(current_pt[3]-point[3])%16 / dic_mean_std[str(3)]
        distances[i] = dist
        
    return sum(sort(distances)[-nb_proches:])
    


# In[217]:


distances = []
nb_points = data.shape[0]
nb_proches = 20

for i in range(nb_points):
    distances.append(distance_cum(data[i], data[0:i, :], nb_proches) + distance_cum(data[i], data[i+1:, :], nb_proches))


# In[211]:


print(argmax(distances))
maxdist = distances[314]


# In[212]:


print(argmin(distances))
mindist = distances[0]


# In[218]:


distances = 1 - ((np.array(distances) - mindist) / maxdist)
distances


# In[184]:


x = np.arange(10).reshape((5,2))


# In[208]:


data[310:317]


# In[194]:


x[0:3:2,:] 


# In[232]:


#ON TENTE un clustering


# In[231]:


from sklearn.cluster import KMeans


# In[268]:


data_clustering = datapd.copy()


# In[269]:


del data_clustering['year']


# In[270]:


data_clustering


# In[271]:


meandata = data_clustering.mean()
mindata = data_clustering.min()
maxdata = data_clustering.max()

data_clustering = (data_clustering - data_clustering.mean()) / (data_clustering.max() - data_clustering.min())


# In[272]:


X = data_clustering.values


# In[273]:


nb_clusters = 10

kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(X)

print(kmeans.labels_)

print(kmeans.cluster_centers_)


# In[287]:


def back_real(vectors):
    n = vectors.shape[1]
    for vec in vectors:
        for i in range(n):
            vec[i] = vec[i]*(maxdata[i]-mindata[i]) + meandata[i]
    return vectors


# In[275]:


back_real(kmeans.cluster_centers_)


# In[292]:


#autres clusters avec seulement les données de terrain et on sort pour chaque clusters la moyenne des dead,caught,fully burried,activity


data_cluster = datapd.copy()
del data_cluster['year']
meandata = data_cluster.mean()
mindata = data_cluster.min()
maxdata = data_cluster.max()

data_clust = (data_cluster - data_cluster.mean()) / (data_cluster.max() - data_cluster.min())

X = data_clust.values[:, :5]
taille = X.shape[0]

nb_clusters = 4
kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(labels)
print(centers)

dics = [{'dead':0, 'count':0, 'caught':0, 'burried':0, 'activity':0} for _ in range(nb_clusters)]

data_clust_np = data_cluster.values

for i in range(taille):
    vect = data_clust_np[i, :]
    clust = labels[i]
    dics[clust]['count'] += 1
    dics[clust]['dead'] += vect[5]
    dics[clust]['burried'] += vect[7]
    dics[clust]['caught'] += vect[6]
    dics[clust]['activity'] += vect[8]
    
for dic in dics:
    dic['dead/count'] = dic['dead']/dic['count']
    dic['caught/count'] = dic['caught']/dic['count']
    dic['burried/count'] = dic['burried']/dic['count']

print(dics)
print(back_real(centers))


# In[289]:


back_real(centers)

