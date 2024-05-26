import numpy as np
import matplotlib.pyplot as plt


def k_means(data,cluster_number,initial_centroids):
    X = data[0]
    y = data[1]
    centroids=initial_centroids
    is_converged_array=[False for i in range(cluster_number)]
    is_converged=False
    cluster_data=[{'centroid':[0,0],
                   'members':[]} for i in range(cluster_number)]
    
    while not is_converged:
        for i in range (cluster_number):
            sum_x=np.array([0,0])
            sum_membership=0

            for j in range(len(X)):
                 member_of=get_membership(X[j],centroids)
                 membership_bool=1 if member_of==i else 0
                 if(membership_bool):
                      cluster_data[i]['members'].append(X[j])
                 sum_x=sum_x+membership_bool*X[j]
                 sum_membership=sum_membership+membership_bool
            
            new_centroid=None
            if(sum_membership!=0):
                 new_centroid=sum_x/sum_membership
            else:
                 continue
            #Converge Check for current cluster
            is_converged_array[i]=np.sqrt(np.sum((new_centroid - centroids[i]) ** 2))<=0.05
            centroids[i]=new_centroid
            cluster_data[i]['centroid']=new_centroid

        #converge criteria
        if(all(x==True for x in is_converged_array)):
             is_converged=True
            

    return cluster_data

def get_membership(x,centroids):
     distance=float('inf')
     membership_index=0
     for i in range(len(centroids)):
          current_dist=np.sqrt(np.sum((x - centroids[i]) ** 2))
          if(current_dist<distance):
               distance=current_dist
               membership_index=i
     return membership_index
          
          

def plot_clusters(cluster_info):
        markers=["o","x","*"]
        colors=["green","red","blue"]
        for i in range(len(cluster_info)):
            members = cluster_info[i]['members']
            centroid = cluster_info[i]['centroid']
            plt.scatter([members[j][0] for j in range(len(members))],
                        [members[j][1] for j in range(len(members))],
                        marker=markers[i], c=colors[i])
            plt.scatter([centroid[0]], [centroid[1]], marker="+", c="black")
       
        plt.show()

def plot_raw_data(data,labels):
        print(data)
        print(labels)
        plt.scatter([data[i][0] for i in range(len(data)) if labels[i]==0],[data[i][1] for i in range(len(data)) if labels[i]==0],marker="o",c="green")
        plt.scatter([data[i][0] for i in range(len(data)) if labels[i]==1],[data[i][1] for i in range(len(data)) if labels[i]==1],marker="x",c="red")
        plt.scatter([data[i][0] for i in range(len(data)) if labels[i]==2],[data[i][1] for i in range(len(data)) if labels[i]==2],marker="+",c="blue")
        plt.show()

    
if __name__ == '__main__':

    data=np.load("./kmeans_data/data.npy")
    label=np.load("./kmeans_data/label.npy")
    clustering_data= [np.array(data),np.array(label)]

    #Plot Raw Data

    #plot_raw_data(clustering_data[0],clustering_data[1])

    #Calculate clustering and get cluster info
    initial_centroids=np.array([[-2.5,2.5],[3.5,0.5],[1,5.5]])
    clustering_info=k_means(clustering_data,3,initial_centroids)
    plot_clusters(clustering_info)
   

