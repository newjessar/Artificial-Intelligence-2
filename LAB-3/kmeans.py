"""kmeans.py"""
import random
import math
from statistics import mean

class Cluster:
    """This class represents the clusters, it contains the
    prototype (the mean of all it's members) and member lists with the
    ID's (which are Integer objects) of the datapoints that are member
    of that cluster. You also want to remember the previous members so
    you can check if the clusters are stable."""
    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()
        self.previous_members = set()
        
    ## Add members to the cluster
    def add_members(self, dataPoint):
        self.current_members.add(dataPoint)
    
    ## Remove members to the cluster
    def remove_members(self, dataPoint):
        self.current_members.discard(dataPoint)

    
    ## Check if there is any change in the cluster
    def changing_members(self):      
        if self.current_members.symmetric_difference(self.previous_members) == set():
            return True
        else:
            return False
            

class KMeans:
    def __init__(self, k, traindata, testdata, dim):
        self.k = k
        self.traindata = traindata
        # print(self.traindata)
        self.testdata = testdata
        self.dim = dim

        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.2
        # An initialized list of k clusters
        self.clusters = [Cluster(dim) for _ in range(k)]

        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0

    ## Initial random clusters  
    def random_initialization(self):
        for test_data, prototype in enumerate(self.traindata):
            idx = random.randint(0, self.k-1)
            self.clusters[idx].add_members(test_data)
            
    ## Manage the movement between all the clusters
    def members_movement(self, cluster_idx, dataPoint_idx):
        for idx, cluster in enumerate(self.clusters): 
            # if is the targeted cluster then add the point
            if idx == cluster_idx:
                cluster.add_members(dataPoint_idx)
                
            # Otherwise, check if the data is part of the cluster to remove it
            elif cluster.current_members.__contains__(dataPoint_idx):
                cluster.remove_members(dataPoint_idx)


    ## Print the partitioning of the clusters
    def print_clusterPart(self, epoch):
        for idx, clus_x in enumerate(self.clusters):
            print("Epoch", epoch, "Partition", idx, "size: ", len(clus_x.current_members))
    
    ## Measures the Euclidean norm
    def euclidean_distance(self, p1, p2):
        return math.sqrt(sum([(a-b)**2 for a, b in zip(p1, p2)]))
    
    
    ## Generate a new partition by assigning each dataPoint to its closest cluster center
    def genrating_partition(self): 
        all_Prototypes = []
        # Gathering all prototypes and Backup the cluster members  
        for cluster in self.clusters:
            all_Prototypes.append(cluster.prototype)
            cluster.previous_members = cluster.current_members.copy()

        # Going through all the dataPoints to check the nearest fit between the prototypes
        for dataPoint_idx, dataPoint in enumerate(self.traindata):
            # Calculating the distance for all prototypes to the dataPoint
            all_Dist_prototypes = [self.euclidean_distance(dataPoint, prototype) 
                        for prototype in all_Prototypes]
            # Get the minimum distance between all the K Prototypes
            nearest_prototype = min(all_Dist_prototypes)
            nearest_prototype_idx = all_Dist_prototypes.index(nearest_prototype)
            # Add the dataPoint to the nearest prototype and remove it from other clusters
            self.members_movement(nearest_prototype_idx, dataPoint_idx)

    ## Calculating the mean value to be assign as a prototype
    def avrage_cluster(self):
        for cluster in self.clusters:
            # Get the indices of all the members
            member_IDXList = [pt for pt in cluster.current_members]         
            # Retrieving all the cluster members using the indices
            mappedMember = list(map(self.traindata.__getitem__, member_IDXList))
            # Calculate the average value of 
            # the cluster to assign it as new prototype
            cluster.prototype = list(map(mean,zip(*mappedMember)))
    
    ## check if it converge
    def converging(self):
        check = 0
        # visit every cluster and check if there are any changes
        for cluster in self.clusters:
            if cluster.changing_members():
                check +=1
        return True if (check == self.k) else False

    
    def train(self):
        epoch = 0
        # implement k-means algorithm here:
        # Step 1: Select an initial random pertaining with k clusters
        self.random_initialization()
        self.avrage_cluster()
                
        # Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
        self.genrating_partition()
        
        # Step 3: recalculate cluster centers
        # Step 4: repeat until cluster membership stabilizes
        while not self.converging():
            epoch += 1
            self.avrage_cluster()
            self.genrating_partition()

        self.print_clusterPart(epoch)

    ## Find the closest prototype
    def prototype_index(self, dataSample_train):  
        for cluster_idx in range(self.k):
            if {dataSample_train}.issubset(self.clusters[cluster_idx].current_members):
                return self.clusters[cluster_idx].prototype
            
    def test(self):
            
        ## Total number of html_prefetched amongst all clients
        html_prefetched = 0
        ## Total number of requests amongst all clients
        total_requests = 0
        ## Total number of hits amongst all clients
        total_hits = 0

        ## Iterate to get the clients and a data sample from the testdata
        for client_idx, testSample in enumerate(self.testdata):
            ## closest cluster
            prototype = self.prototype_index(client_idx)
            ## predicted values
            predict = [(0 if idx < self.prefetch_threshold else 1) for idx in prototype]
            ## sum the entire prefetched to added up
            html_prefetched += sum(predict)
            
            # Calculate the hits & requests
            for n_dim in range(self.dim):
                if testSample[n_dim] == 1:
                    total_requests +=1
                    if predict[n_dim]== 1:
                        total_hits +=1
            
        ## Calculate the Hitrates
        if total_requests > 0: self.hitrate = (total_hits/ total_requests)
        ## Calculate the Accuracy
        if html_prefetched> 0: self.accuracy = (total_hits/ html_prefetched)
        
        # self.print_test

    
    #### Built-in Functions
    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)
        print()

    def print_members(self):
        for i, cluster in enumerate(self.clusters):
            print("Members cluster["+str(i)+"] :", cluster.current_members)
            print()

    def print_prototypes(self):
        for i, cluster in enumerate(self.clusters):
            print("Prototype cluster["+str(i)+"] :", cluster.prototype)
            print()
            
        
