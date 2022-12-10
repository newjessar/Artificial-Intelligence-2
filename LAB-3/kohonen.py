import random as rd

class Cluster:
    """This class represents the clusters, it contains the
    prototype and a set with the ID's (which are Integer objects) 
    of the datapoints that are member of that cluster."""
    def __init__(self, dim):
        self.prototype = [0.0 for _ in range(dim)]
        self.current_members = set()

class Kohonen:
    def __init__(self, n, epochs, traindata, testdata, dim):
        self.n = n
        self.epochs = epochs
        self.traindata = traindata
        self.testdata = testdata
        self.dim = dim

        # A 2-dimensional list of clusters. Size == N x N
        self.clusters = [[Cluster(dim) for _ in range(n)] for _ in range(n)]
        # Threshold above which the corresponding html is prefetched
        self.prefetch_threshold = 0.7
        self.initial_learning_rate = 0.8
        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0
    
    ## Measures the Euclidean Distances
    def euclidean_distance(self, p1, p2):
        return(sum([(a-b)**2 for a, b in zip(p1, p2)]))

    def random_initialization(self):
        for row in range(self.n):
            for col in range(self.n):
                for feature in range(self.dim):
                    value = rd.random()
                    self.clusters[row][col].prototype[feature] = value

    def train(self):
        self.random_initialization()
        ## Step 1: Iterate through every epoch
        for inEpoch in range(self.epochs):      
            ## The radius r of the BMU's neighborhood is calculated
            r = (self.n/2)*(1-inEpoch/self.epochs)
            ## Learning rate calculation according to the given equation
            eTa = (self.initial_learning_rate*(1-inEpoch/self.epochs))
            
            
            ## Step 2: Calculating the square size and the learning rate eTa
            for vInput in self.traindata:
                eucl_dist = None
                
                ## Step 3: Find the BMI related to the given vector
                for clusterRow_IDX, clusterRow in enumerate(self.clusters):
                    for clusterCol_IDX, clusterCol in enumerate(clusterRow):
                        cluster_diff = self.euclidean_distance(clusterCol.prototype, vInput)
                        if eucl_dist == None or eucl_dist > cluster_diff:
                            eucl_dist = cluster_diff
                            bmu_row_idx,bmu_col_idx = clusterRow_IDX, clusterCol_IDX
                    
                ## Step 4: Each node in BMU’s neighborhood is adjusted
                for clusterRow_IDX, clusterRow in enumerate(self.clusters):                 
                    for clusterCol_IDX, clusterCol in enumerate(clusterRow):
                        if self.euclidean_distance(clusterCol.prototype,self.clusters[bmu_row_idx][bmu_col_idx].prototype) < r:
                            ## Step 5: Update the weights pre_prototype following the equation in the assignment
                            for idx in range(self.dim):
                                pre_prototype = clusterCol.prototype[idx]
                                x = vInput[idx]
                                clusterCol.prototype[idx] = (((1 - eTa)*pre_prototype)+(eTa*x))

    def prototype_index(self,  dataPoint):
        all_Prototypes = []
        # Gathering all prototypes and Backup the cluster members  
        for clusterRow in self.clusters:
            for cluster in clusterRow:
                all_Prototypes.append(cluster.prototype)
        # Calculating the distance for all prototypes to the dataPoint
        all_Dist_prototypes = [self.euclidean_distance(dataPoint, prototype) 
                    for prototype in all_Prototypes]
        # Get the minimum distance between all the K Prototypes
        nearest_prototype = min(all_Dist_prototypes)
        nearest_prototype_idx = all_Dist_prototypes.index(nearest_prototype)
        return self.clusters[int(nearest_prototype_idx//self.n)][int(nearest_prototype_idx%self.n)].prototype

    def test(self):
            
        ## Total number of html_prefetched amongst all clients
        html_prefetched = 0
        ## Total number of requests amongst all clients
        total_requests = 0
        ## Total number of hits amongst all clients
        total_hits = 0

        ## Iterate to get the clients and a data sample from the testdata
        for testSample in self.testdata:
            ## closest cluster
        
            prototype = self.prototype_index(testSample)
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
        
        self.print_test            

    def print_test(self):
        print("Prefetch threshold =", self.prefetch_threshold)
        print("Hitrate:", self.hitrate)
        print("Accuracy:", self.accuracy)
        print("Hitrate+Accuracy =", self.hitrate+self.accuracy)
        print()

    def print_members(self):
        for i in range(self.n):
            for j in range(self.n):
                print("Members cluster["+str(i)+"]["+str(j)+"] :", self.clusters[i][j].current_members)
                print()

    def print_prototypes(self):
        for i in range(self.n):
            for j in range(self.n):
               print("Prototype cluster["+str(i)+"]["+str(j)+"] :", self.clusters[i][j].prototype)
               print()
               
