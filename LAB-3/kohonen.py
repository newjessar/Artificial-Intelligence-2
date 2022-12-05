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
        self.prefetch_threshold = 0.5
        self.initial_learning_rate = 0.8
        # The accuracy and hitrate are the performance metrics (i.e. the results)
        self.accuracy = 0
        self.hitrate = 0


    def train(self):
        ## Step 1: Each cluster center is randomly initialized.
        for inEpoch in range(self.epochs):
            # The algorithm requires you to specify the size of the map N (a square map with N × N nodes will be created)
            # and the number of training epochs tmax (the total number of runs over the training data, e.g. 100).
            

            
        ## Step 2: For all vectors from the set: The vector is presented to the map
            for vInput in self.traindata:
                eucl_dist = None
                cluster_diff = 0    
        ## Step 3: All nodes in the map (which represent cluster centers) are examined to find the closest to the input vector 
        # in terms of the Euclidean distance. The winning node is known as the Best Matching Unit (BMU).
                for clusterInput_IDX, clusterInput in enumerate(self.clusters):
                    for clusterValue_idx, clusterValue in enumerate(clusterInput):
                        for dim_idx in range(self.dim):
                            cluster_diff += pow(clusterValue.prototype[dim_idx] - vInput[dim_idx],2)
                        if eucl_dist == None or eucl_dist > cluster_diff:
                            eucl_dist = cluster_diff
                            bmu_node_idx, bmu_in = clusterInput_IDX, clusterValue_idx 
                            
        ## Step 4: set to half the “size” of the map, but diminishes with each training epoch. All the 
        # nodes found within this radius are inside the BMU’s neighborhood.
                ## The radius r of the neighborhood of the BMU is calculated
                r = round((self.n/2)*(1-inEpoch/self.epochs),2)
                for bmu_neighbor_idx, bmu_neighbor in enumerate(self.clusters):
                    if bmu_neighbor_idx < (bmu_node_idx - r) or clusterInput_IDX > (bmu_node_idx + r):
                        continue    
                    for nodeValue_idx, nodeValue in enumerate(bmu_neighbor):
                        if nodeValue_idx < (bmu_in - r) or nodeValue_idx > (bmu_in + r):
                            continue
                        
                        ## Learning rate calculation according to the equation
                        eTa = (0.8*(1-inEpoch/self.epochs))
                        ## Step 5: Each node in BMU’s neighborhood (the nodes found in step 4) is 
                        # adjusted to make it more like the input vector:
                        for idx in range(self.dim):
                            p_OLD = nodeValue.prototype[idx]
                            x = vInput[idx]
                            p_NEW = (((1 - eTa)*p_OLD)+(eTa*x))
                            nodeValue.prototype[idx] = p_NEW
                        
        ## Step 6: Repeat steps 2, 3, 4, 5 running over the whole training dataset several 
        # times (tmax) and at each training epoch decrease the radius of the neighborhood r 
        # and the learning rate η.
    ## Find the closest prototype
    def prototype_index(self, dataSample_test):  
        # for cluster_x in self.clusters:
        #     for cluster_y in cluster_x:
        #         print(cluster_y.prototype)
        #         if cluster_y.prototype == dataSample_test:
        #             return cluster_y.prototype
        for x in range(self.n):
            for y in range(self.n):
                if self.clusters[x][y].prototype == dataSample_test:
                    return self.clusters[x][y].prototype
            
    def test(self):
        # iterate along all clients. Assumption: the same clients are in the same order as in the testData
        # for each client find the cluster of which it is a member
        # get the actual testData (the vector) of this client
        # iterate along all dimensions
        # and count prefetched htmls
        # count number of hits
        # count number of requests
        # set the variables hitrate and accuracy to their appropriate value
            
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