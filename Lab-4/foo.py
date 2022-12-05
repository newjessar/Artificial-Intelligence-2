import mdp
while(True):
    i = input("What type of map do you like to Rrn: RN-> inter 1 or 2D-> inter 2 or 3 for exit:   ")
    if (i == "1"):
        n = input("Choose the Run Value-> inter 1 or Policy-> inter 2:   ")
        if (n == "1"):
            m1 = mdp.makeRNProblem()
            m1.valueIteration()
            m1.printValues()
            m1.printActions()

        else:
            if (n == "2"):
                m2 = mdp.makeRNProblem()
                m2.policyIteration()
                m2.printValues()
                m2.printActions()
    
    elif (i == "2"):        
            n = input("Choose the Run Value-> inter 1 or Policy-> inter 2:   ")
            if (n == "1"):
                m1 = mdp.make2DProblem()
                m1.valueIteration()
                m1.printValues()
                m1.printActions()
            else:
                if (n == "2"):
                    m2 = mdp.make2DProblem()
                    m2.policyIteration()
                    m2.printValues()
                    m2.printActions()
    else:
        if (i == "3"): 
            break
