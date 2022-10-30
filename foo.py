import mdp

# n = input("Choose the Run 1 or 2: ")
# if (n == "1"):
    
m1 = mdp.makeRNProblem()
m1.valueIteration()
m1.printValues()
m1.printActions()
# else:
#     if (n == "2"):
#         m2 = mdp.makeRNProblem()
#         m2.policyIteration()
#         m2.printActions()
        