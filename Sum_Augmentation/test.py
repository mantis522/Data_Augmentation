import pandas as pd

# list of name, degree, score
nme = ["aparna", "pankaj", "sudhir", "Geeku"]
deg = ["MBA", "BCA", "M.Tech", "MBA"]
scr = [90, 40, 80, 98]

# dictionary of lists
dict2 = {'name': nme, 'degree': deg, 'score': scr}

df = pd.DataFrame(dict2)

df.to_csv("test.csv")