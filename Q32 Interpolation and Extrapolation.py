import math

class bcolors:
    ENDC = '\033[0m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



from datetime import datetime
local_dt = datetime.now()
d = str(local_dt.day)
h = str(local_dt.hour)
m = str(local_dt.minute)



def multiplyList(myList): ##function to calculate multiply of all elements
    result = 1
    for x in myList:
        result = result * x
    return result

def Lagrange_interpolation(Table,Point_To_Find):  ##calculate by lagrange
    result=0 ##final result
    str1=" " ##each iteration the string will include
    finalresult=" " ##include all polinoms
    mul=[] ##each iteration includes all xi-xj
    print("----Lagrange_interpolation----")
    print("Table values are : ")
    for i in range(len(Table)):##print the table
        print("x"+str(i)+" is "+str(Table[i][0])+" F(X) is --> "+str(Table[i][1]))
    for i in range(len(Table)):##running for each table index
        val=1
        for j in range(len(Table)):
            if i!=j: ##using the formula according to i!=j
                str1+="(x-"+str(Table[j][0])
                mul.append((Table[i][0]-Table[j][0]))
                if(j==len(Table)-1 or i==len(Table)-1):
                    str1+=")/"
                else:
                    str1+=")"
                val = val * (Point_To_Find - Table[j][0]) / (Table[i][0] - Table[j][0]) ##calculate by formula x-xj/xi-xj
        print(bcolors.OKBLUE+"L"+str(i)+" ="+bcolors.OKGREEN+str1+str(multiplyList(mul)),bcolors.ENDC)
        finalresult+= str1+str(multiplyList(mul))+"*"+str(Table[i][1])
        mul.clear()
        str1=" "
        result= result + val * Table[i][1] ##calculate after inner loop the final result
    print("Final polinom is" + finalresult)
    print("Final solution by Lagrange interpolation " + str(round(result,6) )+ bcolors.FAIL + '00000' + bcolors.OKGREEN  + d + bcolors.OKBLUE + h + m + bcolors.ENDC + '\n')







def Neville_interpolation(Table,pointToFindVal):   ##calcuate Nevil algorithm by the nested formula
    print("----Neville's_interpolation----")
    length = len(Table)
    result = 0 ##final result
    for j in range(1,length):
        for i in range(length-1,j-1,-1): ##running for each pair combination in the table and calculate new value at table
            nevPrevSol = Table[i][1]
            Table[i][1]=((pointToFindVal-Table[i-j][0])*Table[i][1] - (pointToFindVal-Table[i][0])*Table[i-1][1] )/ (Table[i][0]-Table[i-j][0])
            print(bcolors.OKBLUE+"P"+str(i-j)+str(i)+"="+bcolors.ENDC+"(x-"+str(Table[i-j][0])+")*"+str(nevPrevSol)+"- (x-"+str(Table[i][0])+")*"+str(Table[i-1][1])+")/("+str(Table[i][0])+"-"+str(Table[i-j][0])+") ="+ bcolors.OKGREEN +str(Table[i][1]),bcolors.ENDC)
    result = Table[length-1][1]
    print("Final solution by Neville's interpolation " + str(round(result,6)) + bcolors.FAIL + '00000' + bcolors.OKGREEN + d + bcolors.OKBLUE + h + m + bcolors.ENDC + '\n')




#------------------------main-------------------------------------

TableForProject = [[0.2,13.7241], [0.35,13.9776], [0.45 ,14.0625], [0.6,13.9776], [0.75,13.7241] , [0.85,13.3056] , [0.9 , 12.7281]]
point =0.65


Lagrange_interpolation(TableForProject,point)
Neville_interpolation(TableForProject,point)


