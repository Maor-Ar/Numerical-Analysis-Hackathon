"""

 * Authors: Maor Arnon (ID: 205974553) and Neriya Zudi (ID:207073545)
 * Emails: maorar1@ac.sce.ac.il    neriyazudi@Gmail.com
 * Department of Computer Engineering - Assignment 2 - Numeric Analytics
"""

from datetime import datetime

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[90m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # Background colors:
    GREYBG = '\033[100m'
    REDBG = '\033[101m'
    GREENBG = '\033[102m'
    YELLOWBG = '\033[103m'
    BLUEBG = '\033[104m'
    PINKBG = '\033[105m'
    CYANBG = '\033[106m'


local_dt = datetime.now()
d=str(local_dt.day)
h=str(local_dt.hour)
m=str(local_dt.minute)
Five_zeros="00000"

def PrintMatrix(matrix):
    """
    Matrix Printing Function
    :param matrix: Matrix nxn
    """
    for line in matrix:
        line.append('|')
        line.insert(0,'|')
        print('  '.join(map(str, line)))
        line.remove('|',)
        line.remove('|',)
def PrintVectorFinal(vector):
    """
    Matrix Printing Function
    :param matrix: Matrix nxn
    """
    print('Solution:')
    PrintMatrix(vector)
    print()
    for i in range(len(vector)):
        print("Solution to x"+str(i),'value, in the required format.:')
        print(str(vector[i][0])+bcolors.FAIL+'00000'+bcolors.OKGREEN+d+bcolors.OKBLUE+h+m+bcolors.ENDC+'\n')

def Determinant(matrix, mul):
    """
    Recursive function for determinant calculation
    :param matrix: Matrix nxn
    :param mul: The double number
    :return: determinant of matrix
    """
    width = len(matrix)
    # Stop Conditions
    if width == 1:
        return mul * matrix[0][0]
    else:
        sign = -1
        det = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(matrix[j][k])
                m.append(buff)
            # Change the sign of the multiply number
            sign *= -1
            #  Recursive call for determinant calculation
            det = det + mul * Determinant(m, sign * matrix[0][i])
    return det


def MaxNorm(matrix):
    """
    Function for calculating the max-norm of a matrix
    :param matrix: Matrix nxn
    :return:max-norm of a matrix
    """
    max_norm = 0
    for i in range(len(matrix)):
        norm = 0
        for j in range(len(matrix)):
            # Sum of organs per line with absolute value
            norm += abs(matrix[i][j])
        # Maximum row amount
        if norm > max_norm:
            max_norm = norm

    return max_norm


def MultiplyMatrix(matrixA, matrixB):
    """
    Function for multiplying 2 matrices
    :param matrixA: Matrix nxn
    :param matrixB: Matrix nxn
    :return: Multiplication between 2 matrices
    """
    # result matrix initialized as singularity matrix
    result = [[0 for y in range(len(matrixB[0]))] for x in range(len(matrixA))]
    for i in range(len(matrixA)):
        # iterate through columns of Y
        for j in range(len(matrixB[0])):
            # iterate through rows of Y
            for k in range(len(matrixB)):
                result[i][j] += matrixA[i][k] * matrixB[k][j]
    return result


def MakeIMatrix(cols, rows):
    # Initialize a identity matrix
    return [[1 if x == y else 0 for y in range(cols)] for x in range(rows)]


def InverseMatrix(matrix,vector):
    """
    Function for calculating an inverse matrix
    :param matrix:  Matrix nxn
    :return: Inverse matrix
    """
    # Unveri reversible matrix
    if Determinant(matrix, 1) == 0:
        print("Error,Singular Matrix\n")
        return
    # result matrix initialized as singularity matrix
    result = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # turn the pivot into 1 (make elementary matrix and multiply with the result matrix )
        # pivoting process
        matrix, vector = RowXchange(matrix, vector)
        print("Matrix after exchanging rows for the",i+1,"time:")
        PrintMatrix(matrix)
        print("Matrix after making row",i+1,"a pivot - (row",i+1,")/",matrix[i][i],":")
        elementary = MakeIMatrix(len(matrix[0]), len(matrix))
        elementary[i][i] = 1/matrix[i][i]
        result = MultiplyMatrix(elementary, result)
        matrix = MultiplyMatrix(elementary, matrix)
        PrintMatrix(matrix)
        # make elementary loop to iterate for each row and subtracrt the number below (specific) pivot to zero  (make
        # elementary matrix and multiply with the result matrix )
        for j in range(i+1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)
        print("Matrix after subtracting with row",i+1,"making the lower part of the column 0")
        PrintMatrix(matrix)
    # after finishing with the lower part of the matrix subtract the numbers above the pivot with elementary for loop
    # (make elementary matrix and multiply with the result matrix )
    for i in range(len(matrix[0])-1, 0, -1):
        for j in range(i-1, -1, -1):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)
    print("Matrix after subtracting with row",i+1,"making the upper part of the column 0")
    PrintMatrix(matrix)
    return result


def InverseMatrixNoPrints(matrix,vector):
    """
    Function for calculating an inverse matrix, Without printing the steps (for checking cond)
    :param matrix:  Matrix nxn
    :return: Inverse matrix
    """
    # Unveri reversible matrix
    if Determinant(matrix, 1) == 0:
        print("Error,Singular Matrix\n")
        return
    # result matrix initialized as singularity matrix
    result = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # turn the pivot into 1 (make elementary matrix and multiply with the result matrix )
        # pivoting process
        matrix, vector = RowXchange(matrix, vector)
        elementary = MakeIMatrix(len(matrix[0]), len(matrix))
        elementary[i][i] = 1/matrix[i][i]
        result = MultiplyMatrix(elementary, result)
        matrix = MultiplyMatrix(elementary, matrix)
        # make elementary loop to iterate for each row and subtracrt the number below (specific) pivot to zero  (make
        # elementary matrix and multiply with the result matrix )
        for j in range(i+1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)
    # after finishing with the lower part of the matrix subtract the numbers above the pivot with elementary for loop
    # (make elementary matrix and multiply with the result matrix )
    for i in range(len(matrix[0])-1, 0, -1):
        for j in range(i-1, -1, -1):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            elementary[j][i] = -(matrix[j][i])
            matrix = MultiplyMatrix(elementary, matrix)
            result = MultiplyMatrix(elementary, result)
    return result


def RowXchange(matrix, vector):
    """
    Function for replacing rows with both a matrix and a vector
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Replace rows after a pivoting process
    """

    for i in range(len(matrix)):
        max = abs(matrix[i][i])
        for j in range(i, len(matrix)):
            # The pivot member is the maximum in each column
            if abs(matrix[j][i]) > max:
                temp = matrix[j]
                temp_b = vector[j]
                matrix[j] = matrix[i]
                vector[j] = vector[i]
                matrix[i] = temp
                vector[i] = temp_b
                max = abs(matrix[i][i])

    return [matrix, vector]


def GaussJordanElimination(matrix, vector):
    """
    Function for moding a linear equation using gauss's elimination method
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=A(-1)b
    """
    # Pivoting process
    # matrix, vector = RowXchange(matrix, vector)
    # print("Matrix after pivoting - ")
    # PrintMatrix(matrix)
    # Inverse matrix calculation
    invert = InverseMatrix(matrix,vector)
    print("Matrix after inversion - ")
    PrintMatrix(invert)
    return MulMatrixVector(invert, vector)


def MulMatrixVector(InversedMat, b_vector):
    """
    Function for multiplying a vector matrix
    :param InversedMat: Matrix nxn
    :param b_vector: Vector n
    :return: Result vector
    """
    result = []
    # Initialize the x vector
    for i in range(len(b_vector)):
        result.append([])
        result[i].append(0)
    # Multiplication of inverse matrix in the result vector
    for i in range(len(InversedMat)):
        for k in range(len(b_vector)):
            result[i][0] += InversedMat[i][k] * b_vector[k][0]
    return result


def UMatrix(matrix,vector):
    """
    :param matrix: Matrix nxn
    :return:Disassembly into a  U matrix
    """
    # result matrix initialized as singularity matrix
    U = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # pivoting process
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            # Finding the M(ij) to reset the organs under the pivot
            elementary[j][i] = -(matrix[j][i])/matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)
    # U matrix is a doubling of elementary matrices that we used to reset organs under the pivot
    U = MultiplyMatrix(U, matrix)
    return U


def LMatrix(matrix,vector):
    """
       :param matrix: Matrix nxn
       :return:Disassembly into a  L matrix
       """
    # Initialize the result matrix
    L = MakeIMatrix(len(matrix), len(matrix))
    # loop for each row
    for i in range(len(matrix[0])):
        # pivoting process
        matrix, vector = RowXchageZero(matrix, vector)
        for j in range(i + 1, len(matrix)):
            elementary = MakeIMatrix(len(matrix[0]), len(matrix))
            # Finding the M(ij) to reset the organs under the pivot
            elementary[j][i] = -(matrix[j][i])/matrix[i][i]
            # L matrix is a doubling of inverse elementary matrices
            L[j][i] = (matrix[j][i]) / matrix[i][i]
            matrix = MultiplyMatrix(elementary, matrix)

    return L


def RowXchageZero(matrix,vector):
    """
      Function for replacing rows with both a matrix and a vector
      :param matrix: Matrix nxn
      :param vector: Vector n
      :return: Replace rows after a pivoting process
      """

    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            # The pivot member is not zero
            if matrix[i][i] == 0:
                temp = matrix[j]
                temp_b = vector[j]
                matrix[j] = matrix[i]
                vector[j] = vector[i]
                matrix[i] = temp
                vector[i] = temp_b


    return [matrix, vector]



def SolveLU(matrix, vector):
    """
    Function for deconstructing a linear equation by ungrouping LU
    :param matrix: Matrix nxn
    :param vector: Vector n
    :return: Solve Ax=b -> x=U(-1)L(-1)b
    """
    matrixU = UMatrix(matrix,vector)
    matrixL = LMatrix(matrix,vector)
    return MultiplyMatrix(InverseMatrix(matrixU,vector), MultiplyMatrix(InverseMatrix(matrixL,vector), vector))


def Cond(matrix, invert):
    """
    :param matrix: Matrix nxn
    :param invert: Inverted matrix
    :return: CondA = ||A|| * ||A(-1)||
    """
    print("\n|| A ||max = ", MaxNorm(matrix))
    print("\n|| A(-1) ||max = ", MaxNorm(invert))
    return MaxNorm(matrix)*MaxNorm(invert)

def GaussSeidelMethod(matrix, vector, epsilon, previous, counter):

    NextGuess = []
    ImprovedGuess = CopyVector(previous)
    for i in range(len(matrix)):
        ins = 0
        for j in range(len(matrix)):
            if i != j:
                ins = ins + matrix[i][j]*ImprovedGuess[j]
        newGuess = 1/matrix[i][i]*(vector[i][0]-ins)
        ImprovedGuess[i] = newGuess
        NextGuess.append(newGuess)

    print("Iteration no. "+str(counter)+" " +str(NextGuess))

    for i in range(len(matrix)):
        if abs(NextGuess[i] - previous[i]) < epsilon:
            return NextGuess

    return GaussSeidelMethod(matrix, vector, epsilon,NextGuess,counter+1)

def InitVector(size):
    return [0 for index in range(size)]

def CopyVector(vector):
    copy = []
    for i in range(len(vector)):
        copy.append(vector[i])

    return copy

def CheckDominantDiagonal(matrix):
    for i in range(len(matrix)):
        sum = 0
        for j in range(len(matrix)):
            if i != j:
                sum += abs(matrix[i][j])
        if abs(matrix[i][i]) < sum:
            return False
    return True

def DominantDiagonalFix(matrix):
    #Check if we have a dominant for each column
    dom = [0]*len(matrix)
    result = list()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (matrix[i][j] > sum(map(abs,map(int,matrix[i])))-matrix[i][j]) :
                dom[i]=j
    for i in range(len(matrix)):
        result.append([])
        if i not in dom:
            print("Couldn't find dominant diagonal.")
            return matrix
    for i,j in enumerate(dom):
        result[j]=(matrix[i])
    return result

def CheckGeusZaidelGnorm(matrix):
    return 1 > MaxNorm(GeusZaidelG(matrix))

def GeusZaidelG(matrix):
    D, L, U = matrixDLUdissasembly(matrix)
    return MultiplyMatrix(minusMatrix(InverseMatrix(matrixAddition(L, D),[[]])), U)
    #I want to invert but I dont care about a vector so im reusing the function I made and sending an empthy vector
def matrixDLUdissasembly(matrix):
    D, L, U = list(), list(), list()
    for x, row in enumerate(matrix):
        D.append(list()), L.append(list()), U.append(list())
        for y, value in enumerate(row):
            if x == y:
                D[x].append(value), L[x].append(0), U[x].append(0)
            elif x < y:
                D[x].append(0), L[x].append(0), U[x].append(value)
            elif x > y:
                D[x].append(0), L[x].append(value), U[x].append(0)
    return D, L, U

def matrixAddition(matrixA, matrixB):
    return [[a + b for (a, b) in zip(i, j)] for (i, j) in zip(matrixA, matrixB)]

def minusMatrix(matrix):
    return [[-i for i in j] for j in matrix]


matrixA = [[0, 1, 2], [-2 ,1,0.5], [1,-2,-0.5]]
vectorb = [[0], [4], [-4]]
detA = Determinant(matrixA, 1)
print("\nMatrix A: \n")
PrintMatrix(matrixA)
print("\nVector b: \n")
PrintMatrix(vectorb)
print("\nDET(A) = ", detA)
print("\n----The First method according to the elimination of Gauss (of course includes the use of pivoting and the calculation of COND)----")
print("\nCondA = ||A|| * ||A(-1)|| = ", Cond(matrixA, InverseMatrixNoPrints(matrixA,vectorb)))
print("\nGaussJordanElimination\n")
JordanEliminationSol = GaussJordanElimination(matrixA, vectorb)
print("\nfinal result for x=(A^-1) * b  \n")
PrintVectorFinal(JordanEliminationSol)
print("\n----The Second method according to the LU dismantling----\n")
PrintMatrix(matrixA)
print("\n-----Building L and U matrices----\n")
luSolution = SolveLU(matrixA, vectorb)
print("Matrix U: \n")
PrintMatrix(UMatrix(matrixA,vectorb))
print("\nMatrix L: \n")
PrintMatrix(LMatrix(matrixA,vectorb))
print("\nMatrix A=LU: \n")
PrintMatrix(MultiplyMatrix(LMatrix(matrixA,vectorb),UMatrix(matrixA,vectorb)))
print("\nSolve Ax = b: ")
print('LU vector solution:\n')
PrintVectorFinal(luSolution)
print("\n----The Third method according to the Gauss Seidel Method (including finding dominant diagonal)----")
epsilon = 0.00001
#geuss

if CheckDominantDiagonal(matrixA):
    print("There is a dominant diagonal.")
    seidelSolution = GaussSeidelMethod(matrixA,vectorb,epsilon,InitVector(len(vectorb)),1)
    seidelSolution = [[a] for a in seidelSolution]
    PrintVectorFinal(seidelSolution)
else:
    print("There isn't a dominant diagonal.")
    print("We will try to find dominant diagonal.")
    dominantFix=DominantDiagonalFix(matrixA)
    PrintMatrix(dominantFix)
    if dominantFix != matrixA:
        print("Found a dominant diagonal.")
        seidelSolution = GaussSeidelMethod(dominantFix,vectorb,epsilon,InitVector(len(vectorb)),1)
        PrintVectorFinal(seidelSolution)
    else:
        print("didnt find a dominant diagonal.")
        if CheckGeusZaidelGnorm(matrixA):
            print("The matrix convergent.")
            seidelSolution = GaussSeidelMethod(matrixA,vectorb,epsilon,InitVector(len(vectorb)),1)
            PrintVectorFinal(seidelSolution)
        else:
            print(bcolors.FAIL+"\nThe matrix isn't convergent.\n"+bcolors.ENDC)
            print("Can't solve this matrix by Gauss Seidel Method \nYou can find other solutions above this one (LU "
                  "disassembling and elimination of Gauss)")