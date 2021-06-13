import math
from math import e
import sympy as sp
from sympy.utilities.lambdify import lambdify

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



def checkifcontinus(func,x,symbol):
    """
    Function for checking if specific x is in domain of specific function
    :param func: Any function
    :param x: specific x
    :param symbol: x as factor
    :return: True / False
    """
    return (sp.limit(func, symbol, x).is_real)

def printFinalResult(result):
    """
        Function for printing final results according to the requested format
        :param result: The final results (list or number)
        :return: print the result
    """
    from datetime import datetime
    local_dt = datetime.now()
    d = str(local_dt.day)
    h = str(local_dt.hour)
    m = str(local_dt.minute)
    if isinstance(result, list):
        for i in range(len(result)):
            print(str(result[i])+bcolors.FAIL+"00000"+bcolors.OKGREEN+d+bcolors.OKBLUE+h+m+bcolors.ENDC)
        return
    return str(result)+bcolors.FAIL+"00000"+bcolors.OKGREEN+d+bcolors.OKBLUE+h+m+bcolors.ENDC


def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)

    return L
def EvaluateError(startPoint, endPoint):
    """
   This function Helps us find out if we can find a root in a limited amount of tries in a specific range.
   :param startPoint: start of range.
   :param endPoint: end of range.
   :return:
   """
    exp = pow(10, -10)
    if endPoint - startPoint == 0:
        return 100
    return ((-1) * math.log(exp / (endPoint - startPoint), e)) / math.log(2, e)

def SecantMethodInRangeIterations(f, check_range, epsilon=0.0001):
    """
    This function find a root to a function by using the secant method by a given list of values to check beetween.
    :param f: The function (as a python function).
    :param check_range: List of values to check between ; e.g (1,2,3,4,5) it will check between 1-2,2-3,....
    :param epsylon: The tolerance of the deviation of the solution ;
    How precise you want the solution (the smaller the better).
    :return:Returns a list roots by secant method ,
    if it fails to find a solutions in the given tries limit it will return an empty list .
    """

    roots = []
    iterCounter = 0
    for i in check_range:
        startPoint = round(i, 2)
        endPoint = round(i + 0.1, 2)
        local_root = SecantMethod(f, startPoint, endPoint, epsilon, iterCounter)
        if local_root in roots:
            print(bcolors.FAIL, "Already found that root.", bcolors.ENDC)
        elif not (startPoint <= local_root <= endPoint):
            print(bcolors.FAIL, "root out of range.", bcolors.ENDC)
        elif local_root is not None:
            roots += [round(local_root, 6)]

    return roots


def SecantMethod(func, firstGuess, secondGuess, epsilon, iterCounter):
    """
     This function find a root to a function by using the SecantMethod method by a given tow guess.
    :param func: The function on which the method is run
    :param firstGuess: The first guess
    :param secondGuess: The second guess
    :param epsilon: The tolerance of the deviation of the solution
    :param iterCounter: number of tries until the function found the root.
    :return:Returns the local root by Secant method ,
    if it fails to find a solutions in the given tries limit it will return None .
    """

    if iterCounter > 100:
        return

    if abs(secondGuess - firstGuess) < epsilon: #Stop condition
        print("after ", iterCounter, "iterations The root found is: ", bcolors.OKBLUE, round(secondGuess, 6), bcolors.ENDC)
        return round(secondGuess, 6) # Returns the root found

    next_guess = (firstGuess * func(secondGuess) - secondGuess * func(firstGuess)) / (func(secondGuess) - func(firstGuess))
    print(bcolors.OKGREEN, "iteration no.", iterCounter, bcolors.ENDC, "\tXi = ", firstGuess, " \tXi+1 = ", secondGuess,
          "\tf(Xi) = ", func(firstGuess))

    return SecantMethod(func, secondGuess, next_guess, epsilon, iterCounter + 1)


def BisectionMethod(polynomial, startPoint, endPoint, epsilon, iterCounter):
    """
    the bisection method is a root-finding method that applies to any
    continuous functions for which one knows two values with opposite signs
    :param polynomial: The function on which the method is run
    :param startPoint: Starting point of the range
    :param endPoint: End point of the range
    :param epsilon: The tolerance of the deviation of the solution
    :param iterCounter: Counter of the number of iterations performed
    :return: Roots of the equation found
    """
    roots = []
    middle = (startPoint + endPoint) / 2

    if iterCounter > EvaluateError(startPoint, endPoint):
        print(bcolors.FAIL, "The Method isn't convergent.", bcolors.ENDC)
        return roots

    if (abs(endPoint - startPoint)) < epsilon:
        print("after ", iterCounter-1, "iterations The root found is: ", bcolors.OKBLUE, round(middle, 6),
              bcolors.ENDC)
        roots.append(round(middle, 6))
        return roots

    if polynomial(startPoint) * polynomial(middle) > 0:
        print(bcolors.OKGREEN,"iteration no.", iterCounter, bcolors.ENDC, "\ta = ",middle, " \tb = ", endPoint, "\tf(a) = ",polynomial(middle),
              "\tf(b) = ",polynomial(endPoint))
        roots += BisectionMethod(polynomial, middle, endPoint, epsilon, iterCounter + 1)
        return roots
    else:
        print(bcolors.OKGREEN,"iteration no.", iterCounter, bcolors.ENDC, "\ta = ", startPoint, "\tb = ", middle, "\tf(a) = ", polynomial(startPoint),
              "\tf(b) = ", polynomial(middle))
        roots += BisectionMethod(polynomial, startPoint, middle, epsilon, iterCounter + 1)
        return roots


def BisectionMethodSections(func, Cheackrange, epsilon):
    """
    This function dividing the range into small sections and send the to the bisection method
    :param func: Any function
    :param Cheackrange: Range for checking
    :param epsilon: Stop condition

    """
    iterCounter = 0
    result = []
    for i in Cheackrange:
        seperate = round(i, 2)
        next_seperate = round(i+ 0.1, 2)
        if checkifcontinus(my_f,seperate,sp.symbols('x')) == False :
            print(bcolors.FAIL,"this point ", seperate, " not in domain of this function", bcolors.ENDC)
            continue
        if func(seperate) == 0:
            print(bcolors.OKBLUE, "root in ", seperate, bcolors.ENDC)
            result.append(seperate)
        if (func(seperate) * func(next_seperate)) < 0:
            print("sign changing found between ",seperate,'-',next_seperate)
            result += BisectionMethod(func, seperate, next_seperate, epsilon, iterCounter)

    return result

def func(val):
    """
    This fuction get a specific value and return the value of the function on the specific given value (F(x))
    :param val:specific x
    :return: F(x)
    """
    x = sp.symbols('x')
    my_f = (x * sp.exp(-x**2 +5*x)) * (2 * x ** 2 - 3 * x - 5)
    return lambdify(x,my_f)(val)


def TrapezoidalRule(f, n, a, b, tf):
    """
    rapezoidal Rule is a rule that evaluates the area under the curves by dividing the total area
    into smaller trapezoids rather than using rectangles
    :param f: The desired integral function
    :param n: The division number
    :param a: Lower bound
    :param b: Upper bound
    :param tf: Variable to decide whether to perform Error evaluation
    :return: The result of the integral calculation
    """
    h = (b - a) / n
    # if tf:
    #     print(bcolors.FAIL, "Error evaluation En = ", round(TrapezError(func(), b, a, h), 6), bcolors.ENDC)
    integral = 0.5 * (f(a)*f(b))
    for i in range(n):
        integral += f(a+h*i)
    integral *= h
    return integral


def SimpsonRule(func, n, a, b):
    """
    Simpson’s Rule is a numerical method that approximates the value of a definite integral by using quadratic
     functions Simpson’s Rule is based on the fact that given three points,
    we can find the equation of a quadratic through those points (by Lagrange's interpolation)
    :param func: The desired integral function
    :param n: The division number(must be even)
    :param a: Lower bound
    :param b: Upper bound
    :return: The result of the integral calculation
    """
    if n % 2 != 0:
        return 0, False
    h = (b - a) / n
    print("h = ", h)
    str_even = ""
    str_odd = ""
    k2=b
    #print(bcolors.FAIL, "Error evaluation En = ", round(SimpsonError(my_f, b, a, h), 6), bcolors.ENDC)
    integral = func(a) + func(b)
    for i in range(n):
        k = a + i * h
        if i != n-1:
            k2 = a+(i+1)*h
        if i % 2 == 0:
            integral += 2 * func(k)
            str_even = "2 * "+str(func(k))
        else:
            integral += 4 * func(k)
            str_odd = "4 * "+str(func(k))
        print("h/3 ( ", str(func(k)), " + ", str_odd, " + ", str_even, " + ", str(func(k2))," )")
    integral *= (h/3)
    return integral, True

def RombergsMethod(f, n, a, b):
    """
    Romberg integration is an extrapolation technique which allows us to take a sequence
    approximate solutions to an integral and calculate a better approximation.
    This technique assumes that the function we are integrating is sufficiently differentiable
    :param f: The desired integral function
    :param n: The division number
    :param a: Lower bound
    :param b: Upper bound
    :return: The result of the integral calculation
    """
    matrix = [[0 for i in range(n)] for j in range(n)]
    for k in range(0, n):
        # Using the trapezoidal method
        matrix[k][0] = TrapezoidalRule(f, 2**k, a, b, False)
        # Romberg recursive formula Using values that have already been calculated
        for j in range(0, k):
            matrix[k][j + 1] = (4 ** (j + 1) * matrix[k][j] - matrix[k - 1][j]) / (4 ** (j + 1) - 1)
            print("R[{0}][{1}] = ".format(k, j+1), round(matrix[k][j+1], 6))
    return matrix


def MainFunction():

    roots = []
    x = sp.symbols('x')
    my_f = (x * sp.exp(-x**2 +5*x)) * (2 * x ** 2 - 3 * x - 5)
    my_f_diff = lambda a: sp.diff(my_f, x).subs(x, a)
    checkRange = frange(0, 3, 0.1)
    epsilon = 0.00001
    print("\nFinding roots of the equation f(X) = ( Xe^(-X^2 + 5X) ) * (2X^2 - 3X - 5 )\n")
    print("Bisection Method on ( Xe^(-X^2 + 5X) ) * (2X^2 - 3X - 5 ) :\n" )
    print(bcolors.OKGREEN, " ~~ Odd multiplicity Roots ~~", bcolors.ENDC)
    roots += BisectionMethodSections(func, checkRange, epsilon)
    print(bcolors.OKGREEN, "\n ~~ Even multiplicity Roots ~~", bcolors.ENDC)
    root = BisectionMethodSections(my_f_diff, checkRange, epsilon)
    if func(root[0]) == 0:
        roots += root
    else:
        print(bcolors.FAIL, " Not the root of the equation ", bcolors.ENDC)
    print("\nThere are ", bcolors.OKBLUE, len(roots), "roots found by Bisection Method" , bcolors.ENDC)
    printFinalResult(roots)
    roots.clear()
    print("\nSecant Method on ( Xe^(-X^2 + 5X) ) * (2X^2 - 3X - 5 ) :\n")
    roots += SecantMethodInRangeIterations(func, checkRange, 0.0000001)
    print("\nThere are ", bcolors.OKBLUE, len(roots), "roots found by Secant Method ", bcolors.ENDC)
    printFinalResult(roots)



    print("\nFinding area of the equation f(X) = ( Xe^(-X^2 + 5X) ) * (2X^2 - 3X - 5 )\n")
    print("Simpson Rule on ( Xe^(-X^2 + 5X) ) * (2X^2 - 3X - 5 ) :")
    res =SimpsonRule(func,20,0.5,1)
    if res[1]:
        print(bcolors.OKBLUE, "I = ", printFinalResult(round(res[0], 6)), bcolors.ENDC)
        print("\n")
    else:
        print(bcolors.FAIL, "n must be even !", bcolors.ENDC)

    n =20
    print("Rumberg Rule on ( Xe^(-X^2 + 5X) ) * (2X^2 - 3X - 5 ) :\n")
    print(bcolors.OKBLUE, "I = ", printFinalResult(round(RombergsMethod(func, n, 0.5,1)[n - 1][n - 1], 6)), bcolors.ENDC)

x = sp.symbols('x')
my_f = (x * sp.exp(-x**2 +5*x)) * (2 * x ** 2 - 3 * x - 5)
MainFunction()