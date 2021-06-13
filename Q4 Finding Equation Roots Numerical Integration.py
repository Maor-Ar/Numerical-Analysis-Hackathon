import math
import sympy as sp
from sympy.utilities.lambdify import lambdify

x = sp.symbols('x')
my_f = (sp.sin((2 * x ** 3) + (5 * x ** 2) - 6)) / (2 * sp.exp((-2) * x))

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


def SecantMethodInRangeIterations(f, check_range, epsilon=0.0000001):
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
        print(bcolors.HEADER, "Checked range:", startPoint, "-",endPoint, bcolors.ENDC)
        # Send to the Secant Method with 2 guesses
        local_root = SecantMethod(f, startPoint, endPoint, epsilon, iterCounter)
        # If the root has been found in previous iterations
        if round(local_root,6) in roots:
            print(bcolors.FAIL, "Already found that root.", bcolors.ENDC)
        # If the root is out of range tested
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
    # Recursive call with the following guess
    return SecantMethod(func, secondGuess, next_guess, epsilon, iterCounter + 1)


def NewtonsMethod(func, x0, tries=100, epsylon=0.0000001, symbole=sp.symbols('x')):
    """
    This function find a root to a function by using the newton raphson method by a given first guess.
    :param func: The function with sympy symbols.
    :param x0: The first guess.
    :param tries: Number of tries to find the root.
    :param symbole: The symbol you entered in the function (Default is lower case x)
    :param epsylon: The tolerance of the deviation of the solution ;
    How precise you want the solution (the smaller the better).
    :return:Returns the local root by raphson method ,
    if it fails to find a solutions in the given tries limit it will return None .
    """
    if func.subs(symbole, x0) == 0:
        return 0
    for i in range(tries):
        print(bcolors.OKBLUE, "Attempt #", i + 1, ":", bcolors.ENDC)
        print("f({0}) = {1} = {2}".format(x0, func, round(func.subs(symbole, x0), 2)))
        print("f'({0}) = {1} = {2}".format(x0, sp.diff(func, symbole),
                                           round(sp.diff(func, symbole).subs(symbole, x0), 2)))
        if sp.diff(func, symbole).subs(symbole, x0) == 0.0:
            continue
        next_x = (x0 - func.subs(symbole, x0) / sp.diff(func, symbole).subs(symbole, x0))
        print("next_X = ", round(next_x, 2))
        t = abs(next_x - x0)
        if t < epsylon:
            print(bcolors.OKGREEN, "Found a Root Solution ; X =", round(next_x, 8), bcolors.ENDC)
            return next_x
        x0 = next_x
    print(bcolors.FAIL, "Haven't Found a Root Solution ; (returning None)", bcolors.ENDC)
    return None


def NewtonsMethodInRangeIterations(func, check_range, tries=10, epsilon=0.0000001, symbol=sp.symbols('x')):
    """
    This function find a root to a function by using the newton raphson method by a given list of guesses.
    :param func: The function with sympy symbols.
    :param check_range: List of guesses.
    :param tries: Number of tries to find the root.
    :param symbole: The symbol you entered in the function (Default is lower case x)
    :param epsylon: The tolerance of the deviation of the solution ;
    How precise you want the solution (the smaller the better).
    :return:Returns a list roots by raphson method ,
    if it fails to find a solutions in the given tries limit it will return an empty list .
    """
    roots = []
    for i in check_range:
        check_number = round(i, 2)
        print(bcolors.HEADER, "First guess:", check_number , bcolors.ENDC)
        # Send to the Secant Method with one guess
        local_root = NewtonsMethod(func, check_number, tries, epsilon, symbol)
        if round(local_root, 6) in roots:
            print(bcolors.FAIL, "Already found that root.", bcolors.ENDC)
        elif not (check_range[0] <= local_root <= check_range[-1]):
            print(bcolors.FAIL, "root out of range.", bcolors.ENDC)
        elif local_root is not None:
            roots += [round(local_root, 6)]
    return roots


def TrapezoidalRule(my_f, n, a, b, tf):
    """
    rapezoidal Rule is a rule that evaluates the area under the curves by dividing the total area
    into smaller trapezoids rather than using rectangles
    :param my_f: The desired integral function
    :param n: The division number
    :param a: Lower bound
    :param b: Upper bound
    :param tf: Variable to decide whether to perform Error evaluation
    :return: The result of the integral calculation
    """
    h = (b - a) / n
    #if tf:
        #print(bcolors.FAIL, "Error evaluation En = ", round(TrapezError(my_f, b, a, h), 6), bcolors.ENDC)
    integral = 0.5 * (my_f(a) * my_f(b))
    for i in range(n):
        integral += my_f(a + h * i)
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
    k2 = b
    #print(bcolors.FAIL, "Error evaluation En = ", round(SimpsonError(my_f, b, a, h), 6), bcolors.ENDC)
    integral = func(a) + func(b)
    # Calculation of a polynomial lagranz for the sections
    for i in range(n):
        k = a + i * h # new a
        if i != n-1: # new b
            k2 = a+(i+1)*h
        if i % 2 == 0: #even places
            integral += 2 * func(k)
            str_even = "2 * "+str(func(k))
        else: #odd places
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


# def TrapezError(func, b, a, h):
#     """
#     The trapezoidal rule is a method for approximating definite integrals of functions.
#     The error in approximating the integral of a twice-differentiable function by the trapezoidal rule
#     is proportional to the second derivative of the function at some point in the interval.
#     :param func: The desired integral function
#     :param b: Upper bound
#     :param a: Lower bound
#     :param h: The division
#     :return: The error value
#     """
#     xsi = (-1)*(math.pi/2)
#     print("ƒ(x): ", func)
#     f2 = sp.diff(func, x, 2)
#     print("ƒ'': ", f2)
#     diff_2 = lambdify(x, f2)
#     print("ƒ''(", xsi, ") =", diff_2(xsi))
#     return h**2/12*(b-a)*1
#     #return h**2/12*(b-a)*diff_2(xsi)
#
#
# def SimpsonError(func, b, a, h):
#     """
#     The Simpson rule is a method for approximating definite integrals of functions.
#     The error in approximating the integral of a four-differentiable function by the trapezoidal rule
#     is proportional to the second derivative of the function at some point in the interval.
#     :param func: The desired integral function
#     :param b: Upper bound
#     :param a: Lower bound
#     :param h: The division
#     :return: The error value
#     """
#     xsi = 1
#     print("ƒ(x): ", func)
#     f2 = sp.diff(func, x, 4)
#     print("ƒ⁴: ", f2)
#     diff_4 = lambdify(x, f2)
#     print("ƒ⁴(", xsi, ") =", diff_4(xsi))
#
#     return (math.pow(h, 4) / 180)*(b-a)*diff_4(xsi)


def frange(start, end=None, inc=None):
    "Function for a range with incomplete numbers"
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


def MainFunction():

    roots = []
    checkRange = frange(-1, 1.6, 0.1)
    epsilon = 0.00001
    n = 18


    def func(val):
        return lambdify(x, my_f)(val)


    print(bcolors.OKBLUE,"Finding roots of the equation ƒ(X) = sin(2X³+5X²-6) / 2e^-2X\n",bcolors.ENDC)
    print(bcolors.OKGREEN, "\nNewton Raphson Method",bcolors.ENDC)
    roots += NewtonsMethodInRangeIterations(my_f, checkRange, 10, 0.000001)
    print("\nThere are ", bcolors.OKBLUE, len(roots), "roots found by Newton Raphson Method", bcolors.ENDC)
    printFinalResult(roots)
    roots.clear()
    print(bcolors.OKGREEN, "\nSecant Method", bcolors.ENDC)
    roots += SecantMethodInRangeIterations(func, checkRange, 0.0000001)
    print("\nThere are ", bcolors.OKBLUE, len(roots), "roots found by Secant Method" ,bcolors.ENDC)
    printFinalResult(roots)
    print(bcolors.OKGREEN, "\nNumerical Integration",bcolors.ENDC)
    print(bcolors.BOLD, "Division into sections n =", n, bcolors.ENDC)
    print(bcolors.OKBLUE, "Numerical Integration of definite integral in range [0,1] ∫= sin(2X³+5X²-6) / 2e^-2X\n", bcolors.ENDC)
    print(bcolors.OKGREEN, "\n\tSimpson’s Rule", bcolors.ENDC)
    res = SimpsonRule(func, n, 0, 1)
    if res[1]:
        print(bcolors.OKBLUE, "I = ", printFinalResult(round(res[0], 6)), bcolors.ENDC)
    else:
        print(bcolors.FAIL, "n must be even !", bcolors.ENDC)
    print(bcolors.OKGREEN, "\n\tRomberg's method", bcolors.ENDC)
    print(bcolors.OKBLUE, "I = ", printFinalResult(round(RombergsMethod(func, n, 0, 1)[n - 1][n - 1], 6)), bcolors.ENDC)

MainFunction()
