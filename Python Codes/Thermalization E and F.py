import numpy as np
import matplotlib.pyplot as plt

h = 0.0008
N = 7


def ph_0_star(t):
    if N == 3:
        return 0.13182634 * np.exp(0.00000000 * t) + -0.37591213 * np.exp(-0.93459463 * t) + 0.53420318 * np.exp(-1.18340632 * t) + -0.04011739 * np.exp(-2.00056769 * t)
    if N == 5:
        return 0.20965279 * np.exp(-0.00000000 * t) + -1.09156734 * np.exp(-0.94201914 * t) + 1.53133057 * np.exp(-1.31876255 * t) + -0.59064543 * np.exp(-2.00136757 * t) + 0.12352341 * np.exp(-3.00001013 * t) + -0.01562732 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.27955544 * np.exp(0.00000000 * t) + -2.11818299 * np.exp(-0.96607465 * t) + 4.04649863 * np.exp(-1.42723785 * t) + -2.82777519 * np.exp(-2.00076784 * t) + 1.04898706 * np.exp(-3.00000357 * t) + -0.38676060 * np.exp(-4.00000001 * t) + 0.09308419 * np.exp(-5.00000000 * t) + -0.01040654 * np.exp(-6.00000000 * t)
def ph_1_star(t):
    if N == 3:
        return 0.80090435 * np.exp(0.00000000 * t) + -0.06154373 * np.exp(-0.93459463 * t) + -0.75329458 * np.exp(-1.18340632 * t) + 0.26393397 * np.exp(-2.00056769 * t)
    if N == 5:
        return 0.73662935 * np.exp(-0.00000000 * t) + -0.15810643 * np.exp(-0.94201914 * t) + -1.84129730 * np.exp(-1.31876255 * t) + 2.15199222 * np.exp(-2.00136757 * t) + -0.89118095 * np.exp(-3.00001013 * t) + 0.16862977 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.69392837 * np.exp(0.00000000 * t) + -0.14640144 * np.exp(-0.96607465 * t) + -4.38160785 * np.exp(-1.42723785 * t) + 7.11305552 * np.exp(-2.00076784 * t) + -5.25689097 * np.exp(-3.00000357 * t) + 2.90428984 * np.exp(-4.00000001 * t) + -0.93150761 * np.exp(-5.00000000 * t) + 0.13013415 * np.exp(-6.00000000 * t)
def ph_2_star(t):
    if N == 3:
        return 0.06591317 * np.exp(0.00000000 * t) + 0.42060255 * np.exp(-0.93459463 * t) + 0.20794811 * np.exp(-1.18340632 * t) + -0.44446384 * np.exp(-2.00056769 * t)
    if N == 5:
        return 0.05230839 * np.exp(-0.00000000 * t) + 1.18688808 * np.exp(-0.94201914 * t) + 0.28527793 * np.exp(-1.31876255 * t) + -3.02748874 * np.exp(-2.00136757 * t) + 2.28058114 * np.exp(-3.00001013 * t) + -0.61090012 * np.exp(-4.00000004 * t) 
    if N == 7:
        return 0.02610361 * np.exp(0.00000000 * t) + 2.19596659 * np.exp(-0.96607465 * t) + 0.31645551 * np.exp(-1.42723785 * t) + -8.36003208 * np.exp(-2.00076784 * t) + 12.50512224 * np.exp(-3.00000357 * t) + -10.02400072 * np.exp(-4.00000001 * t) + 4.18335902 * np.exp(-5.00000000 * t) + -0.71797418 * np.exp(-6.00000000 * t) 
def ph_3_star(t):
    if N == 3:
        return 0.00135614 * np.exp(0.00000000 * t) + 0.01685331 * np.exp(-0.93459463 * t) + 0.01114329 * np.exp(-1.18340632 * t) + 0.22064726 * np.exp(-2.00056769 * t)
    if N == 5:
        return 0.00139292 * np.exp(-0.00000000 * t) + 0.06170321 * np.exp(-0.94201914 * t) + 0.02416498 * np.exp(-1.31876255 * t) + 1.41257137 * np.exp(-2.00136757 * t) + -2.24895130 * np.exp(-3.00001013 * t) + 0.91578549 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.00040914 * np.exp(0.00000000 * t) + 0.06777195 * np.exp(-0.96607465 * t) + 0.01835516 * np.exp(-1.42723785 * t) + 3.97083908 * np.exp(-2.00076784 * t) + -12.32786883 * np.exp(-3.00000357 * t) + 14.94995271 * np.exp(-4.00000001 * t) + -8.34919771 * np.exp(-5.00000000 * t) + 1.79473848 * np.exp(-6.00000000 * t)
def ph_4_star(t):
    if N == 3:
        return 0
    if N == 5:
        return 0.00001649 * np.exp(-0.00000000 * t) + 0.00107620 * np.exp(-0.94201914 * t) + 0.00052035 * np.exp(-1.31876255 * t) + 0.05309048 * np.exp(-2.00136757 * t) + 0.72248014 * np.exp(-3.00001013 * t) + -0.61051699 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.00000342 * np.exp(0.00000000 * t) + 0.00084063 * np.exp(-0.96607465 * t) + 0.00029636 * np.exp(-1.42723785 * t) + 0.10292991 * np.exp(-2.00076784 * t) + 3.95287239 * np.exp(-3.00000357 * t) + -9.86189231 * np.exp(-4.00000001 * t) + 8.32293341 * np.exp(-5.00000000 * t) + -2.39298382 * np.exp(-6.00000000 * t)
def ph_5_star(t):
    if N == 3:
        return 0
    if N == 5:
        return 0.00000628 * np.exp(-0.94201914 * t) + 0.00000348 * np.exp(-1.31876255 * t) + 0.00048010 * np.exp(-2.00136757 * t) + 0.01354757 * np.exp(-3.00001013 * t) + 0.15262916 * np.exp(-4.00000004 * t) 
    if N == 7:
        return 0.00000523 * np.exp(-0.96607465 * t) + 0.00000218 * np.exp(-1.42723785 * t) + 0.00097864 * np.exp(-2.00076784 * t) + 0.07728636 * np.exp(-3.00000357 * t) + 2.38720133 * np.exp(-4.00000001 * t) + -4.13521162 * np.exp(-5.00000000 * t) + 1.79473786 * np.exp(-6.00000000 * t)
def ph_6_star(t):
    if N == 3:
        return 0
    if N == 5:
        return 0
    if N == 7:
        return 0.00000411 * np.exp(-2.00076784 * t) + 0.00049072 * np.exp(-3.00000357 * t) + 0.03111098 * np.exp(-4.00000001 * t) + 0.81128930 * np.exp(-5.00000000 * t) + -0.71789515 * np.exp(-6.00000000 * t)
def ph_7_star(t):
    if N == 3:
        return 0
    if N == 5:
        return 0
    if N == 7:
        return 0.00000103 * np.exp(-3.00000357 * t) + 0.00009876 * np.exp(-4.00000001 * t) + 0.00525101 * np.exp(-5.00000000 * t) + 0.11964919 * np.exp(-6.00000000 * t) 

def p_0_star(t):
    if N == 3:
        return 0.13209536 * np.exp(0.00000000 * t) + -0.37585246 * np.exp(-0.93443852 * t) + 0.53397573 * np.exp(-1.18384880 * t) + -0.04021863 * np.exp(-2.00057211 * t)
    if N == 5:
        return 0.21014959 * np.exp(-0.00000000 * t) + -1.09128672 * np.exp(-0.94184272 * t) + 1.53263523 * np.exp(-1.31975506 * t) + -0.59309836 * np.exp(-2.00138225 * t) + 0.12394438 * np.exp(-3.00001026 * t) + -0.01567745 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.28034145 * np.exp(-0.00000000 * t) + -2.11766652 * np.exp(-0.96594143 * t) + 4.06042360 * np.exp(-1.42894923 * t) + -2.84675988 * np.exp(-2.00077932 * t) + 1.05412210 * np.exp(-3.00000363 * t) + -0.38849559 * np.exp(-4.00000001 * t) + 0.09348518 * np.exp(-5.00000000 * t) + -0.01045033 * np.exp(-6.00000000 * t)
def p_1_star(t):
    if N == 3:
        return 0.80049458 * np.exp(0.00000000 * t) + -0.06152377 * np.exp(-0.93443852 * t) + -0.75294919 * np.exp(-1.18384880 * t) + 0.26397839 * np.exp(-2.00057211 * t)
    if N == 5:
        return 0.73600056 * np.exp(-0.00000000 * t) + -0.15803721 * np.exp(-0.94184272 * t) + -1.84276885 * np.exp(-1.31975506 * t) + 2.15425023 * np.exp(-2.00138225 * t) + -0.89141742 * np.exp(-3.00001026 * t) + 0.16863937 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.69306602 * np.exp(-0.00000000 * t) + -0.14634660 * np.exp(-0.96594143 * t) + -4.39656411 * np.exp(-1.42894923 * t) + 7.13230004 * np.exp(-2.00077932 * t) + -5.26146695 * np.exp(-3.00000363 * t) + 2.90562182 * np.exp(-4.00000001 * t) + -0.93176760 * np.exp(-5.00000000 * t) + 0.13015739 * np.exp(-6.00000000 * t)
def p_2_star(t):
    if N == 3:
        return 0.06604768 * np.exp(0.00000000 * t) + 0.42048565 * np.exp(-0.93443852 * t) + 0.20780111 * np.exp(-1.18384880 * t) + -0.44433443 * np.exp(-2.00057211 * t)
    if N == 5:
        return 0.05243243 * np.exp(-0.00000000 * t) + 1.18636446 * np.exp(-0.94184272 * t) + 0.28531180 * np.exp(-1.31975506 * t) + -3.02663998 * np.exp(-2.00138225 * t) + 2.27992877 * np.exp(-3.00001026 * t) + -0.61073081 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.02617709 * np.exp(-0.00000000 * t) + 2.19514311 * np.exp(-0.96594143 * t) + 0.31729323 * np.exp(-1.42894923 * t) + -8.35900921 * np.exp(-2.00077932 * t) + 12.50278007 * np.exp(-3.00000363 * t) + -10.02214085 * np.exp(-4.00000001 * t) + 4.18260421 * np.exp(-5.00000000 * t) + -0.71784765 * np.exp(-6.00000000 * t)
def p_3_star(t):
    if N == 3:
        return 0.00136238 * np.exp(0.00000000 * t) + 0.01689059 * np.exp(-0.93443852 * t) + 0.01117236 * np.exp(-1.18384880 * t) + 0.22057467 * np.exp(-2.00057211 * t)
    if N == 5:
        return 0.00140073 * np.exp(-0.00000000 * t) + 0.06187058 * np.exp(-0.94184272 * t) + 0.02429315 * np.exp(-1.31975506 * t) + 1.41176291 * np.exp(-2.00138225 * t) + -2.24818854 * np.exp(-3.00001026 * t) + 0.91552785 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.00041196 * np.exp(-0.00000000 * t) + 0.06801763 * np.exp(-0.96594143 * t) + 0.01854408 * np.exp(-1.42894923 * t) + 3.96915893 * np.exp(-2.00077932 * t) + -12.32473516 * np.exp(-3.00000363 * t) + 14.94679402 * np.exp(-4.00000001 * t) + -8.34761199 * np.exp(-5.00000000 * t) + 1.79442052 * np.exp(-6.00000000 * t)
def p_4_star(t):
    if N == 3:
        return 0
    if N == 5:
        return 0.00001663 * np.exp(-0.00000000 * t) + 0.00108255 * np.exp(-0.94184272 * t) + 0.00052516 * np.exp(-1.31975506 * t) + 0.05324214 * np.exp(-2.00138225 * t) + 0.72214535 * np.exp(-3.00001026 * t) + -0.61034517 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.00000346 * np.exp(-0.00000000 * t) + 0.00084707 * np.exp(-0.96594143 * t) + 0.00030098 * np.exp(-1.42894923 * t) + 0.10331956 * np.exp(-2.00077932 * t) + 3.95122371 * np.exp(-3.00000363 * t) + -9.85938050 * np.exp(-4.00000001 * t) + 8.32124558 * np.exp(-5.00000000 * t) + -2.39255987 * np.exp(-6.00000000 * t)
def p_5_star(t):
    if N == 3:
        return 0
    if N == 5:
        return 0.00000634 * np.exp(-0.94184272 * t) + 0.00000352 * np.exp(-1.31975506 * t) + 0.00048306 * np.exp(-2.00138225 * t) + 0.01358746 * np.exp(-3.00001026 * t) + 0.15258621 * np.exp(-4.00000004 * t)
    if N == 7:
        return 0.00000529 * np.exp(-0.96594143 * t) + 0.00000222 * np.exp(-1.42894923 * t) + 0.00098639 * np.exp(-2.00077932 * t) + 0.07758057 * np.exp(-3.00000363 * t) + 2.38627132 * np.exp(-4.00000001 * t) + -4.13426570 * np.exp(-5.00000000 * t) + 1.79441990 * np.exp(-6.00000000 * t)
def p_6_star(t):
    if N == 3:
        return 0
    if N == 5:
        return 0
    if N == 7:
        return 0.00000416 * np.exp(-2.00077932 * t) + 0.00049462 * np.exp(-3.00000363 * t) + 0.03123023 * np.exp(-4.00000001 * t) + 0.81103892 * np.exp(-5.00000000 * t) + -0.71776796 * np.exp(-6.00000000 * t)
def p_7_star(t):
    if N == 3:
        return 0
    if N == 5:
        return 0
    if N == 7:
        return 0.00000104 * np.exp(-3.00000363 * t) + 0.00009955 * np.exp(-4.00000001 * t) + 0.00527141 * np.exp(-5.00000000 * t) + 0.11962799 * np.exp(-6.00000000 * t)


def p_0_all(t):
    if N == 3:
        return -0.01539468 * np.exp(-3.00000000 * t) + -0.01980289 * np.exp(-1.26332931 * t) + 0.24474732 * np.exp(-0.37474906 * t) + 0.04045025 * np.exp(-0.00000000 * t)
    if N == 5:
        return 0.00071921 * np.exp(-5.00000000 * t) + 0.00168546 * np.exp(-2.99069138 * t) + -0.02474484 * np.exp(-1.68359036 * t) + -0.04245108 * np.exp(-0.84586570 * t) + 0.19791853 * np.exp(-0.21591563 * t) + 0.03353940 * np.exp(-0.00000000 * t)
    if N == 7:
        return -0.00002564 * np.exp(-7.00000000 * t) + -0.00001019 * np.exp(-4.87108387 * t) + 0.00276782 * np.exp(-3.30767533 * t) + 0.00626416 * np.exp(-2.14675147 * t) + -0.02454619 * np.exp(-1.28038338 * t) + -0.04938601 * np.exp(-0.68551323 * t) + 0.02961566 * np.exp(0.00000000 * t) + 0.16032038 * np.exp(-0.14752325 * t)
def p_1_all(t):
    if N == 3:
        return 0.04618403 * np.exp(-3.00000000 * t) + 0.01226417 * np.exp(-1.26332931 * t) + 0.14655077 * np.exp(-0.37474906 * t) + 0.04500102 * np.exp(-0.00000000 * t)
    if N == 5:
        return -0.00359605 * np.exp(-5.00000000 * t) + -0.00448723 * np.exp(-2.99069138 * t) + 0.02824885 * np.exp(-1.68359036 * t) + 0.00708829 * np.exp(-0.84586570 * t) + 0.11200689 * np.exp(-0.21591563 * t) + 0.02740591 * np.exp(-0.00000000 * t) 
    if N == 7:
        return 0.00017947 * np.exp(-7.00000000 * t) + 0.00004705 * np.exp(-4.87108387 * t) + -0.00794663 * np.exp(-3.30767533 * t) + -0.00985279 * np.exp(-2.14675147 * t) + 0.01482761 * np.exp(-1.28038338 * t) + -0.00301949 * np.exp(-0.68551323 * t) + 0.02451324 * np.exp(0.00000000 * t) + 0.10625154 * np.exp(-0.14752325 * t)
def p_2_all(t):
    if N == 3:
        return -0.04618403 * np.exp(-3.00000000 * t) + 0.05067494 * np.exp(-1.26332931 * t) + 0.12415834 * np.exp(-0.37474906 * t) + 0.12135075 * np.exp(-0.00000000 * t)
    if N == 5:
        return 0.00719209 * np.exp(-5.00000000 * t) + 0.00157360 * np.exp(-2.99069138 * t) + 0.02632667 * np.exp(-1.68359036 * t) + 0.03088379 * np.exp(-0.84586570 * t) + 0.07072323 * np.exp(-0.21591563 * t) + 0.02996727 * np.exp(-0.00000000 * t)
    if N == 7:
        return -0.00053840 * np.exp(-7.00000000 * t) + -0.00007552 * np.exp(-4.87108387 * t) + 0.00432214 * np.exp(-3.30767533 * t) + -0.00328826 * np.exp(-2.14675147 * t) + 0.02030543 * np.exp(-1.28038338 * t) + 0.02043546 * np.exp(-0.68551323 * t) + 0.02042635 * np.exp(0.00000000 * t) + 0.06341279 * np.exp(-0.14752325 * t)
def p_3_all(t):
    if N == 3:
        return 0.01539468 * np.exp(-3.00000000 * t) + -0.04313622 * np.exp(-1.26332931 * t) + -0.51545643 * np.exp(-0.37474906 * t) + 0.79319798 * np.exp(-0.00000000 * t)
    if N == 5:
        return -0.00719209 * np.exp(-5.00000000 * t) + 0.00634199 * np.exp(-2.99069138 * t) + 0.00079207 * np.exp(-1.68359036 * t) + 0.06006094 * np.exp(-0.84586570 * t) + 0.05185194 * np.exp(-0.21591563 * t) + 0.05481182 * np.exp(-0.00000000 * t) 
    if N == 7:
        return 0.00089733 * np.exp(-7.00000000 * t) + 0.00002159 * np.exp(-4.87108387 * t) + 0.00626086 * np.exp(-3.30767533 * t) + 0.00455597 * np.exp(-2.14675147 * t) + 0.01461442 * np.exp(-1.28038338 * t) + 0.03429406 * np.exp(-0.68551323 * t) + 0.02221242 * np.exp(0.00000000 * t) + 0.04214335 * np.exp(-0.14752325 * t)
def p_4_all(t):
    if N == 3:
        return 0
    if N == 5:
        return 0.00359605 * np.exp(-5.00000000 * t) + -0.00781591 * np.exp(-2.99069138 * t) + -0.08933385 * np.exp(-1.68359036 * t) + 0.10008189 * np.exp(-0.84586570 * t) + -0.00755849 * np.exp(-0.21591563 * t) + 0.16769698 * np.exp(-0.00000000 * t)
    if N == 7:
        return -0.00089733 * np.exp(-7.00000000 * t) + 0.00008572 * np.exp(-4.87108387 * t) + -0.00172226 * np.exp(-3.30767533 * t) + 0.01013740 * np.exp(-2.14675147 * t) + 0.00196837 * np.exp(-1.28038338 * t) + 0.05155216 * np.exp(-0.68551323 * t) + 0.03404392 * np.exp(0.00000000 * t) + 0.02983202 * np.exp(-0.14752325 * t)
def p_5_all(t):
    if N == 3:
        return 0
    if N == 5:
        return -0.00071921 * np.exp(-5.00000000 * t) + 0.00270210 * np.exp(-2.99069138 * t) + 0.05871110 * np.exp(-1.68359036 * t) + -0.15566383 * np.exp(-0.84586570 * t) + -0.42494211 * np.exp(-0.21591563 * t) + 0.68657862 * np.exp(-0.00000000 * t)
    if N == 7:
        return 0.00053840 * np.exp(-7.00000000 * t) + -0.00012252 * np.exp(-4.87108387 * t) + -0.01492525 * np.exp(-3.30767533 * t) + 0.00645979 * np.exp(-2.14675147 * t) + -0.02775424 * np.exp(-1.28038338 * t) + 0.07738008 * np.exp(-0.68551323 * t) + 0.07353973 * np.exp(0.00000000 * t) + 0.00988402 * np.exp(-0.14752325 * t)
def p_6_all(t):
    if N == 3 or N == 5:
        return 0
    if N == 7:
        return -0.00017947 * np.exp(-7.00000000 * t) + 0.00006818 * np.exp(-4.87108387 * t) + 0.01654225 * np.exp(-3.30767533 * t) + -0.03009575 * np.exp(-2.14675147 * t) + -0.07705985 * np.exp(-1.28038338 * t) + 0.07631477 * np.exp(-0.68551323 * t) + 0.20730964 * np.exp(0.00000000 * t) + -0.06789977 * np.exp(-0.14752325 * t)
def p_7_all(t):
    if N == 3 or N == 5:
        return 0
    if N == 7:
        return 0.00002564 * np.exp(-7.00000000 * t) + -0.00001431 * np.exp(-4.87108387 * t) + -0.00529893 * np.exp(-3.30767533 * t) + 0.01581947 * np.exp(-2.14675147 * t) + 0.07764445 * np.exp(-1.28038338 * t) + -0.20757102 * np.exp(-0.68551323 * t) + 0.58833904 * np.exp(0.00000000 * t) + -0.34394433 * np.exp(-0.14752325 * t)

def ph_0_all(t):
    if N == 3:
        return -0.01542826 * np.exp(-3.00000000 * t) + -0.01986709 * np.exp(-1.26298550 * t) + 0.24491999 * np.exp(-0.37456012 * t) + 0.04037535
    if N == 5:
        return 0.00072120 * np.exp(-5.00000000 * t) + 0.00168740 * np.exp(-2.99008230 * t) + -0.02481474 * np.exp(-1.68300455 * t) + -0.04256707 * np.exp(-0.84561045 * t) + 0.19815016 * np.exp(-0.21566726 * t) + 0.03348972 * np.exp(0.00000000 * t)
    if N == 7:
        return -0.00002577 * np.exp(-7.00000000 * t) + -0.00001057 * np.exp(-4.87033806 * t) + 0.00277607 * np.exp(-3.30675736 * t) + 0.00627787 * np.exp(-2.14594664 * t) + -0.02463642 * np.exp(-1.27979872 * t) + -0.04954979 * np.exp(-0.68534694 * t) + 0.02958536 * np.exp(0.00000000 * t) + 0.16058326 * np.exp(-0.14726586 * t)
def ph_1_all(t):
    if N == 3:
        return 0.04628477 * np.exp(-3.00000000 * t) + 0.01230472 * np.exp(-1.26298550 * t) + 0.14652844 * np.exp(-0.37456012 * t) + 0.04488206
    if N == 5:
        return -0.00360600 * np.exp(-5.00000000 * t) + -0.00449200 * np.exp(-2.99008230 * t) + 0.02833121 * np.exp(-1.68300455 * t) + 0.00713687 * np.exp(-0.84561045 * t) + 0.11197089 * np.exp(-0.21566726 * t) + 0.02732569 * np.exp(0.00000000 * t)
    if N == 7:
        return 0.00018038 * np.exp(-7.00000000 * t) + 0.00004882 * np.exp(-4.87033806 * t) + -0.00796952 * np.exp(-3.30675736 * t) + -0.00987484 * np.exp(-2.14594664 * t) + 0.01489444 * np.exp(-1.27979872 * t) + -0.00297558 * np.exp(-0.68534694 * t) + 0.02444637 * np.exp(0.00000000 * t) + 0.10624992 * np.exp(-0.14726586 * t)
def ph_2_all(t):
    if N == 3:
        return -0.04628477 * np.exp(-3.00000000 * t) + 0.05087127 * np.exp(-1.26298550 * t) + 0.12428744 * np.exp(-0.37456012 * t) + 0.12112606
    if N == 5:
        return 0.00721199 * np.exp(-5.00000000 * t) + 0.00157487 * np.exp(-2.99008230 * t) + 0.02638792 * np.exp(-1.68300455 * t) + 0.03096629 * np.exp(-0.84561045 * t) + 0.07066047 * np.exp(-0.21566726 * t) + 0.02986513 * np.exp(0.00000000 * t)
    if N == 7:
        return -0.00054115 * np.exp(-7.00000000 * t) + -0.00007836 * np.exp(-4.87033806 * t) + 0.00433533 * np.exp(-3.30675736 * t) + -0.00328970 * np.exp(-2.14594664 * t) + 0.02036213 * np.exp(-1.27979872 * t) + 0.02051639 * np.exp(-0.68534694 * t) + 0.02034976 * np.exp(0.00000000 * t) + 0.06334561 * np.exp(-0.14726586 * t)
def ph_3_all(t):
    if N == 3:
        return 0.01542826 * np.exp(-3.00000000 * t) + -0.04330890 * np.exp(-1.26298550 * t) + -0.51573587 * np.exp(-0.37456012 * t) + 0.79361652
    if N == 5:
        return -0.00721199 * np.exp(-5.00000000 * t) + 0.00635047 * np.exp(-2.99008230 * t) + 0.00081208 * np.exp(-1.68300455 * t) + 0.06022720 * np.exp(-0.84561045 * t) + 0.05183753 * np.exp(-0.21566726 * t) + 0.05465138 * np.exp(0.00000000 * t)
    if N == 7:
        return 0.00090192 * np.exp(-7.00000000 * t) + 0.00002240 * np.exp(-4.87033806 * t) + 0.00627561 * np.exp(-3.30675736 * t) + 0.00456305 * np.exp(-2.14594664 * t) + 0.01464945 * np.exp(-1.27979872 * t) + 0.03438458 * np.exp(-0.68534694 * t) + 0.02212157 * np.exp(0.00000000 * t) + 0.04208142 * np.exp(-0.14726586 * t)
def ph_4_all(t):
    if N == 3:
        return 0
    if N == 5:
        return 0.00360600 * np.exp(-5.00000000 * t) + -0.00782707 * np.exp(-2.99008230 * t) + -0.08963250 * np.exp(-1.68300455 * t) + 0.10050551 * np.exp(-0.84561045 * t) + -0.00743386 * np.exp(-0.21566726 * t) + 0.16744860 * np.exp(0.00000000 * t)
    if N == 7:
        return -0.00090192 * np.exp(-7.00000000 * t) + 0.00008896 * np.exp(-4.87033806 * t) + -0.00172340 * np.exp(-3.30675736 * t) + 0.01015380 * np.exp(-2.14594664 * t) + 0.00197945 * np.exp(-1.27979872 * t) + 0.05168656 * np.exp(-0.68534694 * t) + 0.03391627 * np.exp(0.00000000 * t) + 0.02980028 * np.exp(-0.14726586 * t)
def ph_5_all(t):
    if N == 3:
        return 0
    if N == 5:
        return -0.00072120 * np.exp(-5.00000000 * t) + 0.00270633 * np.exp(-2.99008230 * t) + 0.05891603 * np.exp(-1.68300455 * t) + -0.15626879 * np.exp(-0.84561045 * t) + -0.42518519 * np.exp(-0.21566726 * t) + 0.68721948 * np.exp(0.00000000 * t)
    if N == 7:
        return 0.00054115 * np.exp(-7.00000000 * t) + -0.00012716 * np.exp(-4.87033806 * t) + -0.01497440 * np.exp(-3.30675736 * t) + 0.00648406 * np.exp(-2.14594664 * t) + -0.02783057 * np.exp(-1.27979872 * t) + 0.07764848 * np.exp(-0.68534694 * t) + 0.07333911 * np.exp(0.00000000 * t) + 0.00991933 * np.exp(-0.14726586 * t)
def ph_6_all(t):
    if N == 3 or N == 5:
        return 0
    if N == 7:
        return -0.00018038 * np.exp(-7.00000000 * t) + 0.00007077 * np.exp(-4.87033806 * t) + 0.01659799 * np.exp(-3.30675736 * t) + -0.03018037 * np.exp(-2.14594664 * t) + -0.07741414 * np.exp(-1.27979872 * t) + 0.07677167 * np.exp(-0.68534694 * t) + 0.20709750 * np.exp(0.00000000 * t) + -0.06776303 * np.exp(-0.14726586 * t)
def ph_7_all(t):
    if N == 3 or N == 5:
        return 0
    if N == 7:
        return 0.00002577 * np.exp(-7.00000000 * t) + -0.00001486 * np.exp(-4.87033806 * t) + -0.00531767 * np.exp(-3.30675736 * t) + 0.01586613 * np.exp(-2.14594664 * t) + 0.07799564 * np.exp(-1.27979872 * t) + -0.20848230 * np.exp(-0.68534694 * t) + 0.58914407 * np.exp(0.00000000 * t) + -0.34421679 * np.exp(-0.14726586 * t)
   
def derivada_num(fh, f, t):
    return (fh(t) - f(t))/h

def Fisher_Gibbs(system, t):
    if N == 3:
        Fisher_Gibbs_Star = 1.6564888757700964
        Fisher_Gibbs_All = 1.7031830558281775
        
    if N == 5:
        Fisher_Gibbs_Star = 3.202834457001724
        Fisher_Gibbs_All = 3.2691018269425913
        
    if N == 7:
        Fisher_Gibbs_Star = 5.473113087581061
        Fisher_Gibbs_All = 5.026083190883782
    
    if system == 1:
        return Fisher_Gibbs_Star
    
    if system == 2:
        return Fisher_Gibbs_All

E_gibbs_star_values = [-2.547633461679402, -3.61675138119298, -4.972253538113005, -6.636128300625903, -8.577456818079542]
E_gibbs_all_values = [-2.3159161694894985, -2.860143554475746, -3.36097471959862, -3.8271515541718477, -4.266769516294245]

def E_0_star(N,t):
    if N == 3:
        E_0 = 0
        return E_0*(0.1321 -0.3759 * np.exp(-0.9344 * t) + 0.5340 * np.exp(-1.1838 * t) - 0.0402 * np.exp(-2.0006 * t))
    if N == 5:
        E_0 = -2.015
        return E_0*(0.2101 -1.0913 * np.exp(-0.9418 * t) + 1.5326 * np.exp(-1.3198 * t) + -0.5931 * np.exp(-2.0014 * t) + 0.1239 * np.exp(-3.0000 * t) -0.0157 * np.exp(-4.0000 * t))
    if N == 7:
        E_0 = -5.07
        return E_0*(0.2803 - 2.1177 * np.exp(-0.9659 * t) + 4.0604 * np.exp(-1.4289 * t) - 2.8468 * np.exp(-2.0008 * t) + 1.0541 * np.exp(-3.0000 * t) - 0.3885 * np.exp(-4.0000 * t) + 0.0935 * np.exp(-5.0000 * t) - 0.0105 * np.exp(-6.0000 * t))
def E_1_star(N,t):
    if N == 3:
        E_1 = -3.188
        return E_1*(0.8005 - 0.0615 * np.exp(-0.9344 * t) - 0.7529 * np.exp(-1.1838 * t) + 0.2640 * np.exp(-2.0006 * t))
    if N == 5:
        E_1 = -6.040999999999999
        return E_1*(0.7360 * np.exp(-0.0000 * t) + -0.1580 * np.exp(-0.9418 * t) + -1.8428 * np.exp(-1.3198 * t) + 2.1543 * np.exp(-2.0014 * t) -0.8914 * np.exp(-3.0000 * t) + 0.1686 * np.exp(-4.0000 * t))
    if N == 7:
        E_1 = -10.133999999
        return E_1*(0.6931 * np.exp(-0.0000 * t) + -0.1463 * np.exp(-0.9659 * t) + -4.3966 * np.exp(-1.4289 * t) + 7.1323 * np.exp(-2.0008 * t) + -5.2615 * np.exp(-3.0000 * t) + 2.9056 * np.exp(-4.0000 * t) + -0.9318 * np.exp(-5.0000 * t) + 0.1302 * np.exp(-6.0000 * t))
def E_2_star(N,t):
    if N == 3:
        E_2 = 0
        return E_2*(0.0660 * np.exp(0.0000 * t) + 0.4205 * np.exp(-0.9344 * t) + 0.2078 * np.exp(-1.1838 * t) -0.4443 * np.exp(-2.0006 * t))
    if N == 5:
        E_2 = -2.0129999999999995
        return E_2*(0.0524 * np.exp(-0.0000 * t) + 1.1864 * np.exp(-0.9418 * t) + 0.2853 * np.exp(-1.3198 * t) + -3.0266 * np.exp(-2.0014 * t) + 2.2799 * np.exp(-3.0000 * t) + -0.6107 * np.exp(-4.0000 * t))
    if N == 7:
        E_2 = -5.065999999999999
        return E_2*(0.0262 + 2.1951 * np.exp(-0.9659 * t) + 0.3173 * np.exp(-1.4289 * t) - 8.3590 * np.exp(-2.0008 * t) + 12.5028 * np.exp(-3.0000 * t) - 10.0221 * np.exp(-4.0000 * t) + 4.1826 * np.exp(-5.0000 * t) + -0.7178 * np.exp(-6.0000 * t))
def E_3_star(N,t):
    if N == 3:
        E_3 = 3.188
        return E_3*(0.0014 + 0.0169 * np.exp(-0.9344 * t) + 0.0112 * np.exp(-1.1838 * t) + 0.2206 * np.exp(-2.0006 * t))
    if N == 5:
        E_3 = 2.015
        return E_3*(0.0014 + 0.0619 * np.exp(-0.9418 * t) + 0.0243 * np.exp(-1.3198 * t) + 1.4118 * np.exp(-2.0014 * t) + -2.2482 * np.exp(-3.0000 * t) + 0.9155 * np.exp(-4.0000 * t))
    if N == 7:
        E_3 = 0.00200000000000068
        return E_3*(0.0004 + 0.0680 * np.exp(-0.9659 * t) + 0.0185 * np.exp(-1.4289 * t) + 3.9692 * np.exp(-2.0008 * t) + -12.3247 * np.exp(-3.0000 * t) + 14.9468 * np.exp(-4.0000 * t) + -8.3476 * np.exp(-5.0000 * t) + 1.7944 * np.exp(-6.0000 * t))
def E_4_star(N,t):
    if N == 3:
        return 0
    if N == 5:
        E_4 = 3.188
        return E_4*(0.0011 * np.exp(-0.9418 * t) + 0.0005 * np.exp(-1.3198 * t) + 0.0532 * np.exp(-2.0014 * t) + 0.7221 * np.exp(-3.0000 * t) + -0.6103 * np.exp(-4.0000 * t))
    if N == 7:
        E_4 = 5.07
        return E_4*(0.0008 * np.exp(-0.9659 * t) + 0.0003 * np.exp(-1.4289 * t) + 0.1033 * np.exp(-2.0008 * t) + 3.9512 * np.exp(-3.0000 * t) + -9.8594 * np.exp(-4.0000 * t) + 8.3212 * np.exp(-5.0000 * t) + -2.3926 * np.exp(-6.0000 * t))
def E_5_star(N,t):
    if N == 3:
        return 0
    if N == 5:
        E_5 = 10.071
        return E_5*(0.0005 * np.exp(-2.0014 * t) + 0.0136 * np.exp(-3.0000 * t) + 0.1526 * np.exp(-4.0000 * t))
    if N == 7:
        E_5 = 10.138
        return E_5*(0.0010 * np.exp(-2.0008 * t) + 0.0776 * np.exp(-3.0000 * t) + 2.3863 * np.exp(-4.0000 * t) + -4.1343 * np.exp(-5.0000 * t) + 1.7944 * np.exp(-6.0000 * t))
def E_6_star(N,t):
    if N == 3:
        return 0
    if N == 5:
        return 0
    if N == 7:
        E_6 = 15.206
        return E_6*(0.0005 * np.exp(-3.0000 * t) + 0.0312 * np.exp(-4.0000 * t) + 0.8110 * np.exp(-5.0000 * t) + -0.7178 * np.exp(-6.0000 * t))
def E_7_star(N,t):
    if N == 3:
        return 0
    if N == 5:
        return 0
    if N == 7:
        E_7 = 20.274
        return E_7*(0.0001 * np.exp(-4.0000 * t) + 0.0053 * np.exp(-5.0000 * t) + 0.1196 * np.exp(-6.0000 * t))

def E_star(N,t):
    return (E_0_star(N,t) + E_1_star(N,t) + E_2_star(N,t) + E_3_star(N,t) + E_4_star(N,t) + 
            E_5_star(N,t) + E_6_star(N,t) + E_7_star(N,t))

def E_0_all(N,t):
    if N == 3:
        E_0 = -0.0
        return E_0*(-0.0154 * np.exp(-3.0000 * t) + -0.0198 * np.exp(-1.2633 * t) + 0.2447 * np.exp(-0.3747 * t) + 0.0405 * np.exp(-0.0000 * t))
    if N == 5:
        E_0 = -1.5095
        return E_0*(0.0007 * np.exp(-5.0000 * t) + 0.0017 * np.exp(-2.9907 * t) + -0.0247 * np.exp(-1.6836 * t) + -0.0425 * np.exp(-0.8459 * t) + 0.1979 * np.exp(-0.2159 * t) + 0.0335 * np.exp(-0.0000 * t))
    if N == 7:
        E_0 = -2.989
        return E_0*(0.0028 * np.exp(-3.3077 * t) + 0.0063 * np.exp(-2.1468 * t) + -0.0245 * np.exp(-1.2804 * t) + -0.0494 * np.exp(-0.6855 * t) + 0.0296 * np.exp(-0.0000 * t) + 0.1603 * np.exp(-0.1475 * t))
def E_1_all(N,t):
    if N == 3:
        E_1 = 0.992
        return E_1*(0.0462 * np.exp(-3.0000 * t) + 0.0123 * np.exp(-1.2633 * t) + 0.1466 * np.exp(-0.3747 * t) + 0.0450 * np.exp(-0.0000 * t))
    if N == 5:
        E_1 = -0.3019
        return E_1*(-0.0036 * np.exp(-5.0000 * t) + -0.0045 * np.exp(-2.9907 * t) + 0.0282 * np.exp(-1.6836 * t) + 0.0071 * np.exp(-0.8459 * t) + 0.1120 * np.exp(-0.2159 * t) + 0.0274 * np.exp(-0.0000 * t))
    if N == 7:
        E_1 = -0.854
        return E_1*(0.0002 * np.exp(-7.0000 * t) + 0.0000 * np.exp(-4.8711 * t) + -0.0079 * np.exp(-3.3077 * t) + -0.0099 * np.exp(-2.1468 * t) + 0.0148 * np.exp(-1.2804 * t) + -0.0030 * np.exp(-0.6855 * t) + 0.0245 * np.exp(-0.0000 * t) + 0.1063 * np.exp(-0.1475 * t))
def E_2_all(N,t):
    if N == 3:
        E_2 = 0
        return E_2*(-0.0462 * np.exp(-3.0000 * t) + 0.0507 * np.exp(-1.2633 * t) + 0.1242 * np.exp(-0.3747 * t) + 0.1214 * np.exp(-0.0000 * t))
    if N == 5:
        E_2 = 0.9057
        return E_2*(0.0072 * np.exp(-5.0000 * t) + 0.0016 * np.exp(-2.9907 * t) + 0.0263 * np.exp(-1.6836 * t) + 0.0309 * np.exp(-0.8459 * t) + 0.0707 * np.exp(-0.2159 * t) + 0.0300 * np.exp(-0.0000 * t))
    if N == 7:
        E_2 = 0.427
        return E_2*(-0.0005 * np.exp(-7.0000 * t) + -0.0001 * np.exp(-4.8711 * t) + 0.0043 * np.exp(-3.3077 * t) + -0.0033 * np.exp(-2.1468 * t) + 0.0203 * np.exp(-1.2804 * t) + 0.0204 * np.exp(-0.6855 * t) + 0.0204 * np.exp(-0.0000 * t) + 0.0634 * np.exp(-0.1475 * t))
def E_3_all(N,t):
    if N == 3:
        E_3 = -2.976
        return E_3*(0.0154 * np.exp(-3.0000 * t) + -0.0431 * np.exp(-1.2633 * t) + -0.5155 * np.exp(-0.3747 * t) + 0.7932 * np.exp(-0.0000 * t))
    if N == 5:
        E_3 = 0.3019
        return E_3*(-0.0072 * np.exp(-5.0000 * t) + 0.0063 * np.exp(-2.9907 * t) + 0.0008 * np.exp(-1.6836 * t) + 0.0601 * np.exp(-0.8459 * t) + 0.0519 * np.exp(-0.2159 * t) + 0.0548 * np.exp(-0.0000 * t))
    if N == 7:
        E_3 = 0.854
        return E_3*(0.0009 * np.exp(-7.0000 * t) + 0.0000 * np.exp(-4.8711 * t) + 0.0063 * np.exp(-3.3077 * t) + 0.0046 * np.exp(-2.1468 * t) + 0.0146 * np.exp(-1.2804 * t) + 0.0343 * np.exp(-0.6855 * t) + 0.0222 * np.exp(-0.0000 * t) + 0.0421 * np.exp(-0.1475 * t))
def E_4_all(N,t):
    if N == 3:
        return 0
    if N == 5:
        E_4 = -1.5095
        return E_4*(0.0036 * np.exp(-5.0000 * t) + -0.0078 * np.exp(-2.9907 * t) + -0.0893 * np.exp(-1.6836 * t) + 0.1001 * np.exp(-0.8459 * t) + -0.0076 * np.exp(-0.2159 * t) + 0.1677 * np.exp(-0.0000 * t))
    if N == 7:
        E_4 = 0.427
        return E_4*(-0.0009 * np.exp(-7.0000 * t) + 0.0001 * np.exp(-4.8711 * t) + -0.0017 * np.exp(-3.3077 * t) + 0.0101 * np.exp(-2.1468 * t) + 0.0020 * np.exp(-1.2804 * t) + 0.0516 * np.exp(-0.6855 * t) + 0.0340 * np.exp(-0.0000 * t) + 0.0298 * np.exp(-0.1475 * t))
def E_5_all(N,t):
    if N == 3:
        return 0
    if N == 5:
        E_5 = -4.5285
        return E_5*(-0.0007 * np.exp(-5.0000 * t) + 0.0027 * np.exp(-2.9907 * t) + 0.0587 * np.exp(-1.6836 * t) + -0.1557 * np.exp(-0.8459 * t) + -0.4249 * np.exp(-0.2159 * t) + 0.6866 * np.exp(-0.0000 * t))
    if N == 7:
        E_5 = -0.854
        return E_5*(0.0005 * np.exp(-7.0000 * t) + -0.0001 * np.exp(-4.8711 * t) + -0.0149 * np.exp(-3.3077 * t) + 0.0065 * np.exp(-2.1468 * t) + -0.0278 * np.exp(-1.2804 * t) + 0.0774 * np.exp(-0.6855 * t) + 0.0735 * np.exp(-0.0000 * t) + 0.0099 * np.exp(-0.1475 * t))
def E_6_all(N,t):
    if N == 3:
        return 0
    if N == 5:
        return 0
    if N == 7:
        E_6 = -2.989
        return E_6*(-0.0002 * np.exp(-7.0000 * t) + 0.0001 * np.exp(-4.8711 * t) + 0.0165 * np.exp(-3.3077 * t) + -0.0301 * np.exp(-2.1468 * t) + -0.0771 * np.exp(-1.2804 * t) + 0.0763 * np.exp(-0.6855 * t) + 0.2073 * np.exp(-0.0000 * t) + -0.0679 * np.exp(-0.1475 * t))
def E_7_all(N,t):
    if N == 3:
        return 0
    if N == 5:
        return 0
    if N == 7:
        E_7 = -5.978
        return E_7*(-0.0053 * np.exp(-3.3077 * t) + 0.0158 * np.exp(-2.1468 * t) + 0.0776 * np.exp(-1.2804 * t) + -0.2076 * np.exp(-0.6855 * t) + 0.5883 * np.exp(-0.0000 * t) + -0.3439 * np.exp(-0.1475 * t))
    
def E_all(N,t):
    return (E_0_all(N,t) + E_1_all(N,t) + E_2_all(N,t) + E_3_all(N,t) + E_4_all(N,t) + 
            E_5_all(N,t) + E_6_all(N,t) + E_7_all(N,t))
   

t_values = np.linspace(0, 100, 5000)

Energies_Star = []
Energies_All = []

Gibbs_Star = E_gibbs_star_values[N-3]
Gibbs_All = E_gibbs_all_values[N-3]    

Fisher_star = []
Fisher_all = []
Fisher_Star_Gibbs = []
Fisher_All_Gibbs = []
T_F_all = 0 
T_F_star = 0

for t in t_values:
    
    dp0_star_vals = derivada_num(ph_0_star, p_0_star, t)
    p_0_star_vals = p_0_star(t)
    fisher0_star = (dp0_star_vals**2)/p_0_star_vals
    
    dp1_star_vals = derivada_num(ph_1_star, p_1_star, t)
    p_1_star_vals = p_1_star(t)
    fisher1_star = (dp1_star_vals**2)/p_1_star_vals
    
    dp2_star_vals = derivada_num(ph_2_star, p_2_star, t)
    p_2_star_vals = p_2_star(t)
    fisher2_star = (dp2_star_vals**2)/p_2_star_vals
    
    dp3_star_vals = derivada_num(ph_3_star, p_3_star, t)
    p_3_star_vals = p_3_star(t)
    fisher3_star = (dp3_star_vals**2)/p_3_star_vals
    
    dp4_star_vals = derivada_num(ph_4_star, p_4_star, t)
    p_4_star_vals = p_4_star(t)
    
    if p_4_star_vals == 0:
        fisher4_star = 0
    else:
        fisher4_star = (dp4_star_vals**2) / p_4_star_vals
    
    dp5_star_vals = derivada_num(ph_5_star, p_5_star, t)
    p_5_star_vals = p_5_star(t)
    
    if p_5_star_vals == 0:
        fisher5_star = 0
    else:
        fisher5_star = (dp5_star_vals**2) / p_5_star_vals
    
    dp6_star_vals = derivada_num(ph_6_star, p_6_star, t)
    p_6_star_vals = p_6_star(t)
    
    if p_6_star_vals == 0:
        fisher6_star = 0
    else:
        fisher6_star = (dp6_star_vals**2) / p_6_star_vals
    
    dp7_star_vals = derivada_num(ph_7_star, p_7_star, t)
    p_7_star_vals = p_7_star(t)
    
    if p_7_star_vals == 0:
        fisher7_star = 0
    else:
        fisher7_star = (dp7_star_vals**2) / p_7_star_vals
    
    Fisher_Star_t = fisher0_star + fisher1_star + fisher2_star + fisher3_star + fisher4_star + fisher5_star + fisher6_star + fisher7_star    
    
    Fisher_star.append(Fisher_Star_t)
    
    dp0_all_vals = derivada_num(ph_0_all, p_0_all, t)
    p_0_all_vals = p_0_all(t)
    fisher0_all = (dp0_all_vals**2)/p_0_all_vals
    
    dp1_all_vals = derivada_num(ph_1_all, p_1_all, t)
    p_1_all_vals = p_1_all(t)
    fisher1_all = (dp1_all_vals**2)/p_1_all_vals
    
    dp2_all_vals = derivada_num(ph_2_all, p_2_all, t)
    p_2_all_vals = p_2_all(t)
    fisher2_all = (dp2_all_vals**2)/p_2_all_vals
    
    dp3_all_vals = derivada_num(ph_3_all, p_3_all, t)
    p_3_all_vals = p_3_all(t)
    fisher3_all = (dp3_all_vals**2)/p_3_all_vals
    
    dp4_all_vals = derivada_num(ph_4_all, p_4_all, t)
    p_4_all_vals = p_4_all(t)
    
    if p_4_all_vals == 0:
        fisher4_all = 0
    else:
        fisher4_all = (dp4_all_vals**2) / p_4_all_vals
    
    dp5_all_vals = derivada_num(ph_5_all, p_5_all, t)
    p_5_all_vals = p_5_all(t)
    
    if p_5_all_vals == 0:
        fisher5_all = 0
    else:
        fisher5_all = (dp5_all_vals**2) / p_5_all_vals
    
    dp6_all_vals = derivada_num(ph_6_all, p_6_all, t)
    p_6_all_vals = p_6_all(t)
    
    if p_6_all_vals == 0:
        fisher6_all = 0
    else:
        fisher6_all = (dp6_all_vals**2) / p_6_all_vals
    
    dp7_all_vals = derivada_num(ph_7_all, p_7_all, t)
    p_7_all_vals = p_7_all(t)
    
    if p_7_all_vals == 0:
        fisher7_all = 0
    else:
        fisher7_all = (dp7_all_vals**2) / p_7_all_vals
        
    Fisher_All_t = fisher0_all + fisher1_all + fisher2_all + fisher3_all + fisher4_all + fisher5_all + fisher6_all + fisher7_all  
    
    Fisher_all.append(Fisher_All_t)
    
    Fisher_Star_Gibbs.append(Fisher_Gibbs(1, t))
    Fisher_All_Gibbs.append(Fisher_Gibbs(2, t))
    
    E_star_t = E_star(N,t)
    Energies_Star.append(E_star_t)
    
    E_all_t = E_all(N,t)
    Energies_All.append(E_all_t)
       

fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

# First plot (Fisher Information)
line1, = ax1.plot(t_values, Fisher_star, label="Star Model", color='b')
line2, = ax1.plot(t_values, Fisher_all, label="All-To-All", color='g')
line3, = ax1.plot(t_values, Fisher_Star_Gibbs, linestyle='--', color='b')
line4, = ax1.plot(t_values, Fisher_All_Gibbs, linestyle='--', color='g')
line5, = ax1.plot(0, 0, label="Thermal states", linestyle='--', color='black', alpha=1)

ax1.set_ylabel("$\mathcal{F}$", fontsize = 14)
ax1.set_xlim(0, max(t_values))
ax1.set_xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 2)))
ax1.grid()

# Second plot (Energy)
ax2.plot(t_values, Energies_Star, color='b')
ax2.axhline(y=Gibbs_Star, linestyle='--', color='b')
ax2.plot(t_values, Energies_All, color='g')
ax2.axhline(y=Gibbs_All, linestyle='--', color='g')
ax2.plot(0, 0, linestyle='--', color='black', alpha=1)

ax1.set_xlabel("$t$", fontsize = 14)
ax2.set_ylabel("$E$", fontsize = 14)
ax2.set_xlim(0, max(t_values))
ax2.set_xscale('function', functions=(lambda x: np.power(x, 0.5), lambda x: np.power(x, 1/0.5)))
ax2.set_xticks([0, 20, 40, 60, 80])
ax2.set_xticklabels([str(val) for val in [0, 20, 40, 60, 80]])
ax2.grid()

# Unified legend at the top
fig.legend(handles=[line1, line2, line5], labels=["Star Model", "All-To-All", "Thermal states"],
           loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize = 11)

plt.tight_layout(rect=[0, 0, 1, 0.98])  # leave space for legend
plt.show()


