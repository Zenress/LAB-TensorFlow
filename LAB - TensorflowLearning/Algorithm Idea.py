import matplotlib.pyplot as plt
import numpy
import random
from scipy import stats

income_y = numpy.random.normal(272, 30, 1000)
years_x = numpy.random.uniform(1,1000,1000)

for x in income_y:
  random_nr = random.randrange(0,999)
  if x > 272:
    x * 0.975
    if income_y[random_nr] < 272:
      income_y[random_nr] * 1.025

slope, intercept, r,p,std_err = stats.linregress(years_x, income_y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, years_x))



plt.scatter(years_x, income_y)
plt.plot(years_x, mymodel)
plt.show()
