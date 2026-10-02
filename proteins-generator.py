import numpy as np
import random
import string
import sys

letters = np.array(list(chr(ord('A') + i) for i in range(8)))
#print(letters)


f=open('proteins.csv', 'w+')

f.write("protid,enzyme,hydrofob,sequence\n")

print(sys.argv[1])

linesno=int(sys.argv[1])
seed=int(sys.argv[2])
count = 0

random.seed(seed)

for i in range (linesno):
   chars = ''.join([random.choice(letters) for j in range(random.randrange(1, 256, 8))])
#   print(chars)
#id
   f.write(str(i+1))
   f.write(",") 
#enzyme
   ecno=random.randint(1, 10)
   f.write(str(ecno))
   f.write(",") 
#hydrofob
   if ecno <3:
      hidro = random.randint(40, 75)
   elif ecno <6: 
      hidro = random.randint(55, 140)
   elif ecno <8: 
      hidro = random.randint(125, 175)
   else:
      hidro = random.randint(160, 220)
   f.write(str(hidro))
   f.write(",") 
#sequence
   if (random.randrange(1, 8, 1) ==   i % 8):
      count =  count +1
      f.write("ABCD")
      if (count % 2 ==   0):
#         print(i, count, "CDEFGH")
         f.write("CDEFGH")
      if (count % 4 ==   0):
#         print(i, count, "ABCD")
         f.write("ABCD")
   f.write(chars)
   f.write("\n")

