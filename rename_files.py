import os

dir='test/jsons/'

for i, filename in enumerate(os.listdir(dir)):
    os.rename(dir + str(i+1) + '.json' , dir + "/00" + str(i) + ".json")