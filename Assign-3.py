import validators
import urllib.request
tweets = []

# holds indices of cluster centroid
centroids = [] # size of list will be equal to k

# represents cluster number where each tweet belongs to
clas = [] # size of list will equal to number of tweets

# seperates tweet messages intially from given input file and stores all tweets in tweets list
def preprocess():
  url = 'https://drive.google.com/file/d/1LuskkgQdtHZI_gCqvjwxU8aksH1MewPe/view?usp=sharing'
  lines = urllib.request.urlopen(url)
  #fileName = wget.download(url)
  #with open(fileName) as f:
    #lines = []
    #lines = f.readlines()
  for line in lines:
    line = line.decode('utf-8')
    words = line.split("|")[-1].strip().split()
    tweet = []
    for word in words:
      if '@' in word or validators.url(word):
        continue
      tweet.append(word.replace('#','').replace(',','').replace('?','').replace('!','').lower())
    if len(tweet)>0:
      tweets.append(tweet)

# calculates distance between two tweets
def dist(lis1,lis2):
  dic ={}
  count = 0
  for elem in lis1:
    if elem in dic:
      dic[elem]=dic[elem]+1
    else:
      dic[elem]=1
  for elem in lis2:
    if elem in dic:
      count += 1
  return 1.0-(count/(len(lis1)+len(lis2)-count))

# function is used for finding breaking condition for KNN algo i.e when centroids won't changw
def check(prev):
  i=0
  while i<len(prev):
    if prev[i]!=centroids[i]:
      return False
    i = i+1
  return True

# assigns each tweet to particular cluster where they belong
def classify():
  global tweets
  global centroids
  global clas
  i=0
  clas = []
  while i<len(tweets):
    j=0
    min_dist = dist(tweets[i],tweets[centroids[j]])
    min_vert = centroids[j]
    j=j+1
    while j<len(centroids):
      temp = dist(tweets[i],tweets[centroids[j]])
      if temp< min_dist:
        min_dist = temp
        min_vert = centroids[j]
      j=j+1
    clas.append(min_vert)
    i=i+1

# finds centroid for cluster of points
def findcentroids():
  global tweets
  global centroids
  global clas
  j=0
  while j<len(centroids):
    cluster = []
    i=0
    while i<len(tweets):
      if clas[i]==centroids[j]:
        cluster.append(i)
      i=i+1
    min_sum = len(tweets)+3
    min_elem = -1
    for elem in cluster:
      sum = 0
      for rem in cluster:
        sum += dist(tweets[elem],tweets[rem])
      if sum<min_sum:
        min_sum = sum
        min_elem = elem
    centroids[j] = min_elem
    j=j+1

# finds SSE value
def SSE():
  i=0
  res = 0
  while i<len(tweets):
    val = dist(tweets[clas[i]],tweets[i])
    res += (val*val)
    i=i+1
  return res

# prints final output of tweets classification
def print_size():
  dic = {}
  for e in clas:
    if e in dic:
      dic[e] = dic[e]+1
    else:
      dic[e] = 1
  i=0
  while i<len(centroids):
    print("{}: {} tweets".format(i+1,dic[centroids[i]]))
    i=i+1

# runs KNN algorithm on tweets based on k value and returns SSE value for input k
def kmeans(k):
  global tweets
  global centroids
  global clas
  centroids = []
  clas = []
  print("Total no. of tweets are {}".format(len(tweets)))
  i=0
  p=1
  while i<k:
    centroids.append(i)
    i = i+1
  while True:
    classify()
    prev = centroids.copy()
    findcentroids()
    if check(prev):
      break
    p=p+1
  print("no. of iterations KNN algorithm ran before saturation are: ",p)
  return SSE()

k = [5,10,15,20,25,30,35]#,25,30,35,40]#,50,60,70,80,100,120,150,180]
preprocess()
for e in k:
  print(kmeans(e))
  print_size()