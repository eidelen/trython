import math

# print command and string

x = 1
if x == 1:
    print("true")
else:
    print("false")

print(12)

cStr = "aaa" + " " + "bbb" + " " + str(x);
print(cStr)


# multi init
a, b = 5,6
print( "a = " + str(a) + ",  b = " + str(b))

# float - print pi with increasing precision
for pos in range(20):
    formatStr = "%." + str(pos) + "f"
    print(formatStr % math.pi)

myFloat = int(1.0)
if isinstance( myFloat, int):
    print( "myFloat is int")
else:
    if isinstance( myFloat, float ):
        print( "myFloat is float")
    else:
        print( "myFloat is unknown")


# list

myMixedList = []
myMixedList.append(1)
myMixedList.append(2)
myMixedList.append(3.1)
myMixedList.append("abc")

for ele in myMixedList:
    print(ele)

arr1 = [0,1,2,3]
arr2 = [4,5,6,7]
connectedList = arr1 + arr2
print(connectedList)
print(arr1 * 2)

print((arr1 * 2).count(0)) # how many zeros ar in twice list arr1


# Accumlate strings -> cool
strAccum = "h" * 10
print(strAccum)

# more strings

str1 = "abcde"
print( str1[-1]) # prints the last character


# loops

for i in xrange(0,3):
    print(i)
else:
    print( "loop ended")


# functions

def sumTwo(a, b):
    return a + b;

print( sumTwo(1,2) )






