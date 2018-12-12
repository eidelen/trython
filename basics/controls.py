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
