c=0
for x in range(-100, 100):
    for y in range(-100, 100):
        for z in range(-100, 100):
            if x*y*z==60:
                c+=1
print(c)