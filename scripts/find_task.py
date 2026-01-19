import sys, os
site_packages = [p for p in sys.path if 'site-packages' in p]
found = []
for d in site_packages:
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.lower() == 'hand_landmarker.task':
                found.append(os.path.join(root, f))
if found:
    print('FOUND', found[0])
else:
    print('NOT FOUND')
print('Searched site-packages:', site_packages)
