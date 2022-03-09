import sgmdata
from sgmdata import utilities



with open('./docs/Utilities.md', 'w') as f:
    doc = utilities.__doc__
    for lines in doc.split('\n'):
        f.write(f"{lines.lstrip()}\n")

with open('./docs/Load.md', 'w') as f:
    doc = sgmdata.__doc__
    for lines in doc.split('\n'):
        f.write(f"{lines.lstrip()}\n")