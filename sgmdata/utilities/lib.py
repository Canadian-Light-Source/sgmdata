elements = ['EEMs', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
            'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re',
            'Os'
            ]

scan_lib = {k: "" for k in elements}
scan_lib.update({'EEMs': "run_eems 1 544", 'C': "gscan 270 320 col 320", 'N': "gscan 380 440 col 430",
                 'O': "gscan 520 570 col 544", 'F': "gscan 675 705 col 685", 'Na': "gscan 1060 1110 col 1075",
                 'Mg': "gscan 1300 1350 col 1312", 'Al': "gscan 1550 1610 col 1568", 'Si': "gscan 1835 1875 col 1847",
                 'P': "", 'K': "", 'Ca': "gscan 340 360 col 353", 'Ti': "gscan 450 480 col 461",
                 'Mn': "gscan 625 670 col 638", 'Fe': "gscan 690 750 col 711", 'Co': "gscan 770 810 col 780.5",
                 'Ni': "gscan 840 890 col 852.5", 'Cu': "gscan 920 970 col 934", 'Zn': "gscan 1000 1080 col 1060",
                 'Ge': "", 'Se': "gscan 1420 1500 col 1500",
                 })
