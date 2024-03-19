elements = ['EEMs', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
            'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y-L2,3',
            'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In-M4,5', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho-M4,5', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re',
            'Os', 'Zr-M2,3'
            ]

scan_lib = {k: "" for k in elements}
scan_lib.update({'EEMs': "run_eems 1 544", 'C': "gscan 270 320 col 320", 'N': "gscan 380 440 col 430",
                 'O': "gscan 520 570 col 544", 'F': "gscan 680 715 col 695", 'Ne': "gscan 840 900 col 868", 'Na': "gscan 1060 1110 col 1075",
                 'Mg': "gscan 1280 1350 col 1312", 'Al': "gscan 1550 1610 col 1568", 'Si': "gscan 1835 1875 col 1847",
                 'P': "gscan 2140 2200 col 2165", 'K': "gscan 295 305 col 300", 'Ca': "gscan 340 360 col 353", 'Ti': "gscan 450 480 col 461",
                 'Mn': "gscan 625 670 col 638", 'Fe': "gscan 690 750 col 711", 'Co': "gscan 765 810 col 780.5",
                 'Ni': "gscan 840 890 col 852.5", 'Cu': "gscan 920 970 col 934", 'Zn': "gscan 1000 1080 col 1060",
                 'Ge': "gscan 1200 1280 col 1260", 'Se': "gscan 1420 1500 col 1500", 'Br': "gscan 1500 1650 col 1565", 'Cr': "gscan 560 595 col 575",
                 'Ru': "gscan 450 505 col 464", 'La': "gscan 820 870 col 834", 'Ga': "gscan 1100 1180 col 1130", 'V': "gscan 500 530 col 523",
                 'In-M4,5': "gscan 435 470 col 455", 'W': "gscan 1790 1990 col 1970", 'Ho-M4,5': "gscan 1335 1415 col 1392", 
                 'Y-L2,3': "gscan 2050 2200 col 2085", 'Pd-M2,3': "gscan 510 590 col 530", 'Zr-M2,3': "gscan 315 365 col 334",
                 'Sc': "gscan 380 420 col 401.5"
                 })
