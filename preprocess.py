from scripts.preprocessing.nexar_preprocessing import preprocess
from os.path import expanduser
preprocess(nexar_root=expanduser('ROOT TO BDDNexar'),
           dataset_root='/data/datasets/processed')
