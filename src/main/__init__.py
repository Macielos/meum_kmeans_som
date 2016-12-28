"""

TODO
- wczytywanie danych - na wejściu plik, na wyjściu mapa/lista krotek nazwa->array danych
- odpalanie SOM i k-means - na wejsciu krotki jw, na wyjsciu lista klastrow
- statsy, grafy, itd.
"""

# plants
from data_loading.data_loader import SimpleDataLoader, SimpleWithDateDataLoader, LongTextDataLoader


def print_data(loader):
    print('loaded ' + str(len(loader.records)) + ' records')
    print('first 10 records: \n' + (', '.join(map(str, loader.records[0:10]))))
    print('last 10 records: \n' + (', '.join(map(str, loader.records[-10:-1]))))

print('loading data...')

plants_data_loader = SimpleDataLoader()
plants_data_loader.load('../data/plants')
print_data(plants_data_loader)

power_cons_data_loader = SimpleWithDateDataLoader(';')
power_cons_data_loader.load('../data/power_consumption')
print_data(power_cons_data_loader)

reuters_data_loader = LongTextDataLoader(' ')
reuters_data_loader.load('../data/reuters/C50train')
print_data(reuters_data_loader)

print('data loaded, now clustering magic begins (once its done...)')
