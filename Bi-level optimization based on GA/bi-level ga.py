import math
import random
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from deap import base, algorithms
from deap import creator
from deap import tools

# konstanty genetického algoritmu
P_KRIZENI = 0.9 # pravděpodobnost křížení
P_PROMENA = 0.1 # pravděpodobnost mutace
POPULATION_SIZE = 100 # maximální počet jedinců v populaci
MAX_GENERATIONS = 100 # maximální počet generací

# konstanty programu
TYP_AUTA_DODAVKA = 'D'
TYP_AUTA_PIKAP = 'P'
SEZNAM_AUT = [TYP_AUTA_DODAVKA,TYP_AUTA_PIKAP]
CENA_DORUCENI = 80  # cena, kterou účtuje společnost zákazníkovi
ODMENA_DORUCENI_DODAVKOU = 30 # odměna, kterou získává kurýr na dodávce
ODMENA_DORUCENI_PIKAPEM = 45 # odměna, kterou získává kurýr na pikapu
NAKLAD_DORUCENI_DODAVKOU = 10 # palivo a mzda na jednu dodávku (kč/km, kč/h, průměr)
NAKLAD_DORUCENI_PIKAPEM = 15 # palivo a mzda na jeden pikap (kč/km, kč/h, průměr)
VMIN_DODAVKA, VMAX_DODAVKA = 1, 35 # minimální a maximální počet balíků v dodávce
VMIN_PIKAP, VMAX_PIKAP = 1, 15 # minimální a maximální počet balíků v pikapu
CENA_DODAVKA = 20 # pořizovací cena dodávky
CENA_PIKAP = 18 # pořizovací cena pikapu
ROZPOCET = 200 # rozpočet
QMIN_DODAVKA, QMAX_DODAVKA = 0, math.ceil(ROZPOCET/CENA_DODAVKA) # minimální a maximální počet dodávek u společnosti
QMIN_PIKAP, QMAX_PIKAP = 0, math.ceil(ROZPOCET/CENA_PIKAP) # minimální a maximální počet pikapů u společnosti

# registrace funkce obsahující parametry, jez se budou optimalizovat pro auta
creator.create("FitnessMaxCars", base.Fitness, weights=(1.0,))
creator.create("IndividualCar", list, fitness=creator.FitnessMaxCars)

# registrace funkce obsahující parametry, jez se budou optimalizovat pro společností
creator.create("FitnessMaxCompanies", base.Fitness, weights=(1.0,))
creator.create("IndividualCompany", list, fitness=creator.FitnessMaxCompanies)

# cílová funkce kurýra
def oneMaxFitnessKuryr(individual):
    zisk_kuryra, prinos_dodavky, prinos_pikapu = 0, 0, 0 # iniciace proměnných
    typ_auta = individual[0] # zjištění typu auta, na němž jezdi
    pocet_doruceni = individual[1] # odvození dle vstupu počtu doručení timto autem
    # určeni přínosu auta na zaklade jeho typu
    if typ_auta == TYP_AUTA_DODAVKA:
        prinos_dodavky = (ODMENA_DORUCENI_DODAVKOU - NAKLAD_DORUCENI_DODAVKOU) * pocet_doruceni
    elif typ_auta == TYP_AUTA_PIKAP:
        prinos_pikapu = (ODMENA_DORUCENI_PIKAPEM - NAKLAD_DORUCENI_PIKAPEM) * pocet_doruceni
    zisk_kuryra = prinos_dodavky + prinos_pikapu # sečteni přínosu, (přínos + 0) - je zisk kurýra
    return zisk_kuryra, # dvojice ‘tuple’

def oneMaxFitnessSpolecnost(individual):
    zisk_spolecnosti, prinos_dodavky, prinos_pikapu = 0, 0, 0 # iniciace proměnných
    # Počítáme auta každého typu – potřeba pro kontrolu omezení
    count_dodavka = sum(1 for car in individual if car[0] == TYP_AUTA_DODAVKA)
    count_pikap = sum(1 for car in individual if car[0] == TYP_AUTA_PIKAP)

    # Je-li počet aut ‘dodávka’ nebo ‘pikap’ vyšší než přípustné číslo, vypisujeme pokutu na zisk společnosti
    if count_dodavka > QMAX_DODAVKA:
        zisk_spolecnosti -= 1000  # stanovme pokutu na zisku
    if count_pikap > QMAX_PIKAP:
        zisk_spolecnosti -= 1000  # stanovme pokutu na zisku

    for car in individual:
        typ_auta = car[0]
        pocet_doruceni = car[1]

        # počítáme zisk společnosti
        if typ_auta == TYP_AUTA_DODAVKA:
            prinos_dodavky = (CENA_DORUCENI - ODMENA_DORUCENI_DODAVKOU) * pocet_doruceni
        elif typ_auta == TYP_AUTA_PIKAP:
            prinos_pikapu = (CENA_DORUCENI - ODMENA_DORUCENI_PIKAPEM) * pocet_doruceni

        zisk_spolecnosti += prinos_dodavky + prinos_pikapu

        total_cost = sum(CENA_DODAVKA for car in individual if car[0] == TYP_AUTA_DODAVKA) + sum(
            CENA_PIKAP for car in individual if car[0] == TYP_AUTA_PIKAP)

        # pokud celkové náklady překročí rozpočet, aplikujeme pokutu na vhodnost
        if total_cost > ROZPOCET:
            zisk_spolecnosti -= 1000  # zde lze nastavit jakoukoli pokutu, kterou považujete za vhodnou

    return zisk_spolecnosti, # dvojice ‘tuple’

# souprava knihovny DEAP k provedeni genetického algoritmu
toolbox = base.Toolbox()

# definujeme funkci pro tvorbu auta dle vzoru [typ, vytížení]
def createIndividualCarTypeAndCargoVolume(type):
    if type == TYP_AUTA_DODAVKA:
        return [type, random.randint(VMIN_DODAVKA,VMAX_DODAVKA)]
    if type == TYP_AUTA_PIKAP:
        return [type, random.randint(VMIN_PIKAP,VMAX_PIKAP)]

# definujeme funkci pro tvorbu počtu aut společnosti dle vzoru [dodávky, pikapy]
def createIndividualCompanyCars():
    D = random.randint(QMIN_DODAVKA, QMAX_DODAVKA) # náhodný počet aut typu ‘Dodávka’
    P = random.randint(QMIN_PIKAP, QMAX_PIKAP) # náhodný počet aut typu ‘Pikap’
    dodavky = [createIndividualCarTypeAndCargoVolume(TYP_AUTA_DODAVKA) for _ in range(D)] # náhodně vytváříme vytíženi každé dodávce
    pikapy = [createIndividualCarTypeAndCargoVolume(TYP_AUTA_PIKAP) for _ in range(P)] # náhodné vytváříme vytíženi každému pikapu
    auta = dodavky + pikapy # sjednotíme seznamy dodávek a pikapu na jeden seznam
    return auta

toolbox.register("carAssign", createIndividualCompanyCars) # registrujeme v DEAP funkci vytvářeni aut včetně nakladu, viz výše
toolbox.register("individualCompanyCreator", tools.initIterate, creator.IndividualCompany, toolbox.carAssign) # registrujeme v DEAP funkci tvorby jednotlivé společnosti
toolbox.register("populationCompanyCreator", tools.initRepeat, list, toolbox.individualCompanyCreator) # registrujeme funkci pro tvorbu populace společnosti

populationCompanies = toolbox.populationCompanyCreator(n=POPULATION_SIZE) # vytváříme populaci společnosti
# procházíme každou společnost a vytváříme populaci aut z nich
populationCars = [creator.IndividualCar(auto) for spolecnost in [toolbox.carAssign() for _ in range(POPULATION_SIZE)] for auto in spolecnost]

# funkce uživatelského křížení
def customCrossover(ind1, ind2):
    # křížení pro typ auta, vyměníme místa
    ind1[0], ind2[0] = ind2[0], ind1[0]

    # křížení pro počet balíků – bereme průměr
    ind1[1] = (ind1[1] + ind2[1]) // 2
    ind2[1] = ind1[1]
    # před vracením hodnot kontrolujeme na překročení omezení dle typu auta počtu balíků, použijeme minimum z hodnoty nebo omezení
    if ind1[0] == TYP_AUTA_PIKAP:
        ind1[1] = min(ind1[1], VMAX_PIKAP)
    else:
        ind2[1] = min(ind2[1], VMAX_DODAVKA)

    return ind1, ind2

# funkce uživatelské mutace
def customMutation(individual, indpb):
    if random.random() < indpb: # pravděpodobnost mutace

        individual[0] = random.choice(SEZNAM_AUT) # mutace typu auta – náhodný výběr

    # mutace počtu doručení (balíků)
    if random.random() < indpb:
        if individual[0] == TYP_AUTA_DODAVKA:
            individual[1] = random.randint(VMIN_DODAVKA, VMAX_DODAVKA)
        elif individual[0] == TYP_AUTA_PIKAP:
            individual[1] = random.randint(VMIN_PIKAP, VMAX_PIKAP)

    return individual,

# registrace operátorů genetického algoritmu pro kurýra
toolbox.register("evaluate", oneMaxFitnessKuryr)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", customCrossover) # uživatelské křížení aut
toolbox.register("mutate", customMutation, indpb=P_PROMENA) # uživatelská mutace aut

# vypočteme přizpůsobivost každého individua v naše zformované populaci kurýrů
print("Ohodnocení aut...")
fitnessValuesCars = list(map(oneMaxFitnessKuryr, populationCars)) # funkce vytvoří seznam hodnot cílové funkce kurýrů pro každého jedince v současné populaci
# vlastnosti ‘values’, která je uložena ve třidě FitnessMaxCars pro auta předáme vypočtené hodnoty cílové funkce
for individual, fitnessValue in zip(populationCars, fitnessValuesCars):
    individual.fitness.values = fitnessValue

# registrujeme soupravu pro sledovaní statistik
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

# provádíme geneticky algoritmus pro populaci aut
populationCars, logbook = algorithms.eaSimple(populationCars, toolbox,
                                          cxpb=P_KRIZENI,
                                          mutpb=P_PROMENA,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          verbose=True)

# vypočteme přizpůsobivost každého individua v naši populaci společností
print("Ohodnocení společností...")
fitnessValuesCompanies = list(map(oneMaxFitnessSpolecnost, populationCompanies)) # funkce vytvoří seznam hodnot cílové funkce společností pro každého jedince v současné populaci
# vlastnosti ‘values’, která je uložena ve třidě FitnessMaxCompanies pro společností předáme vypočtené hodnoty cílové funkce
for individual, fitnessValue in zip(populationCompanies, fitnessValuesCompanies):
    individual.fitness.values = fitnessValue

# funkce uživatelského křížení
def customMutationCompany(individual, indpb):
    if random.random() < indpb: # pravděpodobnost mutace
        # náhodně vybereme auto
        car_index = random.randint(0, len(individual) - 1)
        car = individual[car_index]

        # ověříme na omezení poctu aut dle typu před mutaci
        count_dodavka = sum(1 for c in individual if c[0] == TYP_AUTA_DODAVKA)
        count_pikap = sum(1 for c in individual if c[0] == TYP_AUTA_PIKAP)

        # pokud počet aut vyhovuje omezení, provedeme mutaci
        if car[0] == TYP_AUTA_DODAVKA and count_dodavka < QMAX_DODAVKA:
            car[0] = random.choice(SEZNAM_AUT)
        elif car[0] == TYP_AUTA_PIKAP and count_pikap < QMAX_PIKAP:
            car[0] = random.choice(SEZNAM_AUT)

        # mutace poctu doručení (balíků)
        if car[0] == TYP_AUTA_DODAVKA:
            car[1] = random.randint(VMIN_DODAVKA, VMAX_DODAVKA)
        elif car[0] == TYP_AUTA_PIKAP:
            car[1] = random.randint(VMIN_PIKAP, VMAX_PIKAP)

    return individual,

# odebráni operátorů genetického algoritmu pro kurýra
toolbox.unregister("evaluate")
toolbox.unregister("select")
toolbox.unregister("mate")
toolbox.unregister("mutate")

# registrace operátorů genetického algoritmu pro společnost
toolbox.register("evaluate", oneMaxFitnessSpolecnost)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", customMutationCompany, indpb=P_PROMENA)

# provádíme geneticky algoritmus pro společnost
populationCompanies, logbook2 = algorithms.eaSimple(populationCompanies, toolbox,
                                          cxpb=P_KRIZENI,
                                          mutpb=P_PROMENA,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          verbose=True)

# zjistíme nejlepšího jedince (společnost) po evoluci
best_individual = tools.selBest(populationCompanies, 1)[0]
print("Nejlepší společnost je ", best_individual)

# vypočteme počet každého typu aut v nejlepší společnosti
count_dodavka = sum(1 for car in best_individual if car[0] == TYP_AUTA_DODAVKA)
count_pikap = sum(1 for car in best_individual if car[0] == TYP_AUTA_PIKAP)

print("Počet dodávek = ", count_dodavka)
print("Počet pikapů = ", count_pikap)

# určíme seznamy pro uchovávaní statistiky práce algoritmu, pro auta
maxFitnessValues = logbook.select("max") # maximální přizpůsobeni v současné populaci aut
meanFitnessValues = logbook.select("avg") # průměrné přizpůsobeni v současné populaci aut
# určíme seznamy pro uchovávaní statistiky práce algoritmu, pro společností
maxFitnessValues2 = logbook2.select("max") # maximální přizpůsobeni v současné populaci společností
meanFitnessValues2 = logbook2.select("avg") # průměrné přizpůsobeni v současné populaci společností

# auta
plot1 = plt.figure(1)
plt.plot(maxFitnessValues, color='gray')
plt.plot(meanFitnessValues, color='red')
plt.xlabel('Generace')
plt.ylabel('Maximální/průměrný zisk kurýra (přizpůsobivost)')
plt.title('Závislost maximální a střední přizpůsobivosti na generaci')

# společnosti
plot2 = plt.figure(2)
plt.plot(maxFitnessValues2, color='gray')
plt.plot(meanFitnessValues2, color='red')
plt.xlabel('Generace')
plt.ylabel('Maximální/průměrný zisk společnosti (přizpůsobivost)')
plt.title('Závislost maximální a střední přizpůsobivosti na generaci')
plt.show()