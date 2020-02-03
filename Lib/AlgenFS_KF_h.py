
import os
import sys
dirX = os.getcwd()
print(dirX)
sys.path.append(dirX+"/lib")
import Algen_lib as lib
import numpy as np

import random

def features_selection(data, label, jumlah_populasi, jumlah_titik, pm,K, alpha, target, 
                        panjang_fitur, fitur, gen_size=10,
                        metode='tf',additional_child=True, pc=0.5, parent_selection=True):
    features_1_bin = [1 for i in range(panjang_fitur)]
    pop = lib.create_population(jumlah_populasi-1, panjang_fitur)
    pop.append(features_1_bin)
    # print(sum())
    fns_1 = lib.fitness_kf(data, label, features_1_bin, fitur, alpha = alpha, metode = metode, K=K)
    print("fitness all features:",fns_1)

    fitnes_pop = list()
    # temp_features_fitness = list()
    for features_bin in pop:
        fns = lib.fitness_kf(data, label, features_bin, fitur, alpha = alpha, metode = metode, K=K)
        fitnes_pop.append(fns)

    # print(len(fitnes_pop),"|",len(pop))

    print("-"*25)
    print("Generasi ke",1 , "[-]")
    print("  "+"___________________________________________")
    print("     Nilai Fitness       |Fitur|ALL  |Generasi")

    pop_used = dict()
    for fi, fe in zip(fitnes_pop, pop):
        print("   ",lib.tampil_finess(fi),"|",str(sum(fe))+"|"+str(len(fe)),"|","Populasi awal")
        key_str = lib.int_to_str(fe)
        pop_used.update({key_str:fi})

    
    popp_0 = list(pop)
    for p, fit in zip(popp_0, fitnes_pop):
        bin_str = lib.int_to_str(p) 
        pop_used.update({bin_str:fit})

    best_jumlah_fitur_list = list()
    gen=1
    while True:
        gen+=1
        print("-"*25)
        print("Generasi ke",gen , "[-]")
        print("  "+"____________________________________________")
        print("     Nilai Fitness       |Fitur|ALL  |Generasi")

        #roulette wheel
        rw = lib.get_roulette_wheel_(fitnes_pop, inc=jumlah_populasi)
        #Menentukan orang tua
        parents = lib.select_parents(rw)

        fitur_fitness_generasi = list()
        for p in parents:
            # best_generasi_list.append(gen)

            mama_index = p[0]
            papa_index = p[1]

            m4ms = fitnes_pop[mama_index]
            m4mr = list(pop[mama_index])
            fitur_fitness_generasi.append([m4mr, m4ms])

            p4ps = fitnes_pop[papa_index]
            p4pr = list(pop[papa_index])
            fitur_fitness_generasi.append([p4pr, p4ps])

            #crossover dan mutasi
            bin_mama = list(pop[mama_index])
            bin_papa = list(pop[papa_index])

            r = random.random()

            if r >= pc:
                anak_binary = lib.crossover(bin_mama, bin_papa, panjang_fitur,  jumlah_titik = jumlah_titik, prob_mutasi = pm)
                for ix, anak_bin in enumerate(anak_binary):
                    if sum(anak_bin)<=0:
                        anak_bin[-1]=1
                        anak_bin[-2]=1
                    str_a = lib.int_to_str(anak_bin)
                    if str_a in pop_used:
                        fitur_fitness_generasi.append([anak_bin, pop_used[str_a]])
                    else:
                        #mencari fitness untuk child
                        fitness_anak_ = lib.fitness_kf(data, label, anak_bin, fitur, alpha = alpha, metode = metode, K=K)
                        # family_fitness.append(fitness_anak_)
                        pop_used.update({str_a:fitness_anak_})
                        fitur_fitness_generasi.append([anak_bin, fitness_anak_])
                    if additional_child == True and ix==2:
                        break
        
        #Rangking fitness setiap individu
        sorted_gen = np.array(lib.sort_data(fitur_fitness_generasi, index=1)).T

        pop_kandidat = list(sorted_gen[0])
        fitnes_pop_kandidat = list(sorted_gen[1])

        str_kandidat_pop = list()
        pop_k1 = list()
        fitnes_pop_k1 = list()
        for fi, fe in zip(fitnes_pop_kandidat, pop_kandidat):
            str_k = lib.int_to_str(fe)
            if  lib.int_to_str(fe) not in str_kandidat_pop:
                str_kandidat_pop.append(str_k)
                pop_k1.append(fe)
                fitnes_pop_k1.append(fi)

        if parent_selection:
            pop = list(pop_k1[:jumlah_populasi])
            fitnes_pop = list(fitnes_pop_k1[:jumlah_populasi])
        else:
            pop = list(pop_k1)#[]:jumlah_populasi])
            fitnes_pop = list(fitnes_pop_k1)#[:jumlah_populasi])

        ix = 0
        for fi, fe in zip(fitnes_pop, pop):
            print("   ",lib.tampil_finess(fi),"|",str(sum(fe))+"|"+str(len(fe)),"|",gen)
            if ix==jumlah_populasi:
                break
            ix+=1

        if fitnes_pop[0]>=target or gen>=gen_size:
            print("="*70)
            print("     Best Fitness:",fitnes_pop[0])
            print("     Jumlah Fitur:",sum(pop[0]))
            print("Jumlah Fitur Asli:",len(pop[0]))
            dict_0919 = {
                'best_score':fitnes_pop[0],
                'best_fitur':np.array(fitur)[lib.get_index(pop[0])],
                'best_fitur_bin':np.array(pop),
            }
            return dict_0919